"""DuckDB 只读沙箱。

职责边界（很重要，别把它写成万能执行器）：
  1. **只读**：连接层 read_only + 语句白名单 + 禁关键字，三层防护。
  2. **可控**：超时中断、行数截断，保证单次查询不会拖垮轨迹预算。
  3. **可诊断**：把失败区分为 illegal / timeout / execution 三类，
     这个区分直接喂给奖励函数（非法 SQL 扣分、执行失败但可恢复则加分）。

它不做的事：不做语义校验、不判断 SQL 是否「答对了业务问题」——
那是奖励模块的职责。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from ..config import SandboxConfig
from ..utils.sql import (
    find_forbidden,
    find_forbidden_function,
    first_keyword,
    split_statements,
    strip_sql_noise,
)

# 错误分类常量
ERR_ILLEGAL = "illegal"
ERR_TIMEOUT = "timeout"
ERR_EXECUTION = "execution"


class SandboxError(Exception):
    """沙箱内部错误基类。"""

    error_type = ERR_EXECUTION


class IllegalStatementError(SandboxError):
    error_type = ERR_ILLEGAL


class QueryTimeoutError(SandboxError):
    error_type = ERR_TIMEOUT


@dataclass
class QueryResult:
    """一次 SQL 执行的完整结果。既是工具返回值，也是轨迹日志的一条记录。"""

    ok: bool
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0
    error: str | None = None
    error_type: str | None = None

    # -------------------------------------------------------------- #
    def to_observation(self, max_chars: int = 2000) -> str:
        """渲染成回灌给模型的观测文本。

        格式刻意保持紧凑且稳定——SFT 阶段模型要学的就是「看到这种格式后如何决策」。
        """
        if not self.ok:
            label = {
                ERR_ILLEGAL: "SQL 被拒绝（只读沙箱不允许该语句）",
                ERR_TIMEOUT: "查询超时",
                ERR_EXECUTION: "SQL 执行报错",
            }.get(self.error_type or "", "执行失败")
            return f"[{label}] {self.error}\n请修正后重试。"

        if self.row_count == 0:
            return "查询成功，但结果为空（0 行）。"

        head = " | ".join(str(c) for c in self.columns)
        lines = [head, "-" * min(len(head), 80)]
        for r in self.rows:
            lines.append(" | ".join(_fmt_cell(v) for v in r))
        body = "\n".join(lines)

        tail = f"\n共 {self.row_count} 行"
        if self.truncated:
            tail += f"（已截断，仅显示前 {len(self.rows)} 行）"
        tail += f"，耗时 {self.elapsed_ms:.0f}ms"

        text = body + tail
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...（输出过长，已截断至 {max_chars} 字符）"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
            "error_type": self.error_type,
        }


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        # 保留有效位但去掉浮点噪声
        return f"{v:.6g}"
    return str(v)


class SQLSandbox:
    """DuckDB 只读沙箱。

    用法::

        with SQLSandbox("data/db/main.duckdb", cfg.sandbox) as sb:
            res = sb.run("SELECT region, SUM(gmv) FROM sales GROUP BY 1")
            print(res.to_observation())
    """

    def __init__(
        self,
        db_path: str | Path,
        config: SandboxConfig | None = None,
        read_only: bool | None = None,
    ) -> None:
        self.cfg = config or SandboxConfig()
        self.db_path = str(db_path)
        # 内存库无法只读打开；文件库默认强制只读
        if read_only is None:
            read_only = self.db_path not in (":memory:", "") and Path(self.db_path).exists()
        self.read_only = read_only

        connect_kwargs: dict[str, Any] = {"read_only": self.read_only}
        try:
            self._con = duckdb.connect(self.db_path, **connect_kwargs)
        except duckdb.Error:
            # 极少数情况（如空文件）只读打开失败，退回可写但靠语句白名单兜底
            self.read_only = False
            self._con = duckdb.connect(self.db_path)

        # 关闭外部访问：即使白名单被绕过，也读不到磁盘/网络
        self.external_access_disabled = self._try_disable_external_access()

        # 运行期统计（供评测指标使用）
        self.stats: dict[str, int] = {
            "total": 0,
            "ok": 0,
            "illegal": 0,
            "timeout": 0,
            "execution_error": 0,
        }

    # ------------------------------------------------------------------ #
    def _try_disable_external_access(self) -> bool:
        for stmt in (
            "SET enable_external_access=false",
            "SET disabled_filesystems='LocalFileSystem'",
        ):
            try:
                self._con.execute(stmt)
            except Exception:
                continue
        try:
            v = self._con.execute(
                "SELECT current_setting('enable_external_access')"
            ).fetchone()
            return str(v[0]).lower() in ("false", "0")
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def validate(self, sql: str) -> None:
        """语句合法性校验。不通过则抛 IllegalStatementError。"""
        if not sql or not sql.strip():
            raise IllegalStatementError("SQL 为空。")

        stmts = split_statements(sql)
        if len(stmts) > 1:
            raise IllegalStatementError(
                f"一次只能执行一条语句，检测到 {len(stmts)} 条。请拆开分次调用。"
            )

        kw = first_keyword(sql)
        if kw not in self.cfg.allowed_statements:
            raise IllegalStatementError(
                f"语句类型 `{kw or 'UNKNOWN'}` 不被允许。"
                f"只读沙箱仅支持：{', '.join(self.cfg.allowed_statements)}。"
            )

        hit = find_forbidden(sql, self.cfg.forbidden_keywords)
        if hit:
            raise IllegalStatementError(
                f"SQL 含被禁关键字 `{hit}`。本沙箱为只读环境，不支持写操作或环境变更。"
            )

        hit_fn = find_forbidden_function(sql, self.cfg.forbidden_functions)
        if hit_fn:
            raise IllegalStatementError(
                f"SQL 含被禁函数 `{hit_fn}`。禁止访问外部文件或网络，"
                f"请只使用已加载的表。"
            )

    # ------------------------------------------------------------------ #
    def run(self, sql: str) -> QueryResult:
        """执行一条 SQL，永不抛异常——所有失败都编码进 QueryResult。

        这样上层（AgentLoop / 奖励）不需要 try/except，也保证轨迹日志完整。
        """
        self.stats["total"] += 1
        t0 = time.perf_counter()

        try:
            self.validate(sql)
        except IllegalStatementError as e:
            self.stats["illegal"] += 1
            return QueryResult(
                ok=False, sql=sql, error=str(e), error_type=ERR_ILLEGAL,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        cur = self._con.cursor()
        try:
            columns, rows, truncated = self._execute_with_timeout(cur, sql)
        except QueryTimeoutError as e:
            self.stats["timeout"] += 1
            return QueryResult(
                ok=False, sql=sql, error=str(e), error_type=ERR_TIMEOUT,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:  # duckdb.Error 及其它
            self.stats["execution_error"] += 1
            return QueryResult(
                ok=False, sql=sql, error=f"{type(e).__name__}: {e}",
                error_type=ERR_EXECUTION,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        finally:
            try:
                cur.close()
            except Exception:
                pass

        self.stats["ok"] += 1
        return QueryResult(
            ok=True, sql=sql, columns=columns, rows=rows,
            row_count=len(rows) if not truncated else self.cfg.max_rows,
            truncated=truncated,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    # ------------------------------------------------------------------ #
    def _execute_with_timeout(self, cur: duckdb.DuckDBPyConnection, sql: str):
        """在工作线程里执行，超时用 cur.interrupt() 中断。

        DuckDB 的 Python 接口没有原生 query timeout，interrupt() 是官方推荐做法。
        工作线程用独立 cursor，避免与主连接争用。
        """
        box: dict[str, Any] = {}

        def worker() -> None:
            try:
                cur.execute(sql)
                desc = cur.description
                # 多取一行用于判断是否被截断
                rows = cur.fetchmany(self.cfg.max_rows + 1)
                box["desc"] = desc
                box["rows"] = rows
            except BaseException as e:  # noqa: BLE001 - 需捕获并转交主线程
                box["err"] = e

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        th.join(self.cfg.timeout_s)

        if th.is_alive():
            try:
                cur.interrupt()
            except Exception:
                pass
            th.join(2.0)
            raise QueryTimeoutError(
                f"查询超时：超过 {self.cfg.timeout_s:g} 秒仍未返回，已中断。"
                f"请缩小扫描范围（加 WHERE 过滤 / 减少 JOIN / 先做聚合）。"
            )

        if "err" in box:
            raise box["err"]

        desc = box.get("desc") or []
        raw = box.get("rows") or []
        columns = [d[0] for d in desc]
        truncated = len(raw) > self.cfg.max_rows
        rows = [list(r) for r in raw[: self.cfg.max_rows]]
        return columns, rows, truncated

    # ------------------------------------------------------------------ #
    def tables(self) -> list[str]:
        try:
            r = self._con.execute("SHOW TABLES").fetchall()
            return [x[0] for x in r]
        except Exception:
            return []

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    def __enter__(self) -> "SQLSandbox":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"SQLSandbox(db={self.db_path!r}, read_only={self.read_only}, "
            f"max_rows={self.cfg.max_rows}, timeout={self.cfg.timeout_s}s)"
        )
