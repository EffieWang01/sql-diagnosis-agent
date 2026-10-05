"""数据层：CSV → DuckDB 装载，以及 schema / 列画像抽取。

约定（来自技术方案）：
  - **CSV 只是原始载体**，Agent 的唯一数据入口是 SQL。
  - 表结构（DDL）进 system prompt；列画像（取值分布）进语义上下文，
    但**业务数值绝不进上下文**，避免污染可验证奖励。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from ..utils.io import ensure_dir


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_ratio: float
    distinct_count: int
    sample_values: list[Any] = field(default_factory=list)

    def to_text(self) -> str:
        s = f"  - {self.name} ({self.dtype}): 空值率 {self.null_ratio:.1%}, 去重 {self.distinct_count} 个"
        if self.sample_values:
            vals = ", ".join(str(v) for v in self.sample_values[:6])
            s += f", 示例: {vals}"
        return s


@dataclass
class TableProfile:
    name: str
    row_count: int
    columns: list[ColumnProfile] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"表 `{self.name}`（{self.row_count} 行）"]
        lines.extend(c.to_text() for c in self.columns)
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
def load_csvs_to_duckdb(
    raw_dir: str | Path,
    db_path: str | Path,
    overwrite: bool = True,
    types: dict[str, dict[str, str]] | None = None,
) -> list[str]:
    """把 raw_dir 下所有 CSV 载入 DuckDB，每张表名 = 文件名（去扩展名）。

    Args:
        types: 可选的列类型覆盖，形如 ``{"sales": {"order_date": "DATE"}}``。
            不传时交给 DuckDB 自动推断。**真实数据集建议显式声明**，
            否则推断结果可能随数据变化而漂移，导致 prompt 里的口径与实际不符。

    返回装载的表名列表。
    """
    raw_dir, db_path = Path(raw_dir), Path(db_path)
    ensure_dir(db_path.parent)
    if overwrite and db_path.exists():
        db_path.unlink()

    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"{raw_dir} 下没有找到 CSV 文件。")

    con = duckdb.connect(str(db_path))
    tables: list[str] = []
    try:
        for csv in csvs:
            table = csv.stem
            type_spec = (types or {}).get(table)
            if type_spec:
                col_list = ", ".join(f"'{k}': '{v}'" for k, v in type_spec.items())
                reader = (
                    f"read_csv_auto(?, header=true, sample_size=20000, "
                    f"types={{{col_list}}})"
                )
            else:
                reader = "read_csv_auto(?, header=true, sample_size=20000)"
            con.execute(
                f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM {reader}',
                [str(csv)],
            )
            tables.append(table)
    finally:
        con.close()
    return tables


# ---------------------------------------------------------------------- #
def column_types(db_path: str | Path, table: str) -> dict[str, str]:
    """取某张表的 {列名: 类型}。供 prompt / 上下文按真实类型生成用法提示。"""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return {c[0]: c[1] for c in con.execute(f'DESCRIBE "{table}"').fetchall()}
    finally:
        con.close()


# ---------------------------------------------------------------------- #
def schema_ddl(db_path: str | Path, tables: list[str] | None = None) -> str:
    """生成给模型看的 DDL 文本（不含数据）。"""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        names = tables or [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        blocks: list[str] = []
        for t in names:
            cols = con.execute(f'DESCRIBE "{t}"').fetchall()
            col_lines = ",\n".join(
                f'  "{c[0]}" {c[1]}' + ("  -- 可空" if c[2] == "YES" else "")
                for c in cols
            )
            blocks.append(f'CREATE TABLE "{t}" (\n{col_lines}\n);')
        return "\n\n".join(blocks)
    finally:
        con.close()


def profile_tables(
    db_path: str | Path,
    tables: list[str] | None = None,
    sample_n: int = 5,
    max_distinct_scan: int = 100_000,
) -> list[TableProfile]:
    """抽取列画像。

    注意：这里读的是**结构信息**（类型、空值率、基数、示例取值），
    不是业务聚合结果，因此可以安全地进上下文。
    """
    con = duckdb.connect(str(db_path), read_only=True)
    profiles: list[TableProfile] = []
    try:
        names = tables or [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        for t in names:
            total = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            cols = con.execute(f'DESCRIBE "{t}"').fetchall()
            cps: list[ColumnProfile] = []
            for c in cols:
                name, dtype = c[0], c[1]
                q = f'SELECT COUNT(*) FROM "{t}" WHERE "{name}" IS NULL'
                nulls = con.execute(q).fetchone()[0]
                null_ratio = (nulls / total) if total else 0.0

                distinct = -1
                if total <= max_distinct_scan:
                    distinct = con.execute(
                        f'SELECT COUNT(DISTINCT "{name}") FROM "{t}"'
                    ).fetchone()[0]

                samples: list[Any] = []
                try:
                    rows = con.execute(
                        f'SELECT DISTINCT "{name}" FROM "{t}" '
                        f'WHERE "{name}" IS NOT NULL LIMIT {int(sample_n)}'
                    ).fetchall()
                    samples = [r[0] for r in rows]
                except Exception:
                    pass

                cps.append(
                    ColumnProfile(
                        name=name, dtype=dtype, null_ratio=null_ratio,
                        distinct_count=distinct, sample_values=samples,
                    )
                )
            profiles.append(TableProfile(name=t, row_count=total, columns=cps))
        return profiles
    finally:
        con.close()
