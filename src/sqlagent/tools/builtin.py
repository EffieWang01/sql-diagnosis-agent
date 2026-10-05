"""三个内置工具：retrieve_context / execute_sql / final_answer。

刻意只保留这三个（技术方案的明确决定）：
  - 不设独立的 search_schema / describe_tables —— 表结构直接进 system prompt，
    列画像走 retrieve_context，避免工具数量膨胀导致小模型选择困难。
"""

from __future__ import annotations

from typing import Any

from ..env.context import ContextStore
from ..env.sandbox import SQLSandbox
from .base import BaseTool, ToolResult


# ====================================================================== #
class RetrieveContextTool(BaseTool):
    """检索业务指标定义与已验证 SQL。"""

    name = "retrieve_context"
    description = (
        "检索业务语义上下文，包括指标口径定义、字段含义、表间关系，以及历史上已验证的查询范式。"
        "当你**不确定某个业务指标的统计口径**（例如 GMV 是否含退款、活跃用户如何定义），"
        "或**不确定该用哪张表哪个字段**时调用。"
        "注意：本工具只返回定义与范式，不返回任何业务数值——具体数字必须自己写 SQL 查。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词，例如「GMV 口径」「华东区 定义」「退款」",
            },
            "top_k": {
                "type": "integer",
                "description": "返回条数，默认 3，最多 5",
            },
        },
        "required": ["query"],
    }

    def __init__(self, store: ContextStore) -> None:
        self.store = store

    def call(self, query: str = "", top_k: int = 3, **_: Any) -> ToolResult:
        if not query.strip():
            return ToolResult(
                ok=False,
                content="query 不能为空，请给出要检索的业务术语。",
                error="empty_query",
            )
        k = max(1, min(int(top_k or 3), 5))
        hits = self.store.search(query, top_k=k)
        text = self.store.render(hits)
        return ToolResult(
            ok=True,
            content=text,
            payload={
                "query": query,
                "hits": [
                    {"id": e.id, "kind": e.kind, "title": e.title, "score": round(s, 4)}
                    for e, s in hits
                ],
                "num_hits": len(hits),
            },
        )


# ====================================================================== #
class ExecuteSQLTool(BaseTool):
    """在只读 DuckDB 沙箱中执行 SQL。"""

    name = "execute_sql"
    description = (
        "在只读 DuckDB 沙箱中执行一条 SQL 并返回结果表格。"
        "支持 SELECT / WITH / DESCRIBE / SUMMARIZE 等只读语句，"
        "不支持任何写操作，也不能读取外部文件。"
        "一次只能提交一条语句。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "要执行的单条只读 SQL 语句",
            },
            "purpose": {
                "type": "string",
                "description": "这条 SQL 想验证什么假设（一句话，便于复盘与证据追溯）",
            },
        },
        "required": ["sql"],
    }

    def __init__(self, sandbox: SQLSandbox, observation_max_chars: int = 2000) -> None:
        self.sandbox = sandbox
        self.observation_max_chars = observation_max_chars

    def call(self, sql: str = "", purpose: str = "", **_: Any) -> ToolResult:
        if not sql.strip():
            return ToolResult(
                ok=False, content="sql 不能为空。", error="empty_sql",
            )
        res = self.sandbox.run(sql)
        payload = res.to_dict()
        payload["purpose"] = purpose
        return ToolResult(
            ok=res.ok,
            content=res.to_observation(self.observation_max_chars),
            payload=payload,
            error=res.error,
        )


# ====================================================================== #
class FinalAnswerTool(BaseTool):
    """提交最终诊断结论，终止循环。"""

    name = "final_answer"
    description = (
        "提交最终的业务异常诊断结论并结束本次分析。"
        "只有在已经用 SQL 收集到足够证据后才调用。"
        "key_findings 中每一条都必须附带你实际执行过的 evidence_sql，"
        "系统会校验该 SQL 是否真的执行过、数值是否对得上。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": "最终结论：一段话说明业务异常是什么、根因在哪",
            },
            "root_cause_dimension": {
                "type": "string",
                "description": "根因所在的维度或切片，例如「手机品类」「华东区-新客」",
            },
            "key_findings": {
                "type": "array",
                "description": "支撑结论的关键发现，每条都要有数值和证据 SQL",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "发现陈述"},
                        "value": {"type": "number", "description": "支撑该发现的关键数值"},
                        "evidence_sql": {
                            "type": "string",
                            "description": "产出该数值的 SQL（须与实际执行过的一致）",
                        },
                    },
                    "required": ["claim", "value", "evidence_sql"],
                },
            },
            "confidence": {
                "type": "number",
                "description": "对结论的置信度，0~1",
            },
        },
        "required": ["diagnosis", "root_cause_dimension", "key_findings"],
    }

    def call(
        self,
        diagnosis: str = "",
        root_cause_dimension: str = "",
        key_findings: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        **_: Any,
    ) -> ToolResult:
        missing = [
            k
            for k, v in (
                ("diagnosis", diagnosis),
                ("root_cause_dimension", root_cause_dimension),
                ("key_findings", key_findings),
            )
            if not v
        ]
        if missing:
            return ToolResult(
                ok=False,
                content=(
                    f"final_answer 缺少必填字段：{', '.join(missing)}。"
                    f"请补全后重新调用。"
                ),
                error="missing_fields",
            )

        findings = list(key_findings or [])
        for i, f in enumerate(findings):
            if not isinstance(f, dict) or "claim" not in f or "value" not in f:
                return ToolResult(
                    ok=False,
                    content=(
                        f"key_findings[{i}] 格式不正确，"
                        f"必须是含 claim / value / evidence_sql 的对象。"
                    ),
                    error="bad_finding",
                )

        payload = {
            "diagnosis": diagnosis,
            "root_cause_dimension": root_cause_dimension,
            "key_findings": findings,
            "confidence": float(confidence) if confidence is not None else None,
        }
        return ToolResult(
            ok=True,
            content="诊断结论已提交，分析结束。",
            payload=payload,
            terminal=True,
        )


# ====================================================================== #
def build_default_tools(
    sandbox: SQLSandbox,
    store: ContextStore,
    observation_max_chars: int = 2000,
) -> list[BaseTool]:
    """装配技术方案规定的三个工具。"""
    return [
        RetrieveContextTool(store),
        ExecuteSQLTool(sandbox, observation_max_chars=observation_max_chars),
        FinalAnswerTool(),
    ]
