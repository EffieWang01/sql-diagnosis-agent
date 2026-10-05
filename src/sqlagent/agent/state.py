"""轨迹状态：记录一条完整的多轮分析过程。

轨迹是**唯一的真相来源**：
  - SFT 阶段：从轨迹里抽取 (上下文 → 下一步动作) 训练样本
  - RL 阶段：奖励函数读轨迹算分
  - 评测阶段：8 个指标全部从轨迹派生

因此这里的字段设计要同时满足三方需求，不要为了省事只存 messages。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..tools.base import ToolCall, ToolResult
from ..utils.sql import normalize_sql, sql_similarity

# 终止原因
STOP_FINAL_ANSWER = "final_answer"
STOP_MAX_TURNS = "max_turns"
STOP_MAX_SQL = "max_sql_calls"
STOP_MAX_CONTEXT = "max_context_calls"
STOP_NO_TOOL_CALL = "no_tool_call"
STOP_ERROR = "error"
STOP_BUDGET_CHARS = "budget_chars"


@dataclass
class Step:
    """一轮「模型输出 → 工具执行」。"""

    turn: int
    assistant_text: str = ""
    thinking: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def tool_name(self) -> str:
        return self.tool_call.name if self.tool_call else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "assistant_text": self.assistant_text,
            "thinking": self.thinking,
            "tool_call": (
                {
                    "name": self.tool_call.name,
                    "arguments": self.tool_call.arguments,
                    "malformed": self.tool_call.malformed,
                    "parse_error": self.tool_call.parse_error,
                }
                if self.tool_call
                else None
            ),
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "elapsed_s": round(self.elapsed_s, 3),
            "warnings": self.warnings,
        }


@dataclass
class Trajectory:
    """一条完整轨迹。"""

    task_id: str
    question: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    final_answer: dict[str, Any] | None = None
    terminated_reason: str = STOP_ERROR
    error: str | None = None

    # 预算
    max_turns: int = 10
    max_sql_calls: int = 8
    max_context_calls: int = 3
    max_chars: int = 60_000

    # 任务自带的参考答案（用于可验证奖励）
    reference: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------ 派生统计
    @property
    def executed_steps(self) -> list[Step]:
        """真正**被分发执行**的步骤（排除被预算拦截的）。

        语义很重要：``n_sql_calls`` 指的是真实执行的 SQL 次数，
        被预算拦下来的尝试不算——否则 SQL 执行率、冗余查询率这些指标会失真。
        """
        out = []
        for s in self.steps:
            if s.tool_result is None:
                continue
            if s.tool_result.error == "over_budget":
                continue
            out.append(s)
        return out

    @property
    def blocked_steps(self) -> list[Step]:
        """被预算拦截的步骤。"""
        return [
            s for s in self.steps
            if s.tool_result is not None and s.tool_result.error == "over_budget"
        ]

    @property
    def sql_calls(self) -> list[str]:
        """所有实际提交给沙箱的 SQL（按执行顺序，不含被拦截的）。"""
        return [
            str(s.tool_call.arguments.get("sql", ""))
            for s in self.executed_steps
            if s.tool_name == "execute_sql" and s.tool_call
        ]

    @property
    def n_sql_calls(self) -> int:
        return len(self.sql_calls)

    @property
    def n_context_calls(self) -> int:
        return sum(1 for s in self.executed_steps if s.tool_name == "retrieve_context")

    @property
    def n_turns(self) -> int:
        return len(self.steps)

    @property
    def illegal_sql_count(self) -> int:
        return sum(
            1 for s in self.executed_steps
            if s.tool_name == "execute_sql" and s.tool_result
            and s.tool_result.payload.get("error_type") == "illegal"
        )

    @property
    def execution_error_count(self) -> int:
        return sum(
            1 for s in self.executed_steps
            if s.tool_name == "execute_sql" and s.tool_result
            and s.tool_result.payload.get("error_type") == "execution"
        )

    @property
    def timeout_count(self) -> int:
        return sum(
            1 for s in self.executed_steps
            if s.tool_name == "execute_sql" and s.tool_result
            and s.tool_result.payload.get("error_type") == "timeout"
        )

    def duplicate_sql_indices(self, threshold: float = 0.95) -> list[int]:
        """找出「与之前某条语义重复」的 SQL 下标。"""
        seen: list[str] = []
        dups: list[int] = []
        for i, sql in enumerate(self.sql_calls):
            norm = normalize_sql(sql)
            if not norm:
                continue
            if any(
                sql_similarity(sql, prev, mask_numbers=False) >= threshold
                or normalize_sql(sql, mask_numbers=True)
                == normalize_sql(prev, mask_numbers=True)
                for prev in seen
            ):
                dups.append(i)
            else:
                seen.append(sql)
        return dups

    @property
    def duplicate_sql_count(self) -> int:
        return len(self.duplicate_sql_indices())

    @property
    def redundant_query_rate(self) -> float:
        return self.duplicate_sql_count / self.n_sql_calls if self.n_sql_calls else 0.0

    def recovery_pairs(self, min_similarity: float = 0.40) -> list[tuple[int, int, float]]:
        """找出「真正的错误恢复」：上一条 SQL 失败，下一条**相似**的 SQL 成功。

        为什么必须要求相似：如果只要「失败后随便来一条成功的」就算恢复，
        模型会学会「先故意写错一条，再写一条能跑的」来白拿恢复分。
        要求修复后的 SQL 与被拒/报错的 SQL 有实质重叠，
        才说明模型是在**修正**而不是**绕过**。

        返回 [(失败下标, 成功下标, 相似度)]。
        """
        sqls = self.sql_calls
        results = [
            s.tool_result
            for s in self.executed_steps
            if s.tool_name == "execute_sql" and s.tool_result
        ]
        pairs: list[tuple[int, int, float]] = []
        for i in range(1, len(sqls)):
            prev, cur = results[i - 1], results[i]
            if prev is None or cur is None:
                continue
            if prev.ok or not cur.ok:
                continue
            sim = sql_similarity(sqls[i - 1], sqls[i], mask_numbers=True)
            if sim >= min_similarity:
                pairs.append((i - 1, i, sim))
        return pairs

    def recovered_indices(self, min_similarity: float = 0.40) -> list[int]:
        """恢复成功的 SQL 下标。"""
        return [cur for _, cur, _ in self.recovery_pairs(min_similarity)]

    @property
    def recovery_count(self) -> int:
        return len(self.recovered_indices())

    @property
    def failed_sql_count(self) -> int:
        return self.illegal_sql_count + self.execution_error_count + self.timeout_count

    @property
    def over_budget(self) -> bool:
        return (
            self.n_sql_calls > self.max_sql_calls
            or self.n_turns > self.max_turns
            or self.n_context_calls > self.max_context_calls
        )

    @property
    def used_all_sql_budget(self) -> bool:
        return self.n_sql_calls >= self.max_sql_calls

    def char_length(self) -> int:
        return sum(len(str(m.get("content", ""))) for m in self.messages)

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "messages": self.messages,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "terminated_reason": self.terminated_reason,
            "error": self.error,
            "budget": {
                "max_turns": self.max_turns,
                "max_sql_calls": self.max_sql_calls,
                "max_context_calls": self.max_context_calls,
                "max_chars": self.max_chars,
            },
            "stats": {
                "n_turns": self.n_turns,
                "n_sql_calls": self.n_sql_calls,
                "n_context_calls": self.n_context_calls,
                "illegal_sql_count": self.illegal_sql_count,
                "execution_error_count": self.execution_error_count,
                "timeout_count": self.timeout_count,
                "duplicate_sql_count": self.duplicate_sql_count,
                "recovery_count": self.recovery_count,
                "redundant_query_rate": round(self.redundant_query_rate, 4),
                "over_budget": self.over_budget,
                "char_length": self.char_length(),
            },
            "reference": self.reference,
            "meta": self.meta,
            "created_at": self.created_at,
        }
