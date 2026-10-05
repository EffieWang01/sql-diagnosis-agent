"""评测指标：全部从轨迹派生，不额外跑模型。

8 个主指标（与技术方案一致）：
  1. Task Success Rate      —— 任务成功率
  2. Root Cause Accuracy    —— 根因维度命中率
  3. Evidence Faithfulness  —— 证据忠实度（声明数值能否追溯到真实查询结果）
  4. SQL Execution Rate     —— SQL 执行成功率
  5. Recovery Success Rate  —— 出错后恢复成功率
  6. Long-horizon SR        —— 长轨迹（SQL 调用次数多）上的成功率
  7. Average Tool Calls     —— 平均工具调用次数
  8. Redundant Query Rate   —— 冗余查询率
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Iterable

from ..agent.state import Trajectory
from ..reward.verifier import RewardBreakdown, Verifier


@dataclass
class MetricReport:
    n: int = 0
    values: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "metrics": {k: round(v, 4) for k, v in self.values.items()},
            "extra": self.extra,
        }

    def to_markdown(self) -> str:
        lines = [f"样本数：{self.n}", "", "| 指标 | 数值 |", "| --- | --- |"]
        for k, v in self.values.items():
            lines.append(f"| {k} | {v:.4f} |")
        if self.extra:
            lines.append("")
            lines.append("**补充统计**")
            lines.append("")
            for k, v in self.extra.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v}")
        return "\n".join(lines)


def _safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return mean(xs) if xs else 0.0


def evaluate(
    trajs: list[Trajectory],
    verifier: Verifier | None = None,
    long_horizon_min_sql: int = 5,
    breakdowns: list[RewardBreakdown] | None = None,
) -> MetricReport:
    """对一批轨迹算指标。

    传入 ``breakdowns`` 可避免重复计算奖励。
    """
    verifier = verifier or Verifier()
    if breakdowns is None:
        breakdowns = [verifier.score(t) for t in trajs]

    rep = MetricReport(n=len(trajs))
    if not trajs:
        return rep

    # 1. Task Success Rate
    rep.values["task_success_rate"] = _safe_mean(
        1.0 if b.task_success else 0.0 for b in breakdowns
    )

    # 2. Root Cause Accuracy（只在有参考答案的样本上统计）
    diag = [
        b.details["diagnosis_match"]
        for b in breakdowns
        if b.details.get("diagnosis_match") is not None
    ]
    rep.values["root_cause_accuracy"] = _safe_mean(diag)
    rep.extra["n_with_reference"] = len(diag)

    # 3. Evidence Faithfulness
    ev = [
        b.components.get("evidence", 0.0) / verifier.cfg.w_evidence
        for b in breakdowns
        if verifier.cfg.w_evidence
    ]
    rep.values["evidence_faithfulness"] = _safe_mean(ev)

    # 4. SQL Execution Rate
    total_sql = sum(t.n_sql_calls for t in trajs)
    ok_sql = 0
    for t in trajs:
        for s in t.executed_steps:
            if s.tool_name == "execute_sql" and s.tool_result and s.tool_result.ok:
                ok_sql += 1
    rep.values["sql_execution_rate"] = ok_sql / total_sql if total_sql else 0.0
    rep.extra["total_sql_calls"] = total_sql

    # 5. Recovery Success Rate（分母 = 出现过 SQL 失败的轨迹数）
    #    注意：恢复的判定要求「修复后的 SQL 与被拒的 SQL 相似」，
    #    否则模型可以靠「先乱写一条再写条对的」白拿分。
    with_failure = [t for t in trajs if t.failed_sql_count > 0]
    recovered = [t for t in with_failure if t.recovery_count > 0]
    rep.values["recovery_success_rate"] = (
        len(recovered) / len(with_failure) if with_failure else 0.0
    )
    rep.extra["n_traj_with_sql_failure"] = len(with_failure)

    # 6. Long-horizon SR
    long_trajs = [t for t in trajs if t.n_sql_calls >= long_horizon_min_sql]
    long_idx = [i for i, t in enumerate(trajs) if t.n_sql_calls >= long_horizon_min_sql]
    rep.values["long_horizon_sr"] = _safe_mean(
        1.0 if breakdowns[i].task_success else 0.0 for i in long_idx
    )
    rep.extra["n_long_horizon"] = len(long_trajs)
    rep.extra["long_horizon_min_sql"] = long_horizon_min_sql

    # 7. Average Tool Calls
    rep.values["avg_tool_calls"] = _safe_mean(t.n_turns for t in trajs)
    rep.values["avg_sql_calls"] = _safe_mean(t.n_sql_calls for t in trajs)
    rep.values["avg_context_calls"] = _safe_mean(t.n_context_calls for t in trajs)
    rep.extra["median_sql_calls"] = median([t.n_sql_calls for t in trajs]) if trajs else 0

    # 8. Redundant Query Rate
    rep.values["redundant_query_rate"] = _safe_mean(t.redundant_query_rate for t in trajs)

    # ---- 补充诊断指标 ----
    rep.values["illegal_sql_rate"] = (
        sum(t.illegal_sql_count for t in trajs) / total_sql if total_sql else 0.0
    )
    rep.values["timeout_rate"] = (
        sum(t.timeout_count for t in trajs) / total_sql if total_sql else 0.0
    )
    rep.values["final_answer_rate"] = _safe_mean(
        1.0 if t.final_answer is not None else 0.0 for t in trajs
    )
    rep.values["avg_reward"] = _safe_mean(b.total for b in breakdowns)
    rep.extra["terminated_reason"] = dict(
        Counter(t.terminated_reason for t in trajs)
    )
    rep.extra["over_budget_count"] = sum(1 for t in trajs if t.over_budget)

    return rep
