"""可验证奖励（Verifiable Reward）。

奖励必须**只依赖可机械验证的事实**，不引入任何模型打分：
  - 结论对不对 → 与参考答案的根因维度比对
  - 数值对不对 → 与参考答案的关键数值在容差内比对
  - 证据实不实 → 该 SQL 是否真的执行过、数值是否真的出现在那次结果里
  - 过程好不好 → 非法 SQL / 重复查询 / 超预算，全部可从轨迹直接统计

权重与《SQL多轮数据分析Agent技术方案》一致：
  最终诊断 +0.55 / 关键数值 +0.20 / 证据 +0.15 / 恢复 +0.10
  非法 -0.03 / 重复 -0.02 / 超预算 -0.10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from ..agent.state import Trajectory
from ..config import RewardConfig
from ..utils.sql import sql_similarity

# 判定证据 SQL「确实执行过」的相似度阈值（比重复查询检测宽松，
# 因为模型复述 SQL 时格式常与原始提交有差异）
EVIDENCE_SQL_MATCH = 0.80


@dataclass
class RewardBreakdown:
    """奖励明细。训练时用 total，分析时看 components。"""

    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    # 便于下游统计
    task_success: bool = False
    has_final_answer: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "details": self.details,
            "task_success": self.task_success,
            "has_final_answer": self.has_final_answer,
        }


# ---------------------------------------------------------------------- #
def numeric_close(a: Any, b: Any, rtol: float = 0.01, atol: float = 1e-9) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if fa == fb:
        return True
    return abs(fa - fb) <= max(atol, rtol * max(abs(fa), abs(fb)))


def text_match(pred: str, gold: str) -> float:
    """文本匹配度 0~1。先看包含关系（中文短词最常见），再看序列相似度。"""
    p, g = (pred or "").strip().lower(), (gold or "").strip().lower()
    if not p or not g:
        return 0.0
    if g in p or p in g:
        return 1.0
    return SequenceMatcher(None, p, g).ratio()


class Verifier:
    """根据轨迹与参考答案计算奖励。"""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()

    # ------------------------------------------------------------------ #
    def score(self, traj: Trajectory) -> RewardBreakdown:
        rb = RewardBreakdown()
        ref = traj.reference or {}
        fa = traj.final_answer
        rb.has_final_answer = fa is not None

        comp: dict[str, float] = {}

        # ---------- 正向：最终诊断 ----------
        if fa and ref.get("root_cause_dimension"):
            s = text_match(
                str(fa.get("root_cause_dimension", "")),
                str(ref["root_cause_dimension"]),
            )
            # 也接受在 diagnosis 全文里提到根因
            if s < 1.0:
                s = max(s, text_match(str(fa.get("diagnosis", "")), str(ref["root_cause_dimension"])))
            comp["final_diagnosis"] = self.cfg.w_final_diagnosis * s
            rb.details["diagnosis_match"] = round(s, 4)
        else:
            comp["final_diagnosis"] = 0.0
            if not ref.get("root_cause_dimension"):
                rb.details["diagnosis_match"] = None  # 参考答案缺失

        # ---------- 正向：关键数值 ----------
        gold_nums = ref.get("key_numbers") or {}
        if fa and gold_nums:
            findings = fa.get("key_findings") or []
            pred_vals = [f.get("value") for f in findings if isinstance(f, dict)]
            hit, detail = 0, {}
            for name, gv in gold_nums.items():
                ok = any(
                    numeric_close(pv, gv, self.cfg.numeric_rtol) for pv in pred_vals
                )
                detail[name] = bool(ok)
                hit += int(ok)
            ratio = hit / len(gold_nums)
            comp["key_numbers"] = self.cfg.w_key_numbers * ratio
            rb.details["key_numbers_hit"] = detail
        else:
            comp["key_numbers"] = 0.0

        # ---------- 正向：证据忠实度 ----------
        if fa and (fa.get("key_findings") or []):
            comp["evidence"] = self.cfg.w_evidence * self._evidence_score(traj, fa, rb)
        else:
            comp["evidence"] = 0.0

        # ---------- 正向：错误恢复 ----------
        n_recover = traj.recovery_count
        comp["recovery"] = self.cfg.w_recovery * (1.0 if n_recover > 0 else 0.0)
        rb.details["recovery_count"] = n_recover

        # ---------- 负向 ----------
        comp["illegal_sql"] = self.cfg.p_illegal_sql * traj.illegal_sql_count
        comp["duplicate_sql"] = self.cfg.p_duplicate_sql * traj.duplicate_sql_count
        comp["over_budget"] = self.cfg.p_over_budget * (1.0 if traj.over_budget else 0.0)
        rb.details["illegal_sql_count"] = traj.illegal_sql_count
        rb.details["duplicate_sql_count"] = traj.duplicate_sql_count
        rb.details["over_budget"] = traj.over_budget
        rb.details["execution_error_count"] = traj.execution_error_count

        rb.components = comp
        rb.total = sum(comp.values())
        rb.task_success = self._is_success(traj, rb)
        return rb

    # ------------------------------------------------------------------ #
    def _evidence_score(
        self, traj: Trajectory, fa: dict[str, Any], rb: RewardBreakdown
    ) -> float:
        """证据忠实度：每条 key_finding 的 evidence_sql 是否真执行过、数值是否对得上。

        分两档：
          - 0.5：SQL 确实执行过（相似度达标）
          - 1.0：在执行过的基础上，声明数值也出现在那次查询的结果里
        """
        executed = traj.sql_calls
        findings = [f for f in (fa.get("key_findings") or []) if isinstance(f, dict)]
        if not findings:
            return 0.0

        per_finding: list[dict[str, Any]] = []
        total = 0.0
        for f in findings:
            ev = str(f.get("evidence_sql", "") or "")
            best_i, best_sim = -1, 0.0
            for i, sql in enumerate(executed):
                sim = sql_similarity(ev, sql) if ev else 0.0
                if sim > best_sim:
                    best_i, best_sim = i, sim

            entry = {"executed": False, "value_grounded": False, "similarity": round(best_sim, 4)}
            if best_i >= 0 and best_sim >= EVIDENCE_SQL_MATCH:
                entry["executed"] = True
                total += 0.5
                # 到执行结果里找这个数值
                res = self._result_at(traj, best_i)
                if res and self._value_in_result(f.get("value"), res):
                    entry["value_grounded"] = True
                    total += 0.5
            per_finding.append(entry)

        rb.details["evidence"] = per_finding
        return total / len(findings)

    @staticmethod
    def _result_at(traj: Trajectory, sql_index: int) -> dict[str, Any] | None:
        """取第 sql_index 条 SQL 对应的工具结果 payload。"""
        k = -1
        for s in traj.executed_steps:
            if s.tool_name == "execute_sql" and s.tool_result:
                k += 1
                if k == sql_index:
                    return s.tool_result.payload
        return None

    @staticmethod
    def _value_in_result(value: Any, payload: dict[str, Any]) -> bool:
        if value is None or not payload.get("ok"):
            return False
        for row in payload.get("rows", []):
            for cell in row:
                if numeric_close(cell, value, rtol=0.01):
                    return True
        return False

    # ------------------------------------------------------------------ #
    def _is_success(self, traj: Trajectory, rb: RewardBreakdown) -> bool:
        """任务成功判定：必须有 final_answer，且根因与关键数值基本正确。

        这个定义会被「Task Success Rate」指标直接使用，改动需同步评测口径。
        """
        if traj.final_answer is None:
            return False
        m = rb.details.get("diagnosis_match")
        if m is None:
            # 无参考答案时退化为「格式完整即成功」，仅用于数据试跑阶段
            return bool(traj.final_answer.get("root_cause_dimension"))
        if m < 1.0:
            return False
        nums = rb.details.get("key_numbers_hit") or {}
        if nums and (sum(nums.values()) / len(nums)) < 0.5:
            return False
        return True
