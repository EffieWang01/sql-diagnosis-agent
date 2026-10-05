"""奖励函数测试。

重点验证两件事：
  1. 正向/负向分量是否按设计生效；
  2. **奖励不能被骗**——尤其是「错误恢复」分。
"""

from __future__ import annotations

from sqlagent.agent.state import Trajectory
from sqlagent.config import RewardConfig
from sqlagent.reward.verifier import Verifier, numeric_close, text_match
from sqlagent.tools.base import ToolCall, ToolResult

SQL_A = "SELECT SUM(gmv) FROM sales WHERE region = '华东'"
SQL_A_FIXED = "SELECT SUM(gmv) FROM sales WHERE region = '华东' AND category = '手机'"
SQL_B = "SELECT COUNT(*) FROM customers"


def make_traj(sql_steps, final_answer=None, reference=None, **kw) -> Trajectory:
    """按 (sql, ok, error_type) 三元组快速构造轨迹。"""
    t = Trajectory(task_id="t", question="q", reference=reference or {}, **kw)
    for i, step in enumerate(sql_steps):
        sql, ok = step[0], step[1]
        etype = step[2] if len(step) > 2 else None
        t.steps.append(
            _step(i, "execute_sql", {"sql": sql}, ok, etype)
        )
    t.final_answer = final_answer
    return t


def _step(i, name, args, ok, error_type=None, rows=None) -> object:
    from sqlagent.agent.state import Step

    s = Step(turn=i)
    s.tool_call = ToolCall(name=name, arguments=args)
    s.tool_result = ToolResult(
        ok=ok,
        content="ok" if ok else "err",
        payload={
            "ok": ok, "sql": args.get("sql", ""), "error_type": error_type,
            # 默认结果里带上 -0.23，让「证据数值可追溯」这条能通过
            "rows": (rows if rows is not None else [[-0.23, 100.0]]) if ok else [],
            "columns": ["change", "gmv"], "row_count": 1,
        },
        error=None if ok else "boom",
    )
    return s


REF = {
    "root_cause_dimension": "手机品类",
    "key_numbers": {"手机品类GMV环比变化率": -0.23},
}

GOOD_FA = {
    "diagnosis": "华东区下滑主因是手机品类",
    "root_cause_dimension": "手机品类",
    "key_findings": [
        {"claim": "手机品类下滑", "value": -0.23, "evidence_sql": SQL_A}
    ],
}


class TestHelpers:
    def test_numeric_close(self):
        assert numeric_close(-0.2301, -0.23, rtol=0.01)
        assert not numeric_close(-0.23, 0.23, rtol=0.01)
        assert numeric_close(100, 100.5, rtol=0.01)
        assert not numeric_close("abc", 1)

    def test_text_match(self):
        assert text_match("手机品类", "手机品类") == 1.0
        assert text_match("下滑主因是手机品类问题", "手机品类") == 1.0
        assert text_match("家电", "手机品类") < 0.6


class TestPositiveComponents:
    def test_full_marks(self):
        t = make_traj([(SQL_A, True)], GOOD_FA, REF)
        rb = Verifier().score(t)
        assert rb.components["final_diagnosis"] == 0.55
        assert rb.components["key_numbers"] == 0.20
        assert rb.components["evidence"] == 0.15  # 执行过 + 数值对得上
        assert rb.task_success is True

    def test_wrong_root_cause(self):
        fa = dict(GOOD_FA, root_cause_dimension="天气", diagnosis="天气不好")
        rb = Verifier().score(make_traj([(SQL_A, True)], fa, REF))
        assert rb.components["final_diagnosis"] == 0.0
        assert rb.task_success is False

    def test_no_final_answer(self):
        rb = Verifier().score(make_traj([(SQL_A, True)], None, REF))
        assert rb.total == 0.0
        assert rb.task_success is False

    def test_key_numbers_partial(self):
        fa = dict(GOOD_FA, key_findings=[
            {"claim": "a", "value": -0.23, "evidence_sql": SQL_A},
            {"claim": "b", "value": 999.0, "evidence_sql": SQL_A},
        ])
        rb = Verifier().score(make_traj([(SQL_A, True)], fa, REF))
        assert rb.components["key_numbers"] == 0.20  # 唯一一个参考答案命中


class TestEvidenceFaithfulness:
    def test_fabricated_sql_gets_nothing(self):
        """证据 SQL 根本没执行过 → 证据分为 0。"""
        fa = dict(GOOD_FA, key_findings=[
            {"claim": "x", "value": -0.23, "evidence_sql": "SELECT 1 -- 没跑过"}
        ])
        rb = Verifier().score(make_traj([(SQL_A, True)], fa, REF))
        assert rb.components["evidence"] == 0.0
        assert rb.details["evidence"][0]["executed"] is False

    def test_executed_but_value_not_grounded(self):
        """SQL 执行过但数值不在结果里 → 只拿一半分。"""
        fa = dict(GOOD_FA, key_findings=[
            {"claim": "x", "value": 12345.0, "evidence_sql": SQL_A}
        ])
        rb = Verifier().score(make_traj([(SQL_A, True)], fa, REF))
        assert rb.components["evidence"] == 0.075  # 0.15 * 0.5
        assert rb.details["evidence"][0]["executed"] is True
        assert rb.details["evidence"][0]["value_grounded"] is False


class TestNegativeComponents:
    def test_illegal_penalty(self):
        t = make_traj([("DROP TABLE sales", False, "illegal"), (SQL_A, True)])
        rb = Verifier().score(t)
        assert rb.components["illegal_sql"] == -0.03

    def test_duplicate_penalty(self):
        t = make_traj([(SQL_A, True), (SQL_A, True)])
        rb = Verifier().score(t)
        assert rb.components["duplicate_sql"] == -0.02
        assert t.duplicate_sql_count == 1

    def test_similar_but_not_identical_is_duplicate(self):
        """只改了空白/大小写也算重复。"""
        t = make_traj([(SQL_A, True), (SQL_A.lower().replace(" ", "  "), True)])
        assert t.duplicate_sql_count == 1

    def test_different_sql_not_duplicate(self):
        t = make_traj([(SQL_A, True), (SQL_B, True)])
        assert t.duplicate_sql_count == 0

    def test_over_budget_penalty(self):
        t = make_traj([(SQL_A, True)] * 2, max_sql_calls=1)
        rb = Verifier().score(t)
        assert rb.components["over_budget"] == -0.10
        assert t.over_budget is True


class TestRecoveryAntiGaming:
    """核心回归测试：恢复分必须要求「真的修了」，不能靠乱写骗分。"""

    def test_unrelated_success_is_not_recovery(self):
        """失败后随便换一条不相关的成功查询 → 不算恢复。"""
        t = make_traj([("DROP TABLE sales", False, "illegal"), (SQL_B, True)])
        assert t.recovery_count == 0
        rb = Verifier().score(t)
        assert rb.components["recovery"] == 0.0

    def test_similar_repair_is_recovery(self):
        """相似 SQL 从失败变成功 → 算恢复。"""
        t = make_traj([(SQL_A, False, "execution"), (SQL_A_FIXED, True)])
        assert t.recovery_count == 1
        rb = Verifier().score(t)
        assert rb.components["recovery"] == 0.10

    def test_failed_then_failed_no_recovery(self):
        t = make_traj([(SQL_A, False, "execution"), (SQL_A_FIXED, False, "execution")])
        assert t.recovery_count == 0

    def test_success_then_success_no_recovery(self):
        t = make_traj([(SQL_A, True), (SQL_A_FIXED, True)])
        assert t.recovery_count == 0


class TestDiscrimination:
    def test_good_beats_bad(self):
        """好轨迹必须显著高于坏轨迹——否则训练信号是坏的。"""
        good = make_traj([(SQL_A, True)], GOOD_FA, REF)
        bad = make_traj(
            [("DROP TABLE sales", False, "illegal"), (SQL_B, True), (SQL_B, True)],
            {"diagnosis": "不知道", "root_cause_dimension": "天气",
             "key_findings": [{"claim": "c", "value": 1.0, "evidence_sql": "SELECT 9"}]},
            REF,
        )
        v = Verifier()
        assert v.score(good).total > v.score(bad).total + 0.5

    def test_total_is_sum_of_components(self):
        rb = Verifier().score(make_traj([(SQL_A, True)], GOOD_FA, REF))
        assert abs(rb.total - sum(rb.components.values())) < 1e-9
