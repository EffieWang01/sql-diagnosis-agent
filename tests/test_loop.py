"""AgentLoop 与工具层测试。"""

from __future__ import annotations

import pytest

from sqlagent.agent.loop import STYLE_STRUCTURED, STYLE_TEXT, AgentLoop
from sqlagent.agent.policy import ScriptedPolicy, format_tool_call_xml
from sqlagent.agent.prompt import PromptBuilder
from sqlagent.agent.state import (
    STOP_FINAL_ANSWER,
    STOP_MAX_SQL,
    STOP_MAX_TURNS,
    STOP_NO_TOOL_CALL,
)
from sqlagent.tools.base import ToolCall, ToolRegistry
from sqlagent.tools.builtin import build_default_tools

SCHEMA = 'CREATE TABLE "sales" (\n  "region" VARCHAR,\n  "gmv" DOUBLE\n);'
QUESTION = "华东区 GMV 为什么下滑？"


def build(script, sandbox, store, agent_cfg, style=STYLE_STRUCTURED):
    registry = ToolRegistry(build_default_tools(sandbox, store, observation_max_chars=500))
    loop = AgentLoop(
        registry=registry,
        policy=ScriptedPolicy(script),
        prompt_builder=PromptBuilder(schema_ddl=SCHEMA),
        config=agent_cfg,
        tool_response_style=style,
    )
    return loop, registry


def sql_call(sql: str, purpose: str = "test") -> str:
    return format_tool_call_xml("execute_sql", {"sql": sql, "purpose": purpose})


def final_call(dim: str = "手机品类") -> str:
    return format_tool_call_xml("final_answer", {
        "diagnosis": f"{dim}下滑导致",
        "root_cause_dimension": dim,
        "key_findings": [{"claim": "c", "value": -0.2, "evidence_sql": "SELECT 1"}],
    })


class TestBasicFlow:
    def test_happy_path(self, sandbox, store, agent_cfg):
        loop, _ = build(
            [sql_call("SELECT region, SUM(gmv) FROM sales GROUP BY 1"), final_call()],
            sandbox, store, agent_cfg,
        )
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.terminated_reason == STOP_FINAL_ANSWER
        assert t.final_answer["root_cause_dimension"] == "手机品类"
        assert t.n_sql_calls == 1
        assert t.n_turns == 2

    def test_tool_call_recorded_with_result(self, sandbox, store, agent_cfg):
        loop, _ = build([sql_call("SELECT 1 AS x")], sandbox, store, agent_cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.steps[0].tool_call.name == "execute_sql"
        assert t.steps[0].tool_result.ok is True

    def test_messages_grow(self, sandbox, store, agent_cfg):
        loop, _ = build([sql_call("SELECT 1"), final_call()], sandbox, store, agent_cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})
        # system + user + (assistant + tool) * 2
        assert len(t.messages) == 6
        assert t.messages[0]["role"] == "system"
        assert t.messages[1]["role"] == "user"

    def test_text_style_uses_user_role(self, sandbox, store, agent_cfg):
        loop, _ = build([sql_call("SELECT 1")], sandbox, store, agent_cfg, style=STYLE_TEXT)
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert all(m["role"] != "tool" for m in t.messages)
        assert any("工具 execute_sql 返回" in str(m.get("content", "")) for m in t.messages)


class TestBudget:
    def test_sql_budget_blocks_execution(self, sandbox, store, agent_cfg):
        """agent_cfg.max_sql_calls=3，第 4 条应被拦截且不真的执行。

        配置里 force_finish_on_budget=True，所以拦截后立即终止循环。
        """
        script = [sql_call("SELECT 1 AS a") for _ in range(5)]
        loop, registry = build(script, sandbox, store, agent_cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})

        assert t.n_sql_calls == 3, f"实际执行了 {t.n_sql_calls} 条"
        # 拦截发生在分发之前，所以工具层只看到 3 次调用
        assert registry.call_counts["execute_sql"] == 3
        assert sandbox.stats["total"] == 3, "被拦截的尝试不应到达沙箱"

        assert len(t.blocked_steps) == 1
        assert "预算上限" in t.blocked_steps[0].tool_result.content
        assert t.terminated_reason == STOP_MAX_SQL
        # 被拦截的不计入真实 SQL 调用
        assert t.n_sql_calls == len(t.executed_steps) - t.n_context_calls

    def test_budget_continue_mode(self, sandbox, store):
        """关掉 force_finish_on_budget 时，拦截后应继续循环。"""
        from sqlagent.config import AgentConfig

        cfg = AgentConfig(max_turns=6, max_sql_calls=2, max_context_calls=1, max_chars=20000)
        cfg.force_finish_on_budget = False
        script = [sql_call("SELECT 1 AS a") for _ in range(4)] + [final_call()]
        loop, _ = build(script, sandbox, store, cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})

        assert t.n_sql_calls == 2
        assert len(t.blocked_steps) == 2
        assert t.terminated_reason == STOP_FINAL_ANSWER

    def test_context_budget(self, sandbox, store, agent_cfg):
        script = [format_tool_call_xml("retrieve_context", {"query": "GMV"}) for _ in range(3)]
        loop, _ = build(script, sandbox, store, agent_cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.n_context_calls == 1  # max_context_calls=1
        assert len(t.blocked_steps) == 1
        assert "预算上限" in t.blocked_steps[0].tool_result.content

    def test_max_turns(self, sandbox, store, agent_cfg):
        loop, _ = build([sql_call("SELECT 1")] * 20, sandbox, store, agent_cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.n_turns <= agent_cfg.max_turns
        assert t.terminated_reason in (STOP_MAX_SQL, STOP_MAX_TURNS)


class TestMalformedOutput:
    def test_no_tool_call_retries_then_stops(self, sandbox, store, agent_cfg):
        loop, _ = build(["我只是随便说说", "还是随便说说", "继续闲聊"], sandbox, store, agent_cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.terminated_reason == STOP_NO_TOOL_CALL
        assert t.error is not None
        # 应该有提醒消息被插入
        assert any("没有调用任何工具" in str(m.get("content", "")) for m in t.messages)

    def test_unknown_tool_returns_error_not_crash(self, sandbox, store, agent_cfg):
        loop, _ = build(
            [format_tool_call_xml("nonexistent_tool", {"x": 1}), final_call()],
            sandbox, store, agent_cfg,
        )
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.steps[0].tool_result is not None
        assert t.steps[0].tool_result.ok is False
        assert "不存在" in t.steps[0].tool_result.content
        assert t.terminated_reason == STOP_FINAL_ANSWER

    def test_non_ascii_tool_name_gets_feedback(self, sandbox, store, agent_cfg):
        """非 ASCII 工具名不能被静默丢弃——模型必须收到明确反馈。"""
        loop, _ = build(
            [format_tool_call_xml("查询工具", {"x": 1}), final_call()],
            sandbox, store, agent_cfg,
        )
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.steps[0].tool_call is not None, "非 ASCII 工具名被丢弃了"
        assert t.steps[0].tool_result.ok is False
        assert "不存在" in t.steps[0].tool_result.content

    def test_bad_arguments_handled(self, sandbox, store, agent_cfg):
        loop, _ = build(
            [format_tool_call_xml("execute_sql", {"wrong_param": "x"}), final_call()],
            sandbox, store, agent_cfg,
        )
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.steps[0].tool_result.ok is False

    def test_missing_required_final_answer_field(self, sandbox, store, agent_cfg):
        bad_final = format_tool_call_xml("final_answer", {"diagnosis": "只有结论"})
        loop, _ = build([bad_final, final_call()], sandbox, store, agent_cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.steps[0].tool_result.ok is False
        assert "缺少必填字段" in t.steps[0].tool_result.content


class TestToolRegistry:
    def test_unknown_tool(self, sandbox, store):
        reg = ToolRegistry(build_default_tools(sandbox, store))
        r = reg.dispatch(ToolCall(name="nope"))
        assert not r.ok and r.error == "unknown_tool"

    def test_malformed_call(self, sandbox, store):
        reg = ToolRegistry(build_default_tools(sandbox, store))
        r = reg.dispatch(ToolCall(name="execute_sql", malformed=True, parse_error="坏了"))
        assert not r.ok and r.error == "malformed_call"

    def test_schemas_are_openai_format(self, sandbox, store):
        reg = ToolRegistry(build_default_tools(sandbox, store))
        schemas = reg.schemas()
        assert len(schemas) == 3
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"] and "parameters" in s["function"]

    def test_tool_names(self, sandbox, store):
        reg = ToolRegistry(build_default_tools(sandbox, store))
        assert set(reg.names()) == {"retrieve_context", "execute_sql", "final_answer"}


class TestRetrieveContext:
    def test_finds_metric(self, sandbox, store, agent_cfg):
        loop, _ = build(
            [format_tool_call_xml("retrieve_context", {"query": "GMV 口径"}), final_call()],
            sandbox, store, agent_cfg,
        )
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.steps[0].tool_result.ok
        assert "GMV" in t.steps[0].tool_result.content
        assert t.steps[0].tool_result.payload["num_hits"] >= 1

    def test_empty_query_rejected(self, sandbox, store):
        from sqlagent.tools.builtin import RetrieveContextTool

        r = RetrieveContextTool(store).call(query="   ")
        assert not r.ok


class TestTrajectoryStats:
    def test_duplicate_detection(self, sandbox, store, agent_cfg):
        sql = "SELECT region, SUM(gmv) FROM sales GROUP BY 1"
        loop, _ = build([sql_call(sql), sql_call(sql), final_call()], sandbox, store, agent_cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})
        assert t.duplicate_sql_count == 1
        assert t.redundant_query_rate == pytest.approx(0.5)

    def test_serializable(self, sandbox, store, agent_cfg):
        import json

        loop, _ = build([sql_call("SELECT 1"), final_call()], sandbox, store, agent_cfg)
        t = loop.run({"task_id": "x", "question": QUESTION})
        blob = json.dumps(t.to_dict(), ensure_ascii=False)
        assert "手机品类" in blob

    def test_reference_carried_through(self, sandbox, store, agent_cfg):
        loop, _ = build([final_call()], sandbox, store, agent_cfg)
        t = loop.run({
            "task_id": "x", "question": QUESTION,
            "reference": {"root_cause_dimension": "手机品类"},
        })
        assert t.reference["root_cause_dimension"] == "手机品类"
