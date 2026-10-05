"""模型输出解析测试。

解析器是「模型 → 工具」的唯一通道，一旦漏解析，轨迹就废了。
"""

from __future__ import annotations

import json

from sqlagent.agent.parser import coerce_value, parse_model_output, strip_thinking

XML_SQL = """<tool_call>
<function=execute_sql>
<parameter=sql>
SELECT region, SUM(gmv) FROM sales GROUP BY 1
</parameter>
<parameter=purpose>
看各区域总量
</parameter>
</function>
</tool_call>"""


class TestXmlFormat:
    def test_basic(self):
        r = parse_model_output(XML_SQL)
        assert r.has_tool_call
        call = r.first
        assert call.name == "execute_sql"
        assert "SELECT region" in call.arguments["sql"]
        assert call.arguments["purpose"] == "看各区域总量"

    def test_multiline_sql_preserved(self):
        text = (
            "<tool_call>\n<function=execute_sql>\n<parameter=sql>\n"
            "SELECT region,\n       SUM(gmv) AS gmv\nFROM sales\nGROUP BY 1\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        r = parse_model_output(text)
        sql = r.first.arguments["sql"]
        assert sql.count("\n") >= 3, "多行 SQL 的换行必须保留"
        assert sql.strip().endswith("GROUP BY 1")

    def test_json_list_argument(self):
        payload = json.dumps(
            [{"claim": "手机下滑", "value": -0.23, "evidence_sql": "SELECT 1"}],
            ensure_ascii=False,
        )
        text = (
            "<tool_call>\n<function=final_answer>\n"
            f"<parameter=key_findings>\n{payload}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        r = parse_model_output(text)
        kf = r.first.arguments["key_findings"]
        assert isinstance(kf, list) and kf[0]["value"] == -0.23

    def test_python_repr_list_argument(self):
        """模型常输出 Python repr（单引号），必须靠 literal_eval 兜住。"""
        text = (
            "<tool_call>\n<function=final_answer>\n"
            "<parameter=key_findings>\n"
            "[{'claim': '手机下滑', 'value': -0.23, 'evidence_sql': 'SELECT 1'}]\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        r = parse_model_output(text)
        kf = r.first.arguments["key_findings"]
        assert isinstance(kf, list), f"解析成了 {type(kf)}"
        assert kf[0]["claim"] == "手机下滑"

    def test_number_and_bool_coercion(self):
        text = (
            "<tool_call>\n<function=f>\n"
            "<parameter=a>\n42\n</parameter>\n"
            "<parameter=b>\n0.75\n</parameter>\n"
            "<parameter=c>\ntrue\n</parameter>\n"
            "<parameter=d>\n-3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        args = parse_model_output(text).first.arguments
        assert args["a"] == 42 and isinstance(args["a"], int)
        assert args["b"] == 0.75 and isinstance(args["b"], float)
        assert args["c"] is True
        assert args["d"] == -3


class TestJsonFormat:
    def test_openai_style(self):
        text = '<tool_call>{"name": "execute_sql", "arguments": {"sql": "SELECT 1"}}</tool_call>'
        r = parse_model_output(text)
        assert r.first.name == "execute_sql"
        assert r.first.arguments["sql"] == "SELECT 1"

    def test_arguments_as_string(self):
        text = '<tool_call>{"name": "f", "arguments": "{\\"x\\": 1}"}</tool_call>'
        r = parse_model_output(text)
        assert r.first.arguments["x"] == 1

    def test_tool_parameters_alias(self):
        text = '<tool_call>{"tool": "execute_sql", "parameters": {"sql": "SELECT 1"}}</tool_call>'
        assert parse_model_output(text).first.name == "execute_sql"


class TestThinking:
    def test_strip_closed(self):
        body, think = strip_thinking("<think>先看区域</think>决定查区域")
        assert think == "先看区域"
        assert "think" not in body
        assert "决定查区域" in body

    def test_strip_unclosed(self):
        """生成被截断时 think 不会闭合。"""
        body, think = strip_thinking("<think>正在思考")
        assert "正在思考" not in body

    def test_thinking_with_tool_call(self):
        text = "<think>需要查数据</think>\n" + XML_SQL
        r = parse_model_output(text)
        assert r.thinking == "需要查数据"
        assert r.has_tool_call

    def test_special_tokens_removed(self):
        r = parse_model_output("<|im_end|>" + XML_SQL + "<|im_end|>")
        assert r.has_tool_call


class TestRobustness:
    def test_empty(self):
        r = parse_model_output("")
        assert not r.has_tool_call

    def test_none(self):
        assert not parse_model_output(None).has_tool_call

    def test_plain_text(self):
        r = parse_model_output("我认为是手机品类下滑导致的。")
        assert not r.has_tool_call
        assert "手机品类" in r.text

    def test_malformed_json_flagged(self):
        r = parse_model_output('<tool_call>{"name": broken}</tool_call>')
        assert r.has_tool_call
        assert r.first.malformed is True

    def test_mentions_tool_without_structure(self):
        """只提到工具名但没有调用结构 → 应产生 warning，供奖励惩罚。"""
        r = parse_model_output(
            "我应该调用 execute_sql 来查一下数据。",
            known_tools=["execute_sql", "final_answer"],
        )
        assert not r.has_tool_call
        assert any("execute_sql" in w for w in r.warnings)

    def test_unclosed_tool_call_loose_parse(self):
        text = "<tool_call>\n<function=execute_sql>\n<parameter=sql>SELECT 1</parameter>"
        r = parse_model_output(text)
        assert r.has_tool_call
        assert r.first.arguments["sql"] == "SELECT 1"

    def test_multiple_calls(self):
        text = XML_SQL + "\n" + XML_SQL.replace("SELECT region, SUM(gmv) FROM sales GROUP BY 1", "SELECT 2")
        r = parse_model_output(text)
        assert len(r.tool_calls) == 2


class TestCoerceValue:
    def test_scalars(self):
        assert coerce_value("42") == 42
        assert coerce_value("3.14") == 3.14
        assert coerce_value("true") is True
        assert coerce_value("False") is False
        assert coerce_value("null") is None

    def test_containers(self):
        assert coerce_value("[1, 2]") == [1, 2]
        assert coerce_value('{"a": 1}') == {"a": 1}
        assert coerce_value("[{'a': 1}]") == [{"a": 1}]

    def test_sql_stays_string(self):
        sql = "SELECT a, b FROM t WHERE x = 1"
        assert coerce_value(sql) == sql

    def test_scientific_notation(self):
        assert coerce_value("1e-3") == 1e-3
