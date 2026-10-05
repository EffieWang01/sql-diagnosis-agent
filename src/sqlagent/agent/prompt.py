"""Prompt 构造。

两条硬约束：
  1. **表结构进 system prompt**，不做成独立工具（技术方案的明确决定）。
  2. **工具定义必须走 chat template 的 tools 参数**，不能手写描述文本——
     Qwen3.5 官方模板依赖这个结构生成正确的 tool_call 提示，
     手写会导致空答或思考旁白（社区已踩坑）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SYSTEM_TEMPLATE = """\
你是一名资深业务数据分析师，使用 SQL 对数据仓库做**目标驱动的异常定位与根因诊断**。

## 你的任务
用户会给出一个业务异常现象（例如「某区域某月 GMV 下滑 12%」）。
你需要通过连续执行 SQL，定位异常出在哪个维度/切片，给出根因结论，并用数据支撑。

## 工作原则
1. **先定位、再下钻**：先用聚合查询确认异常发生在哪个维度（区域 / 品类 / 渠道 / 时间），
   再对最可疑的那个切片做更细的拆分。不要一上来就写复杂多表 JOIN。
2. **每次只推进一个假设**：一条 SQL 回答一个问题。如果结果推翻了假设，就换方向，
   而不是继续在错误分支上加查询。
3. **不要重复查询**：执行前先看历史结果，不要重复提交语义相同的 SQL。
4. **失败要修正而不是放弃**：SQL 报错时，根据错误信息修正（常见原因：字段名写错、
   类型不匹配、需要先聚合再过滤）。最多允许一次修正机会，仍失败就换思路。
5. **证据必须来自你实际执行的查询**：不要凭常识编造数值。

## 预算
- 最多 {max_turns} 轮工具调用，其中 SQL 查询不超过 {max_sql_calls} 次。
- 单次查询最多返回 {max_rows} 行，超时 {timeout_s:.0f} 秒。
- 超出预算会被扣分，请优先做信息量最大的查询。

## 可用表结构
{schema}

## 输出要求
- 需要查数据时，调用 `execute_sql`。
- 不确定业务指标口径时，调用 `retrieve_context`（它只返回定义，不返回数值）。
- 证据充分后，调用 `final_answer` 提交结论，并在 `key_findings` 里附上支撑每条结论的
  `evidence_sql`（必须是你真实执行过的 SQL）。
"""


@dataclass
class PromptBuilder:
    """构造对话消息列表。"""

    schema_ddl: str
    max_turns: int = 10
    max_sql_calls: int = 8
    max_rows: int = 50
    timeout_s: float = 15.0
    extra_rules: str = ""

    def system_prompt(self) -> str:
        text = SYSTEM_TEMPLATE.format(
            max_turns=self.max_turns,
            max_sql_calls=self.max_sql_calls,
            max_rows=self.max_rows,
            timeout_s=self.timeout_s,
            schema=self.schema_ddl.strip() or "（未提供表结构）",
        )
        if self.extra_rules:
            text += "\n\n## 补充规则\n" + self.extra_rules.strip() + "\n"
        return text

    def initial_messages(self, question: str) -> list[dict[str, Any]]:
        """首轮消息。"""
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": question},
        ]


# ---------------------------------------------------------------------- #
def render_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> Any:
    """调用 chat template 渲染。

    兼容 transformers 5.x 的行为差异：``return_tensors="pt"`` 返回的是
    **dict**（含 input_ids / attention_mask）而不是裸张量。
    这个坑在本地实测踩过，这里统一处理掉。
    """
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": True,
        "return_tensors": "pt",
    }
    if tools:
        kwargs["tools"] = tools
    try:
        kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # 老模板不支持 enable_thinking
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def extract_input_ids(encoded: Any) -> Any:
    """从 chat template 的返回值里取出 input_ids。

    transformers 5.x 返回 dict，4.x 返回裸 tensor —— 两种都要能处理。
    """
    if hasattr(encoded, "keys"):
        return encoded["input_ids"]
    return encoded
