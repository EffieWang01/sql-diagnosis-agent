"""模型输出解析：把 Qwen3.5 的原生 tool_call 文本还原成结构化调用。

Qwen3.5 实际输出的格式（已实测）::

    <tool_call>
    <function=execute_sql>
    <parameter=sql>
    SELECT region, SUM(gmv) FROM sales GROUP BY 1
    </parameter>
    </function>
    </tool_call>

同时兼容两种变体：
  - `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`（OpenAI JSON 风格）
  - `<tool_call>{"tool": "...", "parameters": {...}}</tool_call>`

解析器必须**永不抛异常**：格式错误的输出本身就是一个重要的负样本，
要能被记录进轨迹、被奖励函数惩罚，而不是让整个 rollout 崩掉。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..tools.base import ToolCall

# ---------------------------------------------------------------------- #
_THINK_BLOCK = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think\b[^>]*>.*$", re.DOTALL | re.IGNORECASE)
_SPECIAL_TOKENS = re.compile(r"<\|(?:im_start|im_end|endoftext|eot_id|end_of_turn)\|>")

_TOOL_CALL_BLOCK = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call\s*>", re.DOTALL | re.IGNORECASE)
# 函数名/参数名放宽到「非空白非 >」，而不是限制成 ASCII 标识符——
# 否则模型吐出非 ASCII 名字时会被**静默丢弃**，模型收不到任何反馈，
# 只会在循环里空转。宁可解析出来交给注册表报「工具不存在」。
_FUNCTION_XML = re.compile(r"<function\s*=\s*([^\s>]+)\s*>(.*?)</function\s*>", re.DOTALL)
_PARAM_XML = re.compile(
    r"<parameter\s*=\s*([^\s>]+)\s*>(.*?)</parameter\s*>", re.DOTALL
)
# 宽松兜底：模型偶尔会漏掉闭合标签
_FUNCTION_LOOSE = re.compile(r"<function\s*=\s*([^\s>]+)\s*>")


@dataclass
class ParseResult:
    """一次模型输出的解析结果。"""

    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # 工具调用之外的自然语言内容
    text: str = ""
    # 是否检测到「提到了工具名但没有合法调用结构」这类格式问题
    warnings: list[str] = field(default_factory=list)

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_calls)

    @property
    def first(self) -> ToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None


# ---------------------------------------------------------------------- #
def strip_thinking(text: str) -> tuple[str, str | None]:
    """剥离 <think> 块，返回 (剩余文本, 思考内容)。

    即使配置里关了 thinking，模型偶尔仍会吐出 think 块，
    所以这一步是无条件做的防御。
    """
    m = _THINK_BLOCK.search(text)
    thinking = m.group(1).strip() if m else None
    out = _THINK_BLOCK.sub(" ", text)
    # 未闭合的 think（生成被 max_new_tokens 截断）
    out = _THINK_UNCLOSED.sub(" ", out)
    return out.strip(), thinking


def clean_special_tokens(text: str) -> str:
    return _SPECIAL_TOKENS.sub("", text).strip()


def coerce_value(raw: str) -> Any:
    """把 XML 参数里的字符串还原成合适的 Python 类型。

    XML 格式天然丢失类型信息，这里按三级尝试：
      1. 字面量布尔 / None
      2. JSON 解析（标准写法）
      3. ``ast.literal_eval``（模型常输出 Python repr，如单引号包裹的列表/字典）

    三级都失败才保留原字符串。第 3 步很关键——实测模型经常把列表参数
    写成 ``[{'claim': '...'}]`` 这种 Python 风格，只靠 JSON 会解析失败，
    参数被静默降级成字符串，下游校验就会报「格式不正确」。
    """
    import ast

    s = raw.strip()
    if s == "":
        return ""
    lowered = s.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None

    if s[0] in "[{":
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except Exception:
            return s
    if re.fullmatch(r"-?\d*\.\d+(?:[eE][-+]?\d+)?", s) or re.fullmatch(
        r"-?\d+[eE][-+]?\d+", s
    ):
        try:
            return float(s)
        except Exception:
            return s
    return s


# ---------------------------------------------------------------------- #
def _parse_json_tool_call(blob: str) -> ToolCall | None:
    """解析 JSON 风格的 tool_call。"""
    blob = blob.strip()
    if not blob.startswith("{"):
        # 允许 ```json ... ``` 包裹
        m = re.search(r"\{.*\}", blob, re.DOTALL)
        if not m:
            return None
        blob = m.group(0)
    try:
        obj = json.loads(blob)
    except Exception as e:
        return ToolCall(name="", raw=blob, malformed=True, parse_error=f"JSON 解析失败: {e}")
    if not isinstance(obj, dict):
        return None

    name = obj.get("name") or obj.get("tool") or obj.get("function") or ""
    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name=str(name), arguments=args, raw=blob)


def _parse_xml_tool_call(blob: str) -> list[ToolCall]:
    """解析 Qwen 原生 XML 风格 tool_call。"""
    calls: list[ToolCall] = []
    for fm in _FUNCTION_XML.finditer(blob):
        name, body = fm.group(1), fm.group(2)
        args: dict[str, Any] = {}
        for pm in _PARAM_XML.finditer(body):
            args[pm.group(1)] = coerce_value(pm.group(2))
        calls.append(ToolCall(name=name, arguments=args, raw=blob.strip()))
    return calls


# ---------------------------------------------------------------------- #
def parse_model_output(text: str, known_tools: list[str] | None = None) -> ParseResult:
    """解析模型输出。永不抛异常。"""
    if text is None:
        return ParseResult(warnings=["模型输出为 None"])

    text = clean_special_tokens(text)
    body, thinking = strip_thinking(text)
    warnings: list[str] = []

    tool_calls: list[ToolCall] = []
    for block in _TOOL_CALL_BLOCK.finditer(body):
        inner = block.group(1).strip()
        if not inner:
            warnings.append("检测到空的 <tool_call> 块")
            continue
        parsed = _parse_xml_tool_call(inner)
        if parsed:
            tool_calls.extend(parsed)
            continue
        jc = _parse_json_tool_call(inner)
        if jc is not None:
            tool_calls.append(jc)
        else:
            warnings.append(f"无法解析 tool_call 内容: {inner[:80]}")

    # 兜底：没有闭合的 </tool_call>，但出现了 <function=xxx>
    if not tool_calls:
        loose = _FUNCTION_LOOSE.search(body)
        if loose:
            name = loose.group(1)
            tail = body[loose.end():]
            args = {
                pm.group(1): coerce_value(pm.group(2))
                for pm in _PARAM_XML.finditer(tail)
            }
            tool_calls.append(
                ToolCall(name=name, arguments=args, raw=body.strip(),
                         malformed=False)
            )
            warnings.append("tool_call 缺少闭合标签，已按宽松模式解析")

    # 残留的 tool_call 标签说明格式不完整
    residual = _TOOL_CALL_BLOCK.sub(" ", body)
    residual = _FUNCTION_XML.sub(" ", residual)
    residual = _PARAM_XML.sub(" ", residual)
    residual = re.sub(r"</?(?:tool_call|function|parameter)\b[^>]*>", " ", residual)
    plain = re.sub(r"\s+", " ", residual).strip()

    # 完全没解析出调用，但文本里提到了已知工具名 → 明确标记为格式错误
    if not tool_calls and known_tools:
        for t in known_tools:
            if re.search(rf"\b{re.escape(t)}\b", plain):
                warnings.append(
                    f"文本提到工具 `{t}` 但没有合法的 tool_call 结构"
                )
                break

    return ParseResult(
        thinking=thinking, tool_calls=tool_calls, text=plain, warnings=warnings
    )
