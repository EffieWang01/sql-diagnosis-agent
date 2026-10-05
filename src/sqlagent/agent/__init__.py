"""Agent 层：Prompt、解析、轨迹、循环、策略。"""

from .loop import STYLE_STRUCTURED, STYLE_TEXT, AgentLoop
from .parser import ParseResult, parse_model_output, strip_thinking
from .policy import HFPolicy, Policy, ScriptedPolicy, format_tool_call_xml
from .prompt import PromptBuilder, extract_input_ids, render_prompt
from .state import (
    STOP_BUDGET_CHARS,
    STOP_ERROR,
    STOP_FINAL_ANSWER,
    STOP_MAX_SQL,
    STOP_MAX_TURNS,
    STOP_NO_TOOL_CALL,
    Step,
    Trajectory,
)

__all__ = [
    "AgentLoop",
    "STYLE_STRUCTURED",
    "STYLE_TEXT",
    "PromptBuilder",
    "render_prompt",
    "extract_input_ids",
    "parse_model_output",
    "strip_thinking",
    "ParseResult",
    "Policy",
    "ScriptedPolicy",
    "HFPolicy",
    "format_tool_call_xml",
    "Trajectory",
    "Step",
    "STOP_FINAL_ANSWER",
    "STOP_MAX_TURNS",
    "STOP_MAX_SQL",
    "STOP_NO_TOOL_CALL",
    "STOP_BUDGET_CHARS",
    "STOP_ERROR",
]
