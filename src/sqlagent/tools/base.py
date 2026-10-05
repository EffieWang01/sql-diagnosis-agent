"""工具抽象层。

三个工具共用同一套契约，这样 AgentLoop / 训练框架 / 评测脚本
都不需要知道具体工具的实现细节。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """一次工具调用请求（由模型输出解析得到）。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    # 解析失败时为 True，用于奖励里的「格式错误」统计
    malformed: bool = False
    parse_error: str | None = None


@dataclass
class ToolResult:
    """一次工具调用结果。"""

    ok: bool
    # 回灌给模型的观测文本
    content: str
    # 结构化载荷（进轨迹日志，供奖励与评测使用），不直接给模型
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # 是否应终止 Agent 循环
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "content": self.content,
            "payload": self.payload,
            "error": self.error,
            "terminal": self.terminal,
        }


class Tool(Protocol):
    """工具协议。任何实现了 name/description/parameters/call 的对象都是工具。"""

    name: str
    description: str
    parameters: dict[str, Any]

    def call(self, **kwargs: Any) -> ToolResult: ...


class BaseTool:
    """工具基类，提供统一的 JSON Schema 生成。"""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def call(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - 抽象
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def to_openai_schema(self) -> dict[str, Any]:
        """渲染成 chat template 需要的 tools 结构（OpenAI 兼容格式）。

        注意：必须走这条路，不要手写工具描述文本——
        Qwen3.5 官方模板靠这个结构生成正确的 tool_call 提示。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


class ToolRegistry:
    """工具注册表 + 分发器。"""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for t in tools or []:
            self.register(t)
        # 调用计数，供预算控制与评测指标使用
        self.call_counts: dict[str, int] = {}

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("工具必须有 name。")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def dispatch(self, call: ToolCall) -> ToolResult:
        """分发一次工具调用。未知工具与参数错误都返回 ok=False 而非抛异常。"""
        self.call_counts[call.name] = self.call_counts.get(call.name, 0) + 1

        if call.malformed:
            return ToolResult(
                ok=False,
                content=(
                    f"工具调用格式无法解析：{call.parse_error or '未知格式错误'}。"
                    f"请严格使用 <tool_call><function=工具名>"
                    f"<parameter=参数名>值</parameter></function></tool_call> 格式。"
                ),
                error="malformed_call",
            )

        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                ok=False,
                content=(
                    f"不存在名为 `{call.name}` 的工具。"
                    f"可用工具：{', '.join(self.names())}。"
                ),
                error="unknown_tool",
            )

        try:
            return tool.call(**call.arguments)
        except TypeError as e:
            return ToolResult(
                ok=False,
                content=f"调用 `{call.name}` 的参数不正确：{e}",
                error="bad_arguments",
            )
        except Exception as e:  # noqa: BLE001 - 工具内部异常不应中断轨迹
            return ToolResult(
                ok=False,
                content=f"调用 `{call.name}` 时发生内部错误：{type(e).__name__}: {e}",
                error="tool_internal_error",
            )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
