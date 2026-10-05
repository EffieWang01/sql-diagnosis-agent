"""AgentLoop：多轮「模型输出 → 工具执行 → 观测回灌」的主循环。

设计要点：
  - **不直接依赖任何训练框架**，只通过 Policy 协议与模型交互。
  - **预算在分发前拦截**，而不是执行后才判定，避免浪费真实 SQL 调用。
  - **所有异常都编码进轨迹**，绝不向上抛——一条 rollout 崩掉会污染整个 batch。
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import AgentConfig
from ..tools.base import ToolRegistry
from .parser import ParseResult, parse_model_output
from .policy import Policy
from .prompt import PromptBuilder
from .state import (
    STOP_BUDGET_CHARS,
    STOP_ERROR,
    STOP_FINAL_ANSWER,
    STOP_MAX_CONTEXT,
    STOP_MAX_SQL,
    STOP_MAX_TURNS,
    STOP_NO_TOOL_CALL,
    Step,
    Trajectory,
)

# 工具响应回灌风格
STYLE_STRUCTURED = "structured"  # 用 role="tool" + tool_calls，走原生模板
STYLE_TEXT = "text"              # 把观测拼进 user 消息，兼容性最好


class AgentLoop:
    """单条轨迹的执行器。"""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: Policy,
        prompt_builder: PromptBuilder,
        config: AgentConfig | None = None,
        tool_response_style: str = STYLE_STRUCTURED,
        max_format_retries: int = 2,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.prompt = prompt_builder
        self.cfg = config or AgentConfig()
        self.style = tool_response_style
        self.max_format_retries = max_format_retries

    # ------------------------------------------------------------------ #
    def run(self, task: dict[str, Any]) -> Trajectory:
        """执行一条任务，返回完整轨迹。"""
        traj = Trajectory(
            task_id=str(task.get("task_id", "unknown")),
            question=str(task.get("question", "")),
            max_turns=self.cfg.max_turns,
            max_sql_calls=self.cfg.max_sql_calls,
            max_context_calls=self.cfg.max_context_calls,
            max_chars=self.cfg.max_chars,
            reference=task.get("reference", {}) or {},
            meta=task.get("meta", {}) or {},
        )
        traj.messages = self.prompt.initial_messages(traj.question)

        tool_schemas = self.registry.schemas()
        known = self.registry.names()
        format_retries = 0

        try:
            for turn in range(self.cfg.max_turns):
                t0 = time.time()

                # ---- 1. 模型决策
                raw = self.policy.generate(traj.messages, tools=tool_schemas)
                parsed = parse_model_output(raw, known_tools=known)

                # ---- 2. 没有工具调用
                if not parsed.has_tool_call:
                    step = Step(
                        turn=turn, assistant_text=raw, thinking=parsed.thinking,
                        elapsed_s=time.time() - t0, warnings=list(parsed.warnings),
                    )
                    traj.steps.append(step)
                    self._append_plain_assistant(traj, raw)

                    if format_retries < self.max_format_retries:
                        format_retries += 1
                        traj.messages.append({
                            "role": "user",
                            "content": (
                                "你没有调用任何工具。请使用 <tool_call> 格式调用 "
                                "execute_sql 继续分析，或在证据充分时调用 final_answer。"
                            ),
                        })
                        continue

                    traj.terminated_reason = STOP_NO_TOOL_CALL
                    traj.error = "模型连续多轮未产生合法的工具调用"
                    return traj

                # ---- 3. 解析出多个调用时只取第一个
                call = parsed.tool_calls[0]
                warnings = list(parsed.warnings)
                if len(parsed.tool_calls) > 1:
                    warnings.append(
                        f"一轮内解析出 {len(parsed.tool_calls)} 个工具调用，只执行第一个"
                    )

                # ---- 4. 预算拦截（执行前判定）
                blocked = self._check_budget(traj, call.name)
                if blocked is not None:
                    step = Step(
                        turn=turn, assistant_text=raw, thinking=parsed.thinking,
                        tool_call=call, elapsed_s=time.time() - t0,
                        warnings=warnings,
                    )
                    from ..tools.base import ToolResult

                    step.tool_result = ToolResult(
                        ok=False, content=blocked, error="over_budget"
                    )
                    traj.steps.append(step)
                    self._append_exchange(traj, call, raw, blocked)

                    if self.cfg.force_finish_on_budget:
                        # 按**被拦截的工具**判定终止原因，避免多个预算同时耗尽时
                        # 报出一个与实际原因不符的标签
                        traj.terminated_reason = {
                            "execute_sql": STOP_MAX_SQL,
                            "retrieve_context": STOP_MAX_CONTEXT,
                        }.get(call.name, STOP_MAX_TURNS)
                        return traj
                    continue

                # ---- 5. 执行工具
                result = self.registry.dispatch(call)
                step = Step(
                    turn=turn, assistant_text=raw, thinking=parsed.thinking,
                    tool_call=call, tool_result=result,
                    elapsed_s=time.time() - t0, warnings=warnings,
                )
                traj.steps.append(step)
                self._append_exchange(traj, call, raw, result.content)

                # ---- 6. 终止判定
                if result.terminal:
                    traj.final_answer = result.payload
                    traj.terminated_reason = STOP_FINAL_ANSWER
                    return traj

                # ---- 7. 字符预算
                if traj.char_length() > self.cfg.max_chars:
                    traj.terminated_reason = STOP_BUDGET_CHARS
                    traj.error = (
                        f"上下文超过 {self.cfg.max_chars} 字符预算，已强制终止"
                    )
                    return traj

            # 轮次耗尽
            traj.terminated_reason = STOP_MAX_TURNS
            return traj

        except Exception as e:  # noqa: BLE001 - 轨迹级兜底
            traj.terminated_reason = STOP_ERROR
            traj.error = f"{type(e).__name__}: {e}"
            return traj

    # ------------------------------------------------------------------ #
    def _check_budget(self, traj: Trajectory, tool_name: str) -> str | None:
        """返回 None 表示可以执行；返回字符串表示被拦截的原因（回灌给模型）。"""
        if tool_name == "execute_sql" and traj.n_sql_calls >= self.cfg.max_sql_calls:
            return (
                f"已达到 SQL 查询预算上限（{self.cfg.max_sql_calls} 次）。"
                f"请基于已有结果调用 final_answer 给出结论。"
            )
        if tool_name == "retrieve_context" and traj.n_context_calls >= self.cfg.max_context_calls:
            return (
                f"已达到上下文检索预算上限（{self.cfg.max_context_calls} 次）。"
                f"请直接写 SQL 或用已有信息继续。"
            )
        return None

    # ------------------------------------------------------------------ #
    def _append_plain_assistant(self, traj: Trajectory, raw: str) -> None:
        traj.messages.append({"role": "assistant", "content": raw})

    def _append_exchange(
        self,
        traj: Trajectory,
        call: Any,
        raw: str,
        observation: str,
    ) -> None:
        """把「助手调用 + 工具观测」写回消息历史。

        structured 风格走原生 tool 角色（Qwen 模板支持 tool_response）；
        text 风格把观测拼进 user 消息，兼容性最好但丢掉原生结构。
        """
        if self.style == STYLE_TEXT:
            traj.messages.append({"role": "assistant", "content": raw})
            traj.messages.append({
                "role": "user",
                "content": f"【工具 {call.name} 返回】\n{observation}",
            })
            return

        traj.messages.append({
            "role": "assistant",
            "content": raw,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
            ],
        })
        traj.messages.append({"role": "tool", "content": observation})
