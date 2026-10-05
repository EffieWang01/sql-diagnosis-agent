"""策略（Policy）接口 —— AgentLoop 与训练框架之间的唯一耦合点。

AgentLoop 只要求实现 ``generate(messages, tools) -> str``。
这样一来同一套环境代码可以无缝切换：

  - ``ScriptedPolicy``  ：脚本化，离线可跑，用于测试与 demo
  - ``HFPolicy``        ：transformers 本地推理（bf16 或 4-bit）
  - 未来的 vLLM / TRL   ：只需再实现一次 generate

这个设计是刻意为之——技术方案里 P3 要从 TRL 切到 vLLM 做 rollout，
如果 Loop 直接依赖 transformers，那次切换就得重写循环。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .prompt import extract_input_ids, render_prompt


@runtime_checkable
class Policy(Protocol):
    """策略协议。"""

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str: ...


# ====================================================================== #
class ScriptedPolicy:
    """按预设脚本依次返回动作。

    超出脚本长度后返回最后一条（通常是 final_answer），保证循环能收敛。
    这是让 demo 与测试**完全确定性**的关键——不依赖模型、不依赖网络。
    """

    def __init__(self, script: list[str], loop_last: bool = True) -> None:
        self.script = list(script)
        self.loop_last = loop_last
        self.cursor = 0
        self.calls: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            return ""
        if self.cursor < len(self.script):
            out = self.script[self.cursor]
            self.cursor += 1
        else:
            out = self.script[-1] if self.loop_last else ""
        return out

    def reset(self) -> None:
        self.cursor = 0
        self.calls.clear()


# ====================================================================== #
class HFPolicy:
    """基于 transformers 的本地推理策略。

    精度策略（依据实测基准）：
      - 生成用 **bf16 比 4-bit 快约 28%**（4-bit 有反量化开销），
        显存充裕时优先 bf16；显存紧张再退回 4-bit。
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        enable_thinking: bool = False,
        use_cache: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.use_cache = use_cache
        self._call_index = 0

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        load_in_4bit: bool = False,
        device_map: str | dict | None = "auto",
        **gen_kwargs: Any,
    ) -> "HFPolicy":
        """加载模型并构造策略。训练相关依赖只在真正用到这里时才导入。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.eval()
        return cls(model, tokenizer, **gen_kwargs)

    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        import torch

        encoded = render_prompt(
            self.tokenizer,
            messages,
            tools=tools,
            enable_thinking=self.enable_thinking,
        )
        input_ids = extract_input_ids(encoded)
        # transformers 5.x 返回 dict，需要显式带上 attention_mask
        attention_mask = (
            encoded["attention_mask"] if hasattr(encoded, "keys") else None
        )

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        temperature = float(kwargs.get("temperature", self.temperature))
        do_sample = temperature > 1e-6
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(kwargs.get("max_new_tokens", self.max_new_tokens)),
            "do_sample": do_sample,
            "use_cache": self.use_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = float(kwargs.get("top_p", self.top_p))
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)

        new_tokens = out[0][input_ids.shape[-1]:]
        self._call_index += 1
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False)

    def reset(self) -> None:
        self._call_index = 0


# ====================================================================== #
def format_tool_call_xml(name: str, arguments: dict[str, Any]) -> str:
    """把结构化调用渲染成 Qwen 原生 XML 文本。

    ScriptedPolicy 用它来「假装」模型输出，保证测试走的是同一条解析路径。

    非字符串参数必须用 ``json.dumps`` 序列化——直接用 ``str()`` 会得到
    Python repr（单引号），解析端无法当 JSON 读，列表/字典参数会被静默降级成字符串。
    """
    import json as _json

    lines = ["<tool_call>", f"<function={name}>"]
    for k, v in arguments.items():
        if isinstance(v, str):
            payload = v
        else:
            payload = _json.dumps(v, ensure_ascii=False, indent=2)
        lines.append(f"<parameter={k}>")
        lines.append(payload)
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)
