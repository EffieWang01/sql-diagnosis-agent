"""运行时装配：把配置 + 策略组装成一个可直接跑的 Agent。

这是唯一「知道所有部件」的地方。上层脚本（demo / 评测 / 训练数据生成）
都只跟这里打交道，避免到处重复装配逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent.loop import STYLE_STRUCTURED, AgentLoop
from .agent.policy import Policy
from .agent.prompt import PromptBuilder
from .agent.state import Trajectory
from .config import AppConfig, load_config
from .env.context import ContextStore
from .env.database import schema_ddl
from .env.sandbox import SQLSandbox
from .tools.base import ToolRegistry
from .tools.builtin import build_default_tools


@dataclass
class AgentRuntime:
    """装配好的运行时。用完记得 close()。"""

    config: AppConfig
    sandbox: SQLSandbox
    store: ContextStore
    registry: ToolRegistry
    prompt_builder: PromptBuilder
    loop: AgentLoop
    meta: dict[str, Any] = field(default_factory=dict)

    def run(self, task: dict[str, Any]) -> Trajectory:
        return self.loop.run(task)

    def run_many(self, tasks: list[dict[str, Any]]) -> list[Trajectory]:
        return [self.run(t) for t in tasks]

    def close(self) -> None:
        self.sandbox.close()

    def __enter__(self) -> "AgentRuntime":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------- #
def build_runtime(
    policy: Policy,
    config: AppConfig | None = None,
    db_path: str | Path | None = None,
    context_path: str | Path | None = None,
    tool_response_style: str = STYLE_STRUCTURED,
    extra_rules: str = "",
) -> AgentRuntime:
    """按配置装配运行时。"""
    cfg = config or load_config()
    cfg.ensure_dirs()

    db = Path(db_path) if db_path else cfg.paths.db_path
    if not Path(db).exists():
        raise FileNotFoundError(
            f"未找到 DuckDB 数据库：{db}。请先运行 scripts/prepare_data.py。"
        )

    sandbox = SQLSandbox(db, cfg.sandbox)

    ctx_path = Path(context_path) if context_path else (
        cfg.paths.resolve(cfg.paths.context_dir) / "context.json"
    )
    store = ContextStore.from_json(ctx_path)

    registry = ToolRegistry(
        build_default_tools(
            sandbox, store, observation_max_chars=cfg.agent.observation_max_chars
        )
    )

    prompt_builder = PromptBuilder(
        schema_ddl=schema_ddl(db),
        max_turns=cfg.agent.max_turns,
        max_sql_calls=cfg.agent.max_sql_calls,
        max_rows=cfg.sandbox.max_rows,
        timeout_s=cfg.sandbox.timeout_s,
        extra_rules=extra_rules,
    )

    loop = AgentLoop(
        registry=registry,
        policy=policy,
        prompt_builder=prompt_builder,
        config=cfg.agent,
        tool_response_style=tool_response_style,
    )

    return AgentRuntime(
        config=cfg,
        sandbox=sandbox,
        store=store,
        registry=registry,
        prompt_builder=prompt_builder,
        loop=loop,
        meta={
            "db_path": str(db),
            "context_path": str(ctx_path),
            "n_context_entries": len(store),
            "tools": registry.names(),
        },
    )
