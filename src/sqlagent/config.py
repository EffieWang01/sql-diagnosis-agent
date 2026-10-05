"""全局配置：数据类定义 + YAML 加载。

设计原则：所有可调参数集中在此，训练/评测脚本不写死任何魔法数字。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# 项目根目录（本文件位于 <root>/src/sqlagent/config.py）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    """目录约定。相对路径一律相对于项目根目录。"""

    data_dir: str = "data"
    raw_dir: str = "data/raw"
    db_dir: str = "data/db"
    context_dir: str = "data/context"
    outputs_dir: str = "outputs"
    tasks_dir: str = "data/tasks"

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def db_path(self) -> Path:
        return self.resolve(self.db_dir) / "main.duckdb"

    @property
    def trajectory_dir(self) -> Path:
        return self.resolve(self.outputs_dir) / "trajectories"


@dataclass
class SandboxConfig:
    """DuckDB 沙箱约束。这些是「安全边界」，同时也是奖励里的违规判定依据。"""

    # 单次查询最多返回多少行（超出即截断，并置 truncated=True）
    max_rows: int = 50
    # 单次查询超时（秒），超时通过 con.interrupt() 中断
    timeout_s: float = 15.0
    # 允许的语句类型（只读白名单）
    allowed_statements: tuple[str, ...] = (
        "SELECT",
        "WITH",
        "DESCRIBE",
        "SUMMARIZE",
        "EXPLAIN",
        "SHOW",
        "FROM",
        "VALUES",
    )
    # 禁止出现在语句中的关键字（防绕过，如 SELECT ... INTO / ATTACH）
    forbidden_keywords: tuple[str, ...] = (
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE",
        "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "INSTALL", "LOAD",
        "CALL", "PRAGMA", "SET", "BEGIN", "COMMIT", "ROLLBACK", "VACUUM",
    )
    # 禁止在语句中出现的文件/网络函数（防数据外泄与越权读盘）
    forbidden_functions: tuple[str, ...] = (
        "read_csv", "read_parquet", "read_json", "read_csv_auto",
        "read_text", "read_blob", "glob", "httpfs", "install", "load",
    )


@dataclass
class AgentConfig:
    """Agent 循环的预算与行为。"""

    # 最多多少轮「模型输出 → 工具执行」
    max_turns: int = 10
    # 最多调用多少次 execute_sql（与 max_turns 独立，用于「冗余查询率」统计）
    max_sql_calls: int = 8
    # 最多调用多少次 retrieve_context
    max_context_calls: int = 3
    # 是否允许在未调用 final_answer 时因超预算而强制结束
    force_finish_on_budget: bool = True
    # 单条轨迹的字符预算（粗略控制上下文膨胀，训练时再换成 token 计数）
    max_chars: int = 60_000
    # 观测结果截断：超过多少字符就截断写回上下文
    observation_max_chars: int = 2_000


@dataclass
class ModelConfig:
    """学生模型 / 推理配置。"""

    model_path: str = "/root/autodl-tmp/models/Qwen3.5-4B"
    # Qwen3.5 默认开 thinking，多轮 Agent 循环里必须关掉，否则上下文爆炸
    enable_thinking: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    # 生成用 bf16 更快（4-bit 有反量化开销）；训练用 4-bit QLoRA 省显存
    load_in_4bit_for_training: bool = True
    use_cache: bool = True


@dataclass
class RewardConfig:
    """可验证奖励权重。数值与《SQL多轮数据分析Agent技术方案》保持一致。"""

    w_final_diagnosis: float = 0.55
    w_key_numbers: float = 0.20
    w_evidence: float = 0.15
    w_recovery: float = 0.10

    p_illegal_sql: float = -0.03
    p_duplicate_sql: float = -0.02
    p_over_budget: float = -0.10

    # 关键数值判定的相对容差
    numeric_rtol: float = 0.01
    # 判定为「重复查询」的 SQL 归一化后相似度阈值
    duplicate_similarity: float = 0.95


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "AppConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            paths=PathsConfig(**raw.get("paths", {})),
            sandbox=SandboxConfig(**raw.get("sandbox", {})),
            agent=AgentConfig(**raw.get("agent", {})),
            model=ModelConfig(**raw.get("model", {})),
            reward=RewardConfig(**raw.get("reward", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump_yaml(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def ensure_dirs(self) -> None:
        for d in (
            self.paths.raw_dir,
            self.paths.db_dir,
            self.paths.context_dir,
            self.paths.tasks_dir,
            self.paths.trajectory_dir,
        ):
            self.paths.resolve(d).mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """加载配置；不传则用 configs/default.yaml。"""
    if path is None:
        path = PROJECT_ROOT / "configs" / "default.yaml"
    p = Path(path)
    if not p.exists():
        return AppConfig()
    return AppConfig.from_yaml(p)
