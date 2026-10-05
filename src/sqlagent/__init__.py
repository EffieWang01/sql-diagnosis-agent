"""数据库助手 —— 面向业务诊断的多轮 SQL 数据分析 Agent。

分层：
    env/     只读 DuckDB 沙箱 + 数据装载 + 语义上下文检索
    tools/   三个工具（retrieve_context / execute_sql / final_answer）
    agent/   Prompt 构造 + 输出解析 + 轨迹状态 + AgentLoop + Policy 接口
    reward/  可验证奖励
    eval/    8 个评测指标
"""

from .config import AppConfig, load_config

__version__ = "0.1.0"

__all__ = ["AppConfig", "load_config", "__version__"]
