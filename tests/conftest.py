"""pytest 共享 fixture。"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlagent.config import AgentConfig, SandboxConfig  # noqa: E402
from sqlagent.env.context import ContextEntry, ContextStore  # noqa: E402
from sqlagent.env.sandbox import SQLSandbox  # noqa: E402


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """一个只有几行数据的小库，用于沙箱与循环测试。"""
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE sales (order_id INT, order_date DATE, region VARCHAR, "
        "category VARCHAR, gmv DOUBLE)"
    )
    con.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?, ?)",
        [
            (1, "2026-02-01", "华东", "手机", 100.0),
            (2, "2026-02-02", "华东", "家电", 200.0),
            (3, "2026-03-01", "华东", "手机", 70.0),
            (4, "2026-03-02", "华东", "家电", 198.0),
            (5, "2026-03-03", "华北", "手机", 150.0),
        ],
    )
    con.close()
    return db


@pytest.fixture
def sandbox(tmp_db: Path) -> SQLSandbox:
    sb = SQLSandbox(tmp_db, SandboxConfig(max_rows=3, timeout_s=5.0))
    yield sb
    sb.close()


@pytest.fixture
def store() -> ContextStore:
    return ContextStore([
        ContextEntry(
            id="metric.gmv", kind="metric", title="GMV（成交总额）",
            text="GMV 是订单成交金额合计，口径为 sales.gmv 求和，包含退款订单的原始成交额。",
            tags=["gmv", "成交总额"], tables=["sales"],
        ),
        ContextEntry(
            id="metric.region", kind="metric", title="区域口径",
            text="region 取值为 华东 / 华北 / 华南 / 西南。",
            tags=["区域", "华东"], tables=["sales"],
        ),
    ])


@pytest.fixture
def agent_cfg() -> AgentConfig:
    return AgentConfig(max_turns=4, max_sql_calls=3, max_context_calls=1, max_chars=20000)
