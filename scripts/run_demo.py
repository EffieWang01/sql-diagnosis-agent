#!/usr/bin/env python
"""端到端演示：跑通「模型决策 → 工具执行 → 观测回灌 → 提交结论 → 算奖励」。

默认使用 ScriptedPolicy（脚本化策略），**不依赖任何模型与网络**，
这样在任何机器上都能复现同一条轨迹，适合用来验证骨架是否完好。

用法::

    python scripts/run_demo.py                 # 离线脚本化演示
    python scripts/run_demo.py --model         # 用真实模型（需 GPU + 已下载权重）
    python scripts/run_demo.py --style text    # 切换观测回灌风格
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlagent.agent.policy import ScriptedPolicy, format_tool_call_xml  # noqa: E402
from sqlagent.config import load_config  # noqa: E402
from sqlagent.eval.metrics import evaluate  # noqa: E402
from sqlagent.reward.verifier import Verifier  # noqa: E402
from sqlagent.runner import build_runtime  # noqa: E402
from sqlagent.utils.io import read_jsonl, write_json  # noqa: E402


# ---------------------------------------------------------------------- #
def good_script(task: dict) -> list[str]:
    """一条「教科书式」的正确轨迹：先确认异常区域，再下钻到品类，最后给结论。

    说明：demo 里 final_answer 的数值直接取自任务参考答案——
    这是**演示专用的捷径**，真实 Agent 必须自己从 SQL 结果里算出来。
    这里这么写是为了让 demo 能稳定展示「满分路径」。
    """
    nums = (task.get("reference") or {}).get("key_numbers") or {}
    e_chg = nums.get("华东区GMV环比变化率", -0.1)
    p_chg = nums.get("手机品类GMV环比变化率", -0.2)
    contrib = nums.get("手机品类对整体下滑的贡献度", 1.0)

    sql_region = (
        "SELECT region, strftime(order_date, '%Y-%m') AS month, SUM(gmv) AS gmv "
        "FROM sales WHERE strftime(order_date, '%Y-%m') IN ('2026-02', '2026-03') "
        "GROUP BY 1, 2 ORDER BY 1, 2"
    )
    sql_category = (
        "SELECT category, SUM(gmv) AS gmv FROM sales "
        "WHERE region = '华东' AND strftime(order_date, '%Y-%m') = '2026-03' "
        "GROUP BY 1 ORDER BY gmv DESC"
    )
    sql_phone = (
        "SELECT strftime(order_date, '%Y-%m') AS month, SUM(gmv) AS gmv "
        "FROM sales WHERE region = '华东' AND category = '手机' "
        "AND strftime(order_date, '%Y-%m') IN ('2026-02', '2026-03') GROUP BY 1 ORDER BY 1"
    )
    # 一条把三个指标一次算出来的 SQL——三个结论都引用它，
    # 这样每条 key_finding 的数值都能在真实结果里找到出处（证据忠实度满分）。
    sql_metrics = (
        "WITH region_month AS ("
        "  SELECT strftime(order_date, '%Y-%m') AS m, SUM(gmv) AS gmv "
        "  FROM sales WHERE region = '华东' GROUP BY 1"
        "), phone_month AS ("
        "  SELECT strftime(order_date, '%Y-%m') AS m, SUM(gmv) AS gmv "
        "  FROM sales WHERE region = '华东' AND category = '手机' GROUP BY 1"
        ") "
        "SELECT "
        "  (SELECT gmv FROM region_month WHERE m = '2026-03') "
        "    / (SELECT gmv FROM region_month WHERE m = '2026-02') - 1 AS region_gmv_change, "
        "  (SELECT gmv FROM phone_month WHERE m = '2026-03') "
        "    / (SELECT gmv FROM phone_month WHERE m = '2026-02') - 1 AS phone_gmv_change, "
        "  ((SELECT gmv FROM phone_month WHERE m = '2026-03') - (SELECT gmv FROM phone_month WHERE m = '2026-02')) "
        "    / ((SELECT gmv FROM region_month WHERE m = '2026-03') - (SELECT gmv FROM region_month WHERE m = '2026-02')) "
        "    AS phone_contribution"
    )

    return [
        # 1) 先查口径（演示 retrieve_context）
        format_tool_call_xml("retrieve_context", {"query": "GMV 口径 区域 月份"}),

        # 2) 区域 × 月份 总览，确认异常落在华东
        format_tool_call_xml("execute_sql", {
            "sql": sql_region, "purpose": "确认下滑发生在哪个大区",
        }),

        # 3) 下钻到华东区品类
        format_tool_call_xml("execute_sql", {
            "sql": sql_category, "purpose": "在异常区域内部按品类拆解",
        }),

        # 4) 量化手机品类环比降幅
        format_tool_call_xml("execute_sql", {
            "sql": sql_phone, "purpose": "量化手机品类环比降幅",
        }),

        # 5) 一次性算出三个关键指标
        format_tool_call_xml("execute_sql", {
            "sql": sql_metrics, "purpose": "计算环比变化率与贡献度",
        }),

        # 6) 提交结论（evidence_sql 必须与真实执行过的 SQL 一致）
        format_tool_call_xml("final_answer", {
            "diagnosis": (
                "华东区 2026 年 3 月 GMV 下滑的主因是手机品类。"
                "手机品类环比大幅下滑，降幅明显大于华东区整体，"
                "且贡献了全部降幅，说明下滑高度集中在这一单一品类，"
                "其余品类基本持平甚至微涨。"
            ),
            "root_cause_dimension": "手机品类",
            "key_findings": [
                {
                    "claim": "华东区整体 GMV 环比下滑",
                    "value": e_chg,
                    "evidence_sql": sql_metrics,
                },
                {
                    "claim": "华东区手机品类 GMV 环比下滑幅度显著大于整体",
                    "value": p_chg,
                    "evidence_sql": sql_metrics,
                },
                {
                    "claim": "手机品类贡献了华东区整体降幅的全部（贡献度 ≥100%）",
                    "value": contrib,
                    "evidence_sql": sql_metrics,
                },
            ],
            "confidence": 0.9,
        }),
    ]


def bad_script() -> list[str]:
    """一条「反面教材」：非法语句 + 重复查询 + 无证据结论。

    用来验证负向奖励确实生效——如果奖励函数写错了，这条轨迹也能拿高分，
    那就是训练信号坏了。
    """
    dup_sql = "SELECT SUM(gmv) FROM sales WHERE region = '华东'"
    return [
        format_tool_call_xml("execute_sql", {"sql": "DROP TABLE sales", "purpose": "试试"}),
        format_tool_call_xml("execute_sql", {"sql": dup_sql, "purpose": "看总数"}),
        format_tool_call_xml("execute_sql", {"sql": dup_sql, "purpose": "再看一次"}),
        format_tool_call_xml("execute_sql", {"sql": "SELECT * FROM 不存在的表", "purpose": "瞎试"}),
        format_tool_call_xml("final_answer", {
            "diagnosis": "可能是天气原因导致的。",
            "root_cause_dimension": "天气",
            "key_findings": [
                {"claim": "我觉得是天气", "value": 12345.0,
                 "evidence_sql": "SELECT 1 -- 这条根本没执行过"},
            ],
            "confidence": 0.99,
        }),
    ]


# ---------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="端到端演示")
    ap.add_argument("--task", default=None, help="任务文件路径")
    ap.add_argument("--script", choices=["good", "bad"], default="good")
    ap.add_argument("--style", choices=["structured", "text"], default="structured")
    ap.add_argument("--model", action="store_true", help="使用真实模型推理")
    ap.add_argument("--save", action="store_true", help="保存轨迹与报告")
    args = ap.parse_args()

    cfg = load_config()
    task_path = Path(args.task) if args.task else (
        cfg.paths.resolve(cfg.paths.tasks_dir) / "demo_tasks.jsonl"
    )
    tasks = list(read_jsonl(task_path))
    if not tasks:
        print(f"未找到任务：{task_path}\n请先运行 python scripts/prepare_data.py --demo")
        return 1
    task = tasks[0]

    # ---- 选择策略 ----
    if args.model:
        from sqlagent.agent.policy import HFPolicy

        print(f"[模型] 加载 {cfg.model.model_path} ...")
        policy = HFPolicy.from_pretrained(
            cfg.model.model_path,
            load_in_4bit=False,  # 生成用 bf16 更快（实测快约 28%）
            max_new_tokens=cfg.model.max_new_tokens,
            temperature=cfg.model.temperature,
            top_p=cfg.model.top_p,
            enable_thinking=cfg.model.enable_thinking,
        )
    else:
        script = good_script(task) if args.script == "good" else bad_script()
        policy = ScriptedPolicy(script)
        print(f"[策略] ScriptedPolicy（{args.script} 轨迹，{len(script)} 步）")

    style = "structured" if args.style == "structured" else "text"
    runtime = build_runtime(policy, cfg, tool_response_style=style)

    print(f"\n{'=' * 70}\n任务：{task['question']}\n{'=' * 70}")
    print(f"可用工具：{', '.join(runtime.registry.names())}")
    print(f"语义上下文条目：{len(runtime.store)}\n")

    traj = runtime.run(task)

    # ---- 打印轨迹 ----
    for s in traj.steps:
        print(f"\n--- 第 {s.turn + 1} 轮 ---")
        if s.tool_call:
            print(f"  调用 {s.tool_call.name}")
            for k, v in s.tool_call.arguments.items():
                v_str = str(v).replace("\n", " ")
                print(f"    {k}: {v_str[:150]}{'...' if len(v_str) > 150 else ''}")
        if s.tool_result:
            obs = s.tool_result.content.replace("\n", "\n    ")
            print(f"  观测（{'成功' if s.tool_result.ok else '失败'}）:\n    {obs[:600]}")
        if s.warnings:
            print(f"  ⚠ {s.warnings}")

    # ---- 奖励 ----
    verifier = Verifier(cfg.reward)
    rb = verifier.score(traj)

    print(f"\n{'=' * 70}\n终止原因：{traj.terminated_reason}")
    if traj.error:
        print(f"错误：{traj.error}")
    print(f"\n最终结论：{(traj.final_answer or {}).get('diagnosis', '（未提交）')}")
    print(f"根因维度：{(traj.final_answer or {}).get('root_cause_dimension', '（无）')}")

    print(f"\n奖励明细（总分 {rb.total:+.4f}）：")
    for k, v in rb.components.items():
        print(f"  {k:20s} {v:+.4f}")
    print(f"  任务成功：{rb.task_success}")

    # ---- 指标 ----
    report = evaluate([traj], verifier, breakdowns=[rb])
    print(f"\n{'=' * 70}\n评测指标\n{report.to_markdown()}")

    # ---- 保存 ----
    if args.save:
        out_dir = cfg.paths.trajectory_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        tpath = write_json(traj.to_dict(), out_dir / f"{traj.task_id}_{args.script}.json")
        rpath = write_json(rb.to_dict(), out_dir / f"{traj.task_id}_{args.script}_reward.json")
        print(f"\n已保存：\n  {tpath}\n  {rpath}")

    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
