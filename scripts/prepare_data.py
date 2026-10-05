#!/usr/bin/env python
"""数据准备：生成演示数据 → 装载 DuckDB → 构建语义上下文 → 生成任务文件。

关键设计：**参考答案不是手写的，而是用参考 SQL 从真实数据里跑出来的。**
这样保证 tasks.jsonl 里的数值与数据库严格一致，
可验证奖励不会因为人工笔误而出现「永远拿不到分」的假失败。

用法::

    python scripts/prepare_data.py --demo          # 生成演示数据并全流程准备
    python scripts/prepare_data.py                 # 只用 data/raw 下已有的 CSV
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# 让脚本能直接以 `python scripts/xxx.py` 运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb  # noqa: E402

from sqlagent.config import load_config  # noqa: E402
from sqlagent.env.context import ContextEntry, ContextStore  # noqa: E402
from sqlagent.env.database import (  # noqa: E402
    column_types,
    load_csvs_to_duckdb,
    profile_tables,
    schema_ddl,
)
from sqlagent.utils.io import ensure_dir, write_jsonl  # noqa: E402

REGIONS = ["华东", "华北", "华南", "西南"]
CATEGORIES = ["手机", "家电", "服饰", "食品"]
CHANNELS = ["APP", "小程序", "线下"]

# 各区域 / 品类的基础日 GMV（单位：元）
BASE_REGION = {"华东": 9_100_000, "华北": 7_900_000, "华南": 6_100_000, "西南": 3_400_000}
# 华东区各品类占比：手机占大头，这样手机下滑才能拖垮整体
CATEGORY_SHARE = {"手机": 0.49, "家电": 0.22, "服饰": 0.18, "食品": 0.11}

MONTHS = ["2026-01", "2026-02", "2026-03"]
DAYS_PER_MONTH = 28


# ---------------------------------------------------------------------- #
def generate_demo_csvs(raw_dir: Path) -> None:
    """生成带植入异常的销售数据。

    植入的异常：2026-03 华东区「手机」品类 GMV 相对 2026-02 下滑约 22%，
    其余区域/品类基本持平，从而在区域总盘上表现为约 -12%。
    """
    rng = random.Random(20261005)
    ensure_dir(raw_dir)

    sales_rows: list[list] = []
    customers: dict[int, dict] = {}
    products: dict[int, dict] = {}

    pid = 1
    for cat in CATEGORIES:
        for i in range(1, 6):
            products[pid] = {
                "product_id": pid,
                "category": cat,
                "brand": f"{cat}品牌{i}",
                "price": round(rng.uniform(80, 5200), 2),
            }
            pid += 1

    # 预建客户池：一个客户会有多笔订单，这样 sales JOIN customers 才有分析价值
    CUSTOMERS_PER_REGION = 900
    pool: dict[str, list[int]] = {}
    cid = 1
    for region in REGIONS:
        pool[region] = []
        for _ in range(CUSTOMERS_PER_REGION):
            customers[cid] = {
                "customer_id": cid,
                "region": region,
                "segment": rng.choices(
                    ["高价值", "成长", "普通"], weights=[0.15, 0.35, 0.50]
                )[0],
                "register_date": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            }
            pool[region].append(cid)
            cid += 1

    oid = 1
    for month in MONTHS:
        for day in range(1, DAYS_PER_MONTH + 1):
            date = f"{month}-{day:02d}"
            for region in REGIONS:
                region_base = BASE_REGION[region]
                for cat in CATEGORIES:
                    cat_base = region_base * CATEGORY_SHARE[cat]
                    # 基础乘子
                    mult = 1.0
                    # —— 植入异常 ——
                    if month == "2026-03" and region == "华东" and cat == "手机":
                        mult *= 0.78  # 下滑约 22%
                    elif month == "2026-03" and region == "华东":
                        mult *= rng.uniform(0.97, 1.00)  # 其它品类基本持平
                    elif month == "2026-03":
                        mult *= rng.uniform(1.00, 1.04)  # 其它区域略涨
                    elif month == "2026-01":
                        mult *= rng.uniform(0.96, 1.02)  # 1 月作噪声基线

                    day_gmv = cat_base / DAYS_PER_MONTH * mult
                    # 拆成若干订单
                    n_orders = max(3, int(day_gmv / rng.uniform(900, 2600)))
                    for _ in range(n_orders):
                        for ch in CHANNELS:
                            if rng.random() > 0.55:
                                continue
                            share = {"APP": 0.55, "小程序": 0.28, "线下": 0.17}[ch]
                            amount = round(
                                day_gmv * share / max(n_orders, 1) * rng.uniform(0.6, 1.4), 2
                            )
                            if amount <= 0:
                                continue
                            is_new = rng.random() < 0.22
                            cust = rng.choice(pool[region])
                            refund = round(amount * rng.uniform(0, 0.06), 2) if rng.random() < 0.08 else 0.0
                            sales_rows.append([
                                oid, date, region, cat, ch,
                                products[rng.choice(list(products))]["product_id"],
                                cust, amount, refund, int(is_new),
                            ])
                            oid += 1

    _write_csv(
        raw_dir / "sales.csv",
        ["order_id", "order_date", "region", "category", "channel",
         "product_id", "customer_id", "gmv", "refund_amount", "is_new_customer"],
        sales_rows,
    )
    _write_csv(
        raw_dir / "customers.csv",
        ["customer_id", "region", "segment", "register_date"],
        [list(c.values()) for c in customers.values()],
    )
    _write_csv(
        raw_dir / "products.csv",
        ["product_id", "category", "brand", "price"],
        [list(p.values()) for p in products.values()],
    )
    print(f"[数据] 生成 {len(sales_rows)} 条订单 / {len(customers)} 个客户 / {len(products)} 个商品")


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------- #
def build_context(db_path: Path, store_path: Path) -> ContextStore:
    """构建语义上下文：指标口径 + 表关系 + 已验证 SQL。

    注意这里**只写定义与范式，不写任何业务数值**。

    日期用法提示是**按实际推断出的列类型动态生成**的——
    否则换个数据集（DATE vs VARCHAR）口径就会写错，
    模型照着错的范式写 SQL 会一直被沙箱拒绝。
    """
    dtype = column_types(db_path, "sales")
    od_type = dtype.get("order_date", "VARCHAR").upper()
    is_temporal = od_type.startswith("DATE") or "TIMESTAMP" in od_type

    if is_temporal:
        month_expr = "strftime(order_date, '%Y-%m')"
        date_note = (
            f"order_date 是 **{od_type} 类型**（不是字符串）。按月筛选/分组请用 "
            f"`strftime(order_date, '%Y-%m') = '2026-03'`，"
            f"或 `date_trunc('month', order_date)`。"
            f"**不要用 substr(order_date, ...)，会报类型错误。**"
            f"比较日期请写 `order_date >= DATE '2026-03-01'`。"
        )
    else:
        month_expr = "substr(order_date, 1, 7)"
        date_note = (
            f"order_date 是 **{od_type} 类型**的字符串，格式 'YYYY-MM-DD'。"
            f"按月筛选请用 `substr(order_date, 1, 7) = '2026-03'`。"
        )

    entries = [
        ContextEntry(
            id="metric.gmv",
            kind="metric",
            title="GMV（成交总额）",
            text=(
                "GMV 指订单成交金额合计，口径为 sales.gmv 字段求和，"
                "**包含已退款订单的原始成交额**（退款需单独用 refund_amount 扣减）。"
                "不要用 gmv - refund_amount 冒充 GMV，那叫「净成交额」。"
            ),
            tags=["gmv", "成交总额", "销售额", "成交额"],
            tables=["sales"],
        ),
        ContextEntry(
            id="metric.net_gmv",
            kind="metric",
            title="净成交额",
            text="净成交额 = SUM(gmv) - SUM(refund_amount)。只有明确要求「扣除退款」时才用这个口径。",
            tags=["净成交额", "退款", "扣退款"],
            tables=["sales"],
        ),
        ContextEntry(
            id="metric.region",
            kind="metric",
            title="区域口径",
            text="region 字段取值为：华东 / 华北 / 华南 / 西南，共 4 个大区，不存在更细的省市粒度。",
            tags=["区域", "大区", "华东", "region"],
            tables=["sales"],
        ),
        ContextEntry(
            id="metric.category",
            kind="metric",
            title="品类口径",
            text="category 字段取值为：手机 / 家电 / 服饰 / 食品，共 4 个一级品类，没有二级品类维度。",
            tags=["品类", "类目", "手机", "category"],
            tables=["sales"],
        ),
        ContextEntry(
            id="metric.new_customer",
            kind="metric",
            title="新客判定",
            text="is_new_customer = 1 表示该订单来自新客。不要用注册日期推断，统一用这个字段。",
            tags=["新客", "新用户", "is_new_customer"],
            tables=["sales"],
        ),
        ContextEntry(
            id="metric.channel",
            kind="metric",
            title="渠道口径",
            text="channel 字段取值为：APP / 小程序 / 线下。",
            tags=["渠道", "channel", "app", "小程序", "线下"],
            tables=["sales"],
        ),
        ContextEntry(
            id="relation.sales_customers",
            kind="relation",
            title="sales 与 customers 的关联",
            text="sales.customer_id = customers.customer_id。customers 表有 segment（高价值/成长/普通）可用于客户分层分析。",
            tags=["关联", "join", "customer_id", "客户分层", "segment"],
            tables=["sales", "customers"],
        ),
        ContextEntry(
            id="relation.sales_products",
            kind="relation",
            title="sales 与 products 的关联",
            text="sales.product_id = products.product_id。products 表有 brand（品牌）与 price（标价）。",
            tags=["关联", "join", "product_id", "品牌", "brand"],
            tables=["sales", "products"],
        ),
        ContextEntry(
            id="verified_sql.monthly_region_trend",
            kind="verified_sql",
            title="已验证：区域月度 GMV 环比",
            text=(
                "按月聚合区域 GMV，用于确认异常发生在哪个大区：\n"
                f"SELECT region, {month_expr} AS month, SUM(gmv) AS gmv\n"
                "FROM sales GROUP BY 1, 2 ORDER BY 1, 2;"
            ),
            tags=["月度", "环比", "区域趋势", "trend"],
            tables=["sales"],
        ),
        ContextEntry(
            id="verified_sql.category_breakdown",
            kind="verified_sql",
            title="已验证：指定区域指定月的品类拆解",
            text=(
                "在确定异常区域后，下钻到品类维度：\n"
                "SELECT category, SUM(gmv) AS gmv FROM sales\n"
                f"WHERE region = '华东' AND {month_expr} = '2026-03'\n"
                "GROUP BY 1 ORDER BY gmv DESC;"
            ),
            tags=["品类拆解", "下钻", "drilldown", "category"],
            tables=["sales"],
        ),
        ContextEntry(
            id="verified_sql.channel_split",
            kind="verified_sql",
            title="已验证：按渠道拆分",
            text=(
                "SELECT channel, SUM(gmv) AS gmv FROM sales\n"
                f"WHERE region = '华东' AND {month_expr} = '2026-03'\n"
                "GROUP BY 1 ORDER BY gmv DESC;"
            ),
            tags=["渠道拆解", "channel"],
            tables=["sales"],
        ),
        ContextEntry(
            id="table_note.date_format",
            kind="table_note",
            title="日期字段用法",
            text=date_note,
            tags=["日期", "order_date", "月份", "时间", "strftime", "date_trunc", "substr"],
            tables=["sales"],
        ),
    ]

    # 追加自动抽取的表画像（结构信息，不含业务聚合值）
    try:
        for tp in profile_tables(db_path):
            lines = [f"{c.name}（{c.dtype}）" for c in tp.columns]
            entries.append(
                ContextEntry(
                    id=f"profile.{tp.name}",
                    kind="table_note",
                    title=f"表 {tp.name} 的列清单",
                    text=f"共 {tp.row_count} 行。字段：{', '.join(lines)}",
                    tags=[tp.name] + [c.name for c in tp.columns],
                    tables=[tp.name],
                )
            )
    except Exception as e:  # 画像失败不影响主流程
        print(f"[上下文] 表画像抽取失败（已跳过）：{e}")

    store = ContextStore(entries)
    store.to_json(store_path)
    print(f"[上下文] 写入 {len(store)} 条语义条目 → {store_path}")
    return store


# ---------------------------------------------------------------------- #
def _scalar(con, sql: str, params: list | None = None):
    row = con.execute(sql, params or []).fetchone()
    return row[0] if row else None


def build_tasks(db_path: Path, task_path: Path) -> int:
    """用参考 SQL 从真实数据反算答案，生成任务文件。

    这样做的好处：参考答案永远与数据库一致，不会因为手写笔误
    导致可验证奖励出现「怎么答都拿不到分」的假失败。
    """
    # 按实际列类型选择月份表达式
    od_type = column_types(db_path, "sales").get("order_date", "VARCHAR").upper()
    if od_type.startswith("DATE") or "TIMESTAMP" in od_type:
        M = "strftime(order_date, '%Y-%m')"
    else:
        M = "substr(order_date, 1, 7)"

    con = duckdb.connect(str(db_path), read_only=True)

    def month_sum(region: str, month: str, category: str | None = None):
        sql = f"SELECT SUM(gmv) FROM sales WHERE region = ? AND {M} = ?"
        params: list = [region, month]
        if category:
            sql += " AND category = ?"
            params.append(category)
        return _scalar(con, sql, params)

    try:
        e02 = month_sum("华东", "2026-02")
        e03 = month_sum("华东", "2026-03")
        e_chg = (e03 - e02) / e02 if e02 else 0.0

        p02 = month_sum("华东", "2026-02", "手机")
        p03 = month_sum("华东", "2026-03", "手机")
        p_chg = (p03 - p02) / p02 if p02 else 0.0

        # 手机品类对整体下滑的贡献度
        contrib = (p03 - p02) / (e03 - e02) if (e03 - e02) else 0.0

        # 其它品类（用于确认只有手机异常）
        others = con.execute(
            f"SELECT category, SUM(gmv) FROM sales WHERE region = '华东' "
            f"AND {M} = '2026-03' AND category <> '手机' GROUP BY 1"
        ).fetchall()
    finally:
        con.close()

    tasks = [
        {
            "task_id": "demo_east_china_2026_03",
            "question": (
                f"2026 年 3 月华东区的 GMV 相比 2 月下滑了约 {abs(e_chg):.1%}，"
                "管理层要求定位原因。请找出这次下滑主要出在哪个维度，"
                "给出根因结论，并用数据说明该维度贡献了多少降幅。"
            ),
            "reference": {
                "root_cause_dimension": "手机品类",
                "key_numbers": {
                    "华东区GMV环比变化率": round(float(e_chg), 6),
                    "手机品类GMV环比变化率": round(float(p_chg), 6),
                    "手机品类对整体下滑的贡献度": round(float(contrib), 6),
                },
                "expected_sql_fragments": [
                    "region='华东'",
                    "category='手机'",
                ],
            },
            "meta": {
                "difficulty": "easy",
                "hop": 2,
                "generated_by": "prepare_data.py",
                "ground_truth_debug": {
                    "east_2026_02": float(e02 or 0),
                    "east_2026_03": float(e03 or 0),
                    "phone_2026_02": float(p02 or 0),
                    "phone_2026_03": float(p03 or 0),
                    "other_categories_2026_03": {c: float(v) for c, v in others},
                },
            },
        }
    ]

    n = write_jsonl(tasks, task_path)
    print(f"[任务] 写入 {n} 条任务 → {task_path}")
    print(f"       华东 2026-03 环比 {e_chg:+.2%} | 手机品类 {p_chg:+.2%} | 贡献度 {contrib:.1%}")
    return n


# ---------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="准备数据、上下文与任务")
    ap.add_argument("--demo", action="store_true", help="生成演示数据")
    ap.add_argument("--config", default=None, help="配置文件路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()

    raw_dir = cfg.paths.resolve(cfg.paths.raw_dir)
    db_path = cfg.paths.db_path
    store_path = cfg.paths.resolve(cfg.paths.context_dir) / "context.json"
    task_path = cfg.paths.resolve(cfg.paths.tasks_dir) / "demo_tasks.jsonl"

    if args.demo or not list(raw_dir.glob("*.csv")):
        generate_demo_csvs(raw_dir)

    tables = load_csvs_to_duckdb(raw_dir, db_path)
    print(f"[数据库] 装载 {len(tables)} 张表 → {db_path}：{', '.join(tables)}")

    build_context(db_path, store_path)
    build_tasks(db_path, task_path)

    print("\n[Schema]\n" + schema_ddl(db_path))
    print("\n完成。下一步：python scripts/run_demo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
