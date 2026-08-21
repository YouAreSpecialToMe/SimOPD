#!/usr/bin/env python3
"""把一波 run 的 in-loop 全键曲线导成 make_dynamics_page.py 吃的 CSV。

    # 集群上跑(wandb 凭据在 simopd_env.sh 里)
    python scripts/export_wave_metrics.py --since 2026-08-19 \
        --out docs/data/training_metrics_corr_allkeys.csv.gz

两份已入库的 CSV(training_metrics_16k_allkeys / _exp_allkeys)覆盖的是 corr 波之前的
43 条臂;corr/n0/c5 这一整波(vanilla_corr、各 *_corr、*_n0、c4_rep/c4_carrier、
c5_union_*)一条都不在里面,所以动态页看不到当前战役。这个脚本补那一份。

schema 与既有两份一致:arm(去掉 _s<seed>[_16k] 后缀)、seed、step,其余列是 wandb 原键。
同名 run 取按创建时间排序后逐步覆盖(重投会留下同名的 crashed 旧 run,后写的赢),
这与 export_corr.py 的既有做法相同。
"""
import argparse
import os
import re
import sys

import pandas as pd
import wandb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_KEY = "training/global_step"
SKIP = ("_rehearsal", "_dry", "_smoke")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("PROJECT_NAME", "simopd"))
    ap.add_argument("--since", default="2026-08-19",
                    help="只取创建时间 >= 该日期的 run(corr 波起点)")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/data/training_metrics_corr_allkeys.csv.gz"))
    ap.add_argument("--samples", type=int, default=4000)
    # 名字撞车会静默毁掉动态页:make_dynamics_page 把几份 dump concat 起来后按
    # (arm, seed) 分组再 set_index("step"),同名同 seed 的两份数据会让 index 出现重复,
    # ser.get(step) 于是返回 Series 而不是标量,r4() 里 pd.isna 直接抛 ValueError。
    # 实测撞的是 a2_coldstart:入库那条是 16k 战役的,08-19 之后又按 v2 契约重跑过一轮,
    # 同名不同实验 —— 合成一条曲线是错的,所以默认排除掉已在其它 dump 里的臂。
    ap.add_argument("--dedupe-against", nargs="*", default=[
        os.path.join(ROOT, "docs/data/training_metrics_16k_allkeys.csv.gz"),
        os.path.join(ROOT, "docs/data/training_metrics_exp_allkeys.csv.gz")],
        help="这些 dump 里已有的臂名一律排除;传空串关闭")
    a = ap.parse_args()

    taken = set()
    for p in a.dedupe_against or []:
        if p and os.path.exists(p):
            taken |= set(pd.read_csv(p, usecols=["arm"]).arm.unique())

    api = wandb.Api(timeout=120)
    ent = os.environ.get("WANDB_ENTITY") or api.default_entity
    proj = f"{ent}/{a.project}" if ent else a.project
    runs = list(api.runs(proj, per_page=800))
    print(f"project {proj}: {len(runs)} runs", file=sys.stderr)

    want = []
    for r in runs:
        if any(s in r.name for s in SKIP):
            continue
        if str(r.created_at) < a.since:
            continue
        m = re.search(r"_s(\d+)(_16k)?$", r.name)
        if not m:
            continue
        arm = r.name[: m.start()]
        if arm in taken:
            continue
        want.append((arm, int(m.group(1)), r))
    print(f"命中 {len(want)} run,{len({w[0] for w in want})} 臂", file=sys.stderr)

    # 先抓历史,再决定取舍 —— 因为"从头重开"要靠各 run 的起始 step 才能认出来。
    hists = {}
    for arm, seed, r in sorted(want, key=lambda x: str(x[2].created_at)):
        try:
            hists[id(r)] = r.history(samples=a.samples, pandas=False)
        except Exception as e:
            print(f"  ! {r.name}: history 失败 {e!r}", file=sys.stderr)
            hists[id(r)] = []

    # 断代规则(2026-08-22):同名 run 里,最后一次从 step<=2 起跑的那个是当前轨迹的
    # 原点,在它之前创建的一律丢弃。重投续跑的 run 起始 step 远大于 2,不受影响;
    # 只有"挪开 ckpt 从头重开"才切断历史 —— 而那正是语义变了的时候。
    # 不这么做的后果是实测过的:h9_prune_adapt_n0 修复前后各跑过一轮,修复后的两段是
    # 2-49 与 52-251,step 50-51 是空档,坏中继那轮的数据就补了进去 —— 一条曲线上
    # 混着"预算没人写、长度 12k"和"预算 1000、长度 800"两种语义。宁可留空档。
    origin = {}
    for arm, seed, r in sorted(want, key=lambda x: str(x[2].created_at)):
        steps = [row.get(STEP_KEY) for row in hists[id(r)] if row.get(STEP_KEY) is not None]
        if steps and min(steps) <= 2:
            origin[(arm, seed)] = str(r.created_at)
    kept, dropped = [], []
    for arm, seed, r in sorted(want, key=lambda x: str(x[2].created_at)):
        o = origin.get((arm, seed))
        (dropped if (o and str(r.created_at) < o) else kept).append((arm, seed, r))
    for arm, seed, r in dropped:
        print(f"  - 断代丢弃 {r.name} (创建于 {str(r.created_at)[:19]},早于本轨迹原点 "
              f"{origin[(arm, seed)][:19]})", file=sys.stderr)

    # (arm, seed, step) -> {key: value};同名 run 按创建时间升序,后写覆盖先写
    table = {}
    for arm, seed, r in kept:
        n = 0
        for row in hists[id(r)]:
            st = row.get(STEP_KEY)
            if st is None:
                continue
            dst = table.setdefault((arm, seed, int(st)), {})
            for k, v in row.items():
                if k.startswith("_") or k == STEP_KEY or v is None:
                    continue
                if isinstance(v, (int, float, bool)):
                    dst[k] = v
            n += 1
        print(f"  {r.name:<40} {r.state:<9} {n:>4} 行", file=sys.stderr)

    if not table:
        sys.exit("没有任何数据 —— 检查 --since 或 wandb 凭据")
    rows = [{"arm": arm, "seed": seed, "step": step, **vals}
            for (arm, seed, step), vals in sorted(table.items())]
    df = pd.DataFrame(rows)
    front = ["arm", "seed", "step"]
    df = df[front + [c for c in df.columns if c not in front]]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False, compression="gzip")
    print(f"\n{len(df)} 行 x {len(df.columns)} 列 -> {a.out}", file=sys.stderr)
    print(f"臂: {sorted(df.arm.unique())}", file=sys.stderr)


if __name__ == "__main__":
    main()
