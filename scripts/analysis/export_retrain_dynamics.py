#!/usr/bin/env python3
"""把 results/<date>/metrics_full/*.tsv(逐臂全键逐步表)转成 make_dynamics_page.py 吃的 CSV。

    python scripts/analysis/export_retrain_dynamics.py --results results/20260909
    python scripts/make_dynamics_page.py \
        --csv docs/data/training_metrics_retrain_20260909_allkeys.csv.gz \
        --suite-md /dev/null --cells /dev/null \
        --title "2026-09 名册重训 · 训练动力学" --out docs/retrain-dynamics.html

schema 与 docs/data/training_metrics_*_allkeys.csv.gz 一致:arm, seed, step + verl/wandb 原键。
臂名取文件名(已带 _corr/_n0 载体后缀),seed 固定 0(名册单种子)。每臂每步一行 —— extract.py
已按"后写覆盖先写"折叠续跑重叠,这里再断言一次,重复即拒绝:同名同步两行会让动态页的
ser.get(step) 返回 Series、r4() 抛错(export_wave_metrics.py 记过的同一条教训)。

**不要**把这份和 training_metrics_corr_allkeys.csv.gz 合进同一张页:重训臂与 08 月 corr 波
同名(vanilla_corr、*_corr、*_n0),是同名不同实验;要并排比较先给一边加后缀。
离线套件面板同理:--suite-md / --cells 必须指向空文件,否则旧波的套件曲线会按臂名挂到重训臂上。
"""
import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "results/20260909"),
                    help="结果数据集目录(含 metrics_full/*.tsv)")
    ap.add_argument("--out", default=None,
                    help="默认 docs/data/training_metrics_retrain_<目录名>_allkeys.csv.gz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = os.path.basename(os.path.normpath(a.results))
    out = a.out or os.path.join(ROOT, f"docs/data/training_metrics_retrain_{tag}_allkeys.csv.gz")

    frames = []
    for p in sorted(glob.glob(os.path.join(a.results, "metrics_full", "*.tsv"))):
        arm = os.path.basename(p)[:-4]
        df = pd.read_csv(p, sep="\t")
        if "step" not in df.columns:
            sys.exit(f"{p}: 没有 step 列")
        dup = sorted(set(df.step[df.step.duplicated()]))
        if dup:
            sys.exit(f"{p}: 同一步出现多行 {dup[:5]}{' ...' if len(dup) > 5 else ''} —— 先折叠再导")
        df.insert(0, "seed", a.seed)
        df.insert(0, "arm", arm)
        frames.append(df)
    if not frames:
        sys.exit(f"{a.results}/metrics_full/ 下没有 *.tsv")
    allrows = pd.concat(frames, ignore_index=True, sort=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allrows.to_csv(out, index=False, compression="gzip")
    print(f"wrote {out}: {len(frames)} arms, {len(allrows)} rows, {allrows.shape[1]} columns, "
          f"max step {int(allrows.step.max())}")


if __name__ == "__main__":
    main()
