#!/usr/bin/env python3
"""终止塌缩的现状表:谁塌了、谁没塌、修正载体到底管不管用。

    python scripts/analysis/collapse_status.py            # 打表
    python scripts/analysis/collapse_status.py --write    # 顺便写 docs/data/collapse_status.csv

只读两张归档表(docs/data/inloop_wave_dynamics.csv、post_eval_bystep.csv),不连 wandb、
不碰集群 —— MECHANISMS.md 里 "M-I cure" 一节的每个数字都由这个脚本重算得出。

判据用【晚期窗口】(最后 25 步的均值),不用早期阈值。原因是负结果本身:
eos_p<0.3 / 熵<0.2 / 截断>0.6 这些阈值在健康臂上同样早早触发(vanilla_corr 在 35-48 步
把三个全触发过一遍,却健康跑到 250),所以早期阈值没有判别力,--early-warning 复现它。
"""
import argparse

import numpy as np
import pandas as pd

DYN = "docs/data/inloop_wave_dynamics.csv"
BYSTEP = "docs/data/post_eval_bystep.csv"
LATE = 25          # 晚期窗口宽度(步)


def classify(dyn):
    rows = []
    for run, g in dyn.groupby("run"):
        g = g.sort_values("step")
        mx = int(g.step.max())
        late = g[g.step > mx - LATE]
        if late.empty:
            continue
        clip = late.resp_len_clip_frac.mean()
        ln = late.resp_len.mean()
        # 自带生成预算的臂(h7_gen512 / h9_prune_adapt 之流)天天撞自己的小帽:
        # 截断率 ~1 而长度远低于训练帽 16k —— 那是设计,不是塌缩,单列出来。
        if clip > 0.6 and ln < 4000:
            st = "预算封顶"
        elif clip > 0.6:
            st = "塌缩"
        elif clip > 0.3:
            st = "危险"
        else:
            st = "健康"
        rows.append(dict(run=run, max_step=mx, clip_late=clip, len_late=ln,
                         eos_p_late=late.eos_p_at_stop.mean(),
                         eos_q_late=late.eos_q_at_stop.mean(),
                         entropy_late=late.entropy.mean(), status=st))
    return pd.DataFrame(rows).sort_values(["status", "clip_late"], ascending=[True, False])


def early_warning(dyn, runs):
    """阈值先后表:证明早期阈值不能预警(健康臂一样触发)。"""
    def cross(g, col, thr, above):
        g = g.dropna(subset=[col]).sort_values("step")
        m = (g[col] > thr) if above else (g[col] < thr)
        hit = g[m.rolling(5).sum() >= 5]        # 连续 5 步,防抖
        return int(hit.step.iloc[0]) if len(hit) else None
    out = []
    for run in runs:
        g = dyn[dyn.run == run]
        if g.empty:
            continue
        out.append(dict(run=run,到步=int(g.step.max()),
                        eos_p_lt_0p3=cross(g, "eos_p_at_stop", .30, False),
                        entropy_lt_0p2=cross(g, "entropy", .20, False),
                        clip_gt_0p6=cross(g, "resp_len_clip_frac", .60, True),
                        len_gt_12k=cross(g, "resp_len", 12000, True)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="写 docs/data/collapse_status.csv")
    a = ap.parse_args()
    dyn = pd.read_csv(DYN)
    t = classify(dyn)

    print("== 晚期窗口分类(最后 %d 步均值)" % LATE)
    print(f"{'run':<40}{'到步':>5}{'截断':>7}{'长度':>8}{'eos_p':>7}{'熵':>7}  状态")
    for r in t.itertuples():
        ep = "  n/a" if np.isnan(r.eos_p_late) else f"{r.eos_p_late:>7.2f}"
        print(f"{r.run:<40}{r.max_step:>5}{r.clip_late:>7.2f}{r.len_late:>8.0f}{ep}{r.entropy_late:>7.2f}  {r.status}")
    print("\n  合计:", t.status.value_counts().to_dict())

    print("\n== 治愈对照:legacy vanilla(k1_rec)vs vanilla_corr(k1_termfix)")
    b = pd.read_csv(BYSTEP)
    for arm in ["vanilla", "vanilla_corr"]:
        for sd, g in b[b.arm == arm].groupby("seed"):
            g = g.sort_values("step")
            print(f"  {arm:<13} s{sd} composite " + "  ".join(f"{int(r.step)}:{r.composite:.3f}" for r in g.itertuples()))
            print(f"  {arm:<13} s{sd} 截断      " + "  ".join(f"{int(r.step)}:{r.trunc_rate:.2f}" for r in g.itertuples()))
    print("  契约核对(必须同为 off 才可比):",
          {arm: sorted(b[b.arm == arm].stop_set.astype(str).unique()) for arm in ["vanilla", "vanilla_corr"]})

    print("\n== 负结果:早期阈值不能预警(健康臂一样触发)")
    print(early_warning(dyn, ["vanilla_corr_s0_16k", "n2_corr_s0_16k", "f1_soft_log_corr_s0_16k",
                              "n2_termcal_s0_16k", "h2_last_segment_corr_s0_16k",
                              "e2_set_coverage_a0_corr_s0_16k"]).to_string(index=False))

    if a.write:
        t.to_csv("docs/data/collapse_status.csv", index=False)
        print("\nwrote docs/data/collapse_status.csv  rows=%d" % len(t))


if __name__ == "__main__":
    main()
