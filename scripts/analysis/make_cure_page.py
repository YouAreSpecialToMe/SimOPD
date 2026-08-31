#!/usr/bin/env python3
"""生成「终止塌缩判读台」单文件 HTML(docs/collapse-cure.html)。

    python scripts/analysis/make_cure_page.py

数据全部来自 docs/data/ 的归档表(不连集群、不连 wandb),分类与早期阈值那两张表直接
复用 collapse_status.py 的函数 —— 页面上的数字与 MECHANISMS.md「M-I cure」同源同算法,
不会各说各话。产物内嵌全部数据,双击即开;也可直接当 artifact 发布。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collapse_status import classify, early_warning          # noqa: E402  同源

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "collapse-cure.html")
EVERY = 2          # 曲线抽稀:每 2 步一个点,肉眼无差、文件小一半

STATUS_KEY = {"健康": "ok", "危险": "warn", "塌缩": "bad", "预算封顶": "na"}


def series(g, col, r=4):
    g = g.dropna(subset=[col]).sort_values("step")
    g = g[g.step % EVERY == 0]
    return [[int(s), round(float(v), r)] for s, v in zip(g.step, g[col])]


def main():
    dyn = pd.read_csv(os.path.join(ROOT, "docs/data/inloop_wave_dynamics.csv"))
    bys = pd.read_csv(os.path.join(ROOT, "docs/data/post_eval_bystep.csv"))
    st = classify(dyn)

    # --- 治愈对照:legacy 三种子(均值 + 带)vs corr ---
    hero = {}
    for key, arm in (("legacy", "vanilla"), ("corr", "vanilla_corr")):
        g = bys[bys.arm == arm]
        rows = []
        for step, gg in g.groupby("step"):
            rows.append(dict(step=int(step),
                             comp=float(gg.composite.mean()), comp_lo=float(gg.composite.min()),
                             comp_hi=float(gg.composite.max()),
                             trunc=float(gg.trunc_rate.mean()), trunc_lo=float(gg.trunc_rate.min()),
                             trunc_hi=float(gg.trunc_rate.max()),
                             len_mean=float(gg.len_mean.mean()), n=int(len(gg))))
        hero[key] = sorted(rows, key=lambda r: r["step"])

    # --- 全波动态 ---
    runs = []
    for r in st.itertuples():
        g = dyn[dyn.run == r.run]
        runs.append(dict(
            run=r.run, arm=r.run.replace("_s0_16k", "").replace("_s0", ""),
            status=STATUS_KEY[r.status], max_step=int(r.max_step),
            clip_late=round(float(r.clip_late), 3), len_late=round(float(r.len_late)),
            eos_late=None if np.isnan(r.eos_p_late) else round(float(r.eos_p_late), 3),
            ent_late=round(float(r.entropy_late), 3),
            m=dict(clip=series(g, "resp_len_clip_frac"), len=series(g, "resp_len", 0),
                   eos=series(g, "eos_p_at_stop"), ent=series(g, "entropy", 3))))

    ew = early_warning(dyn, ["vanilla_corr_s0_16k", "f1_soft_log_corr_s0_16k", "n2_corr_s0_16k",
                             "n2_termcal_s0_16k", "h2_last_segment_corr_s0_16k",
                             "e2_set_coverage_a0_corr_s0_16k"])
    ew_rows = [dict(run=r.run.replace("_s0_16k", ""), maxstep=int(r.到步),
                    fate=("健康" if r.run in ("vanilla_corr_s0_16k", "f1_soft_log_corr_s0_16k",
                                             "n2_corr_s0_16k") else "塌缩"),
                    eos=None if pd.isna(r.eos_p_lt_0p3) else int(r.eos_p_lt_0p3),
                    ent=None if pd.isna(r.entropy_lt_0p2) else int(r.entropy_lt_0p2),
                    clip=None if pd.isna(r.clip_gt_0p6) else int(r.clip_gt_0p6))
               for r in ew.itertuples()]

    # --- 药效归零:同一臂在 legacy 载体 / corr 载体上,相对各自 vanilla 的增益 ---
    s0 = bys[bys.seed == 0]
    cur = {a: g.set_index("step").composite for a, g in s0.groupby("arm")}
    van_l = bys[bys.arm == "vanilla"].groupby("step").composite.mean()   # legacy 三种子均值
    van_c = cur.get("vanilla_corr")
    med = []
    for arm in sorted(cur):
        base = arm[:-5] if arm.endswith("_corr") else (arm[:-3] if arm.endswith("_n0") else None)
        if base is None or base not in cur or base == "vanilla":
            continue
        shared = sorted(set(cur[arm].index) & set(cur[base].index) & set(van_l.index) & set(van_c.index))
        if len(shared) < 2:
            continue
        med.append(dict(arm=arm, base=base, pts=[
            dict(step=int(t), d_legacy=round(float(cur[base][t] - van_l[t]), 4),
                 d_corr=round(float(cur[arm][t] - van_c[t]), 4)) for t in shared]))

    # 同批臂在同一步上的极差:修正前后各算一次
    batch = [a for a in ["b2_forward_kl", "c2_quantile_budget", "c4_pi_tail_budget"] if a in cur]
    spread = []
    for step in (75, 100, 125, 150):
        L = [float(van_l[step])] + [float(cur[a][step]) for a in batch if step in cur[a].index]
        C = ([float(van_c[step])] if step in van_c.index else []) + \
            [float(cur[a + "_corr"][step]) for a in batch if a + "_corr" in cur and step in cur[a + "_corr"].index]
        if len(L) >= 3 and len(C) >= 3:
            spread.append(dict(step=step, n_l=len(L), n_c=len(C),
                               s_l=round(max(L) - min(L), 4), s_c=round(max(C) - min(C), 4)))

    # --- 吸引子:健康臂的晚期均长挤到多窄 ---
    okr = [r for r in runs if r["status"] == "ok"]
    band = [r for r in okr if 8500 < r["len_late"] < 9700]
    exc = sorted([r for r in okr if not (8500 < r["len_late"] < 9700)], key=lambda r: r["len_late"])
    attractor = dict(inband=len(band), total=len(okr),
                     exceptions=[dict(arm=r["arm"], len_late=r["len_late"], clip=r["clip_late"],
                                      ent=r["ent_late"]) for r in exc])

    # --- 覆盖账:这一波每条 run 评了多少、评到哪 —— 「已跑出来」与「还没跑」一目了然 ---
    cells = pd.read_csv(os.path.join(ROOT, "docs/data/post_eval_cells.csv"))
    cov = []
    for r in runs:
        # 按 arm 关联,不按 run:本波是单种子,而 run 名的后缀习惯不统一
        # (c4_carrier_s0 / vanilla_corr_s0_16k / c2_quantile_budget_corr_s0),
        # 用 run 关联会把后缀猜错的那几条判成「未评」—— 正是本页要避免的那类漏算。
        g = cells[cells.arm == r["arm"]]
        nb = bys[bys.arm == r["arm"]]
        steps = sorted(int(x) for x in g.step.unique()) if len(g) else []
        cov.append(dict(arm=r["arm"], run=r["run"], status=r["status"], train_step=r["max_step"],
                        cells=int(len(g)), complete=int(len(nb)),
                        eval_max=(max(steps) if steps else None), steps=steps))
    cov.sort(key=lambda x: (-x["cells"], x["arm"]))

    # --- 已产出但不入主对照:w 对(8B-Base<-32B,cap 8192)与 diag_* 一次性诊断 ---
    others = []
    for run, g in cells[cells.run.str.endswith("_w") | cells.run.str.startswith("diag_")].groupby("run"):
        for step, gg in g.groupby("step"):
            others.append(dict(run=run, step=int(step),
                               benches=[dict(b=x.bench, acc=round(float(x.avg_at_k), 3),
                                             tr=round(float(x.trunc_rate), 2)) for x in gg.itertuples()]))
    others.sort(key=lambda x: (x["run"], x["step"]))

    counts = st.status.value_counts().to_dict()
    payload = dict(hero=hero, runs=runs, ew=ew_rows, med=med, spread=spread, attractor=attractor,
                   coverage=cov, others=others,
                   counts={STATUS_KEY[k]: int(v) for k, v in counts.items()})

    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cure_page.tpl.html")).read()
    html = tpl.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT}  {os.path.getsize(OUT)/1024:.0f} KB  runs={len(runs)}  counts={counts}")


if __name__ == "__main__":
    main()
