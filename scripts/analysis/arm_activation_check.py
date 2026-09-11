#!/usr/bin/env python3
"""每条臂的旋钮到底有没有"咬上":从 results/<date>/metrics_full 读运行时证据,出一张 markdown 表。

    python scripts/analysis/arm_activation_check.py --results results/20260909

背景(2026-09-11):重训里十来条臂贴着载体走,用户问"是不是旋钮根本没生效"。arms.tsv 的 env 增量只是
arms.yaml 的登记,不是运行时。运行时的证据是**内核自己吐的仪表**:每个 loss mode 只在自己的分支里
emit 特定键(qb_budget、c4_budget、gkd_lambda_realized、gate_keep_frac、clip_hit_rate……),键在且在动,
分支就跑了;再看"咬合度"—— 旋钮实际改动了多大比例的信号(hit rate / keep frac / selected frac),
这决定了"≈载体"是**旋钮没开**还是**旋钮在修正载体上没东西可咬**(两者的论文结论完全不同)。
位置分箱(delta_ell_absmean_pos*)是 H 轴分段臂的直接证据:h1 只该在 0–100 有信号。

只读 TSV、纯 pandas;每列解释见表头下的注。
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POS = ["pos0_100", "pos100_500", "pos500_2k", "pos2k_up"]
# 旋钮 -> "咬合度"仪表(键后缀)与读法。没有专属仪表的臂写 None,用间接证据(Δℓ 幅度 / 位置分布)。
BITE = {
    "c2_quantile_budget_corr": ("qb_captured_mass", "教师块被选中列覆盖的质量;budget 见 qb_budget"),
    "c2_qb_perseq_corr":       ("qb_captured_mass", "同上,tau 按序列"),
    "c2_qb_fixed8_corr":       ("qb_captured_mass", "固定前 8 列覆盖的教师质量;掉到 <0.5 = 归一化目标失真"),
    "c4_pi_tail_budget_corr":  ("c4_budget", "自适应预算列数;c4_eps_missed 为漏掉的尾质量"),
    "c4_hq":                   ("c4_hq_freeze", "agreed-freeze 触发的位置占比(逐 token,天然极小)"),
    "c4_state":                ("c4_rep_freeze", "复读段冻结的位置占比"),
    "c5_union_rkl":            ("un_p_imend", "学生对教师终止符的概率(预注册:看它升不升)"),
    "c5_union_fkl":            ("un_p_imend", "同上"),
    "n2_corr":                 ("eos_aux_term", "校准项(按全 batch token 数归一 → 结构上 ~1/L 权重)"),
    "b2_forward_kl_corr":      ("teacher_mass_min", "教师 top-k 块最小覆盖"),
    "c3_intersection_corr":    ("c3_inter_size", "交集支撑列数(两个 top-34 的交)"),
    "d2_selectkd_corr":        ("d_selected_frac", "采样 token 落在教师 top-5 的占比(其余 ×0.01)"),
    "d3_teachability_corr":    ("d_selected_frac", "保留的 token 占比(登记 5%)"),
    "e2_set_coverage_a0_corr": ("e2_coverage", "覆盖目标满足度(=1 时损失 ~0)"),
    "f2_hard_clip_corr":       ("clip_hit_rate", "|Δℓ|>10 被截的 token 占比"),
    "f3_power_corr":           ("power_dead_frac", "exp 下溢为 0 的 token 占比(a=1 时 loss=p_θ−p_T)"),
    "g1_verified_only_corr":   ("gate_keep_frac", "被 verifier 放行进 loss 的占比"),
    "h1_first_segment_corr":   ("firstseg_covered_frac", "前 100 token 占全部 token 的比例"),
    "h2_last_segment_corr":    ("lastseg_covered_frac", "末 100 token 占比"),
    "h3_random_segment_corr":  ("randseg_covered_frac", "随机 100 段占比"),
    "h4_random_scatter_corr":  ("randscatter_covered_frac", "随机散点 100 个占比"),
    "a1_gkd_mix0.5_n0":        ("gkd_lambda_realized", "实际教师混合比(目标 0.5)"),
    "a3_offpolicy_n0":         ("gkd_lambda_realized", "实际教师混合比(目标 1.0)"),
    "a4_dagger_anneal_n0":     ("gkd_lambda_realized", "实际教师混合比(线性退火)"),
    "a5_aggrevate_n0":         ("gkd_mixed", "本步走混合前缀的序列数(/128)"),
    "vanilla_te":              ("eos_dl_at_stop", "事件级 Δℓ 面板在(TE=1 生效;legacy k1_rec 无此键且会晚塌)"),
    "b1_skew_kl_corr":         (None, "无专属仪表;间接:Δℓ 幅度分箱明显低于载体(skew α=0.1)"),
    "g6_seqmean_corr":         (None, "无专属仪表;间接:seq-mean 聚合改变位置权重分布"),
    "vanilla_corr":            (None, "载体"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "results/20260909"))
    ap.add_argument("--window", type=int, default=40, help="末窗步数(位置分箱与咬合度取末窗均值)")
    a = ap.parse_args()
    arms_tsv = pd.read_csv(os.path.join(a.results, "arms.tsv"), sep="\t").set_index("arm")
    order = list(pd.read_csv(os.path.join(a.results, "progress.tsv"), sep="\t").arm)
    F = {arm: pd.read_csv(os.path.join(a.results, "metrics_full", arm + ".tsv"), sep="\t").sort_values("step") for arm in order}
    base = set(F["vanilla_corr"].columns)
    print("| 臂 | 登记旋钮(相对载体) | 专属键数 | 咬合度仪表 | 末窗值 [全程范围] | Δℓ/loss 位置份额 0–100 / 100–500 / 500–2k / 2k+ |")
    print("|---|---|---|---|---|---|")
    for arm in order:
        f = F[arm]; last = int(f.step.max()); w = f[f.step > last - a.window]
        extra = [c for c in f.columns if c not in base and f[c].notna().any()]
        delta = str(arms_tsv.env_delta_vs_carrier.get(arm, "")).replace("DISTILLATION_LOSS_MODE=", "mode=").replace("SIMOPD_", "")
        delta = "(载体)" if delta == "nan" else delta.replace(";", "; ")
        key, note = BITE.get(arm, (None, ""))
        if key:
            cols = [c for c in f.columns if c.endswith(key)]
            if cols:
                s = f[cols[0]].dropna(); val = f"{w[cols[0]].mean():.3g} [{s.min():.2g}..{s.max():.2g}]"
            else:
                val = "**键缺失**"
        else:
            val = "—"
        share = ""
        for pre in ("actor/distillation/delta_ell_absmean_", "actor/distillation/loss_absmean_"):
            if pre + POS[0] in f.columns and f[pre + POS[0]].notna().any():
                bins = np.array([w[pre + p].mean() if pre + p in f.columns else np.nan for p in POS], float)
                tot = np.nansum(bins)
                share = " / ".join(f"{b / tot:.2f}" if tot else "–" for b in bins)
                break
        print(f"| `{arm}` | {delta} | {len(extra)} | {key or '—'} | {val} | {share} |")
    print("\n注:专属键数 = 该臂 metrics_full 里有而 vanilla_corr 里没有的键(内核分支跑了的直接证据);"
          "咬合度 = 旋钮实际改动的信号比例,读法见脚本 BITE 表;位置份额 = 末窗内 |Δℓ|(k1 族)或 |loss|(top-k 族)"
          "按位置分箱的相对份额,H 轴分段臂应只在自己的段里有份额。")


if __name__ == "__main__":
    main()
