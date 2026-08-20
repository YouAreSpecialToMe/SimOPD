#!/usr/bin/env python3
"""逐位置台账的四张表:谁扛着 loss、次序对齐到哪、以及哪些位置翻了。

    python scripts/analysis/token_ledger_report.py docs/data/token_ledger_*.parquet

输入是 token_ledger_probe.py 的输出(可以给多份,按 ckpt_step 排成时间序列)。
表 D(翻转)需要至少两份同一 txt_step 的台账 —— 位置靠 (pid, pos) 内联,所以
两份必须来自同一批文本,否则内联为空,脚本会说明而不是给一张空表。
"""
import sys

import pandas as pd

FILES = sys.argv[1:]
if not FILES:
    print(__doc__)
    sys.exit(2)
d = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
d["absdl"] = d.dl_sampled.abs()
steps = sorted(d.ckpt_step.unique())
print(f"台账 {len(d)} 行,arm={sorted(d.arm.unique())},文本 step={sorted(d.txt_step.unique())},"
      f"权重 step={steps}\n")

# ---- A. loss 按 token 类别:占比,不是均值 —— 我们问的是"谁扛着",那是质量占比。
print("=" * 78)
print("A  |Δℓ| 的质量占比 × token 类别(括号内是该类的位置占比)")
print("=" * 78)
tot = d.groupby("ckpt_step").absdl.sum()
cnt = d.groupby("ckpt_step").size()
rows = []
for cls, g in d.groupby("tok_class"):
    r = {"class": cls}
    for s in steps:
        gs = g[g.ckpt_step == s]
        r[s] = (gs.absdl.sum() / tot[s] if tot.get(s) else 0.0, len(gs) / cnt[s] if cnt.get(s) else 0.0)
    rows.append(r)
rows.sort(key=lambda r: -r[steps[-1]][0])
print(f"{'class':<12} " + " ".join(f"{'step ' + str(s):>18}" for s in steps))
for r in rows:
    print(f"{r['class']:<12} " + " ".join(f"{r[s][0]*100:>7.1f}% ({r[s][1]*100:>5.1f}%)" for s in steps))

# ---- A2. 具体是哪些词。类别是稳的那一层,但"word 扛了 46%"没法行动 —— 要能行动
# 就得知道是哪些词。按 |Δℓ| 的总量排(不是均值:一个出现两次的词均值再高也不重要),
# 同时给出现次数和均值,好把"高频平庸"和"低频剧毒"分开。
TOPN = int(__import__("os").environ.get("LEDGER_TOPN", "25"))
print()
print("=" * 78)
print(f"A2  扛 |Δℓ| 最多的 {TOPN} 个 token(按总量;share=占全部 |Δℓ| 的比例)")
print("=" * 78)
for s in steps:
    ds = d[d.ckpt_step == s]
    g = ds.groupby(["token_str", "tok_class"]).agg(
        mass=("absdl", "sum"), n=("absdl", "size"), mean=("absdl", "mean"),
        agree=("agree", "mean"), ent=("t_ent", "mean")).reset_index()
    g["share"] = g["mass"] / ds.absdl.sum() * 100
    g = g.sort_values("mass", ascending=False).head(TOPN)
    print(f"\n-- step {s}(共 {len(ds)} 位置,{ds.token_str.nunique()} 个不同 token)")
    print(f"{'token':<18} {'class':<11} {'share':>7} {'n':>6} {'mean|dl|':>9} {'agree':>7} {'教师熵':>7}")
    for _, r in g.iterrows():
        print(f"{repr(r.token_str)[:18]:<18} {r.tok_class:<11} {r.share:>6.2f}% {int(r.n):>6} "
              f"{r['mean']:>9.3f} {r.agree*100:>6.1f}% {r.ent:>7.3f}")

# ---- A3. 每一类内部再看头部,免得 word 这种大类把别的类挤掉
print()
print("=" * 78)
print("A3  每个类别内部扛 |Δℓ| 最多的 token(末个 ckpt)")
print("=" * 78)
ds = d[d.ckpt_step == steps[-1]]
for cls, g0 in sorted(ds.groupby("tok_class"), key=lambda kv: -kv[1].absdl.sum()):
    g = g0.groupby("token_str").agg(mass=("absdl", "sum"), n=("absdl", "size"),
                                    agree=("agree", "mean")).reset_index()
    g["share"] = g["mass"] / ds.absdl.sum() * 100
    g = g.sort_values("mass", ascending=False).head(8)
    head = "  ".join(f"{repr(r.token_str)[:14]}({r.share:.1f}%/{int(r.n)}/{r.agree*100:.0f}%)"
                     for _, r in g.iterrows())
    print(f"{cls:<12} 占 {g0.absdl.sum()/ds.absdl.sum()*100:>5.1f}% | {head}")
print("  (括号内 = share / 出现次数 / top1 一致率)")

# ---- B. 按位置段
print()
print("=" * 78)
print("B  |Δℓ| 均值 × 位置段(in-loop 面板的离线复核,并加上 top1 一致率)")
print("=" * 78)
BANDS = ["0-100", "100-500", "500-2k", "2k+"]
print(f"{'band':<10} " + " ".join(f"{'step ' + str(s):>20}" for s in steps))
print(f"{'':<10} " + " ".join(f"{'|dl|   agree':>20}" for _ in steps))
for b in BANDS:
    g = d[d.pos_band == b]
    if not len(g):
        continue
    cells = []
    for s in steps:
        gs = g[g.ckpt_step == s]
        cells.append(f"{gs.absdl.mean():>8.3f} {gs.agree.mean()*100:>6.1f}%" if len(gs) else f"{'-':>20}")
    print(f"{b:<10} " + " ".join(f"{c:>20}" for c in cells))

# ---- C. 次序对齐,分层
print()
print("=" * 78)
print("C  次序:top1 一致率 / rank_S(y_T^top1) 中位 / M_t 分位,分层")
print("=" * 78)


def strata(dd):
    q95 = dd.absdl.quantile(0.95)
    return [("全体", dd),
            ("高 loss (p95+)", dd[dd.absdl >= q95]),
            ("教师熵低 (<0.3)", dd[dd.t_ent < 0.3]),
            ("教师熵高 (>1.0)", dd[dd.t_ent > 1.0]),
            ("重复段内", dd[dd.is_rep]),
            ("答对", dd[dd.correct == 1]),
            ("答错", dd[dd.correct == 0])]


print(f"{'层':<16} " + " ".join(f"{'step ' + str(s):>26}" for s in steps))
print(f"{'':<16} " + " ".join(f"{'agree  rank50   M_t p50':>26}" for _ in steps))
names = [n for n, _ in strata(d[d.ckpt_step == steps[0]])]
for i, name in enumerate(names):
    cells = []
    for s in steps:
        g = strata(d[d.ckpt_step == s])[i][1]
        cells.append(f"{g.agree.mean()*100:>6.1f}% {int(g.rank_t_top1.median()):>6} {g.M_t.median():>9.3f}"
                     if len(g) else f"{'-':>26}")
    print(f"{name:<16} " + " ".join(f"{c:>26}" for c in cells))

# ---- D. 翻转
print()
print("=" * 78)
print("D  翻转:同一 (pid, pos) 上 M_t 的符号变化")
print("=" * 78)
if len(steps) < 2:
    print("只有一个 ckpt_step,无法数翻转 —— 给两份同一 txt_step 的台账。")
else:
    for a, b in zip(steps[:-1], steps[1:]):
        A = d[d.ckpt_step == a][["pid", "pos", "M_t", "absdl", "agree", "correct", "tok_class", "t_ent"]]
        B = d[d.ckpt_step == b][["pid", "pos", "M_t", "agree"]]
        j = A.merge(B, on=["pid", "pos"], suffixes=("_a", "_b"))
        if not len(j):
            print(f"{a} -> {b}: 内联为空(两份台账不是同一批文本?)")
            continue
        gain = (j.M_t_a < 0) & (j.M_t_b > 0)
        lose = (j.M_t_a > 0) & (j.M_t_b < 0)
        q95 = j.absdl.quantile(0.95)
        print(f"\n{a} -> {b}  内联 {len(j)} 个位置")
        print(f"   翻正 (M_t<0 -> >0): {gain.sum():>6}  ({gain.mean()*100:.2f}%)")
        print(f"   翻负 (M_t>0 -> <0): {lose.sum():>6}  ({lose.mean()*100:.2f}%)")
        print(f"   净: {int(gain.sum() - lose.sum()):+}   一致率 {j.agree_a.mean()*100:.1f}% -> {j.agree_b.mean()*100:.1f}%")
        sub = [("高 loss (p95+)", j[j.absdl >= q95]),
               ("原本不一致", j[j.M_t_a < 0]),
               ("教师熵低 (<0.3)", j[j.t_ent < 0.3]),
               ("答对", j[j.correct == 1]), ("答错", j[j.correct == 0])]
        print(f"   {'层':<16} {'n':>7} {'翻正%':>8} {'翻负%':>8} {'净%':>8}")
        for name, g in sub:
            if not len(g):
                continue
            gp = ((g.M_t_a < 0) & (g.M_t_b > 0)).mean() * 100
            ln = ((g.M_t_a > 0) & (g.M_t_b < 0)).mean() * 100
            print(f"   {name:<16} {len(g):>7} {gp:>7.2f}% {ln:>7.2f}% {gp-ln:>+7.2f}%")
        cls = j[gain].tok_class.value_counts(normalize=True).head(4)
        print("   翻正最集中的 token 类:" + ", ".join(f"{k} {v*100:.0f}%" for k, v in cls.items()))
