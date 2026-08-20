#!/usr/bin/env python3
"""
c2/c4 后期在学什么 —— 同一批文本、同一批位置,几个 ckpt 的账本逐位对齐后作差。

回答的问题(2026-08-21,loss/entropy/clip/length 全平但 eval acc 在涨):
损失是质量加权的平均,一个位置把质量从错误词挪到正确词,KL 变化可以近乎对冲;
但 greedy 解码只看 argmax,这一挪就是 0/1 翻转。所以"学了什么"要在排序里找,
不在损失里找。这里把翻面的位置逐词点名。

用法(在舰队树里,CPU 秒级):
  DIFF_PREFIX=/tmp/ledger_c4_ckpt DIFF_CKPTS=50,100,150,250 \
      python scripts/analysis/token_ledger_diff.py

口径与告警:
  * 文本是 TXT_ARM@TXT_STEP 的固定 dump —— 每个 ckpt 都在同一批状态上测,
    晚期 ckpt 的数字是"在参考策略访问的状态上"的表现,不是它自己会走的路。
  * kl_arm 是臂自己的支撑限制 reverse KL,与 kernel 同向同权 —— "贡献 loss"
    一律以它为准;dl_sampled = 教师 - 学生 在文本 token 上的 logprob 差,
    <0 即学生过自信(reverse KL 罚的那一侧)。
"""
import os

import numpy as np
import pandas as pd

PREFIX = os.environ.get("DIFF_PREFIX", "/tmp/ledger_c4_ckpt")
CKPTS = [int(x) for x in os.environ.get("DIFF_CKPTS", "50,100,150,250").split(",")]
TENT = float(os.environ.get("DIFF_TENT", "0.3"))     # 决策位:教师熵 < TENT
TOPN = int(os.environ.get("DIFF_TOPN", "14"))
pd.set_option("display.width", 200)

DYN = ["M_t", "agree", "rank_t_top1", "kl_arm", "dl_sampled", "s_top1", "in_sup", "sup_size"]
STATIC = ["token_str", "tok_class", "t_top1_str", "t_ent", "pos_band", "correct", "is_rep"]

dfs = {s: pd.read_parquet(f"{PREFIX}{s}.parquet") for s in CKPTS}
w = dfs[CKPTS[-1]][["pid", "pos"] + STATIC].copy()
for s in CKPTS:
    d = dfs[s][["pid", "pos"] + DYN].rename(columns={c: f"{c}_{s}" for c in DYN})
    w = w.merge(d, on=["pid", "pos"], how="inner")
a, b = CKPTS[0], CKPTS[-1]
print(f"P0 对齐:{ {s: len(dfs[s]) for s in CKPTS} } -> 交集 {len(w)} 位;"
      f"轨迹 {w['pid'].nunique()} 条(答对 {w[w.correct==1]['pid'].nunique()}/答错 {w[w.correct==0]['pid'].nunique()})"
      f" —— 样本薄,词表看方向不看小数点")

# ---- P1 总量:loss 平、排序在动,两句话放在同一张表里
rows = []
for s in CKPTS:
    dis = w[~w[f"agree_{s}"]]
    rows.append(dict(
        ckpt=s,
        kl_arm_mean=w[f"kl_arm_{s}"].mean(), kl_arm_p50=w[f"kl_arm_{s}"].median(),
        agree=w[f"agree_{s}"].mean(), frac_Mt_pos=(w[f"M_t_{s}"] > 0).mean(),
        rank_med_at_disagree=dis[f"rank_t_top1_{s}"].median(),
        M_t_margin=w[f"M_t_{s}"].mean(), sup_size=w[f"sup_size_{s}"].mean(),
    ))
print("\nP1 总量面板(loss 列平不平、agree/M_t 列动没动,自己对照)")
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---- P2 ΔKL 按 agree 迁移分解:净变化 ~0 是不是"翻正大降 + 风格微涨"的对冲
def decomp(x, y):
    d = (w[f"kl_arm_{y}"] - w[f"kl_arm_{x}"])
    g = w.groupby([w[f"agree_{x}"], w[f"agree_{y}"]])
    out = g.apply(lambda t: pd.Series(dict(
        n=len(t), dKL_sum=(t[f"kl_arm_{y}"] - t[f"kl_arm_{x}"]).sum(),
        dKL_mean=(t[f"kl_arm_{y}"] - t[f"kl_arm_{x}"]).mean())), include_groups=False)
    out.index = [{(False, False): "FF 一直错", (False, True): "FT 翻正",
                  (True, False): "TF 翻错", (True, True): "TT 一直对"}[i] for i in out.index]
    print(f"  {x}->{y}: 净ΔKL={d.sum():+.1f}(均值 {d.mean():+.4f}/位)")
    print(out.round(3).to_string())

print(f"\nP2 ΔKL 分解(kl_arm,臂自己的口径)")
for x, y in [(CKPTS[1], b), (CKPTS[2], b)]:
    decomp(x, y)

# ---- P3 决策位翻正词表:窗口逐段,"文本走了X/教师要Y,现在学生也要Y了"
print(f"\nP3 决策位翻正(教师熵<{TENT} 且窗口起点不同意 -> 终点同意)")
for x, y in zip(CKPTS[:-1], CKPTS[1:]):
    m = (w.t_ent < TENT) & ~w[f"agree_{x}"] & w[f"agree_{y}"]
    f = w[m]
    print(f"  {x}->{y}: 翻正 {m.sum()} 位"
          + ("" if not m.sum() else ";词表(文本token -> 教师要的):"))
    if m.sum():
        g = (f.groupby([f.token_str.str.replace("\n", "\\n"), f.t_top1_str.str.replace("\n", "\\n")])
             .agg(n=("pos", "size"), rank_from=(f"rank_t_top1_{x}", "median"),
                  dKL=("pos", lambda i: (f.loc[i.index, f"kl_arm_{y}"] - f.loc[i.index, f"kl_arm_{x}"]).sum())))
        print(g.sort_values("n", ascending=False).head(TOPN).round(2).to_string())

# ---- P4 后期(150->250)还在降 KL 的词,不要求翻面 —— "在学但未必学会"
x, y = CKPTS[2], b
w["_d"] = w[f"kl_arm_{y}"] - w[f"kl_arm_{x}"]
print(f"\nP4 {x}->{y} 按词聚的 ΔKL(负=后期还在这词上降损;n>=3 才上榜)")
g = (w.groupby(w.token_str.str.replace("\n", "\\n"))
     .agg(n=("_d", "size"), dKL_sum=("_d", "sum"), kl_250=(f"kl_arm_{b}", "sum"),
          overconf_150=(f"dl_sampled_{x}", lambda v: (v < 0).mean())))
g = g[g.n >= 3]
print("  降得最多:"); print(g.sort_values("dKL_sum").head(TOPN).round(2).to_string())
print("  涨得最多(支撑变形/风格残差):"); print(g.sort_values("dKL_sum", ascending=False).head(6).round(2).to_string())

# ---- P5 到 250 还没学会的决策位:接近翻面(rank<=3)还是深埋
st = w[(w.t_ent < TENT) & ~w[f"agree_{b}"]]
print(f"\nP5 顽固决策位 @{b}:{len(st)} 位,占 kl_arm 总量 "
      f"{st[f'kl_arm_{b}'].sum() / w[f'kl_arm_{b}'].sum():.1%};"
      f"其中 rank<=3(一步之遥){(st[f'rank_t_top1_{b}'] <= 3).mean():.0%}")
g = (st.groupby([st.token_str.str.replace("\n", "\\n"), st.t_top1_str.str.replace("\n", "\\n")])
     .agg(n=("pos", "size"), rank_250=(f"rank_t_top1_{b}", "median"),
          kl_250=(f"kl_arm_{b}", "sum"), t_ent=("t_ent", "mean")))
print(g.sort_values("kl_250", ascending=False).head(TOPN).round(2).to_string())

# ---- P6 倒退与 churn:翻错的位置是决策位还是高熵风格位
for x, y in [(CKPTS[2], b)]:
    m = w[f"agree_{x}"] & ~w[f"agree_{y}"]
    tf = w[m]
    print(f"\nP6 {x}->{y} 翻错 {m.sum()} 位;其中教师熵<{TENT} 的 {(tf.t_ent < TENT).sum()} 位"
          f"(其余是高熵风格位,教师自己都不坚持,来回摆无害)")
    if (tf.t_ent < TENT).sum():
        gg = tf[tf.t_ent < TENT].groupby(tf.t_top1_str.str.replace("\n", "\\n")).size()
        print("  低熵倒退词:", dict(gg.sort_values(ascending=False).head(8)))

# ---- P7 答对/答错轨迹分开(1 条答错轨迹,只看方向)
x = CKPTS[1]
print(f"\nP7 {x}->{b} 翻正率:答对轨迹 "
      f"{(~w[w.correct==1][f'agree_{x}'] & w[w.correct==1][f'agree_{b}']).mean():.2%},"
      f"答错轨迹 {(~w[w.correct==0][f'agree_{x}'] & w[w.correct==0][f'agree_{b}']).mean():.2%};"
      f"决策位密度 答对 {((w.correct==1)&(w.t_ent<TENT)&~w[f'agree_{x}']).sum()/max((w.correct==1).sum(),1)*100:.1f}%"
      f" vs 答错 {((w.correct==0)&(w.t_ent<TENT)&~w[f'agree_{x}']).sum()/max((w.correct==0).sum(),1)*100:.1f}%(每百位)")

# ---- P8 一直对的位置在干嘛:置信收紧(reverse KL 的残差形状项)。
# 置信用 M_t(教师 top1 对最强对手的 logit 余量)—— s_top1 存的是 token id,别用。
tt = w[np.logical_and.reduce([w[f"agree_{s}"] for s in CKPTS])]
rows = [dict(ckpt=s, M_t_margin=tt[f"M_t_{s}"].mean(), kl_arm=tt[f"kl_arm_{s}"].mean(),
             overconf=(tt[f"dl_sampled_{s}"] < 0).mean()) for s in CKPTS]
print(f"\nP8 全程一直对的 {len(tt)} 位({len(tt)/len(w):.0%}):决策余量与残差")
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---- P9 终止符三位,逐 ckpt 点名(wave 21 预测(2) 的基线)
tm = w[w.tok_class == "terminator"]
if len(tm):
    cols = ["pid"] + [f"kl_arm_{s}" for s in CKPTS] + [f"in_sup_{s}" for s in CKPTS]
    print("\nP9 终止符位置(kl_arm 逐 ckpt;in_sup=文本 token 是否在支撑内)")
    print(tm[cols].round(2).to_string(index=False))
