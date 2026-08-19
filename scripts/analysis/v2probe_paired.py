#!/usr/bin/env python3
"""v2 定性探针配对分析:复刻 late-training-collapse.md 的 §1/§2 算术 + §4 文本读法。
对每臂 (peak, late):总分、P(finish)、acc|finish、配对转移矩阵、子群 Δ、
截断响应的 boxed 统计(首个 boxed 位置、boxed 重复次数、尾部循环)。"""
import glob, os, re, sys
import pandas as pd

OUT = "/mgfs/shared/Group_GY/changhao/simopd_data/evals_v2probe"
PAIRS = [("a1_gkd_mix0.5_s0_16k", 50, 125),
         ("a3_offpolicy_s0_16k", 50, 250),
         ("h6_gen_sched_s0_16k", 25, 175)]

def load(run, st):
    fs = sorted(glob.glob(f"{OUT}/{run}__math500__step{st}__*.parquet"))
    return pd.read_parquet(fs[-1]) if fs else None

def summ(df):
    fin = df["truncated"] == 0
    return dict(score=df["correct"].mean(), pfin=fin.mean(),
                acc_fin=df.loc[fin, "correct"].mean() if fin.any() else float("nan"),
                acc_tr=df.loc[~fin, "correct"].mean() if (~fin).any() else float("nan"),
                lmean=df["resp_len"].mean(), lmed=df["resp_len"].median())

def tail_cycle(t, w=200):
    tail = t[-3000:]
    seg = tail[-w:]
    return tail.count(seg) if seg else 0   # >1 => 尾部片段在末 3k 字符里重复

for run, sp, sl in PAIRS:
    a, b = load(run, sp), load(run, sl)
    if a is None or b is None:
        print(f"== {run}: MISSING artifacts (peak={a is not None}, late={b is not None})"); continue
    sa, sb = summ(a), summ(b)
    print(f"\n== {run}  step{sp} -> step{sl}")
    print(f"   score {sa['score']:.3f} -> {sb['score']:.3f}   P(finish) {sa['pfin']:.3f} -> {sb['pfin']:.3f}")
    print(f"   acc|finish {sa['acc_fin']:.3f} -> {sb['acc_fin']:.3f}   acc|trunc {sa['acc_tr']:.3f} -> {sb['acc_tr']:.3f}")
    print(f"   len mean/med {sa['lmean']:.0f}/{sa['lmed']:.0f} -> {sb['lmean']:.0f}/{sb['lmed']:.0f}")
    m = a.set_index("problem_id")[["truncated","correct","resp_len","response"]].join(
        b.set_index("problem_id")[["truncated","correct","resp_len","response"]], lsuffix="_p", rsuffix="_l")
    ff = (m.truncated_p==0)&(m.truncated_l==0); ft = (m.truncated_p==0)&(m.truncated_l==1)
    tf = (m.truncated_p==1)&(m.truncated_l==0); tt = (m.truncated_p==1)&(m.truncated_l==1)
    print(f"   转移: fin->fin {ff.sum()}  fin->trunc {ft.sum()}  trunc->fin {tf.sum()}  trunc->trunc {tt.sum()}")
    for name, msk in [("fin@both", ff), ("fin->trunc", ft), ("trunc->fin", tf), ("trunc@both", tt)]:
        if msk.sum():
            print(f"   {name:<11} n={msk.sum():<4} acc {m.loc[msk,'correct_p'].mean():.3f} -> {m.loc[msk,'correct_l'].mean():.3f}")
    # late 截断响应的文本读法
    tr = m[m.truncated_l==1]
    if len(tr):
        first_box, nbox, cyc = [], [], 0
        for t in tr["response_l"].astype(str):
            i = t.find("\\boxed{"); first_box.append(i if i>=0 else None)
            nbox.append(t.count("\\boxed"))
            if tail_cycle(t) > 1: cyc += 1
        fb = [x for x in first_box if x is not None]
        print(f"   late 截断响应 n={len(tr)}: 有 boxed {len(fb)}/{len(tr)}, 首 boxed 位置中位 {pd.Series(fb).median() if fb else '-'} 字符,")
        print(f"     boxed 次数中位 {pd.Series(nbox).median():.0f}, 尾部循环占比 {cyc}/{len(tr)}")
    # late 完成但答错、peak 曾答对的问题(a3 类问题的核心切片):存 3 例全文
    lost = m[(m.correct_p==1)&(m.correct_l==0)&(m.truncated_l==0)]
    print(f"   完成但由对变错 n={len(lost)}(late 未截断)")
    ex = lost.head(3)
    for pid, r in ex.iterrows():
        fp = f"{OUT}/ex_{run}_{re.sub('[^a-z0-9]','_',pid.lower())}.txt"
        with open(fp,"w") as f:
            f.write(f"### {pid}\n--- step{sp} (correct, len {r.resp_len_p}):\n{r.response_p}\n\n--- step{sl} (wrong, len {r.resp_len_l}):\n{r.response_l}\n")
        print(f"     例存 {os.path.basename(fp)}")
