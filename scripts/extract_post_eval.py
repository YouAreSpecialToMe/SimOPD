#!/usr/bin/env python3
"""把 post-eval 的逐样本 parquet 压成两张分析表。只读,不动任何产物。

产出(写到 simopd_data/,再由人拷进仓库 docs/data/):
  post_eval_cells.csv   逐 (arm, seed, step, benchmark) —— 最细一层
  post_eval_bystep.csv  逐 (arm, step) 汇总,seed 与 benchmark 池化

为什么不只存 acc:逐样本 parquet 自 2026-08-01 就带 resp_len / truncated /
finish_reason,489 个完成格全覆盖。这些列能回答 in-loop 面板回答不了的问题 ——
离线 tau=0.7、32k 预算下长度是否失控、错答是不是比对答长(“不会停”假说的直接检验)。
`response` 原文列 8/11 才加,所以这里只记它在不在(has_text),不依赖它。

口径与 eval_suite.py 严格一致:
  - 每个 (run, step, bench) 取时间戳最新的一份产物
  - avg@k = 逐题先对 k 个采样取均值,再对题目取均值
  - pass@k = 逐题取 max,再对题目取均值
  - composite = aime(24+25 合并池) / amc23 / minerva / math500 四组分等权
"""
import os
import re
import sys
from collections import defaultdict

import pandas as pd

ROOT = "/mgfs/shared/Group_GY/changhao/simopd_data"
EVAL = os.path.join(ROOT, "evals")
Q = os.path.join(ROOT, "evalq")
BENCH = ["aime24", "aime25", "amc23", "minerva", "math500"]
COMP = [("aime", ["aime24", "aime25"]), ("amc23", ["amc23"]),
        ("minerva", ["minerva"]), ("math500", ["math500"])]
COLS = ["problem_id", "sample_idx", "resp_len", "truncated", "finish_reason", "correct"]

roster = set(l.strip() for l in open(os.path.join(Q, "roster.txt")) if l.strip())
pat = re.compile(
    r"^(?P<run>.+?)__(?P<bench>aime24|aime25|amc23|minerva|math500)"
    r"__step(?P<step>\d+)__seed\d+__(?P<ts>\d{8}T\d{6}Z)\.parquet$")

newest = {}
for f in os.listdir(EVAL):
    m = pat.match(f)
    if m and m.group("run") in roster:
        k = (m.group("run"), int(m.group("step")), m.group("bench"))
        if k not in newest or m.group("ts") > newest[k][0]:
            newest[k] = (m.group("ts"), os.path.join(EVAL, f))

print("artifacts to read: %d" % len(newest), flush=True)

arm_of = lambda r: re.sub(r"_s\d+_16k$", "", r)
seed_of = lambda r: int(re.search(r"_s(\d+)_16k$", r).group(1))

rows, per_problem_acc = [], {}
for i, ((run, step, bench), (ts, path)) in enumerate(sorted(newest.items()), 1):
    if i % 250 == 0:
        print("  %d/%d" % (i, len(newest)), flush=True)
    try:
        import pyarrow.parquet as pq
        names = set(pq.ParquetFile(path).schema_arrow.names)
        df = pd.read_parquet(path, columns=[c for c in COLS if c in names])
    except Exception as e:
        print("SKIP %s (%s)" % (os.path.basename(path), e), flush=True)
        continue

    pp = df.groupby("problem_id")["correct"].mean()
    per_problem_acc[(run, step, bench)] = pp
    ok = df["correct"] == 1
    tr = df["truncated"] if "truncated" in df else pd.Series(0, index=df.index)
    fr = df["finish_reason"] if "finish_reason" in df else pd.Series("", index=df.index)

    rows.append({
        "arm": arm_of(run), "seed": seed_of(run), "run": run, "step": step,
        "bench": bench, "ts": ts,
        "n_problems": df["problem_id"].nunique(), "n_samples": len(df),
        "avg_at_k": float(pp.mean()),
        "pass_at_k": float(df.groupby("problem_id")["correct"].max().mean()),
        "len_mean": float(df.resp_len.mean()),
        "len_std": float(df.resp_len.std()),
        "len_p50": float(df.resp_len.quantile(.50)),
        "len_p90": float(df.resp_len.quantile(.90)),
        "len_max": int(df.resp_len.max()),
        "trunc_rate": float(tr.mean()),
        # 只看自然停下来的样本:撞帽会把长度压平,这一列才反映真实的“想写多长”
        "len_mean_untrunc": float(df.loc[tr == 0, "resp_len"].mean()) if (tr == 0).any() else None,
        # “不会停”假说的直接检验:错答是否显著更长 / 更容易撞帽
        "len_mean_correct": float(df.loc[ok, "resp_len"].mean()) if ok.any() else None,
        "len_mean_wrong": float(df.loc[~ok, "resp_len"].mean()) if (~ok).any() else None,
        "trunc_rate_correct": float(tr[ok].mean()) if ok.any() else None,
        "trunc_rate_wrong": float(tr[~ok].mean()) if (~ok).any() else None,
        "fr_stop": float((fr == "stop").mean()),
        "fr_length": float((fr == "length").mean()),
        "has_text": int("response" in names),
    })

cells = pd.DataFrame(rows)
if cells.empty:
    sys.exit("no artifacts parsed")
cells = cells.sort_values(["arm", "seed", "step", "bench"])
cells.to_csv(os.path.join(ROOT, "post_eval_cells.csv"), index=False)
print("wrote post_eval_cells.csv  rows=%d" % len(cells))

# ---- 逐 (arm, seed, step):只在 5 个 benchmark 齐全时算 composite ----
out = []
for (run, step), g in cells.groupby(["run", "step"]):
    if set(g.bench) != set(BENCH):
        continue
    comp = {}
    for name, bs in COMP:
        pooled = pd.concat([per_problem_acc[(run, step, b)] for b in bs])
        comp[name] = float(pooled.mean())
    w = g.n_samples
    out.append({
        "arm": g.arm.iloc[0], "seed": g.seed.iloc[0], "run": run, "step": step,
        "composite": sum(comp.values()) / 4.0,
        **{"acc_" + k: v for k, v in comp.items()},
        # 按样本数加权:AIME 32 采样的权重本就该大于 Minerva 的 3 采样
        "len_mean": float((g.len_mean * w).sum() / w.sum()),
        "len_p90": float((g.len_p90 * w).sum() / w.sum()),
        "trunc_rate": float((g.trunc_rate * w).sum() / w.sum()),
        "len_mean_correct": float((g.len_mean_correct * w).sum() / w.sum()),
        "len_mean_wrong": float((g.len_mean_wrong * w).sum() / w.sum()),
        "pass_minus_avg": float(((g.pass_at_k - g.avg_at_k) * w).sum() / w.sum()),
        "n_samples": int(w.sum()), "has_text": int(g.has_text.min()),
    })
bystep = pd.DataFrame(out).sort_values(["arm", "seed", "step"])
bystep.to_csv(os.path.join(ROOT, "post_eval_bystep.csv"), index=False)
print("wrote post_eval_bystep.csv  rows=%d (complete cells)" % len(bystep))
print("arms=%d  steps=%s" % (bystep.arm.nunique(), sorted(bystep.step.unique())))
print("has_text 覆盖: %d/%d 格" % (int(bystep.has_text.sum()), len(bystep)))
