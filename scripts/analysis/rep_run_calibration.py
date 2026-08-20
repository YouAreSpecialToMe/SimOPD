"""Calibrate the rep-freeze detector on REAL responses: run-length distribution of the
exact-8-gram repetition mask, at both noise regimes.

  greedy 32k truncated responses  = the deterministic end (late collapse, entropy ~0.1)
  tau=1.0 16k responses           = the training-noise end (where the gate must fire early)

For each response: rep mask (same rolling-hash rule as _rep_runs_packed), run-length
histogram, and for candidate rules the RECALL on loop text (fraction of rep mass inside
positions the rule freezes) vs FALSE-POSITIVE on healthy text (frozen fraction among
correct-stopped responses). Rules: MINRUN in {16,32,64,128}; trailing-density W=128
rho in {0.5,0.6,0.7}.

  ARM=c4_pi_tail_budget GREEDY_ROOT=$D/n2/probe_evals/mt32768 T1_ROOT=$D/n2/probe_evals/t1 \
      python scripts/analysis/rep_run_calibration.py
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

D = "/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ARM = os.environ.get("ARM", "c4_pi_tail_budget")
N_GRAM = int(os.environ.get("REP_N", "8"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")


def rep_mask(ids, n=N_GRAM):
    ids = torch.tensor(ids, dtype=torch.long)
    L = ids.shape[0]
    rep = torch.zeros(L, dtype=torch.bool)
    if L < n:
        return rep.numpy()
    win = ids.unfold(0, n, 1)
    h = win[:, 0].clone()
    for j in range(1, n):
        h = h * 1000003 + win[:, j]
    uniq, inv = torch.unique(h, return_inverse=True)
    ar = torch.arange(h.shape[0])
    first = torch.full((uniq.shape[0],), h.shape[0], dtype=torch.long).scatter_reduce(0, inv, ar, reduce="amin", include_self=True)
    rep[n - 1:] = ar > first[inv]
    return rep.numpy()


def runs(mask):
    out, c = [], 0
    for v in mask:
        if v:
            c += 1
        elif c:
            out.append(c); c = 0
    if c:
        out.append(c)
    return out


def freeze_minrun(mask, minrun):
    out = np.zeros_like(mask)
    c = 0
    for i, v in enumerate(mask):
        c = c + 1 if v else 0
        if c >= minrun:
            out[i - c + 1:i + 1] = True
    return out


def freeze_density(mask, W=128, rho=0.6):
    cs = np.concatenate([[0], np.cumsum(mask)])
    idx = np.arange(len(mask))
    lo = np.maximum(0, idx - W + 1)
    dens = (cs[idx + 1] - cs[lo]) / np.minimum(W, idx + 1)
    return dens >= rho


def newest(root, run_id):
    hits = sorted(glob.glob(f"{root}/{run_id}__math500__*seed*.parquet"))
    for h in reversed(hits):
        df = pd.read_parquet(h)
        if "response" in df.columns:
            return df
    return None


def analyse(name, df, loop_sel, healthy_sel, tail_only):
    loops = df[loop_sel].head(24)
    healthy = df[healthy_sel].sample(n=min(30, int(healthy_sel.sum())), random_state=0)
    print(f"\n=== {name}: loops n={len(loops)} healthy n={len(healthy)}")
    L_masks = []
    for _, r in loops.iterrows():
        ids = tok.encode(r.response, add_special_tokens=False)
        if tail_only:
            ids = ids[-8000:]
        L_masks.append(rep_mask(ids))
    H_masks = [rep_mask(tok.encode(r.response, add_special_tokens=False)) for _, r in healthy.iterrows()]
    all_runs = [r for m in L_masks for r in runs(m)]
    if all_runs:
        q = np.percentile(all_runs, [50, 90, 99])
        print(f"  loop-run lengths: n={len(all_runs)} med={q[0]:.0f} p90={q[1]:.0f} p99={q[2]:.0f} max={max(all_runs)}"
              f" | rep frac (loops)={np.mean([m.mean() for m in L_masks]):.2f} (healthy)={np.mean([m.mean() for m in H_masks]):.2f}")
    hdr = f"  {'rule':<16}{'loop recall':>12}{'healthy FP':>12}"
    print(hdr)
    for mr in (16, 32, 64, 128):
        rec = np.mean([freeze_minrun(m, mr)[m].mean() if m.any() else 0.0 for m in L_masks])
        fp = np.mean([freeze_minrun(m, mr).mean() for m in H_masks])
        print(f"  {'minrun=' + str(mr):<16}{rec:>12.2f}{fp:>12.4f}")
    for rho in (0.5, 0.6, 0.7):
        rec = np.mean([freeze_density(m, 128, rho)[m].mean() if m.any() else 0.0 for m in L_masks])
        fp = np.mean([freeze_density(m, 128, rho).mean() for m in H_masks])
        print(f"  {'dens W128 r' + str(rho):<16}{rec:>12.2f}{fp:>12.4f}")


import re
g = newest(os.environ.get("GREEDY_ROOT", f"{D}/n2/probe_evals/mt32768"), f"{ARM}_s0_16k")
if g is not None:
    analyse("greedy 32k @250 (deterministic loops)", g,
            g.finish_reason == "length", (g.finish_reason == "stop") & (g.correct == 1), tail_only=True)
root = os.environ.get("T1_ROOT", f"{D}/n2/probe_evals/t1")
cells = {}
for f in sorted(glob.glob(f"{root}/*__math500__step*__seed*.parquet")):
    m = re.match(r"(.+)_s0_16k__math500__step(\d+)__", os.path.basename(f))
    if m:
        cells[(m.group(1), int(m.group(2)))] = f
for (arm, st), f in sorted(cells.items()):
    df = pd.read_parquet(f)
    if "response" not in df.columns:
        continue
    analyse(f"tau=1.0 {arm}@{st} (trunc={float((df.finish_reason == 'length').mean()):.2f})", df,
            df.finish_reason == "length", (df.finish_reason == "stop") & (df.correct == 1), tail_only=True)
print("\nDONE")
