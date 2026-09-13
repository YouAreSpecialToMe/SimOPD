#!/usr/bin/env python3
"""Is the per-token signal r_t = log pi_theta - log pi_T lower at late positions?

Under token-mean aggregation the loss is the mean of r over every token in the
batch.  A token whose r sits BELOW the batch mean lowers the loss simply by
existing, so if continuation tokens at late positions carry systematically
smaller r than early ones, the objective is served by writing more of them.

Reads traj/step_<n>.parquet per-token arrays (stu_lp, tch_lp, ent) and reports
mean r by position bin, plus what share of tokens in each bin fall below the
batch mean.  Also splits out the stop token so its cost is visible separately.
"""
import glob
import math
import os
import re
import statistics as st
import sys
import pyarrow.parquet as pq

root = sys.argv[1] if len(sys.argv) > 1 else "/home/zz865/opd_pack/extracted"
arm = sys.argv[2] if len(sys.argv) > 2 else "vanilla_corr_s0"
steps = [int(x) for x in sys.argv[3:]] or None

BINS = [(0, 100), (100, 500), (500, 2000), (2000, 8000), (8000, 16400)]


def ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


files = sorted(glob.glob(os.path.join(root, arm, "traj", "step_*.parquet")),
               key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
if steps:
    files = [f for f in files if int(re.search(r"step_(\d+)", f).group(1)) in steps]

print(f"arm = {arm}")
print("r_t = stu_lp - tch_lp  (archive convention; the loss is the batch mean of r)")
print("ent = student entropy at that position\n")

for f in files:
    step = int(re.search(r"step_(\d+)", f).group(1))
    t = pq.read_table(f)
    cn = t.column_names
    if not all(c in cn for c in ("stu_lp", "tch_lp", "response_ids")):
        continue
    slp = t.column("stu_lp").to_pylist()
    tlp = t.column("tch_lp").to_pylist()
    ent = t.column("ent").to_pylist() if "ent" in cn else [None] * len(slp)
    rid = t.column("response_ids").to_pylist()

    binvals = {b: [] for b in BINS}
    binent = {b: [] for b in BINS}
    stopvals = []
    allr = []
    for s, tt, e, r in zip(slp, tlp, ent, rid):
        if not s or not tt:
            continue
        L = min(len(s), len(tt), len(r))
        for j in range(L):
            if not (ok(s[j]) and ok(tt[j])):
                continue
            v = s[j] - tt[j]
            allr.append(v)
            if j == L - 1:
                stopvals.append(v)
                continue
            for b in BINS:
                if b[0] <= j < b[1]:
                    binvals[b].append(v)
                    if e and j < len(e) and ok(e[j]):
                        binent[b].append(e[j])
                    break
    if not allr:
        continue
    mu = st.mean(allr)
    print(f"step {step}:  batch mean r = {mu:.4f}   tokens = {len(allr)}   "
          f"stop-token r = {(st.mean(stopvals) if stopvals else float('nan')):.2f} "
          f"(n={len(stopvals)})")
    print("   %-14s %9s %9s %9s %9s" % ("position", "n", "mean r", "frac<mean", "mean ent"))
    for b in BINS:
        v = binvals[b]
        if not v:
            continue
        below = sum(1 for x in v if x < mu) / len(v)
        me = st.mean(binent[b]) if binent[b] else float("nan")
        print("   %-14s %9d %9.4f %9.3f %9.3f" %
              (f"[{b[0]},{b[1]})", len(v), st.mean(v), below, me))
    print()
