#!/usr/bin/env python3
"""Sanity-check the SimOPD trajectory pack against what its README claims.

Run on the cluster:  python3 pack_verify.py /home/zz865/opd_pack/extracted

Checks, in order of how much they would hurt if wrong:
  1. arm count and file count vs the README (29 arms, 18242 files)
  2. every arm carries the four artefact families (traj/, val_gen/, metrics/, manifests)
  3. tch_lp_nan == 0 in the teacher columns  -- the 09-04 index fix; the README
     claims the whole pack is from the post-fix rerun, this is the receipt
  4. div/tok_step*.parquet present  -- results/20260909 reported n_tok_parquet=0,
     which we believe is an extract.py glob bug (missing the div/ level)
  5. per-arm max step, so the pack can be reconciled against progress.tsv
"""
import os
import sys
import glob
import json
import re
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "/home/zz865/opd_pack/extracted"

arms = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
print(f"arms: {len(arms)}  (README says 29)")

total_files = 0
rows = []
missing = defaultdict(list)

for a in arms:
    p = os.path.join(root, a)
    n = sum(len(f) for _, _, f in os.walk(p))
    total_files += n

    traj = os.path.join(p, "traj")
    ids = glob.glob(os.path.join(traj, "ids_*.parquet"))
    summ = glob.glob(os.path.join(traj, "summary_*.parquet"))
    step = glob.glob(os.path.join(traj, "step_*.parquet"))
    tok = glob.glob(os.path.join(traj, "div", "tok_step*.parquet"))
    rank = glob.glob(os.path.join(traj, "div", "rank*.jsonl"))
    vg = glob.glob(os.path.join(p, "val_gen", "*.jsonl"))
    met = glob.glob(os.path.join(p, "metrics", "launch_*.jsonl"))

    def maxstep(paths, pat):
        s = [int(m.group(1)) for f in paths for m in [re.search(pat, os.path.basename(f))] if m]
        return max(s) if s else 0

    for label, got in (("traj", ids), ("val_gen", vg), ("metrics", met)):
        if not got:
            missing[label].append(a)

    rows.append((a, n, len(ids), len(summ), len(step), len(tok), len(rank), len(vg),
                 maxstep(ids, r"ids_(\d+)"), maxstep(vg, r"(\d+)\.jsonl")))

print(f"files: {total_files}  (README says 18242)")
print()
hdr = ("arm", "files", "ids", "summ", "step", "tok", "rank", "vgen", "max_ids", "max_vg")
print("%-28s %7s %5s %5s %5s %5s %5s %5s %8s %7s" % hdr)
for r in rows:
    print("%-28s %7d %5d %5d %5d %5d %5d %5d %8d %7d" % r)

if missing:
    print("\nMISSING artefact families:")
    for k, v in missing.items():
        print(f"  {k}: {', '.join(v)}")
else:
    print("\nall arms carry traj/ + val_gen/ + metrics/")

print("\n--- teacher columns (tch_lp_nan) ---")
try:
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow not available; skipped (install or run under a venv that has it)")
    sys.exit(0)

bad = 0
checked = 0
for a in arms:
    files = sorted(glob.glob(os.path.join(root, a, "traj", "summary_*.parquet")))
    if not files:
        continue
    for f in (files[0], files[-1]):
        try:
            t = pq.read_table(f)
        except Exception as e:
            print(f"  READ-FAIL {a} {os.path.basename(f)}: {e}")
            bad += 1
            continue
        cols = t.column_names
        nan_cols = [c for c in cols if "nan" in c.lower()]
        checked += 1
        for c in nan_cols:
            col = t.column(c).to_pylist()
            nz = [v for v in col if v]
            if nz:
                print(f"  NONZERO {a} {os.path.basename(f)} {c}: {len(nz)}/{len(col)} rows, e.g. {nz[:3]}")
                bad += 1
print(f"checked {checked} summary files across {len(arms)} arms; problems: {bad}")
if bad == 0:
    print("tch_lp_nan == 0 everywhere sampled -- consistent with the post-09-04-fix rerun")
