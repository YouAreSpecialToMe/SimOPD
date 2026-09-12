#!/usr/bin/env python3
"""Where does the teacher put terminator mass -- only at the end, or everywhere?

This is the discriminating measurement registered in docs/AUDIT-20260911-setup.md
section 10.4.  If a teacher assigns non-negligible probability to a terminator at
ORDINARY mid-sequence positions, then sampling one there earns a positive
advantage and "stop early" is reinforced everywhere -- the mechanism we floated
as a candidate for Demystifying's Mode B.  If the mass sits only at genuine end
positions, that channel is closed.

Reads traj/step_<n>.parquet, which carries per-token arrays:
    response_ids, stu_lp, ent, tch_lp, tch_top1_id, tch_lp_151643, tch_lp_151645

Splits each response into BODY (all but the final token) and END (final token),
and reports the distribution of the teacher's terminator mass in each.
"""
import os
import sys
import glob
import re
import math
import statistics as st
import pyarrow.parquet as pq

root = sys.argv[1] if len(sys.argv) > 1 else "/home/zz865/opd_pack/extracted"
arm = sys.argv[2] if len(sys.argv) > 2 else "vanilla_corr_s0"
want_steps = [int(x) for x in sys.argv[3:]] or None

EOT, IMEND = 151643, 151645


def qs(v, ps=(0.5, 0.9, 0.99, 0.999, 1.0)):
    v = sorted(v)
    if not v:
        return [float("nan")] * len(ps)
    return [v[max(0, min(len(v) - 1, int(round(p * (len(v) - 1)))))] for p in ps]


files = sorted(glob.glob(os.path.join(root, arm, "traj", "step_*.parquet")),
               key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
if want_steps:
    files = [f for f in files if int(re.search(r"step_(\d+)", f).group(1)) in want_steps]

print(f"arm = {arm}")
print("q_T(term) = teacher prob on <|endoftext|>(151643) or <|im_end|>(151645)")
print("BODY = every response position except the last; END = the last position\n")
print(f"{'step':>4} {'seqs':>4} {'body_tok':>9} | "
      f"{'BODY q(ime): p50':>16} {'p90':>8} {'p99':>8} {'p99.9':>8} {'max':>8} | "
      f"{'frac>.01':>8} {'frac>.1':>8} {'frac>.5':>8} | {'END q(ime)':>10} {'END q(eot)':>10}")

for f in files:
    step = int(re.search(r"step_(\d+)", f).group(1))
    t = pq.read_table(f)
    cn = t.column_names
    need = ["response_ids", "tch_lp_151645", "tch_lp_151643"]
    if any(c not in cn for c in need):
        print(f"{step:4d}  (missing columns: {[c for c in need if c not in cn]})")
        continue
    rid = t.column("response_ids").to_pylist()
    ime = t.column("tch_lp_151645").to_pylist()
    eot = t.column("tch_lp_151643").to_pylist()

    body_ime, end_ime, end_eot = [], [], []
    nseq = 0
    for r, mi, me in zip(rid, ime, eot):
        if not r or not mi:
            continue
        nseq += 1
        L = min(len(r), len(mi), len(me))
        for j in range(L - 1):
            v = mi[j]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                body_ime.append(math.exp(v))
        if L:
            for src, dst in ((mi, end_ime), (me, end_eot)):
                v = src[L - 1]
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    dst.append(math.exp(v))

    if not body_ime:
        print(f"{step:4d}  (no body tokens)")
        continue
    p50, p90, p99, p999, mx = qs(body_ime)
    f01 = sum(1 for x in body_ime if x > 0.01) / len(body_ime)
    f1 = sum(1 for x in body_ime if x > 0.1) / len(body_ime)
    f5 = sum(1 for x in body_ime if x > 0.5) / len(body_ime)
    print(f"{step:4d} {nseq:4d} {len(body_ime):9d} | "
          f"{p50:16.2e} {p90:8.2e} {p99:8.2e} {p999:8.2e} {mx:8.3f} | "
          f"{f01:8.4f} {f1:8.4f} {f5:8.4f} | "
          f"{(st.mean(end_ime) if end_ime else float('nan')):10.3f} "
          f"{(st.mean(end_eot) if end_eot else float('nan')):10.2e}")
