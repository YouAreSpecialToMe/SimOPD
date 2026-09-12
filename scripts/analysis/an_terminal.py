#!/usr/bin/env python3
"""AGENT.md task 3: terminal behaviour + dl_last distribution, all arms.

Reads traj/summary_<step>.parquet (128 rows/step, every 25 steps) and reports,
per arm and per step:
  stop_frac    fraction of sequences that ended on a stop token (last_is_stop)
  trunc_frac   fraction truncated at the cap
  dl_last      the EVENT-level delta-ell at the stop position (mean / p10 / p90)
  p_eot,p_ime  student's prob on 151643 / 151645 at the last position
  tch_top1     how often the teacher's top-1 at the last position IS a terminator
  rep4         4-gram repetition
  ent_last     entropy at the last position

The p_last_151643 / p_last_151645 pair is the direct read on the terminator
question: with the corrected carrier the student still stops on <|endoftext|>,
and we can see what the teacher wanted there.
"""
import os
import sys
import glob
import re
import statistics as st
import pyarrow.parquet as pq

root = sys.argv[1] if len(sys.argv) > 1 else "/home/zz865/opd_pack/extracted"
only = sys.argv[2:] if len(sys.argv) > 2 else None

EOT, IMEND = 151643, 151645
COLS = ["resp_len", "last_is_stop", "truncated", "dl_last", "dl_mean", "dl_max",
        "rep4", "ent_last", "p_last_151643", "p_last_151645",
        "tch_top1_last_id", "tch_lp_last", "score", "n_stop_body"]


def frac(v):
    return sum(1 for x in v if x) / len(v) if v else float("nan")


def q(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return float("nan")
    i = max(0, min(len(v) - 1, int(round(p * (len(v) - 1)))))
    return v[i]


arms = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
if only:
    arms = [a for a in arms if any(o in a for o in only)]

print(f"{'arm':26} {'step':>4} {'stop%':>6} {'trunc%':>6} {'len':>6} "
      f"{'dl_last':>8} {'dl_p10':>8} {'dl_p90':>8} {'p_eot':>7} {'p_ime':>7} "
      f"{'tchT1=term%':>11} {'rep4':>6} {'ent_l':>6}")

for a in arms:
    files = sorted(glob.glob(os.path.join(root, a, "traj", "summary_*.parquet")),
                   key=lambda f: int(re.search(r"summary_(\d+)", f).group(1)))
    if not files:
        continue
    for f in files:
        step = int(re.search(r"summary_(\d+)", f).group(1))
        t = pq.read_table(f)
        have = [c for c in COLS if c in t.column_names]
        d = {c: t.column(c).to_pylist() for c in have}
        n = len(d.get("resp_len", []))
        if not n:
            continue
        dl = [x for x in d.get("dl_last", []) if x is not None]
        t1 = d.get("tch_top1_last_id", [])
        tterm = sum(1 for x in t1 if x in (EOT, IMEND)) / len(t1) if t1 else float("nan")
        print(f"{a:26} {step:4d} "
              f"{100*frac(d.get('last_is_stop',[])):6.1f} "
              f"{100*frac(d.get('truncated',[])):6.1f} "
              f"{st.mean(d['resp_len']):6.0f} "
              f"{(st.mean(dl) if dl else float('nan')):8.2f} "
              f"{q(dl,0.10):8.2f} {q(dl,0.90):8.2f} "
              f"{(st.mean([x for x in d.get('p_last_151643',[]) if x is not None]) if d.get('p_last_151643') else float('nan')):7.3f} "
              f"{(st.mean([x for x in d.get('p_last_151645',[]) if x is not None]) if d.get('p_last_151645') else float('nan')):7.5f} "
              f"{100*tterm:11.1f} "
              f"{(st.mean([x for x in d.get('rep4',[]) if x is not None]) if d.get('rep4') else float('nan')):6.3f} "
              f"{(st.mean([x for x in d.get('ent_last',[]) if x is not None]) if d.get('ent_last') else float('nan')):6.3f}")
    print()
