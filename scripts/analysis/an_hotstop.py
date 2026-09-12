#!/usr/bin/env python3
"""How often does the teacher say "you should have stopped here" mid-response?

A BODY position is "hot" when the teacher's probability on <|im_end|> exceeds
0.5 there -- the teacher is more likely than not to end the turn, and the student
kept writing.  Reports, per arm and step:

  seq_with_hot   % of responses containing at least one hot body position
  hot/seq        mean number of hot body positions per response
  1st_hot@       median index of the first hot position
  tail_after     median number of tokens the student wrote AFTER that point
"""
import glob
import re
import math
import statistics as st
import sys
import pyarrow.parquet as pq

root = sys.argv[1] if len(sys.argv) > 1 else "."
arms = sys.argv[2:] or ["vanilla_corr_s0", "d3_teachability_corr_s0",
                        "g1_verified_only_corr_s0", "h1_first_segment_corr_s0"]
THRESH = 0.5


def ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


for arm in arms:
    fs = sorted(glob.glob(f"{root}/{arm}/traj/step_*.parquet"),
                key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    if not fs:
        continue
    print("=" * 78)
    print(arm)
    print("%4s %5s %9s %13s %9s %10s %11s" %
          ("step", "seqs", "len_mean", "seq_with_hot", "hot/seq", "1st_hot@", "tail_after"))
    for f in fs:
        step = int(re.search(r"step_(\d+)", f).group(1))
        t = pq.read_table(f)
        if "tch_lp_151645" not in t.column_names:
            continue
        rid = t.column("response_ids").to_pylist()
        ime = t.column("tch_lp_151645").to_pylist()
        nseq = 0
        withhot = 0
        hots, firsts, tails, lens = [], [], [], []
        for r, mi in zip(rid, ime):
            if not r or not mi:
                continue
            nseq += 1
            L = min(len(r), len(mi))
            lens.append(L)
            idx = [j for j in range(L - 1) if ok(mi[j]) and math.exp(mi[j]) > THRESH]
            hots.append(len(idx))
            if idx:
                withhot += 1
                firsts.append(idx[0])
                tails.append(L - 1 - idx[0])
        if not nseq:
            continue
        print("%4d %5d %9.0f %12.1f%% %9.2f %10s %11s" %
              (step, nseq, st.mean(lens), 100 * withhot / nseq, st.mean(hots),
               ("%.0f" % st.median(firsts)) if firsts else "-",
               ("%.0f" % st.median(tails)) if tails else "-"))
