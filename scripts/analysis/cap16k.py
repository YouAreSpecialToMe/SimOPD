#!/usr/bin/env python3
"""Would a 16k budget have lost these answers?

The in-loop validation ran under max_response_length=16384; the re-run evals ran at
32768.  If the in-loop curve under-measures late checkpoints, the mechanism should
be visible directly: responses that are correct at 32k but longer than 16k would
have been truncated -- and a truncated response has no \\boxed{}, which the grader
scores as 0 unconditionally (measured: 0.0% correct without it, n=54,000).

Re-scores each cell under a simulated 16k cap and prints the gap.
"""
import glob
import os
import re
import sys

import pyarrow.parquet as pq

EV = sys.argv[1] if len(sys.argv) > 1 else "/share/rush/zz865/opd/evals"
ARM = sys.argv[2] if len(sys.argv) > 2 else "vanilla_corr_s0"
CAP = 16384

print(f"{ARM} · MATH500 · 模拟 16k 截断\n")
print("%6s %8s %9s %9s %8s %9s" %
      ("step", "acc@32k", "acc@16k", "差", ">16k占比", "对的里>16k"))
rows = []
for f in sorted(glob.glob(os.path.join(EV, f"{ARM}__math500__step*.parquet")),
                key=lambda p: int(re.search(r"step(\d+)", p).group(1))):
    step = int(re.search(r"step(\d+)", f).group(1))
    t = pq.read_table(f, columns=["correct", "resp_len"])
    cor = t.column("correct").to_pylist()
    ln = t.column("resp_len").to_pylist()
    n = len(cor)
    a32 = sum(1 for c in cor if c) / n
    # under a 16k cap the long ones get cut off mid-answer -> no \boxed{} -> 0
    a16 = sum(1 for c, L in zip(cor, ln) if c and L <= CAP) / n
    over = sum(1 for L in ln if L > CAP) / n
    ncor = sum(1 for c in cor if c)
    corover = (sum(1 for c, L in zip(cor, ln) if c and L > CAP) / ncor) if ncor else 0
    rows.append((step, 100 * a32, 100 * a16, 100 * (a32 - a16), 100 * over, 100 * corover))
    print("%6d %8.1f %9.1f %9.1f %8.1f %9.1f" % rows[-1])

print("\n口径:acc@16k = 把长度超过 16384 的响应一律记 0(截断后写不出 \\boxed{},")
print("判分器对无 box 的响应实测 0.0% 正确)。这是在环 val 会看到的上界式模拟,")
print("不是重新判分——真正的 16k 生成会在不同位置停,不必然与截断等价。")
