"""Compare arms at MATCHED steps, against vanilla's own seed spread.

Ranking arms by "best value anywhere on the curve" is what the raw scan does and it is
wrong twice over: a run killed at step 125 gets credit for a step-25 peak it then lost,
and an arm with one point at step 25 outranks an arm measured to 250. So: fix the step,
require the arm to have a point there, and print vanilla's three seeds at that same step
as the band anything has to clear.
"""
import re, glob, os, sys
from collections import defaultdict
from statistics import mean, pstdev

RUN = re.compile(r"^#+ RUN: ([A-Za-z0-9_.]+)")
VAL = re.compile(r"step:(\d+) .*?val-core/[^:]*acc/mean@1:np\.float64\(([0-9.]+)\)")

curves = defaultdict(dict)
for f in sorted(glob.glob("logs/*/lane*.log")) + sorted(glob.glob("logs/lane*.log")):
    cur = None
    with open(f, errors="replace") as fh:
        for line in fh:
            body = re.sub(r"^\([^)]*\)\s*", "", line.lstrip())
            m = RUN.match(body)
            if m:
                cur = m.group(1); continue
            if cur:
                m = VAL.search(body)
                if m:
                    curves[cur][int(m.group(1))] = float(m.group(2))

def arm_seed(n):
    m = re.match(r"^(.*)_s(\d+)$", n)
    return (m.group(1), int(m.group(2))) if m else (n, -1)

by_arm = defaultdict(dict)
for name, c in curves.items():
    a, s = arm_seed(name)
    by_arm[a][s] = c

STEPS = [int(x) for x in sys.argv[1:]] or [25, 50, 100]
for STEP in STEPS:
    van = [c[STEP] for s, c in sorted(by_arm.get("vanilla", {}).items()) if STEP in c]
    if not van:
        continue
    lo, hi = min(van), max(van)
    band = hi - lo
    print(f"\n===== step {STEP} =====  vanilla seeds {['%.3f' % v for v in van]}"
          f"  mean {mean(van):.3f}  spread {band:.3f}")
    rows = []
    for a, seeds in by_arm.items():
        vals = [c[STEP] for s, c in sorted(seeds.items()) if STEP in c]
        if not vals or a == "vanilla":
            continue
        rows.append((a, vals))
    rows.sort(key=lambda r: -mean(r[1]))
    for a, vals in rows:
        m = mean(vals)
        d = m - mean(van)
        # "outside the noise floor" = the arm's mean is outside vanilla's OWN observed
        # seed range at this step. With 1-3 seeds that is the honest bar; a t-test on
        # three points would dress it up as more than it is.
        mark = "  >" if m > hi else (" <" if m < lo else "  ~")
        print(f"  {mark} {a:28s} {m:6.3f}  d={d:+.3f}  n={len(vals)}  {['%.3f' % v for v in vals]}")
