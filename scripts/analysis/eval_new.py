#!/usr/bin/env python3
"""What the evals/ tree in Jerrycool/opd actually adds.

The September handover package shipped no eval results at all -- its Table B only
listed which cells still needed running.  This dataset now carries 155 result
parquets, all stamped 2026-09-17, so the useful question is which (arm, step,
benchmark) cells they cover and which of the protocol's grid points are still
blank.
"""
import collections
import os
import re
import sys

EV = sys.argv[1] if len(sys.argv) > 1 else "/share/rush/zz865/opd/evals"

FULL = ["aime24", "aime25", "amc23", "minerva", "math500"]   # anchor steps
LIGHT = ["amc23", "minerva", "math500"]                      # 25 / 50 / 150
STEPS = [25, 50, 100, 150, 200, 250]
def want_at(s):
    return set(FULL) if s in (100, 200, 250) else set(LIGHT)

cells = collections.defaultdict(set)
hours = collections.Counter()
for f in os.listdir(EV):
    m = re.match(r"^(.+?)__(.+?)__step(\d+)__seed(\d+)__(\d{8}T\d{6}Z)", f)
    if not m:
        continue
    arm, bench, step, _, t = m.groups()
    cells[arm].add((int(step), bench))
    hours[t[:11]] += 1

print(f"{sum(len(v) for v in cells.values())} 个新评测单元，覆盖 {len(cells)} 条臂\n")
head = "".join(f"{s:>7}" for s in STEPS)
print(f"{'arm':<28}{head}{'合计':>7}")
print("-" * (28 + 7 * len(STEPS) + 7))
for a in sorted(cells):
    row = ""
    for s in STEPS:
        bs = {b for st, b in cells[a] if st == s}
        row += f"{'—' if not bs else ('全' if bs >= want_at(s) else f'{len(bs)}/{len(want_at(s))}'):>7}"
    print(f"{a:<28}{row}{len(cells[a]):>7}")

print("\n没跑齐的格子：")
any_gap = False
for a in sorted(cells):
    gaps = []
    for s in STEPS:
        bs = {b for st, b in cells[a] if st == s}
        if bs:
            miss = want_at(s) - bs
            if miss:
                gaps.append(f"step{s} 缺 " + ",".join(sorted(miss)))
    if gaps:
        any_gap = True
        print(f"   {a:<28} {'; '.join(gaps)}")
if not any_gap:
    print("   （凡是跑了的步，基准都齐）")

print("\n完全没评的步（该臂训练到了但这批没跑）：")
for a in sorted(cells):
    have = sorted({st for st, _ in cells[a]})
    blank = [s for s in STEPS if s not in have and s <= max(have)]
    if blank:
        print(f"   {a:<28} {blank}")

print("\n生成时间（UTC）：")
for k, v in sorted(hours.items()):
    print(f"   {k}:00   {v} 个")
