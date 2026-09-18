#!/usr/bin/env python3
"""Reconcile the handover's outstanding-eval table against what actually landed.

Table B in opd-ckpt/README.md lists, per arm, which (step, benchmark) cells still
need running.  evals/ in the opd dataset now carries 155 result parquets stamped
today.  This says which of those cells are closed, which are still open, and
whether anything arrived that the table did not ask for.

Filenames are <run>__<bench>__step<n>__seed<s>__<ts>.parquet.
"""
import collections
import os
import re
import sys

README = sys.argv[1] if len(sys.argv) > 1 else "/share/rush/zz865/opd-ckpt/README.md"
EVALS = sys.argv[2] if len(sys.argv) > 2 else "/share/rush/zz865/opd/evals"

# --- the backlog, straight out of the table --------------------------------
want = set()            # (arm, step, bench)
status = {}             # arm -> "需续训 …" / "已完成"
rows = 0
for line in open(README, encoding="utf-8"):
    m = re.match(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*$", line)
    if not m:
        continue
    arm, st, cells = m.groups()
    if "(" not in cells:                       # not the eval table
        continue
    rows += 1
    status[arm] = st
    for step, benches in re.findall(r"(\d+)\(([^)]*)\)", cells):
        for b in benches.split(","):
            want.add((arm, int(step), b.strip()))

# --- what exists -----------------------------------------------------------
have = set()
extra = []
for f in os.listdir(EVALS):
    if not f.endswith(".parquet"):
        continue
    m = re.match(r"^(.+?)__(.+?)__step(\d+)__seed(\d+)__", f)
    if not m:
        extra.append(f)
        continue
    run, bench, step, _ = m.groups()
    arm = run[:-3] if run.endswith("_s0") else run
    have.add((arm, int(step), bench))

done = want & have
open_ = want - have
unasked = have - want

print(f"表 B: {rows} 行臂, {len(want)} 个 (臂,步,bench) 单元")
print(f"evals/: {len(have)} 个单元 ({len(os.listdir(EVALS))} 个 parquet)")
print(f"\n已补上   {len(done):4d}  ({100*len(done)/len(want):.1f}%)")
print(f"仍缺     {len(open_):4d}")
print(f"表外多出 {len(unasked):4d}")

print("\n--- 逐臂 ---")
print("%-28s %-14s %6s %6s" % ("臂", "训练状态", "已补", "仍缺"))
arms = sorted({a for a, _, _ in want})
for a in arms:
    d = sum(1 for x in done if x[0] == a)
    o = sum(1 for x in open_ if x[0] == a)
    flag = "  <<< 全齐" if o == 0 else ""
    print("%-28s %-14s %6d %6d%s" % (a, status.get(a, "?")[:14], d, o, flag))

if unasked:
    print("\n--- 表外多出的(按臂) ---")
    for a, c in collections.Counter(x[0] for x in unasked).most_common():
        print(f"   {a:28} {c}")
if extra:
    print(f"\n--- 文件名不合规: {len(extra)} 个 ---")
    for f in extra[:5]:
        print("   ", f)

print("\n--- 仍缺的明细(前 30)---")
for a, s, b in sorted(open_)[:30]:
    print(f"   {a:28} step{s:<4} {b}")
