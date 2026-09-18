#!/usr/bin/env python3
"""Per-arm remaining work, computed from the artifacts rather than transcribed.

  targets   configs/retrain_roster.tsv (arm, steps, prio)
  current   <arm>_s0/latest_checkpointed_iteration.txt from the unpacked runs/
  evals     evals/*.parquet, one file per (arm, bench, step)
  grid      light: 25/50/150 -> amc23 minerva math500; 100/200/250 -> all five

Prints markdown rows plus the totals, so they can be checked against the handoff
manual's own numbers (650 steps; 150 cells, 36 complete, 114 missing, of which
99 have weights on disk, 15 need training first, 11 are partially done).
"""
import collections
import os
import re
import sys

ROSTER, RUNS, EVALS = sys.argv[1:4]
FIVE = ("aime24", "aime25", "amc23", "minerva", "math500")
THREE = ("amc23", "minerva", "math500")
def need(step):
    return set(FIVE) if step in (100, 200, 250) else set(THREE)
def grid(target):
    return [s for s in (25, 50, 100, 150, 200, 250) if s <= target]

roster = []
for line in open(ROSTER, encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    arm, steps, prio, note = (line.rstrip("\n").split("\t") + [""] * 4)[:4]
    roster.append((arm, int(steps), int(prio), note))

have = collections.defaultdict(set)
for f in os.listdir(EVALS):
    m = re.match(r"^(.+?)_s0__(.+?)__step(\d+)__", f)
    if m:
        have[(m.group(1), int(m.group(3)))].add(m.group(2))

# measured per-cell wall time on one H100 (HANDOFF-20260917 section 8.2); a partially
# done cell costs less, so per-tier totals below are upper bounds
HOURS_NONANCHOR, HOURS_ANCHOR = 1.6, 2.5

tot = collections.Counter()
tier_cells = collections.Counter()
tier_hours = collections.Counter()
rows = []
for arm, target, prio, note in roster:
    p = os.path.join(RUNS, f"{arm}_s0", "latest_checkpointed_iteration.txt")
    cur = int(open(p).read().strip()) if os.path.exists(p) else -1
    todo_train = max(0, target - cur)
    done = partial = ready = blocked = 0
    parts = []
    for s in grid(target):
        got = have.get((arm, s), set())
        miss = need(s) - got
        tot["cells"] += 1
        if not miss:
            done += 1
            continue
        if s > cur:
            blocked += 1
            parts.append(f"{s}*")
            continue
        ready += 1
        tier_cells[prio] += 1
        tier_hours[prio] += HOURS_ANCHOR if s in (100, 200, 250) else HOURS_NONANCHOR
        if got:
            partial += 1
            parts.append(f"{s}(缺 {','.join(b for b in FIVE if b in miss)})")
        else:
            parts.append(f"{s}")
    tot["done"] += done
    tot["ready"] += ready
    tot["blocked"] += blocked
    tot["partial"] += partial
    tot["steps"] += todo_train
    tot["arms_training"] += todo_train > 0
    rows.append((prio, arm, cur, target, todo_train, done, len(grid(target)),
                 ready, blocked, " ".join(parts) or "—"))

rows.sort(key=lambda r: (r[0], -r[4], -(r[7] + r[8]), r[1]))
print("| 优先级 | 臂 | 落档 / 目标 | 还差步 | 评测已齐 | 现在能评 | 等训练 | 待评的步 |")
print("|---|---|---|---:|---|---:|---:|---|")
for prio, arm, cur, target, tt, done, cells, ready, blocked, parts in rows:
    print(f"| {prio} | `{arm}` | {cur} / {target} | {tt or '—'} | {done} / {cells} | "
          f"{ready or '—'} | {blocked or '—'} | {parts} |")

missing = tot["ready"] + tot["blocked"]
print(f"\n训练:{tot['arms_training']} 条臂还差 {tot['steps']} 步")
print(f"评测:应有 {tot['cells']} 格,已齐 {tot['done']},缺 {missing}"
      f"(现在能评 {tot['ready']},其中部分完成 {tot['partial']};等训练 {tot['blocked']})")
PRIO = {0: "判决基线/回归对照", 1: "自研方法", 2: "代表面板", 3: "A 轴"}
print("\n现在能评的格,按优先级(H100 单格实测耗时,部分完成的格按整格计,故为上界):")
for p in sorted(tier_cells):
    print(f"  prio {p} {PRIO.get(p, ''):<10} {tier_cells[p]:3d} 格  <= {tier_hours[p]:6.1f} GPU·h")
print(f"  合计         {sum(tier_cells.values()):3d} 格  <= {sum(tier_hours.values()):6.1f} GPU·h")
ok = (tot["steps"], tot["cells"], tot["done"], missing, tot["ready"], tot["blocked"],
      tot["partial"]) == (650, 150, 36, 114, 99, 15, 11)
print("与交接手册的数字" + ("一致" if ok else "**不一致** —— 以本脚本为准,手册需要更正"))
