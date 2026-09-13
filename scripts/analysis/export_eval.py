#!/usr/bin/env python3
"""Export the in-loop eval so one problem can be followed across checkpoints.

val_gen/<step>.jsonl holds the SAME 500 MATH500 problems at every eval, in the
same order, so row i is the same problem at every step.  That is the one place
in the archive where "how did this answer change as training went on" is
answerable; the training rollouts draw a fresh batch each step.

Writes:
  <out>            index: per problem, acc / output length / truncation per step
  <out>_prob/<i>.json   that problem's full answer at every step
"""
import glob
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/zz865/opd_pack/extracted"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/zz865/eval_data.json"
ARMS = sys.argv[3:] or ["vanilla_corr_s0"]
PROBDIR = os.path.splitext(OUT)[0] + "_prob"

os.makedirs(PROBDIR, exist_ok=True)
index = {"arms": {}}

for arm in ARMS:
    vg = os.path.join(ROOT, arm, "val_gen")
    if not os.path.isdir(vg):
        continue
    name = arm[:-3] if arm.endswith("_s0") else arm
    files = sorted(glob.glob(os.path.join(vg, "*.jsonl")),
                   key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    steps = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]

    per_step = []
    for f in files:
        per_step.append([json.loads(l) for l in open(f, encoding="utf-8")])
    n = min(len(rs) for rs in per_step) if per_step else 0
    if not n:
        continue

    probs = []
    for i in range(n):
        accs, lens, answers = [], [], {}
        for s, rows in zip(steps, per_step):
            r = rows[i]
            out = r.get("output", "") or ""
            accs.append(r.get("acc"))
            lens.append(len(out))
            answers[str(s)] = out
        first = per_step[0][i]
        stmt = (first.get("input", "") or "")
        # the render carries a chat wrapper; keep the question itself
        stmt = re.sub(r"^\s*(system|user|assistant)\s*\n", "", stmt).strip()
        gt = str(first.get("gts", ""))

        probs.append({"i": i, "acc": accs, "len": lens,
                      "q": stmt[:400], "gt": gt[:120]})
        with open(os.path.join(PROBDIR, f"{name}__{i}.json"), "w", encoding="utf-8") as fh:
            json.dump({"q": stmt, "gt": gt, "steps": steps, "acc": accs,
                       "answers": answers}, fh, separators=(",", ":"), ensure_ascii=False)

    flips = sum(1 for p in probs if len(set(a for a in p["acc"] if a is not None)) > 1)
    index["arms"][name] = {"steps": steps, "probs": probs}
    print(f"{name}: {n} problems x {len(steps)} steps, {flips} flipped")

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(index, fh, separators=(",", ":"), ensure_ascii=False)

sz = sum(os.path.getsize(os.path.join(PROBDIR, f)) for f in os.listdir(PROBDIR))
print("index:", OUT, os.path.getsize(OUT), "bytes")
print("per-problem:", PROBDIR, len(os.listdir(PROBDIR)), "files,", sz, "bytes")
