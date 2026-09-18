#!/usr/bin/env python3
"""Mirror Jerrycool/opd onto the cluster, verify it, and unpack the per-arm runs.

The 2026-09-12 pass took a zip of this dataset to /home/zz865/opd_pack.  Since then
the repo grew an evals/ tree -- 155 parquet files, all stamped 20260917 -- so this
re-pulls the whole repo rather than diffing by hand, then reports what is new
relative to the September tree.

runs/<arm>.tar.zst unpack into extracted/<arm>/; evals/*.parquet are already in
their final form and stay where they land.
"""
import csv
import os
import subprocess
import sys

from huggingface_hub import snapshot_download

REPO = "Jerrycool/opd"
DEST = sys.argv[1] if len(sys.argv) > 1 else "/share/rush/zz865/opd"
OLD = "/home/zz865/opd_pack/extracted"      # the September tree, for the diff

os.makedirs(DEST, exist_ok=True)
print(f"repo = {REPO}\ndest = {DEST}", flush=True)

snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=DEST,
                  max_workers=4, resume_download=True)
print("snapshot done", flush=True)

# --- verify against the manifest the repo ships -----------------------------
man = os.path.join(DEST, "MANIFEST.tsv")
ok = bad = miss = 0
if os.path.exists(man):
    rows = list(csv.reader(open(man, encoding="utf-8"), delimiter="\t"))
    for r in rows:
        if len(r) < 2 or not r[1].strip().isdigit():
            continue                      # header or a non-size column
        f = os.path.join(DEST, r[0])
        if not os.path.exists(f):
            miss += 1
            if miss <= 5:
                print("MISSING", r[0])
        elif os.path.getsize(f) == int(r[1]):
            ok += 1
        else:
            bad += 1
            if bad <= 5:
                print(f"SIZE MISMATCH {r[0]}: {os.path.getsize(f)} != {r[1]}")
    print(f"manifest: ok={ok} mismatched={bad} missing={miss}", flush=True)
else:
    print("no MANIFEST.tsv in the repo copy", flush=True)

# --- unpack the per-arm archives -------------------------------------------
runs = os.path.join(DEST, "runs")
out = os.path.join(DEST, "extracted")
os.makedirs(out, exist_ok=True)
if os.path.isdir(runs):
    tars = sorted(f for f in os.listdir(runs) if f.endswith(".tar.zst"))
    for i, t in enumerate(tars, 1):
        arm = t[: -len(".tar.zst")]
        if os.path.isdir(os.path.join(out, arm)):
            print(f"  [{i}/{len(tars)}] {arm} already unpacked", flush=True)
            continue
        print(f"  [{i}/{len(tars)}] unpacking {arm}", flush=True)
        # -I zstd rather than --zstd: tar on this box predates the built-in flag
        subprocess.run(["tar", "-I", "zstd", "-xf", os.path.join(runs, t), "-C", out],
                       check=True)

# --- what is new since the September tree ----------------------------------
new_arms = old_arms = set()
if os.path.isdir(OLD):
    old_arms = {d for d in os.listdir(OLD) if os.path.isdir(os.path.join(OLD, d))}
now_arms = {d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))}
print(f"\n臂: 9/12 树 {len(old_arms)} 条, 这次 {len(now_arms)} 条")
if now_arms - old_arms:
    print("  仅在新包里:", ", ".join(sorted(now_arms - old_arms)))
if old_arms - now_arms:
    print("  仅在旧树里:", ", ".join(sorted(old_arms - now_arms)))

ev = os.path.join(DEST, "evals")
if os.path.isdir(ev):
    import collections
    fs = os.listdir(ev)
    per_arm = collections.Counter(f.split("__")[0] for f in fs)
    per_bench = collections.Counter(f.split("__")[1] for f in fs if "__" in f)
    print(f"\nevals: {len(fs)} 个 parquet")
    print("  按 bench:", dict(per_bench.most_common()))
    print(f"  覆盖 {len(per_arm)} 条臂")
print("\nDONE", flush=True)
