#!/usr/bin/env python3
"""Pull the handover checkpoints onto /share/rush, then verify against MANIFEST.tsv.

Jerrycool/opd-ckpt is 865 GiB across 668 files: resume/ (10 arms that still need
training, with optimiser state) and eval/ (93 inference-only weights for the 99
outstanding eval cells).  /home/zz865 has only 1.5T free, so this lands on
/share/rush, which the September cleanup took back to 18T.

Resumable: snapshot_download skips files already present with the right size.
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = "Jerrycool/opd-ckpt"
DEST = sys.argv[1] if len(sys.argv) > 1 else "/share/rush/zz865/opd-ckpt"

os.makedirs(DEST, exist_ok=True)
print(f"repo = {REPO}\ndest = {DEST}", flush=True)

p = snapshot_download(
    repo_id=REPO,
    repo_type="model",
    local_dir=DEST,
    max_workers=4,
    resume_download=True,
)
print("downloaded to", p, flush=True)

# verify every file against the manifest the upstream shipped
man = os.path.join(DEST, "MANIFEST.tsv")
if not os.path.exists(man):
    print("no MANIFEST.tsv; skipping verification")
    sys.exit(0)

import csv

ok = bad = miss = 0
rows = list(csv.reader(open(man, encoding="utf-8"), delimiter="\t"))
hdr = rows[0] if rows and not rows[0][0].startswith(("resume/", "eval/")) else None
for r in rows[1:] if hdr else rows:
    if len(r) < 2:
        continue
    rel, size = r[0], r[1]
    try:
        want = int(size)
    except ValueError:
        continue
    f = os.path.join(DEST, rel)
    if not os.path.exists(f):
        miss += 1
        if miss <= 5:
            print("MISSING", rel)
        continue
    got = os.path.getsize(f)
    if got == want:
        ok += 1
    else:
        bad += 1
        if bad <= 5:
            print(f"SIZE MISMATCH {rel}: {got} != {want}")
print(f"\nmanifest check: ok={ok} mismatched={bad} missing={miss}")
