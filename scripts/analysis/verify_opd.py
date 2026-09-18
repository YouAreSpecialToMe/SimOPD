#!/usr/bin/env python3
"""Verify the unpacked opd tree against MANIFEST.tsv.

The manifest is (archive, path_in_archive, bytes) -- it describes the CONTENTS of
each runs/*.tar.zst, not the archives themselves, so the check has to run against
extracted/ rather than against the downloaded files.  The first pass got ok=0
because it looked for a size in column 2, which holds a path.
"""
import csv
import collections
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/share/rush/zz865/opd"
EXT = os.path.join(ROOT, "extracted")

ok = bad = miss = 0
per_arm_bad = collections.Counter()
examples = []
with open(os.path.join(ROOT, "MANIFEST.tsv"), encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    hdr = next(rd)
    assert hdr[:3] == ["archive", "path_in_archive", "bytes"], hdr
    for arch, rel, size in (r[:3] for r in rd if len(r) >= 3):
        want = int(size)
        f = os.path.join(EXT, rel)
        arm = rel.split("/")[0]
        if not os.path.exists(f):
            miss += 1
            per_arm_bad[arm] += 1
            if len(examples) < 6:
                examples.append("MISSING " + rel)
        elif os.path.getsize(f) == want:
            ok += 1
        else:
            bad += 1
            per_arm_bad[arm] += 1
            if len(examples) < 6:
                examples.append(f"SIZE {rel}: {os.path.getsize(f)} != {want}")

print(f"MANIFEST 共 {ok + bad + miss} 条")
print(f"  一致   {ok}")
print(f"  不符   {bad}")
print(f"  缺失   {miss}")
for e in examples:
    print("   ", e)
if per_arm_bad:
    print("\n有问题的臂:")
    for a, c in per_arm_bad.most_common(10):
        print(f"   {a:30} {c}")

# anything on disk the manifest does not mention
on_disk = set()
for dp, _, fs in os.walk(EXT):
    for f in fs:
        on_disk.add(os.path.relpath(os.path.join(dp, f), EXT))
listed = set()
with open(os.path.join(ROOT, "MANIFEST.tsv"), encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    next(rd)
    listed = {r[1] for r in rd if len(r) >= 3}
extra = on_disk - listed
print(f"\n磁盘上有 {len(on_disk)} 个文件, 清单列了 {len(listed)} 个, 清单外 {len(extra)} 个")
for e in sorted(extra)[:8]:
    print("   ", e)
