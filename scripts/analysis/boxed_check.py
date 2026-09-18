#!/usr/bin/env python3
"""Does the student still emit \\boxed{} -- and does the grader depend on it?

The 2026-09-12 trajectory read showed \\boxed{} appearing in ~3% of late-training
rollouts, which matters because the MATH grader parses exactly that.  If the
graded responses lost it too, every accuracy number downstream is suspect.  The
re-run evals carry the full response text, so this measures it directly instead
of inferring it from rollouts.

Per (arm, step): boxed rate, accuracy, truncation rate, and the accuracy split by
whether the response contains \\boxed{}.
"""
import collections
import glob
import os
import re
import sys

import pyarrow.parquet as pq

EV = sys.argv[1] if len(sys.argv) > 1 else "/share/rush/zz865/opd/evals"
BENCH = sys.argv[2] if len(sys.argv) > 2 else "math500"

rows = []
meta = collections.Counter()
for f in sorted(glob.glob(os.path.join(EV, f"*__{BENCH}__*.parquet"))):
    b = os.path.basename(f)
    m = re.match(r"^(.+?)__(.+?)__step(\d+)__", b)
    arm, _, step = m.group(1), m.group(2), int(m.group(3))
    t = pq.read_table(f, columns=["response", "correct", "truncated",
                                  "git_sha", "max_tokens", "stop_contract_source",
                                  "finish_reason", "resp_len"])
    resp = t.column("response").to_pylist()
    cor = t.column("correct").to_pylist()
    tru = t.column("truncated").to_pylist()
    meta[("git_sha", t.column("git_sha")[0].as_py())] += 1
    meta[("max_tokens", t.column("max_tokens")[0].as_py())] += 1
    meta[("stop_contract", t.column("stop_contract_source")[0].as_py())] += 1
    for fr in set(t.column("finish_reason").to_pylist()):
        meta[("finish_reason", fr)] += 1

    n = len(resp)
    boxed = [bool(r) and "\\boxed{" in r for r in resp]
    nb = sum(boxed)
    acc = sum(1 for c in cor if c) / n
    trunc = sum(1 for x in tru if x) / n
    accb = ([c for c, k in zip(cor, boxed) if k])
    accn = ([c for c, k in zip(cor, boxed) if not k])
    rows.append((arm, step, n, 100 * nb / n, 100 * acc, 100 * trunc,
                 100 * sum(1 for c in accb if c) / len(accb) if accb else float("nan"),
                 100 * sum(1 for c in accn if c) / len(accn) if accn else float("nan"),
                 sum(t.column("resp_len").to_pylist()) / n))

print(f"=== {BENCH}: {len(rows)} 格 ===\n")
print("%-28s %5s %6s %8s %7s %8s %9s %9s %8s" %
      ("arm", "step", "n", "boxed%", "acc%", "截断%", "有box acc", "无box acc", "均长"))
for r in sorted(rows, key=lambda x: (x[0], x[1])):
    print("%-28s %5d %6d %8.1f %7.1f %8.1f %9.1f %9.1f %8.0f" % r)

print("\n=== 这批评测的固定量 ===")
for (k, v), c in sorted(meta.items()):
    print(f"   {k:16} {str(v)[:60]:60} {c} 个文件")
