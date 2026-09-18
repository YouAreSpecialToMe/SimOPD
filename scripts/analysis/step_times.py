#!/usr/bin/env python3
"""How much machine time do the unfinished arms still need?

Reads perf/time_per_step out of each arm's metrics JSONL rather than estimating:
step time grows with response length over a run, so the late-window median is the
number that applies to the steps still to come -- and it makes the result a floor,
not a point estimate.

Lane width is 2 by default (one student GPU + one teacher GPU), which is what the
checkpoints were trained under: their FSDP shards are world_size_1.
"""
import glob
import json
import os
import statistics as st
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/zz865/opd_pack/extracted"
LANE = int(sys.argv[2]) if len(sys.argv) > 2 else 2
LATE = 20                      # steps in the late window

# arm -> steps still owed, from the handover README section A
REMAINING = {
    "a4_dagger_anneal_n0_s0": 25,
    "g1_verified_only_corr_s0": 25,
    "c5_union_fkl_s0": 50,
    "h3_random_segment_corr_s0": 50,
    "h4_random_scatter_corr_s0": 50,
    "e2_set_coverage_a0_corr_s0": 75,
    "g6_seqmean_corr_s0": 75,
    "a5_aggrevate_n0_s0": 100,
    "c5_union_rkl_s0": 100,
    "h2_last_segment_corr_s0": 100,
}

print("%-30s %6s %10s %10s %10s" % ("arm", "还差", "中位步时s", "末窗中位s", "GPU·h"))
total = owed = 0
for arm, need in sorted(REMAINING.items(), key=lambda kv: -kv[1]):
    ts = []
    for f in glob.glob(os.path.join(ROOT, arm, "metrics", "*.jsonl")):
        for line in open(f):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            v = d.get("data", {}).get("perf/time_per_step")
            if v:
                ts.append((d.get("step", 0), v))
    if not ts:
        print("%-30s %6d  <无 metrics>" % (arm, need))
        continue
    ts.sort()
    med = st.median([v for _, v in ts])
    late = st.median([v for _, v in ts[-LATE:]])
    gh = need * late * LANE / 3600
    total += gh
    owed += need
    print("%-30s %6d %10.1f %10.1f %10.1f" % (arm, need, med, late, gh))

print("\n合计 %d 步,%.0f GPU·h(末窗步时 × %d 卡/lane)" % (owed, total, LANE))
print("步时随长度上涨,所以这是下界 —— 预算按 %.0f–%.0f 更稳。"
      % (total * 1.1, total * 1.25))
