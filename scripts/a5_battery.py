#!/usr/bin/env python3
"""CPU battery for a5_aggrevate's pure machinery. No GPU, no verl, no ray.

Covers the kappa draw (uniformity, determinism, endpoint coverage), the T_max
ramp, the telemetry bucket's derived fields, and install()'s refusal paths --
everything the 3-step GPU rehearsal does NOT need to re-prove. The wrapper's
routing (sentinel strip, membership gate, stitch + scoring) is rehearsal
territory by design: it needs live engines.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

os.environ.pop("SIMOPD_GKD_CACHE", None)
os.environ["SIMOPD_A5_TMAX_SCHEDULE"] = "mode=linear,start=0,end=16384,decay=250"
os.environ["SIMOPD_A5_KEYS"] = "/nonexistent-until-checked"

from simopd import a5_aggrevate as a5  # noqa: E402

PASS = 0


def ok(cond, msg):
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1


# 1. T_max ramp: integers, endpoints, midpoint; step 0 is the paper's pure-teacher
#    cold start (T_max=0 => kappa=0 => teacher writes everything).
ok(a5._tmax_at(0) == 0, "tmax(0) == 0 (iter-0 cold start)")
ok(a5._tmax_at(125) == 8192, "tmax midpoint")
ok(a5._tmax_at(250) == 16384 and a5._tmax_at(999) == 16384, "tmax cap")

# 2. kappa: deterministic, in range, uniform-ish, fresh across steps.
ok(a5.kappa("k", 10, 0) == 0, "tmax=0 -> kappa=0")
draws = [a5.kappa(f"key{i}", 7, 100) for i in range(20000)]
ok(all(0 <= d <= 100 for d in draws), "kappa in U{0, tmax} inclusive range")
ok(min(draws) == 0 and max(draws) == 100, "kappa endpoints reached")
mean = sum(draws) / len(draws)
ok(abs(mean - 50.0) < 1.5, f"kappa mean {mean:.1f} ~ 50 (uniform)")
ok(a5.kappa("kx", 3, 500) == a5.kappa("kx", 3, 500), "kappa deterministic per (key, step)")
ok(len({a5.kappa("kx", s, 500) for s in range(50)}) > 30, "kappa fresh across steps")

# 3. bucket flush: derived kappa_mean and tail_token_frac (the honest dose).
with tempfile.TemporaryDirectory() as d:
    os.environ["SIMOPD_GKD_STATS"] = os.path.join(d, "s.jsonl")
    a5._roll_bucket(1, 40)
    a5._bucket.update(n_seen=4, kappa_sum=80, mixed=3, pure_student=1,
                      prefix_tokens=300, tail_tokens=700)
    a5._roll_bucket(2, 44)   # rollover flushes step 1
    row = json.loads(open(os.environ["SIMOPD_GKD_STATS"]).read().strip())
    ok(row["step"] == 1 and row["tmax"] == 40, "flushed row carries step + tmax")
    ok(abs(row["kappa_mean"] - 20.0) < 1e-12, "kappa_mean derived at flush")
    ok(abs(row["tail_token_frac"] - 0.7) < 1e-12, "tail_token_frac = teacher share of delivered mix")
    del os.environ["SIMOPD_GKD_STATS"]

# 4. install refusals fire BEFORE any verl machinery is touched.
os.environ["SIMOPD_GKD_CACHE"] = "/tmp/x.parquet"
try:
    a5.install()
    ok(False, "install accepted a5 + gkd cache together")
except RuntimeError:
    ok(True, "")
del os.environ["SIMOPD_GKD_CACHE"]

os.environ["SIMOPD_A5_KEYS"] = ""
try:
    a5.install()
    ok(False, "install accepted a5 without a membership key set")
except RuntimeError:
    ok(True, "")

# 5. a5 disarmed (schedule unset) -> install is a no-op even with bad siblings.
os.environ["SIMOPD_A5_TMAX_SCHEDULE"] = ""
a5.install()
ok(True, "disarmed install no-ops")

print(f"a5 battery {PASS}/{PASS} pass")
