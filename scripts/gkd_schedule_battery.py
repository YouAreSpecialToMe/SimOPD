#!/usr/bin/env python3
"""CPU battery for a4_dagger_anneal's schedule + telemetry machinery. No GPU, no verl.

Covers: gkd_schedule math (all four modes, warmup, the registered a4 shapes and
their mean doses), parse refusals (every malformation must be LOUD), the
gkd_stats relay (incremental tail, torn-line resilience), and the end-to-end
dose check -- gkd_mix.coin() realised frequency tracks lambda(step) from the
schedule, which is the property the whole arm rests on.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from simopd import gkd_schedule, gkd_stats  # noqa: E402

PASS = 0


def ok(cond, msg):
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1


def mean_dose(s, horizon=250):
    return sum(s.value_at(t) for t in range(horizon)) / horizon


# 1. a4 primary: full-window linear. Endpoints, midpoint, and the matched-dose
#    claim against a1's constant 0.5 (exact value over steps 0..249 is 0.502).
s = gkd_schedule.parse("mode=linear,start=1.0,end=0.0,warmup=0,decay=250")
ok(s.value_at(0) == 1.0 and s.value_at(250) == 0.0, "linear250 endpoints")
ok(abs(s.value_at(125) - 0.5) < 1e-12, "linear250 midpoint")
ok(abs(mean_dose(s) - 0.502) < 1e-12, "linear250 mean dose == 0.502 (a1-matched)")

# 2. a4 adjudicator: half-window (the published agentic recipe, 0.2/round x 10
#    iters, in normalized time). Anneal done at 125, second half pure on-policy.
h = gkd_schedule.parse("mode=linear,start=1.0,end=0.0,decay=125")
ok(h.value_at(125) == 0.0 and h.value_at(200) == 0.0, "linear125 consolidation half")
ok(abs(mean_dose(h) - 0.252) < 1e-12, "linear125 mean dose == 0.252")

# 3. warmup holds start; decay begins after it.
w = gkd_schedule.parse("mode=linear,start=1.0,end=0.0,warmup=50,decay=200")
ok(w.value_at(49) == 1.0 and w.value_at(50) == 1.0, "warmup holds start")
ok(abs(w.value_at(150) - 0.5) < 1e-12 and w.value_at(250) == 0.0, "warmup shifts the ramp")

# 4. cosine: endpoints, midpoint = (start+end)/2, monotone non-increasing.
c = gkd_schedule.parse("mode=cosine,start=1.0,end=0.0,decay=250")
ok(c.value_at(0) == 1.0 and c.value_at(250) == 0.0, "cosine endpoints")
ok(abs(c.value_at(125) - 0.5) < 1e-12, "cosine midpoint")
vals = [c.value_at(t) for t in range(251)]
ok(all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])), "cosine monotone")

# 5. exponential: STACX 1e-8 clamp quirk preserved -- inside the window it decays
#    toward ~1e-8, and lands exactly on end when the window closes.
e = gkd_schedule.parse("mode=exponential,start=1.0,end=0.0,decay=250")
ok(e.value_at(0) == 1.0 and e.value_at(250) == 0.0, "exponential endpoints")
ok(0.0 < e.value_at(249) < 1e-7, "exponential 1e-8 clamp quirk (documented)")

# 6. step mode flips at the decay-window midpoint (a2-analog inside the budget).
st = gkd_schedule.parse("mode=step,start=1.0,end=0.0,decay=250")
ok(st.value_at(124) == 1.0 and st.value_at(125) == 0.0, "step flips at midpoint")

# 7. ascending ramp (a5's T_max reuses this evaluator).
r = gkd_schedule.parse("mode=linear,start=0,end=16384,decay=250")
ok(abs(r.value_at(125) - 8192.0) < 1e-9 and r.value_at(250) == 16384.0, "ascending ramp")

# 8. parse refusals: every malformation is loud.
for bad in ("mode=linear,start=1.0,end=0.0",            # missing decay
            "mode=warp,start=1,end=0,decay=250",         # unknown mode
            "mode=linear,start=1,end=0,decay=250,k=1",   # unknown key
            "start=1,start=0.5,end=0,decay=250",         # duplicate key
            "start=one,end=0,decay=250",                 # non-numeric
            "start=1,end=0,decay=0",                     # decay < 1
            "start=1 end=0 decay=250"):                  # not k=v
    try:
        gkd_schedule.parse(bad)
        ok(False, f"parse accepted malformed spec {bad!r}")
    except ValueError:
        ok(True, "")

# 9. gkd_stats relay: incremental tail + torn-line resilience.
with tempfile.TemporaryDirectory() as d:
    os.environ["SIMOPD_GKD_STATS"] = os.path.join(d, "s.jsonl")
    gkd_stats._read.update(offset=0, row=None)
    ok(gkd_stats.latest() is None, "no file -> None")
    gkd_stats.append({"step": 1, "hit": 3})
    gkd_stats.append({"step": 2, "hit": 5})
    ok(gkd_stats.latest()["step"] == 2, "latest returns newest complete row")
    with open(os.environ["SIMOPD_GKD_STATS"], "ab") as f:
        f.write(b'{"step": 3, "hi')                      # torn line
    ok(gkd_stats.latest()["step"] == 2, "torn line keeps previous row")
    with open(os.environ["SIMOPD_GKD_STATS"], "ab") as f:
        f.write(b't": 7}\n')
    ok(gkd_stats.latest()["step"] == 3, "completed line picked up incrementally")
    del os.environ["SIMOPD_GKD_STATS"]

# 10. end-to-end dose: coin() realised frequency tracks lambda(step) from the a4
#     primary schedule, and the coin stays deterministic per (key, step).
from simopd import gkd_mix  # noqa: E402  (pure imports only on this path)

keys = [gkd_mix.prompt_key([i, i + 1, i + 2]) for i in range(20000)]
for step, lo, hi in ((0, 0.995, 1.0), (125, 0.49, 0.51), (200, 0.19, 0.21), (250, 0.0, 0.005)):
    lam = s.value_at(step)
    freq = sum(gkd_mix.coin(k, step, lam) for k in keys) / len(keys)
    ok(lo <= freq <= hi, f"coin freq {freq:.4f} tracks lambda {lam:.2f} at step {step}")
ok(gkd_mix.coin(keys[0], 42, 0.5) == gkd_mix.coin(keys[0], 42, 0.5), "coin deterministic")

print(f"gkd_schedule battery {PASS}/{PASS} pass")
