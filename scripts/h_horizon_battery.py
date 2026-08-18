#!/usr/bin/env python3
"""CPU battery for h6_gen_sched (h_horizon). No GPU, no real verl, no ray.

Layers: schedule math, install refusals (mutual exclusion with A-arm knobs,
missing membership keys, out-of-bounds horizons), then a stub-harness pass over
the full wrapped generate(): scoring passthrough, val-miss full-budget
exemption (the h5-confound fix), clamp math incl. caller/config caps, cap-hit
accounting, step-rollover flush, converged-to-vanilla at ramp end, and the
cloudpickle co_names structural guard (closure state only via module fns).
"""

import asyncio
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

os.environ.pop("SIMOPD_GKD_CACHE", None)
os.environ.pop("SIMOPD_A5_TMAX_SCHEDULE", None)
CANON = "mode=linear,start=128,end=16384,warmup=0,decay=250"
os.environ["SIMOPD_H_SCHEDULE"] = CANON
os.environ["SIMOPD_H_KEYS"] = "/placeholder-until-harness"

from simopd import h_horizon as hh  # noqa: E402

PASS = 0


def ok(cond, msg):
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1


# ---------------------------------------------------------------- pure math --
ok(hh._h_at(0) == 128, "H(0) == 128")
ok(hh._h_at(125) == 8256, "H(125) == 8256 (linear midpoint)")
ok(hh._h_at(250) == 16384 and hh._h_at(999) == 16384, "H caps at 16384")

# ------------------------------------------------------------ install gates --
os.environ["SIMOPD_GKD_CACHE"] = "/tmp/x.parquet"
try:
    hh.install()
    ok(False, "install accepted h-arm + gkd cache together")
except RuntimeError:
    ok(True, "")
os.environ.pop("SIMOPD_GKD_CACHE", None)

os.environ["SIMOPD_A5_TMAX_SCHEDULE"] = "mode=linear,start=0,end=1,decay=1"
try:
    hh.install()
    ok(False, "install accepted h-arm + a5 together")
except RuntimeError:
    ok(True, "")
os.environ.pop("SIMOPD_A5_TMAX_SCHEDULE", None)

os.environ.pop("SIMOPD_H_KEYS", None)
try:
    hh.install()
    ok(False, "install accepted missing membership keys (val would be clamped)")
except RuntimeError:
    ok(True, "")
os.environ["SIMOPD_H_KEYS"] = "/placeholder-until-harness"

os.environ["SIMOPD_H_SCHEDULE"] = "mode=linear,start=0,end=16384,decay=250"
try:
    hh.install()
    ok(False, "install accepted H=0 endpoint (empty-rollout crash class)")
except RuntimeError:
    ok(True, "")
os.environ["SIMOPD_H_SCHEDULE"] = "mode=linear,start=128,end=20000,decay=250"
try:
    hh.install()
    ok(False, "install accepted horizon above protocol window")
except RuntimeError:
    ok(True, "")
os.environ["SIMOPD_H_SCHEDULE"] = CANON

_saved = os.environ.pop("SIMOPD_H_SCHEDULE")
hh.install()   # knob absent -> silent no-op, no verl needed
ok(True, "clean no-op without the knob (zero impact on other arms)")
os.environ["SIMOPD_H_SCHEDULE"] = _saved

# ------------------------------------------------------------ stub harness --
class TokenOutput:
    def __init__(self, token_ids, stop_reason="completed"):
        self.token_ids = token_ids
        self.stop_reason = stop_reason


class vLLMHttpServer:
    pass


def _register(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


_register("verl.workers.rollout.vllm_rollout.vllm_async_server", vLLMHttpServer=vLLMHttpServer)

SCRIPT = []
CALLS = []


async def _orig(self, prompt_ids, sampling_params, request_id, *a, **kw):
    CALLS.append((list(prompt_ids), dict(sampling_params) if isinstance(sampling_params, dict) else sampling_params, request_id))
    return SCRIPT.pop(0)(list(prompt_ids))


vLLMHttpServer.generate = _orig

tmpdir = tempfile.mkdtemp()
os.environ["SIMOPD_GKD_STATS"] = os.path.join(tmpdir, "stats.jsonl")
os.environ["SIMOPD_H_KEYS"] = os.path.join(tmpdir, "keys.parquet")

import pandas as pd  # noqa: E402

P1, PV = [11, 12, 13], [91, 92]   # P1 keyed (training), PV unkeyed (val)
pd.DataFrame({"prefix_hash": [hh.prompt_key(P1)]}).to_parquet(os.environ["SIMOPD_H_KEYS"])
hh._keys = None
hh._sched = None

hh.install()
ok(getattr(vLLMHttpServer.generate, "_simopd_h_horizon", False), "wrapper armed on stub server")

server = vLLMHttpServer()
server.config = types.SimpleNamespace(response_length=16384, max_model_len=None)
server.global_steps = 0


def run(prompt, sp=None, rid="req1"):
    return asyncio.run(vLLMHttpServer.generate(server, prompt, {} if sp is None else sp, rid))


# scoring passthrough: params untouched, nothing counted in the bucket.
SCRIPT.append(lambda p: TokenOutput([1]))
run(P1, {"prompt_logprobs": 0})
ok("max_tokens" not in CALLS[-1][1], "scoring passthrough leaves params untouched")
ok(hh._bucket["n_train"] == 0, "scoring path counts nothing")

# val miss: FULL budget, params untouched -- the h5-confound fix.
SCRIPT.append(lambda p: TokenOutput([1, 2, 3]))
run(PV)
ok("max_tokens" not in CALLS[-1][1], "val/unseen prompt generates at full budget")
ok(hh._bucket["n_miss"] == 1 and hh._bucket["n_train"] == 0, "miss counted, not train")

# training prompt at step 0: clamp to H(0)=128, clamped flag on.
SCRIPT.append(lambda p: TokenOutput(list(range(128))))
run(P1)
ok(CALLS[-1][1]["max_tokens"] == 128, "train prompt clamped to H(0)=128")
ok(hh._bucket["n_train"] == 1 and hh._bucket["clamped_n"] == 1, "train + clamped counted")
ok(hh._bucket["cap_hits"] == 1 and hh._bucket["gen_tokens"] == 128, "cap hit + tokens recorded")

# natural stop below H: no cap hit.
SCRIPT.append(lambda p: TokenOutput(list(range(50))))
run(P1)
ok(hh._bucket["cap_hits"] == 1 and hh._bucket["gen_tokens"] == 178, "natural stop is not a cap hit")

# caller max_tokens combines by min.
SCRIPT.append(lambda p: TokenOutput([1]))
run(P1, {"max_tokens": 64})
ok(CALLS[-1][1]["max_tokens"] == 64, "caller max_tokens wins when tighter")

# engine config cap combines by min.
server.config = types.SimpleNamespace(response_length=100, max_model_len=None)
SCRIPT.append(lambda p: TokenOutput([1]))
run(P1)
ok(CALLS[-1][1]["max_tokens"] == 100, "engine response_length cap wins when tighter")
server.config = types.SimpleNamespace(response_length=16384, max_model_len=None)

# step rollover flushes the finished bucket to the sideband.
server.global_steps = 1
SCRIPT.append(lambda p: TokenOutput([1]))
run(P1)
rows = [json.loads(x) for x in open(os.environ["SIMOPD_GKD_STATS"])]
r0 = [r for r in rows if r.get("step") == 0][-1]
ok(r0["n_train"] == 4 and r0["n_miss"] == 1 and r0["h_target"] == 128 and "pid" in r0,
   "step-0 bucket flushed with counts + pid")
ok(hh._bucket["step"] == 1 and hh._bucket["h_target"] == 193,
   "fresh bucket at step 1 with H(1)=193")

# ramp end: eff == cap -> unclamped, converged to vanilla.
server.global_steps = 250
SCRIPT.append(lambda p: TokenOutput([1]))
run(P1)
ok(CALLS[-1][1]["max_tokens"] == 16384, "H(250) == full window")
ok(hh._bucket["clamped_n"] == 0, "ramp end is unclamped (vanilla-converged)")

# Structural guard (cloudpickle by-value seam, 2026-08-18 zero-row incident):
# the closure must touch state ONLY via module functions.
def _code_names(code):
    names = set(code.co_names)
    for c in code.co_consts:
        if hasattr(c, "co_names"):
            names |= _code_names(c)
    return names


_gen = next(c for c in hh.install.__code__.co_consts
            if getattr(c, "co_name", "") == "generate")
_bad = {"_bucket", "_stats", "_flush_state"} & _code_names(_gen)
ok(not _bad, f"closure touches bare state globals directly: {_bad}")

print(f"h_horizon battery {PASS}/{PASS} pass")
