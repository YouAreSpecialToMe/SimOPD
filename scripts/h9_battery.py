#!/usr/bin/env python3
"""CPU battery for h9_prune_adapt: reliable-length math (pure reference),
h_budget relay round-trip and clamps, the h_horizon budget-mode clamp via the
stub harness, the two-knobs refusal, and (when torch is importable) the
vectorized controller against the pure reference. No GPU, no verl, no ray.
"""

import asyncio
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

for v in ("SIMOPD_GKD_CACHE", "SIMOPD_A5_TMAX_SCHEDULE", "SIMOPD_H_SCHEDULE"):
    os.environ.pop(v, None)
tmpdir = tempfile.mkdtemp()
os.environ["SIMOPD_H9_ADAPT"] = "1"
os.environ["SIMOPD_H_KEYS"] = os.path.join(tmpdir, "keys.parquet")
os.environ["SIMOPD_GKD_STATS"] = os.path.join(tmpdir, "stats.jsonl")
os.environ["SIMOPD_H_BUDGET"] = os.path.join(tmpdir, "budget.jsonl")

from simopd import h9_controller as h9  # noqa: E402
from simopd import h_budget  # noqa: E402

PASS = 0


def ok(cond, msg):
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1


# ------------------------------------------------------- reliable_len (pure) --
ok(h9.reliable_len([0.0] * 10, tau=-9.2, m=8) == 10, "no lost tokens -> full length")
row = [-10.0] * 8 + [0.0, 0.0]
ok(h9.reliable_len(row, tau=-9.2, m=8) == 8, "8 straight lost -> reliable len 8")
row = [0.0, -10.0] * 8            # lost at odd positions: 8th lost at index 15
ok(h9.reliable_len(row, tau=-9.2, m=8) == 16, "interleaved: m-th event position")
ok(h9.reliable_len([-9.2] * 20, tau=-9.2, m=8) == 20, "lp == tau is NOT lost (strict <)")
ok(h9.reliable_len([-9.3] * 3, tau=-9.2, m=8) == 3, "fewer than m events -> full length")

# --------------------------------------------------------- h_budget relay -----
ok(h_budget.budget() == 16384, "cold start budget == full window (adapt DOWN)")
h_budget.append({"budget": 300, "lq": 240.0, "calls": 1})
ok(h_budget.budget() == 300, "budget follows the newest row")
h_budget.append({"budget": 90000})
ok(h_budget.budget() == 16384, "over-window budget clamped to cap")
h_budget.append({"budget": "garbage"})
ok(h_budget.budget() == 16384, "garbage row falls back to default")
h_budget.append({"budget": 300})
ok(h_budget.budget() == 300, "relay recovers after garbage")

_p = os.environ.pop("SIMOPD_H_BUDGET")
_n = os.environ.pop("EXPERIMENT_NAME", None)
try:
    h_budget.path()
    ok(False, "path() accepted a shared default (lane-mixing class)")
except RuntimeError:
    ok(True, "")
os.environ["SIMOPD_H_BUDGET"] = _p
if _n is not None:
    os.environ["EXPERIMENT_NAME"] = _n

# ------------------------------------------- wrapper budget mode (stub) -------
class TokenOutput:
    def __init__(self, token_ids, stop_reason="completed"):
        self.token_ids = token_ids
        self.stop_reason = stop_reason


class vLLMHttpServer:
    pass


m = types.ModuleType("verl.workers.rollout.vllm_rollout.vllm_async_server")
m.vLLMHttpServer = vLLMHttpServer
sys.modules["verl.workers.rollout.vllm_rollout.vllm_async_server"] = m

SCRIPT = []
CALLS = []


async def _orig(self, prompt_ids, sampling_params, request_id, *a, **kw):
    CALLS.append((list(prompt_ids), dict(sampling_params) if isinstance(sampling_params, dict) else sampling_params, request_id))
    return SCRIPT.pop(0)(list(prompt_ids))


vLLMHttpServer.generate = _orig

from simopd import h_horizon as hh  # noqa: E402

# two-knobs refusal first (schedule + adapt together).
os.environ["SIMOPD_H_SCHEDULE"] = "mode=linear,start=128,end=16384,decay=250"
try:
    hh.install()
    ok(False, "install accepted schedule + adaptive budget together")
except RuntimeError:
    ok(True, "")
os.environ.pop("SIMOPD_H_SCHEDULE", None)

import pandas as pd  # noqa: E402

P1, PV = [21, 22, 23], [91, 92]
pd.DataFrame({"prefix_hash": [hh.prompt_key(P1)]}).to_parquet(os.environ["SIMOPD_H_KEYS"])
hh._keys = None

hh.install()
ok(getattr(vLLMHttpServer.generate, "_simopd_h_horizon", False), "wrapper armed in budget mode")

server = vLLMHttpServer()
server.config = types.SimpleNamespace(response_length=16384, max_model_len=None)
server.global_steps = 3


def run(prompt, sp=None, rid="req1"):
    return asyncio.run(vLLMHttpServer.generate(server, prompt, {} if sp is None else sp, rid))


# budget row 300 is the newest -> train prompts clamp to it.
SCRIPT.append(lambda p: TokenOutput(list(range(300))))
run(P1)
ok(CALLS[-1][1]["max_tokens"] == 300, "train prompt clamped to relayed budget 300")

# val stays exempt at full budget in budget mode too.
SCRIPT.append(lambda p: TokenOutput([1]))
run(PV)
ok("max_tokens" not in CALLS[-1][1], "val exempt in budget mode")

# budget moves -> next request follows.
h_budget.append({"budget": 512})
SCRIPT.append(lambda p: TokenOutput([1]))
run(P1)
ok(CALLS[-1][1]["max_tokens"] == 512, "clamp follows budget updates")

# ------------------------------------------ torch controller (if available) ---
try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False
    print("note: torch not importable here -- observe() covered by cluster battery run")

if HAVE_TORCH:
    h9._state["ema"] = None
    h9._state["calls"] = 0
    lp = torch.zeros(3, 20)
    lp[0, :8] = -10.0            # reliable len 8
    lp[1, :] = 0.0               # never lost -> len 20
    lp[2, 1::2] = -10.0          # interleaved -> 8th event at pos 16
    mask = torch.ones(3, 20, dtype=torch.bool)
    row = h9.observe(lp, mask)
    ok(row is not None and row["n"] == 3, "observe returns a row for 3 sequences")
    import numpy as np  # noqa: F401  (torch quantile is the reference here)
    L_ref = sorted([8, 20, 16])
    q_ref = float(torch.quantile(torch.tensor([8.0, 20.0, 16.0]), 0.9)) * 1.25
    ok(abs(row["lq"] - round(q_ref, 1)) < 0.11, f"lq matches reference quantile ({row['lq']} vs {q_ref:.1f})")
    ok(256 <= row["budget"] <= 16384, "budget within [floor, cap]")
    # masked-out rows are excluded
    mask2 = torch.zeros(1, 20, dtype=torch.bool)
    ok(h9.observe(torch.zeros(1, 20), mask2) is None, "fully-masked batch -> no row")
    # floor clamp: everything lost immediately
    h9._state["ema"] = None
    lp3 = torch.full((4, 300), -10.0)
    mask3 = torch.ones(4, 300, dtype=torch.bool)
    row3 = h9.observe(lp3, mask3)
    ok(row3["budget"] == 256, "hopeless batch clamps to FLOOR, never starves to 0")

print(f"h9 battery {PASS}/{PASS} pass")
