#!/usr/bin/env python3
"""CPU battery for a5_aggrevate. No GPU, no real verl, no ray.

Two layers. Pure math first (kappa draw, T_max ramp, flush arithmetic, install
refusals). Then a stub harness -- fake vLLMHttpServer + TokenOutput injected
into sys.modules, fake teacher handle -- drives the FULL wrapped generate()
through every branch: sentinel strip, membership miss, natural stop, the
EOS-exactly-at-kappa probe case (review 2026-08-15 #6), mixed stitch + scoring,
kappa=0 full-teacher, cap_full, teacher/scoring/prefix aborts, the outcome-sum
invariant, and miss-only-step flushing (review #7). The review demonstrated all
of these are CPU-reachable; now they are regression-locked.
"""

import asyncio
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

os.environ.pop("SIMOPD_GKD_CACHE", None)
os.environ["SIMOPD_A5_TMAX_SCHEDULE"] = "mode=linear,start=0,end=16384,decay=250"
os.environ["SIMOPD_A5_KEYS"] = "/placeholder-until-harness"

from simopd import a5_aggrevate as a5  # noqa: E402

PASS = 0


def ok(cond, msg):
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1


# ---------------------------------------------------------------- pure math --
ok(a5._tmax_at(0) == 0, "tmax(0) == 0 (iter-0 cold start)")
ok(a5._tmax_at(125) == 8192, "tmax midpoint")
ok(a5._tmax_at(250) == 16384 and a5._tmax_at(999) == 16384, "tmax cap")

ok(a5.kappa("k", 10, 0) == 0, "tmax=0 -> kappa=0")
draws = [a5.kappa(f"key{i}", 7, 100) for i in range(20000)]
ok(all(0 <= d <= 100 for d in draws), "kappa in U{0, tmax} inclusive range")
ok(min(draws) == 0 and max(draws) == 100, "kappa endpoints reached")
mean = sum(draws) / len(draws)
ok(abs(mean - 50.0) < 1.5, f"kappa mean {mean:.1f} ~ 50 (uniform)")
ok(a5.kappa("kx", 3, 500) == a5.kappa("kx", 3, 500), "kappa deterministic per (key, step)")
ok(len({a5.kappa("kx", s, 500) for s in range(50)}) > 30, "kappa fresh across steps")

# install refusals fire before any sys.modules lookup.
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

# ------------------------------------------------------------ stub universe --
class TokenOutput:
    def __init__(self, token_ids=None, log_probs=None, routed_experts=None,
                 stop_reason=None, extra_fields=None):
        self.token_ids = token_ids
        self.log_probs = log_probs
        self.routed_experts = routed_experts
        self.stop_reason = stop_reason
        self.extra_fields = extra_fields or {}


def _register(name, **attrs):
    m = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent:
        _register(parent, **{child: m})
    return m


_register("verl.workers.rollout.replica", TokenOutput=TokenOutput)


class vLLMHttpServer:
    pass


_register("verl.workers.rollout.vllm_rollout.vllm_async_server", vLLMHttpServer=vLLMHttpServer)

SCRIPT = []          # queued behaviors for the ORIGINAL generate
CALLS = []           # (prompt_ids, sampling_params) the original saw
TEACHER = {"behavior": None, "calls": []}


async def _orig(self, prompt_ids, sampling_params, request_id, *a, **kw):
    CALLS.append((list(prompt_ids), dict(sampling_params), request_id))
    return SCRIPT.pop(0)(list(prompt_ids), dict(sampling_params))


vLLMHttpServer.generate = _orig


class _TeacherHandle:
    class generate:
        @staticmethod
        def remote(request_id, prompt_ids, sampling_params):
            TEACHER["calls"].append((list(prompt_ids), dict(sampling_params), request_id))

            async def _run():
                return TEACHER["behavior"](list(prompt_ids), dict(sampling_params))

            return _run()


def scored_output(n_rows):
    return TokenOutput(token_ids=[0], stop_reason="completed",
                       extra_fields={"prompt_logprobs": [0.5] * n_rows, "prompt_ids": []})


tmpdir = tempfile.mkdtemp()
os.environ["SIMOPD_GKD_STATS"] = os.path.join(tmpdir, "stats.jsonl")
os.environ["SIMOPD_A5_KEYS"] = os.path.join(tmpdir, "keys.parquet")

import pandas as pd  # noqa: E402

P1, P2, PV = [11, 12, 13], [21, 22, 23, 24], [91, 92]   # PV stays un-keyed
pd.DataFrame({"prefix_hash": [a5.prompt_key(P1), a5.prompt_key(P2)]}).to_parquet(
    os.environ["SIMOPD_A5_KEYS"])
a5._keys = None
a5._sched = None
a5._handles = [_TeacherHandle]

a5.install()
ok(getattr(vLLMHttpServer.generate, "_simopd_a5", False), "wrapper armed on stub server")

server = vLLMHttpServer()
server.config = types.SimpleNamespace(response_length=100, max_model_len=None)
server.global_steps = 10

_real_kappa = a5.kappa


def run(prompt, sp=None, rid="req1"):
    return asyncio.run(vLLMHttpServer.generate(server, prompt, {} if sp is None else sp, rid))


def force_kappa(v):
    a5.kappa = lambda key, step, tmax: v


# ---------------------------------------------------------------- scenarios --
# sentinel strip: key-eligible prompt with the tail marker goes straight through
# with the marker removed and touches no counters.
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[7], stop_reason="completed"))
out = run(P1, {a5._TAIL_SENTINEL: 1, "temperature": 1.0})
ok(a5._TAIL_SENTINEL not in CALLS[-1][1] and CALLS[-1][1].get("temperature") == 1.0,
   "sentinel stripped, rest of params intact")
ok(a5._bucket["n_seen"] == 0, "sentinel path counts nothing")

# scoring passthrough.
SCRIPT.append(lambda p, sp: scored_output(5))
run(P1, {"prompt_logprobs": 0})
ok(a5._bucket["n_seen"] == 0, "scoring path counts nothing")

# membership miss (validation prompt).
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[1, 2], stop_reason="completed"))
run(PV)
ok(a5._bucket["miss"] == 1 and a5._bucket["n_seen"] == 0, "miss counted, not seen")

# natural stop strictly under budget: kappa=5, probe must ask for 6, got 3.
force_kappa(5)
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[1, 2, 3], stop_reason="completed"))
out = run(P1)
ok(CALLS[-1][1]["max_tokens"] == 6, "probe requests kappa+1 tokens")
ok(out.token_ids == [1, 2, 3] and a5._bucket["pure_student"] == 1, "natural stop delivered as-is")

# EOS exactly at kappa (review #6): got exactly 5 of the 6 allowed -> natural.
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[1, 2, 3, 4, 5], stop_reason="completed"))
out = run(P1)
ok(out.token_ids == [1, 2, 3, 4, 5] and a5._bucket["pure_student"] == 2
   and not TEACHER["calls"], "EOS-at-kappa is a natural stop, no teacher tail glued")

# mixed: probe fills kappa+1 -> cut to 5, teacher continues, scoring stitches.
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[1, 2, 3, 4, 5, 6], stop_reason="completed"))
TEACHER["behavior"] = lambda p, sp: TokenOutput(token_ids=[7, 8, 9], stop_reason="completed")
SCRIPT.append(lambda p, sp: scored_output(len(P1) + 8 + 1))
out = run(P1)
ok(out.token_ids == [1, 2, 3, 4, 5, 7, 8, 9], "stitched = 5-token prefix + teacher tail")
ok(len(out.log_probs) == 8, "student logprobs cover the whole stitched response")
tp, tsp, trid = TEACHER["calls"][-1]
ok(tp == P1 + [1, 2, 3, 4, 5] and tsp["max_tokens"] == 95 and trid == "req1",
   "teacher sees prompt+prefix, budget cap-kappa, and the ORIGINAL request id")
ok(a5._TAIL_SENTINEL in tsp, "teacher call carries the sentinel")
ok(CALLS[-1][1]["prompt_logprobs"] == 0 and CALLS[-1][0] == P1 + [1, 2, 3, 4, 5, 7, 8, 9],
   "scoring call covers prompt+stitched")
ok(CALLS[-2][2] == "req1" and CALLS[-1][2] == "req1-a5s",
   "probe keeps the original id, scoring is suffixed (NEW-ISSUE 3)")
ok(a5._bucket["mixed"] == 1 and a5._bucket["prefix_tokens"] >= 5
   and a5._bucket["tail_tokens"] == 3, "mixed outcome + token provenance")

# kappa=0 cold start: teacher writes everything, student only scores.
force_kappa(0)
TEACHER["behavior"] = lambda p, sp: TokenOutput(token_ids=[7, 7, 7, 7], stop_reason="completed")
SCRIPT.append(lambda p, sp: scored_output(len(P2) + 4 + 1))
out = run(P2)
ok(out.token_ids == [7, 7, 7, 7] and a5._bucket["full_teacher"] == 1,
   "kappa=0 -> teacher writes all, counted full_teacher")
ok(TEACHER["calls"][-1][0] == P2, "teacher generates from the bare prompt")

# cap_full: kappa >= cap keeps the whole window for the student.
force_kappa(150)
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=list(range(100)), stop_reason="completed"))
out = run(P1)
ok(CALLS[-1][1]["max_tokens"] == 100 and a5._bucket["cap_full"] == 1,
   "kappa clamped to cap: single student call, cap_full")

# teacher abort degrades to the delivered student prefix.
force_kappa(5)
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[1, 2, 3, 4, 5, 6], stop_reason="completed"))
TEACHER["behavior"] = lambda p, sp: TokenOutput(token_ids=[], stop_reason="aborted")
out = run(P1)
ok(out.token_ids == [1, 2, 3, 4, 5, 6] and a5._bucket["degraded"] == 1,
   "teacher abort -> deliver student prefix, counted degraded")

# scoring abort likewise degrades.
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[1, 2, 3, 4, 5, 6], stop_reason="completed"))
TEACHER["behavior"] = lambda p, sp: TokenOutput(token_ids=[9], stop_reason="completed")
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[0], stop_reason="aborted"))
out = run(P1)
ok(out.token_ids == [1, 2, 3, 4, 5, 6] and a5._bucket["degraded"] == 2,
   "scoring abort -> deliver student prefix, counted degraded")

# prefix abort is counted (review #10), not silent.
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[], stop_reason="aborted"))
out = run(P1)
ok(out.stop_reason == "aborted" and a5._bucket["aborted"] == 1, "prefix abort counted")


# teacher RAISING (over-window / transport) degrades instead of killing the
# request (verification NEW-ISSUE 2).
def _boom(p, sp):
    raise ValueError("prompt > teacher max_model_len")


force_kappa(5)
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[1, 2, 3, 4, 5, 6], stop_reason="completed"))
TEACHER["behavior"] = _boom
out = run(P1)
ok(out.token_ids == [1, 2, 3, 4, 5, 6] and a5._bucket["degraded"] == 3,
   "teacher exception -> deliver student prefix, counted degraded")

# ... and with no prefix (kappa=0): a COUNTED student-fallback generation --
# an empty/aborted output would crash verl's as_dict (rm_scores[-1] on size 0,
# measured 2026-08-18, 110x, killed the first a5 rehearsal batch).
force_kappa(0)
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[42, 43], stop_reason="completed"))
out = run(P1)
ok(out.token_ids == [42, 43] and a5._bucket["degraded"] == 4,
   "teacher exception at kappa=0 -> counted student generation, never empty")

# outcome-sum invariant: every eligible request lands in exactly one bucket.
b = a5._bucket
outcomes = b["mixed"] + b["pure_student"] + b["full_teacher"] + b["cap_full"] + b["degraded"] + b["aborted"]
ok(outcomes == b["n_seen"] == 10, f"outcome sum {outcomes} == n_seen {b['n_seen']}")

# miss-only step still flushes (review #7), and rows carry derived fields + pid.
server.global_steps = 11
SCRIPT.append(lambda p, sp: TokenOutput(token_ids=[1], stop_reason="completed"))
run(PV)
server.global_steps = 12
force_kappa(0)
TEACHER["behavior"] = lambda p, sp: TokenOutput(token_ids=[7], stop_reason="completed")
SCRIPT.append(lambda p, sp: scored_output(len(P1) + 1 + 1))
run(P1)   # first request of step 12 flushes step 11
rows = [json.loads(x) for x in open(os.environ["SIMOPD_GKD_STATS"])]
ok(rows[-2]["step"] == 10 and rows[-1]["step"] == 11, "step buckets flushed in order")
ok(rows[-1]["miss"] == 1 and rows[-1]["n_seen"] == 0, "miss-only step 11 flushed (not dropped)")
ok(abs(rows[-2]["tail_token_frac"] - (rows[-2]["tail_tokens"] /
       (rows[-2]["tail_tokens"] + rows[-2]["prefix_tokens"]))) < 1e-12
   and "pid" in rows[-1], "derived fields + writer pid present")

a5.kappa = _real_kappa

# Structural guard (2026-08-18 zero-row sideband incident; twin of the gkd_mix
# battery's): the closure is cloudpickled BY VALUE into the actor, bare-dict
# globals become private copies -- all state mutations must route through
# module-level functions (pickled by reference). Recurses into nested code
# objects, so _student_fallback is covered too.
def _code_names(code):
    names = set(code.co_names)
    for c in code.co_consts:
        if hasattr(c, "co_names"):
            names |= _code_names(c)
    return names


_gen = next(c for c in a5.install.__code__.co_consts
            if getattr(c, "co_name", "") == "generate")
_bad = {"_bucket", "_stats", "_flush_state"} & _code_names(_gen)
ok(not _bad, f"a5 closure touches bare state globals directly: {_bad}")

print(f"a5 battery {PASS}/{PASS} pass")
