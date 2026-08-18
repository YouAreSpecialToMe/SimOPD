"""Battery for simopd.stop_set (dual-terminator contract wrapper). CPU-only.

Covers the classes that have actually bitten this repo: env-parse garbage
accepted silently, wrapper mutating the caller's dict, scoring-probe params
drifting from the recorded protocol, double-install double-wrap, composition
with the h_horizon-style inner wrapper, and module-level mutable state in a
closure that cloudpickle would copy by value (the gkd_mix zero-row incident).
Run: python3 scripts/stop_set_battery.py  -> prints N/N PASS, exit 0.
"""

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from simopd import stop_set  # noqa: E402

VERL_MOD = "verl.workers.rollout.vllm_rollout.vllm_async_server"
PASS = 0


def ok(name, cond):
    global PASS
    if not cond:
        print(f"FAIL: {name}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {name}")


def fresh_server():
    """Fabricate the verl module + class the installer looks up."""
    mod = types.ModuleType(VERL_MOD)

    class vLLMHttpServer:
        def __init__(self):
            self.seen = []

        async def generate(self, prompt_ids, sampling_params, request_id, *a, **kw):
            self.seen.append(sampling_params)
            return types.SimpleNamespace(token_ids=[1, 2, 3], sp=sampling_params)

    mod.vLLMHttpServer = vLLMHttpServer
    sys.modules[VERL_MOD] = mod
    return mod


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---- parse_ids -------------------------------------------------------------
ok("unset -> None", stop_set.parse_ids(None) is None)
ok("empty -> None", stop_set.parse_ids("") is None)
ok("off -> None (case-insensitive)", stop_set.parse_ids("OFF") is None)
ok("dual parses ordered", stop_set.parse_ids("151643,151645") == (151643, 151645))
ok("dupes deduped", stop_set.parse_ids("5,5,7") == (5, 7))
ok("spaces tolerated", stop_set.parse_ids(" 151643 , 151645 ") == (151643, 151645))
for bad in ("abc", "151643,x", "0", "-3", ",,,"):
    try:
        stop_set.parse_ids(bad)
        ok(f"refuses {bad!r}", False)
    except RuntimeError:
        ok(f"refuses {bad!r}", True)

# ---- no-op paths -----------------------------------------------------------
mod = fresh_server()
orig = mod.vLLMHttpServer.generate
os.environ.pop("SIMOPD_STOP_IDS", None)
stop_set.install()
ok("env unset -> class untouched", mod.vLLMHttpServer.generate is orig)
os.environ["SIMOPD_STOP_IDS"] = "off"
stop_set.install()
ok("env off -> class untouched", mod.vLLMHttpServer.generate is orig)

# ---- injection -------------------------------------------------------------
mod = fresh_server()
os.environ["SIMOPD_STOP_IDS"] = "151643,151645"
stop_set.install()
srv = mod.vLLMHttpServer()
caller_sp = {"max_tokens": 100}
out = run(srv.generate("p" * 4, caller_sp, "req1"))
ok("stop ids injected", out.sp["stop_token_ids"] == [151643, 151645])
ok("caller dict not mutated", "stop_token_ids" not in caller_sp)
ok("other keys preserved", out.sp["max_tokens"] == 100)

out = run(srv.generate("p" * 4, {"stop_token_ids": [999], "max_tokens": 5}, "req2"))
ok("existing ids merged ahead, ours appended", out.sp["stop_token_ids"] == [999, 151643, 151645])
out = run(srv.generate("p" * 4, {"stop_token_ids": [151645, 42]}, "req3"))
ok("overlap deduped, order stable", out.sp["stop_token_ids"] == [42, 151643, 151645])

probe = {"prompt_logprobs": 1, "max_tokens": 1}
out = run(srv.generate("p" * 4, probe, "req4"))
ok("scoring probe passes through untouched (same object)", out.sp is probe and "stop_token_ids" not in probe)

marker = object()
out = run(srv.generate("p" * 4, marker, "req5"))
ok("non-dict sampling_params passthrough", out.sp is marker)

wrapped_once = mod.vLLMHttpServer.generate
stop_set.install()
ok("double install is a no-op", mod.vLLMHttpServer.generate is wrapped_once)

# ---- composition with an h_horizon-style inner wrapper ---------------------
mod = fresh_server()
inner_orig = mod.vLLMHttpServer.generate

async def h_style(self, prompt_ids, sampling_params, request_id, *a, **kw):
    sp = dict(sampling_params) if isinstance(sampling_params, dict) else sampling_params
    if isinstance(sp, dict):
        sp["max_tokens"] = 64
    return await inner_orig(self, prompt_ids, sp, request_id, *a, **kw)

mod.vLLMHttpServer.generate = h_style
stop_set.install()
srv = mod.vLLMHttpServer()
out = run(srv.generate("p" * 4, {"max_tokens": 4096}, "req6"))
ok("composes with inner cap wrapper (both effects land)",
   out.sp["stop_token_ids"] == [151643, 151645] and out.sp["max_tokens"] == 64)

# ---- statelessness (cloudpickle by-value law) ------------------------------
mutables = [n for n, v in vars(stop_set).items()
            if not n.startswith("__") and isinstance(v, (dict, list, set))]
ok("module holds no mutable container state", mutables == [])
gen = mod.vLLMHttpServer.generate
names = set(gen.__code__.co_names)
ok("wrapper touches no module globals (closure-only)",
   names <= {"isinstance", "dict", "get", "list"})
ok("ids live in closure freevars", "ids" in gen.__code__.co_freevars and "fn" in gen.__code__.co_freevars)

os.environ.pop("SIMOPD_STOP_IDS", None)
sys.modules.pop(VERL_MOD, None)
print(f"{PASS}/{PASS} PASS")
