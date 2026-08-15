"""a5_aggrevate: single-switch roll-in -- student writes tokens 0..kappa-1, the
teacher completes ONLINE from kappa. kappa ~ U{0, T_max(step)}, T_max on a
gkd_schedule ramp. arXiv 2605.12913 Eq.9 (STACX aggrevate_pure), single-turn image.

Seam: the same vLLMHttpServer.generate wrap as gkd_mix, plus one new route -- the
wrapper calls a TEACHER server actor (resolved through teacher_registry) for the
continuation, token ids in / TokenOutput out, so the handoff is token-exact with
no re-tokenization boundary. Guards, in order:

  scoring passthrough   prompt_logprobs requests untouched (gkd_mix's guard).
  tail sentinel         sampling_params carrying _TAIL_SENTINEL are OUR OWN
                        continuation calls arriving at the TEACHER server, whose
                        wrapper (sitecustomize installs everywhere) must strip the
                        sentinel and generate normally. Without this, a kappa=0
                        tail call -- whose prompt is a bare training prompt, hence
                        key-eligible -- would recurse into the teacher's own
                        wrapper forever.
  membership gate       prefix-hash keys from SIMOPD_A5_KEYS (any parquet with a
                        prefix_hash column; `gen_offpolicy.py --dry` emits one
                        CPU-only, so a5's unlock does NOT need the GPU cache).
                        Validation prompts miss and generate normally.

Mixed-sequence bookkeeping: the student must report log_probs over the WHOLE
stitched response (verl's behavior-policy term), so after stitching we reuse
gkd_mix's score-not-generate trick over prompt+prefix+tail. Loss needs no
surgery: the protocol estimator follows the executor per token (audit r6).

Failure posture: teacher unreachable at install/first-mix -> RuntimeError (an a5
that cannot mix must die, not train as vanilla); a single aborted tail or
scoring call degrades THAT sequence to its student prefix and counts it, so one
flaky request does not kill a lane.
"""

import atexit
import hashlib
import os
import sys
import uuid

from simopd import gkd_schedule, gkd_stats, teacher_registry
from simopd.gkd_mix import prompt_key, tail_logprobs

_MARK = "_simopd_a5"
_TAIL_SENTINEL = "simopd_a5_tail"

_keys = None
_sched = None
_handles = None
_stats = {"mixed": 0, "pure_student": 0, "full_teacher": 0, "miss": 0, "degraded": 0}
_bucket = {"step": None, "tmax": 0, "kappa_sum": 0, "n_seen": 0, "mixed": 0,
           "pure_student": 0, "full_teacher": 0, "prefix_tokens": 0, "tail_tokens": 0,
           "miss": 0, "degraded": 0}


def _load_keys():
    global _keys
    if _keys is None:
        import pandas as pd

        _keys = set(pd.read_parquet(os.environ["SIMOPD_A5_KEYS"],
                                    columns=["prefix_hash"])["prefix_hash"])
        print(f"[simopd] a5_aggrevate: {len(_keys)} training-prompt keys loaded",
              file=sys.stderr, flush=True)
    return _keys


def _tmax_at(step):
    global _sched
    if _sched is None:
        _sched = gkd_schedule.parse(os.environ["SIMOPD_A5_TMAX_SCHEDULE"])
    return max(0, int(round(_sched.value_at(step))))


def kappa(key, step, tmax):
    """Deterministic kappa ~ U{0, tmax} per (prompt, step) -- fresh each visit,
    reproducible across relaunches (the gkd_mix coin discipline)."""
    if tmax <= 0:
        return 0
    h = hashlib.sha1(f"{key}:{step}:a5".encode()).hexdigest()
    return int(h[:12], 16) % (tmax + 1)


def _flush_bucket():
    b = _bucket
    if b["step"] is None or b["n_seen"] == 0:
        return
    row = dict(b)
    row["kappa_mean"] = (b["kappa_sum"] / b["n_seen"]) if b["n_seen"] else 0.0
    tot = b["prefix_tokens"] + b["tail_tokens"]
    # Teacher-token share of the DELIVERED mix: the honest dose the registration
    # says comparisons must run on (kappa~U front-loads vs the nominal ramp).
    row["tail_token_frac"] = (b["tail_tokens"] / tot) if tot else 0.0
    gkd_stats.append(row)


def _roll_bucket(step, tmax):
    if step != _bucket["step"]:
        _flush_bucket()
        _bucket.update(step=step, tmax=tmax, kappa_sum=0, n_seen=0, mixed=0,
                       pure_student=0, full_teacher=0, prefix_tokens=0,
                       tail_tokens=0, miss=0, degraded=0)


async def _teacher_generate(prompt_ids, budget, request_id):
    global _handles
    if _handles is None:
        _handles = await teacher_registry.resolve()
    pick = _handles[int(hashlib.sha1(str(request_id).encode()).hexdigest()[:8], 16) % len(_handles)]
    # Protocol rollout params, pinned explicitly (gen_offpolicy's discipline,
    # applied live): the tail stands in for training-regime teacher samples.
    params = {"temperature": 1.0, "top_p": 1.0, "max_tokens": int(budget),
              "n": 1, "logprobs": None, _TAIL_SENTINEL: 1}
    return await pick.generate.remote(request_id=f"{uuid.uuid4().hex}-a5t",
                                      prompt_ids=list(prompt_ids),
                                      sampling_params=params)


def install():
    if os.environ.get("SIMOPD_A5_TMAX_SCHEDULE", "") == "":
        return
    if os.environ.get("SIMOPD_GKD_CACHE", ""):
        raise RuntimeError("a5_aggrevate: SIMOPD_GKD_CACHE is also set -- a5 and the "
                           "a1/a3/a4 mixer are different arms and never share a run")
    if os.environ.get("SIMOPD_A5_KEYS", "") == "":
        raise RuntimeError("a5_aggrevate: SIMOPD_A5_KEYS unset -- without the membership "
                           "gate every validation prompt would get a teacher tail")
    gkd_schedule.parse(os.environ["SIMOPD_A5_TMAX_SCHEDULE"])  # typos die here, loudly
    mod = sys.modules.get("verl.workers.rollout.vllm_rollout.vllm_async_server")
    if mod is None:
        return
    cls = getattr(mod, "vLLMHttpServer", None)
    fn = getattr(cls, "generate", None) if cls else None
    if fn is None or getattr(fn, _MARK, False):
        if fn is None:
            raise RuntimeError("a5_aggrevate: vLLMHttpServer.generate not found -- verl "
                               "moved it; the arm cannot mix and would train as vanilla")
        return
    atexit.register(_flush_bucket)

    async def generate(self, prompt_ids, sampling_params, request_id, *a, **kw):
        params_is_dict = isinstance(sampling_params, dict)
        if params_is_dict and sampling_params.get("prompt_logprobs") is not None:
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)
        if params_is_dict and sampling_params.get(_TAIL_SENTINEL) is not None:
            # Our own continuation call, now on the TEACHER server: strip and run.
            clean = {k: v for k, v in sampling_params.items() if k != _TAIL_SENTINEL}
            return await fn(self, prompt_ids, clean, request_id, *a, **kw)
        key = prompt_key(prompt_ids)
        step = getattr(self, "global_steps", 0) or 0
        tmax = _tmax_at(step)
        _roll_bucket(step, tmax)
        if key not in _load_keys():
            _stats["miss"] += 1
            _bucket["miss"] += 1
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)

        cfg = getattr(self, "config", None)
        rl = int(getattr(cfg, "response_length", 0) or 0) or 16384
        k = min(kappa(key, step, tmax), rl)
        _bucket["n_seen"] += 1
        _bucket["kappa_sum"] += k

        prefix = []
        prefix_out = None
        if k > 0:
            sp = dict(sampling_params) if params_is_dict else {}
            sp["max_tokens"] = k
            out = await fn(self, prompt_ids, sp, request_id, *a, **kw)
            if getattr(out, "stop_reason", None) == "aborted":
                return out
            prefix = list(out.token_ids or [])
            # Finished under the kappa budget = the student stopped on its own:
            # a pure-student sequence, exactly Eq.9's "kappa past termination".
            if len(prefix) < k:
                _stats["pure_student"] += 1
                _bucket["pure_student"] += 1
                _bucket["prefix_tokens"] += len(prefix)
                return out
            prefix_out = out

        budget = rl - len(prefix)
        if budget <= 0:
            _stats["pure_student"] += 1
            _bucket["pure_student"] += 1
            _bucket["prefix_tokens"] += len(prefix)
            return prefix_out

        tail = await _teacher_generate(list(prompt_ids) + prefix, budget, request_id)
        tail_ids = list(getattr(tail, "token_ids", None) or [])
        if getattr(tail, "stop_reason", None) == "aborted" or not tail_ids:
            if k == 0:
                # No prefix to fall back to; surface the abort as-is.
                _stats["degraded"] += 1
                _bucket["degraded"] += 1
                return tail
            _stats["degraded"] += 1
            _bucket["degraded"] += 1
            _bucket["pure_student"] += 1
            _bucket["prefix_tokens"] += len(prefix)
            return prefix_out

        stitched = prefix + tail_ids
        score_params = dict(sampling_params) if params_is_dict else {}
        score_params.update({"max_tokens": 1, "prompt_logprobs": 0, "logprobs": None,
                             "temperature": 0.0, "n": 1})
        scored = await fn(self, list(prompt_ids) + stitched, score_params,
                          f"{uuid.uuid4().hex}-a5s", *a, **kw)
        extra = dict(getattr(scored, "extra_fields", None) or {})
        if getattr(scored, "stop_reason", None) == "aborted" or "prompt_logprobs" not in extra:
            if k == 0:
                _stats["degraded"] += 1
                _bucket["degraded"] += 1
                return scored
            _stats["degraded"] += 1
            _bucket["degraded"] += 1
            _bucket["pure_student"] += 1
            _bucket["prefix_tokens"] += len(prefix)
            return prefix_out

        lps = tail_logprobs(extra, len(stitched))
        extra.pop("prompt_logprobs", None)
        extra.pop("prompt_ids", None)
        _stats["mixed" if k > 0 else "full_teacher"] += 1
        _bucket["mixed" if k > 0 else "full_teacher"] += 1
        _bucket["prefix_tokens"] += len(prefix)
        _bucket["tail_tokens"] += len(tail_ids)
        seen = sum(_stats.values())
        if seen % 500 == 1:
            print(f"[simopd] a5_aggrevate: {_stats} (step {step}, tmax {tmax})",
                  file=sys.stderr, flush=True)
        from verl.workers.rollout.replica import TokenOutput

        return TokenOutput(token_ids=[int(t) for t in stitched], log_probs=lps,
                           routed_experts=None,
                           stop_reason=getattr(tail, "stop_reason", None) or "completed",
                           extra_fields=extra)

    setattr(generate, _MARK, True)
    cls.generate = generate
    print(f"[simopd] a5_aggrevate armed on vLLMHttpServer.generate "
          f"(tmax schedule[{os.environ['SIMOPD_A5_TMAX_SCHEDULE']}]; "
          f"stats -> {gkd_stats.path()})", file=sys.stderr, flush=True)
