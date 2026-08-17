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
                        wrapper (sitecustomize installs everywhere in the lane's
                        shared env) strips the sentinel and generates normally.
                        Without this, a kappa=0 tail call -- whose prompt is a
                        bare training prompt, hence key-eligible -- would recurse
                        into the teacher's own wrapper forever.
  membership gate       prefix-hash keys from SIMOPD_A5_KEYS (any parquet with a
                        prefix_hash column; `gen_offpolicy.py --dry` emits one
                        CPU-only, so a5's unlock does NOT need the GPU cache).
                        Validation prompts miss and generate normally.

Natural-stop discrimination (review 2026-08-15 #6): this verl maps BOTH vllm
finish reasons ("stop", "length") to stop_reason="completed", so the cap and a
natural stop are indistinguishable after the fact. The wrapper therefore probes
with max_tokens = kappa+1: a return of <= kappa tokens PROVES the student
stopped on its own (delivered as-is); kappa+1 tokens proves it wanted to
continue, so the prefix is cut at kappa and the teacher takes over. Exact
discrimination for one token of budget, which kappa+1 <= cap always affords.

Window discipline (review #4, audit F5's twin): every generation and scoring
budget is clamped against the engine window (max_model_len - len(prompt) - 1)
and any caller-supplied max_tokens, mirroring gkd_mix. The teacher's own window
is not knowable from here; an over-window tail call fails per-request and lands
in the degraded path rather than killing the lane.

Mixed-sequence bookkeeping: the student must report log_probs over the WHOLE
stitched response (verl's behavior-policy term), so after stitching we reuse
gkd_mix's score-not-generate trick over prompt+prefix+tail. Loss needs no
surgery: the protocol estimator follows the executor per token (audit r6).

Outcome accounting (review #8/#10): every key-eligible request increments
EXACTLY ONE of {mixed, pure_student, full_teacher, cap_full, degraded, aborted},
so their sum equals n_seen -- the lost-sequence detector.

  mixed         student prefix + teacher tail delivered
  pure_student  natural stop within the kappa budget (the Eq.9 overshoot case)
  full_teacher  kappa == 0: teacher writes everything (cold-start end)
  cap_full      kappa consumed the whole window: no room for a tail
  degraded      teacher tail or scoring failed; the student prefix was delivered
  aborted       nothing usable delivered (abort surfaced to the caller)

Failure posture: teacher route DEAD (registry unresolvable after full retry)
-> TeacherRouteDead raised out of the rollout request, the lane dies -- round 4
measured the alternative (331/331 sequences silently degraded to student-only,
tail_token_frac 0, exit 0: a vanilla arm under a green banner). A single failed
tail degrades THAT sequence and is counted, never silent.
"""

import atexit
import hashlib
import os
import sys

from simopd import gkd_schedule, gkd_stats, teacher_registry
from simopd.gkd_mix import prompt_key, tail_logprobs

_MARK = "_simopd_a5"
_TAIL_SENTINEL = "simopd_a5_tail"

_keys = None
_sched = None
_handles = None
_stats = {"mixed": 0, "pure_student": 0, "full_teacher": 0, "cap_full": 0,
          "degraded": 0, "aborted": 0, "miss": 0}
_bucket = {"step": None, "tmax": 0, "kappa_sum": 0, "n_seen": 0, "mixed": 0,
           "pure_student": 0, "full_teacher": 0, "cap_full": 0, "degraded": 0,
           "aborted": 0, "prefix_tokens": 0, "tail_tokens": 0, "miss": 0}


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


_flush_state = {"at": 0.0}


def _flush_bucket():
    b = _bucket
    # miss-only steps (val sweeps) still flush, matching gkd_mix (review #7).
    if b["step"] is None or (b["n_seen"] + b["miss"]) == 0:
        return
    row = dict(b)
    row["kappa_mean"] = (b["kappa_sum"] / b["n_seen"]) if b["n_seen"] else 0.0
    tot = b["prefix_tokens"] + b["tail_tokens"]
    # Teacher-token share of the DELIVERED mix: the honest dose the registration
    # says comparisons must run on (kappa~U front-loads vs the nominal ramp).
    row["tail_token_frac"] = (b["tail_tokens"] / tot) if tot else 0.0
    gkd_stats.append(row)


def _roll_bucket(step, tmax):
    # Synchronous check-flush-reset (no awaits): two rollovers cannot interleave.
    # Requests that awaited across a step boundary credit the NEW bucket --
    # misattribution only, tolerated; the sync loop's generate->train barrier
    # keeps it rare (same argument as gkd_mix's bucket).
    # SNAPSHOT BELT: same as gkd_mix (2026-08-18 empty-sideband incident) --
    # Ray SIGKILLs actors, the atexit flush is not guaranteed, so every 120s a
    # cumulative snapshot of the live bucket lands; the reader takes the last.
    import time

    now = time.monotonic()
    if step != _bucket["step"]:
        _flush_bucket()
        _bucket.update(step=step, tmax=tmax, kappa_sum=0, n_seen=0, mixed=0,
                       pure_student=0, full_teacher=0, cap_full=0, degraded=0,
                       aborted=0, prefix_tokens=0, tail_tokens=0, miss=0)
        _flush_state["at"] = now
    elif now - _flush_state["at"] > 120.0:
        _flush_bucket()
        _flush_state["at"] = now


def _flush_if_stale(limit=30.0):
    # Completion-time belt (see gkd_mix._flush_if_stale): the final step's
    # bucket must reach disk while requests are still completing, because Ray
    # SIGKILLs the actor at teardown and atexit never runs there.
    import time

    now = time.monotonic()
    if now - _flush_state["at"] > limit:
        _flush_bucket()
        _flush_state["at"] = now


# CLOSURE/STATE SEAM: same law as gkd_mix (2026-08-18 zero-row sideband
# incident) -- the wrapper closure is cloudpickled BY VALUE into the serving
# actor, so bare-dict globals it touches directly are private copies invisible
# to the real module. All state mutation routes through module-level functions
# (pickled by reference); the battery pins this shape (co_names guard).
def _mark_miss():
    _stats["miss"] += 1
    _bucket["miss"] += 1


def _mark_seen(k, step, tmax):
    _bucket["n_seen"] += 1
    _bucket["kappa_sum"] += k
    seen = sum(_stats.values())
    if seen % 500 == 1:
        print(f"[simopd] a5_aggrevate: {_stats} (step {step}, tmax {tmax})",
              file=sys.stderr, flush=True)


def _add_tokens(where, n):
    if n:
        _bucket[where] += n
    _flush_if_stale()


def _degraded_seen():
    return _stats["degraded"] > 0


def _outcome(name, ntok=0, where=None):
    _stats[name] += 1
    _bucket[name] += 1
    if where and ntok:
        _bucket[where] += ntok
    _flush_if_stale()


async def _teacher_generate(prompt_ids, budget, request_id):
    global _handles
    if _handles is None:
        _handles = await teacher_registry.resolve()
    pick = _handles[int(hashlib.sha1(str(request_id).encode()).hexdigest()[:8], 16) % len(_handles)]
    # Protocol rollout params, pinned explicitly (gen_offpolicy's discipline,
    # applied live): the tail stands in for training-regime teacher samples.
    # The incoming request_id is reused for traceability (review #9); the
    # teacher engine has its own id space, so no collision with the student.
    params = {"temperature": 1.0, "top_p": 1.0, "max_tokens": int(budget),
              "n": 1, "logprobs": None, _TAIL_SENTINEL: 1}
    return await pick.generate.remote(request_id=request_id,
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
    # Config errors (unresolvable sideband path) die HERE at bringup; the IO
    # helpers below swallow everything (verification NEW-ISSUE 1). NO
    # reset_file() (2026-08-18): late-spawning armed processes were truncating
    # the serving process's rows -- the sideband is append-only, launchers
    # delete it before a fresh run.
    gkd_stats.path()
    atexit.register(_flush_bucket)

    async def generate(self, prompt_ids, sampling_params, request_id, *a, **kw):
        params_is_dict = isinstance(sampling_params, dict)
        if params_is_dict and sampling_params.get("prompt_logprobs") is not None:
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)
        if params_is_dict and sampling_params.get(_TAIL_SENTINEL) is not None:
            # Our own continuation call, now on the TEACHER server: strip and run.
            # (Assumes the lane-shared env armed this wrapper on the teacher too;
            # the 3-step rehearsal checks exactly this before any real launch.)
            clean = {k: v for k, v in sampling_params.items() if k != _TAIL_SENTINEL}
            return await fn(self, prompt_ids, clean, request_id, *a, **kw)
        key = prompt_key(prompt_ids)
        step = getattr(self, "global_steps", 0) or 0
        tmax = _tmax_at(step)
        _roll_bucket(step, tmax)
        if key not in _load_keys():
            _mark_miss()
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)

        # The window discipline: never ask either engine to overflow. cap is the
        # largest response this request may deliver (audit F5's clamp, plus the
        # caller's own max_tokens where one arrives).
        cfg = getattr(self, "config", None)
        rl = int(getattr(cfg, "response_length", 0) or 0) or 16384
        cap = rl
        mml = getattr(cfg, "max_model_len", None)
        if mml:
            cap = min(cap, int(mml) - len(prompt_ids) - 1)
        mt_in = sampling_params.get("max_tokens") if params_is_dict else None
        if mt_in:
            cap = min(cap, int(mt_in))
        k = min(kappa(key, step, tmax), max(cap, 0))
        _mark_seen(k, step, tmax)
        sp = dict(sampling_params) if params_is_dict else {}

        async def _student_fallback(reason):
            # kappa=0 infra failure has no student prefix to deliver, and an
            # EMPTY output crashes verl's as_dict (rm_scores[-1] on a size-0
            # response -- measured 2026-08-18, 110x, killed the first a5
            # rehearsal batch). The honest degrade is a COUNTED on-policy
            # student generation: this sequence trains as vanilla, the
            # telemetry says so, and the lane lives.
            if not _degraded_seen():
                print(f"[simopd] a5_aggrevate: {reason}; delivering a counted "
                      f"student generation instead", file=sys.stderr, flush=True)
            _outcome("degraded")
            fb = await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)
            ftok = getattr(fb, "token_ids", None)
            if ftok:
                _add_tokens("prefix_tokens", len(ftok))
            return fb

        if cap <= 0 or k >= cap:
            # kappa consumed the whole window (or there is none): the student
            # keeps the entire budget, a teacher tail cannot fit.
            if cap > 0:
                sp["max_tokens"] = cap
                out = await fn(self, prompt_ids, sp, request_id, *a, **kw)
            else:
                out = await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)
            tok = list(getattr(out, "token_ids", None) or [])
            if getattr(out, "stop_reason", None) == "aborted":
                _outcome("aborted")
                return out
            _outcome("cap_full" if (cap <= 0 or len(tok) >= cap) else "pure_student",
                     len(tok), "prefix_tokens")
            return out

        prefix = []
        prefix_out = None
        if k > 0:
            # kappa+1 probe: <= k tokens proves a natural stop, k+1 proves the
            # student wanted more. k < cap guarantees the probe fits the window.
            sp["max_tokens"] = k + 1
            out = await fn(self, prompt_ids, sp, request_id, *a, **kw)
            if getattr(out, "stop_reason", None) == "aborted":
                _outcome("aborted")
                return out
            got = list(out.token_ids or [])
            if len(got) <= k:
                # Natural stop within budget: the genuine Eq.9 overshoot case.
                _outcome("pure_student", len(got), "prefix_tokens")
                return out
            prefix = got[:k]
            prefix_out = out

        budget = cap - len(prefix)   # >= 1 because len(prefix) == k < cap
        # The teacher is a DIFFERENT model with its own window, unknowable from
        # here; a tail call it rejects (over-window prompt, transport error) must
        # land in the degrade path, not kill the rollout request (verification
        # NEW-ISSUE 2 -- the docstring promised this, now the code delivers it).
        try:
            tail = await _teacher_generate(list(prompt_ids) + prefix, budget, request_id)
        except teacher_registry.TeacherRouteDead:
            # Route dead (resolve exhausted its retry): die loudly, never train
            # as vanilla -- kappa=0's student fallback is for transient infra,
            # not for a run whose teacher can never arrive.
            raise
        except Exception as e:
            if not _degraded_seen():
                print(f"[simopd] a5_aggrevate: teacher call failed ({e!r}); degrading "
                      f"this and any further failures to the student prefix",
                      file=sys.stderr, flush=True)
            tail = None
        tail_ids = list(getattr(tail, "token_ids", None) or [])
        if tail is None or getattr(tail, "stop_reason", None) == "aborted" or not tail_ids:
            if prefix_out is None:
                return await _student_fallback("teacher tail unavailable at kappa=0")
            # The k+1-token student output is a valid capped on-policy sample;
            # deliver it rather than fail the request (counted, never silent).
            _outcome("degraded", len(prefix_out.token_ids or []), "prefix_tokens")
            return prefix_out

        stitched = prefix + tail_ids
        score_params = dict(sampling_params) if params_is_dict else {}
        score_params.update({"max_tokens": 1, "prompt_logprobs": 0, "logprobs": None,
                             "temperature": 0.0, "n": 1})
        # Suffixed id: the probe already used request_id on THIS engine, and vLLM
        # rejects an id it still tracks -- sequential-and-awaited should have
        # released it, but "should" is not a contract (verification NEW-ISSUE 3).
        # Abort reachability stays where it matters: the probe and the 16k teacher
        # tail keep the original id; this scoring call is max_tokens=1.
        scored = await fn(self, list(prompt_ids) + stitched, score_params,
                          f"{request_id}-a5s", *a, **kw)
        extra = dict(getattr(scored, "extra_fields", None) or {})
        if getattr(scored, "stop_reason", None) == "aborted" or "prompt_logprobs" not in extra:
            if prefix_out is None:
                return await _student_fallback("scoring unavailable at kappa=0")
            _outcome("degraded", len(prefix_out.token_ids or []), "prefix_tokens")
            return prefix_out

        lps = tail_logprobs(extra, len(stitched))
        extra.pop("prompt_logprobs", None)
        extra.pop("prompt_ids", None)
        _outcome("mixed" if prefix else "full_teacher")
        _add_tokens("prefix_tokens", len(prefix))
        _add_tokens("tail_tokens", len(tail_ids))
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
