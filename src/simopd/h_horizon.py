"""h6_gen_sched: progressive rollout horizon -- per-request max_tokens clamped
to H(step) on a gkd_schedule ramp (POPD 2605.31490 / FastOPD-schedule
2602.15260 form), at the same vLLMHttpServer.generate seam as gkd_mix/a5.

Guards, in order (the A-axis seam discipline):

  scoring passthrough   prompt_logprobs requests untouched (gkd_mix's guard).
  membership gate       SIMOPD_H_KEYS (prefix-hash parquet; the a5 .dry file)
                        marks TRAINING prompts; validation/unseen prompts pass
                        through at FULL budget. This is the registered
                        improvement over h5_gen100's recorded confound, where
                        the env-level cap also truncated in-loop validation --
                        here the clamp can only ever touch training rollouts,
                        so the in-loop val column stays main-table comparable.
  window discipline     effective horizon = min(H(step), engine response cap,
                        max_model_len headroom, caller max_tokens), floor 1 --
                        a 0-token rollout is the verl as_dict crash class
                        (rm_scores[-1] on size 0, measured 2026-08-18 on a5).

Mutual exclusion: an h-arm never shares a run with the A-axis mixers --
install() refuses if SIMOPD_GKD_CACHE or SIMOPD_A5_TMAX_SCHEDULE is also set
(the two-knobs-one-dose ambiguity class arm_lint exists for).

CLOSURE/STATE SEAM: same law as gkd_mix (2026-08-18 zero-row sideband
incident) -- the wrapper closure is cloudpickled BY VALUE into the serving
actor, so bare-dict globals it touches directly would be private copies. All
state mutation routes through module-level functions (pickled by reference);
the battery pins this shape (co_names guard).

Telemetry: per-step sideband rows via gkd_stats (append-only; 30s
completion-time flush belt because Ray SIGKILLs the actor at teardown), relayed
to wandb by losses._gkd_relay_metrics' h key set. h_target is the schedule
value; realized lengths live in gen_tokens/cap_hits (natural stops shorter
than H are the point, not an error).
"""

import atexit
import os
import sys

from simopd import gkd_schedule, gkd_stats
from simopd.gkd_mix import prompt_key

_MARK = "_simopd_h_horizon"

_keys = None
_sched = None
_stats = {"pass": 0, "miss": 0, "train": 0}
_bucket = {"step": None, "h_target": 0, "n_train": 0, "n_miss": 0,
           "clamped_n": 0, "gen_tokens": 0, "cap_hits": 0, "max_len": 0}
_flush_state = {"at": 0.0}


def _load_keys():
    global _keys
    if _keys is None:
        import pandas as pd

        _keys = set(pd.read_parquet(os.environ["SIMOPD_H_KEYS"],
                                    columns=["prefix_hash"])["prefix_hash"])
        print(f"[simopd] h_horizon: {len(_keys)} training-prompt keys loaded",
              file=sys.stderr, flush=True)
    return _keys


def _adapt_mode():
    # h9_prune_adapt: horizon comes from the trainer's budget relay instead of
    # a schedule. Env-keyed (process-stable), so the closure may branch on the
    # FUNCTION safely under the cloudpickle law.
    return os.environ.get("SIMOPD_H9_ADAPT", "") != ""


def _h_at(step):
    if _adapt_mode():
        from simopd import h_budget

        return h_budget.budget()
    global _sched
    if _sched is None:
        _sched = gkd_schedule.parse(os.environ["SIMOPD_H_SCHEDULE"])
    return max(1, int(round(_sched.value_at(step))))


def _flush_bucket():
    b = _bucket
    if b["step"] is None or (b["n_train"] + b["n_miss"]) == 0:
        return
    gkd_stats.append(dict(b))


def _roll_bucket(step, h):
    # Step boundaries flush the finished bucket (sync loop = barrier); the
    # 30s completion belt below covers the final step (atexit is not
    # guaranteed under Ray SIGKILL). Pattern-copy of gkd_mix._roll_bucket.
    import time

    now = time.monotonic()
    if step != _bucket["step"]:
        _flush_bucket()
        _bucket.update(step=step, h_target=h, n_train=0, n_miss=0,
                       clamped_n=0, gen_tokens=0, cap_hits=0, max_len=0)
        _flush_state["at"] = now
    elif now - _flush_state["at"] > 120.0:
        _flush_bucket()
        _flush_state["at"] = now


def _flush_if_stale(limit=30.0):
    import time

    now = time.monotonic()
    if now - _flush_state["at"] > limit:
        _flush_bucket()
        _flush_state["at"] = now


def _mark_pass():
    _stats["pass"] += 1


def _mark_miss():
    _stats["miss"] += 1
    _bucket["n_miss"] += 1


def _mark_train(eff, clamped, h, step):
    _stats["train"] += 1
    _bucket["n_train"] += 1
    if clamped:
        _bucket["clamped_n"] += 1
    seen = _stats["train"] + _stats["miss"]
    if seen % 500 == 1:
        print(f"[simopd] h_horizon: h_target={h} eff={eff} @ step {step}; "
              f"train {_stats['train']} / miss {_stats['miss']} (val exempt)",
              file=sys.stderr, flush=True)


def _mark_train_tokens(n, hit_cap):
    _bucket["gen_tokens"] += n
    if hit_cap:
        _bucket["cap_hits"] += 1
    if n > _bucket["max_len"]:
        _bucket["max_len"] = n
    _flush_if_stale()


def install():
    sched_spec = os.environ.get("SIMOPD_H_SCHEDULE", "")
    adapt = os.environ.get("SIMOPD_H9_ADAPT", "") != ""
    if not sched_spec and not adapt:
        return
    if sched_spec and adapt:
        raise RuntimeError("h_horizon: both SIMOPD_H_SCHEDULE and SIMOPD_H9_ADAPT are "
                           "set -- two knobs claiming one horizon; an arm registers "
                           "exactly one curve (h6: schedule; h9: adaptive budget)")
    if os.environ.get("SIMOPD_GKD_CACHE", ""):
        raise RuntimeError("h_horizon: SIMOPD_GKD_CACHE is also set -- an h-arm and "
                           "the a1/a3/a4 mixer never share a run")
    if os.environ.get("SIMOPD_A5_TMAX_SCHEDULE", ""):
        raise RuntimeError("h_horizon: SIMOPD_A5_TMAX_SCHEDULE is also set -- an h-arm "
                           "and a5 never share a run")
    if os.environ.get("SIMOPD_H_KEYS", "") == "":
        raise RuntimeError("h_horizon: SIMOPD_H_KEYS unset -- without the membership "
                           "gate the clamp would also truncate in-loop validation "
                           "(h5's recorded confound; this arm registered the fix)")
    if sched_spec:
        # Eager parse + bounds: a horizon of 0 delivers empty rollouts (the verl
        # as_dict crash class); above the protocol cap it silently no-ops under
        # an h-arm's name. Both die HERE, not at step 0 in the server actor.
        s = gkd_schedule.parse(sched_spec)
        if not (1 <= min(s.start, s.end) and max(s.start, s.end) <= 16384):
            raise RuntimeError(f"h_horizon: schedule endpoints ({s.start}, {s.end}) outside "
                               f"[1, 16384] -- the horizon must stay in the protocol window")
    else:
        # Budget mode: the relay path must resolve at bringup (config errors die
        # here, not at step 0 three layers deep); budget() itself is clamped to
        # [1, 16384] by h_budget, cold-start default = full window (adapt DOWN).
        from simopd import h_budget

        h_budget.path()
    mod = sys.modules.get("verl.workers.rollout.vllm_rollout.vllm_async_server")
    if mod is None:
        return
    cls = getattr(mod, "vLLMHttpServer", None)
    fn = getattr(cls, "generate", None) if cls else None
    if fn is None or getattr(fn, _MARK, False):
        if fn is None:
            raise RuntimeError("h_horizon: vLLMHttpServer.generate not found -- verl "
                               "moved it; the arm cannot clamp and would train as vanilla")
        return
    gkd_stats.path()
    atexit.register(_flush_bucket)

    async def generate(self, prompt_ids, sampling_params, request_id, *a, **kw):
        if isinstance(sampling_params, dict) and sampling_params.get("prompt_logprobs") is not None:
            _mark_pass()
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)
        key = prompt_key(prompt_ids)
        step = getattr(self, "global_steps", 0) or 0
        h = _h_at(step)
        _roll_bucket(step, h)
        if key not in _load_keys():
            # Validation / unseen prompt: FULL budget, params untouched.
            _mark_miss()
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)
        cfg = getattr(self, "config", None)
        rl = int(getattr(cfg, "response_length", 0) or 0) or 16384
        cap = rl
        mml = getattr(cfg, "max_model_len", None)
        if mml:
            cap = min(cap, int(mml) - len(prompt_ids) - 1)
        mt_in = sampling_params.get("max_tokens") if isinstance(sampling_params, dict) else None
        if mt_in:
            cap = min(cap, int(mt_in))
        eff = max(1, min(h, cap))
        sp = dict(sampling_params) if isinstance(sampling_params, dict) else {}
        sp["max_tokens"] = eff
        _mark_train(eff, eff < cap, h, step)
        out = await fn(self, prompt_ids, sp, request_id, *a, **kw)
        tok = getattr(out, "token_ids", None)
        if tok is not None:
            _mark_train_tokens(len(tok), len(tok) >= eff)
        return out

    setattr(generate, _MARK, True)
    cls.generate = generate
    knob = f"schedule[{sched_spec}]" if sched_spec else "adaptive budget (h9 relay)"
    print(f"[simopd] h_horizon armed on vLLMHttpServer.generate ({knob}; "
          f"stats -> {gkd_stats.path()})", file=sys.stderr, flush=True)
