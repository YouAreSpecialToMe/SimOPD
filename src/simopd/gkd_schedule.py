"""Step-indexed schedules for the A-axis mixing knob (a4_dagger_anneal, a5 T_max).

Faithful port of STACX rl_engine/rollout/utils/dagger_schedule.py::DAggerSchedule
(the DAgger-for-agents recipe, arXiv 2605.12913): same four modes, same
warmup/decay semantics, same equations -- a schedule quoted in that paper's units
transfers to SimOPD by renaming "iteration" to "training step". Direction note:
STACX's beta and our SIMOPD_GKD_LAMBDA are both P(teacher-sourced), so
start=1.0 -> end=0.0 is the DAgger anneal in BOTH repos with no inversion
(contrast with the paper-lambda trap documented at gkd_mix._lam()).

Spec grammar (one string, env-friendly):

    "mode=linear,start=1.0,end=0.0,warmup=0,decay=250"

  mode    linear | cosine | exponential | step   (default linear)
  start   value held during warmup and at decay begin          (required)
  end     value from step warmup+decay onward                  (required)
  warmup  steps held at start before decay begins              (default 0)
  decay   steps over which the value moves start -> end        (required, >= 1)

Registered a4 shapes: primary "mode=linear,start=1.0,end=0.0,warmup=0,decay=250"
(mean dose 0.502 over 250 steps == a1's constant 0.5: the matched-average-dose
pair) and the adjudicator "...decay=125" (the published agentic recipe's 0.2/round
x 10 iters in normalized time: anneal done at 50%, second half pure on-policy).

Values may ascend (a5's T_max ramp: start=0,end=16384). STACX quirk preserved
on purpose for cross-repo comparability: exponential clamps both endpoints at
1e-8 before taking the ratio, so exponential-to-zero decays to ~1e-8 inside the
window and lands exactly on `end` only when the window closes.
"""

import math
from dataclasses import dataclass

MODES = ("linear", "cosine", "exponential", "step")
_KEYS = ("mode", "start", "end", "warmup", "decay")


@dataclass(frozen=True)
class Schedule:
    mode: str
    start: float
    end: float
    warmup: int
    decay: int

    def value_at(self, step):
        """Value at a 0-indexed training step. Equations verbatim from STACX
        DAggerSchedule.beta(); `step` mode flips at the decay window midpoint."""
        if step < self.warmup:
            return self.start
        d = step - self.warmup
        if d >= self.decay:
            return self.end
        progress = d / max(self.decay, 1)
        if self.mode == "linear":
            return self.start + (self.end - self.start) * progress
        if self.mode == "cosine":
            return self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * progress))
        if self.mode == "exponential":
            ratio = max(self.end, 1e-8) / max(self.start, 1e-8)
            return self.start * ratio ** progress
        # parse() has already validated mode, so this is the "step" branch.
        return self.start if progress < 0.5 else self.end


def parse(spec):
    """Parse a spec string; every malformation is a loud ValueError.

    Silent fallback here would be the two-vanillas-under-a1's-name failure class
    (arm_lint r6): a typo'd schedule must kill the run at install, not train a
    different dose curve than the registered one.
    """
    kv = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"gkd_schedule: '{part}' is not key=value (spec {spec!r})")
        k, v = (s.strip() for s in part.split("=", 1))
        if k not in _KEYS:
            raise ValueError(f"gkd_schedule: unknown key {k!r}; allowed {_KEYS} (spec {spec!r})")
        if k in kv:
            raise ValueError(f"gkd_schedule: duplicate key {k!r} (spec {spec!r})")
        kv[k] = v
    for req in ("start", "end", "decay"):
        if req not in kv:
            raise ValueError(f"gkd_schedule: missing required key {req!r} (spec {spec!r})")
    mode = kv.get("mode", "linear")
    if mode not in MODES:
        raise ValueError(f"gkd_schedule: mode {mode!r} not in {MODES} (spec {spec!r})")
    try:
        sched = Schedule(mode=mode, start=float(kv["start"]), end=float(kv["end"]),
                         warmup=int(kv.get("warmup", "0")), decay=int(kv["decay"]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"gkd_schedule: non-numeric field in {spec!r}: {e}")
    if sched.decay < 1:
        raise ValueError(f"gkd_schedule: decay must be >= 1 (spec {spec!r})")
    if sched.warmup < 0:
        raise ValueError(f"gkd_schedule: warmup must be >= 0 (spec {spec!r})")
    return sched
