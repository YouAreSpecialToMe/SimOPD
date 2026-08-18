"""h9_prune_adapt trainer-side controller: adapt the NEXT rollouts' response
budget from how far the teacher stays "on the thread" of student trajectories.

PROVENANCE HONESTY (registry note repeats this): Prune-OPD (2605.07804)
computes per-position teacher/student TOP-K SET OVERLAP and turns repeated
low-overlap events into a reliability signal that adapts the response budget.
Our protocol loss is sampled-token k1 -- the trainer sees log pi_T(y_t) and
log pi_S(y_t) SCALARS, not distributions, so the top-k sets are simply not
present at this seam. The lite controller keeps Prune's architecture (local
compatibility events -> reliable length -> adapted budget) but swaps the event
signal for what the protocol already measures: a position where the TEACHER
assigns the student's token log-prob below TAU_LOST is a "teacher lost the
thread" event (ESR 2605.27028's off-policy-teacher-decay story, measured
directly; the delta-ell position panels chart the same quantity). The M-th
lost event marks the sequence's reliable length.

Controller (documented constants, not knobs -- wave 1 registers ONE curve):

    L_i        = position of the M-th lost token (else full length)
    L_q        = quantile_Q({L_i} over the micro-batch) * MARGIN
    ema        = BETA * ema + (1-BETA) * L_q          (cold start: CAP)
    budget     = clip(int(ema), FLOOR, CAP)  -> h_budget relay -> wrapper clamp

Starts at CAP and adapts DOWN as evidence accumulates (Prune's posture: only
shorten when compatibility says so; if the teacher stays on-thread the window
stays long). Micro-batch granularity: k1_rec runs per micro-batch and the
reader takes the newest row -- the EMA absorbs the extra variance. Single-rank
assumption: the protocol lane trains with n_gpus_per_node=1, so no cross-rank
reduction is needed (revisit if lanes ever go DP>1; the budget file would need
a single-writer rule).
"""

import os
import sys

TAU_LOST = -9.2       # log pi_T(y_t) < ~1e-4: the teacher would essentially
                      # never have produced this token from this prefix
M_EVENTS = 8          # the M-th lost token ends the reliable region
QUANTILE = 0.9        # budget covers 90% of sequences' reliable lengths
MARGIN = 1.25         # headroom above the quantile
BETA = 0.7            # EMA smoothing across micro-batches
FLOOR = 256           # never starve rollouts below this
CAP = 16384           # protocol response window

_state = {"ema": None, "calls": 0}


def reliable_len(teacher_lp_row, tau=TAU_LOST, m=M_EVENTS):
    """Pure-python reference (battery-tested; the torch path must agree)."""
    lost = 0
    for i, lp in enumerate(teacher_lp_row):
        if lp < tau:
            lost += 1
            if lost >= m:
                return i + 1
    return len(teacher_lp_row)


def observe(teacher_lp, mask):
    """k1_rec hook: teacher_lp/mask are [batch, seq] torch tensors (sampled-token
    teacher log-probs and the response mask). Returns the appended row (for
    metrics) or None on empty batches. Never raises past the guard in caller.
    """
    import torch

    with torch.no_grad():
        lp = teacher_lp.detach().float()
        msk = mask.bool()
        lens = msk.sum(dim=-1)
        keep = lens > 0
        if not bool(keep.any()):
            return None
        lp = lp[keep]
        msk = msk[keep]
        lens = lens[keep]
        # lost events inside the mask; cumulative count per position
        lost = ((lp < TAU_LOST) & msk).to(torch.int32)
        cum = torch.cumsum(lost, dim=-1)
        reached = cum >= M_EVENTS
        # first position reaching M events -> index+1; else the row's length
        any_hit = reached.any(dim=-1)
        first = torch.argmax(reached.to(torch.int8), dim=-1) + 1
        L = torch.where(any_hit, first, lens).float()
        lq = torch.quantile(L, QUANTILE).item() * MARGIN
    ema = _state["ema"]
    ema = lq if ema is None else BETA * ema + (1.0 - BETA) * lq
    _state["ema"] = ema
    _state["calls"] += 1
    budget = int(max(FLOOR, min(CAP, ema)))
    from simopd import h_budget

    row = {"budget": budget, "lq": round(lq, 1), "ema": round(ema, 1),
           "n": int(L.numel()), "calls": _state["calls"]}
    h_budget.append(row)
    if _state["calls"] == 1:
        print(f"[simopd] h9_controller: first budget {budget} "
              f"(lq {lq:.0f}, n {int(L.numel())}) -> {h_budget.path()}",
              file=sys.stderr, flush=True)
    return row


def armed():
    return os.environ.get("SIMOPD_H9_ADAPT", "") != ""
