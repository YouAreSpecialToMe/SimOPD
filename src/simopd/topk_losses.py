"""C-axis arms that need the full student distribution (the top-k path).

verl computes top-k losses inside the logits processor, and `compute_topk_loss`
hardcodes `compute_forward_kl_topk` regardless of loss_mode -- there is no
dispatch seam, so a registry entry alone cannot change the top-k objective. We
add that seam by rebinding `compute_topk_loss` at import time (see `install()`),
which keeps the verl checkout unmodified and rebasable. Registry entries whose
names appear in TOPK_DISPATCH get our kernel; everything else falls through to
verl's original function untouched.
"""

import os

import torch
import torch.nn.functional as F

# verl is imported inside functions, never at module scope: importing it here
# would re-enter our own install hook (which imports simopd.losses, which imports
# this module) and hit a partially-initialised module.

# "renorm" divides both sides by their top-k mass; "tailbucket" instead appends a
# single bucket carrying the leftover mass. The casefile pre-registers these as an
# internal ablation of the same arm, not as two separate arms.
SUPPORT_MODE = os.environ.get("SIMOPD_SUPPORT_MODE", "renorm")

_PAD_LOGPROB_THRESHOLD = -1e15
_WARNED = False


def _warn_padding(n):
    global _WARNED
    _WARNED = True
    print(f"[simopd] top-k arm saw {n} padded teacher slots; they are excluded from the support")


def compute_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config,
    data_format: str,
):
    """Truncated reverse KL over the teacher's top-k support (LSM 2603.25562).

    Mirrors verl's compute_forward_kl_topk plumbing (nested tensors, sequence
    parallel slicing, the same diagnostic outputs) and changes only the objective:
    KL(student || teacher) restricted to the teacher's top-k, with both sides made
    into proper distributions over that support first.

    Returns the same dict keys verl's version does, so the downstream registry
    function and all existing overlap metrics keep working.
    """
    from verl.trainer.distillation.fsdp.losses import kl_divergence
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor

    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    loss_config = config.distillation_loss

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)

    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    # Guard mirrored from EasyOPD's kl_renorm_topk (methods/opcd/core.py), which this
    # arm was checked against. Note it is INERT under verl's current plumbing: verl
    # fills every rank slot and pads only along the sequence dimension, with 0.0 rather
    # than -inf, and those positions are dropped by response_mask downstream. Kept as a
    # cheap invariant that would catch a change of padding convention instead of
    # silently deflating every probability in the row.
    valid = teacher_topk_log_probs > _PAD_LOGPROB_THRESHOLD
    n_pad = (~valid).sum()

    def _renorm(log_probs):
        masked = torch.where(valid, log_probs, torch.full_like(log_probs, -1e20))
        return log_probs - torch.logsumexp(masked, dim=-1, keepdim=True)

    if SUPPORT_MODE == "renorm":
        # Both sides become distributions over S; the tail is dropped, not modelled.
        stu, tch = _renorm(student_topk_log_probs), _renorm(teacher_topk_log_probs)
    elif SUPPORT_MODE == "tailbucket":
        # One extra bucket holds the off-support mass, so the tail contributes to the
        # divergence instead of vanishing (RSKD 2503.16870's correction, in spirit).
        def with_tail(log_probs):
            mass = log_probs.exp().sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-6)
            tail = (1.0 - mass).clamp_min(1e-9).log()
            return torch.cat([log_probs, tail], dim=-1)

        stu, tch = with_tail(student_topk_log_probs), with_tail(teacher_topk_log_probs)
    else:
        raise ValueError(f"SIMOPD_SUPPORT_MODE must be 'renorm' or 'tailbucket', got {SUPPORT_MODE!r}")

    # kl_divergence(log_q, log_p) = sum p (log p - log q); student as p gives reverse KL.
    if SUPPORT_MODE == "renorm":
        stu = torch.where(valid, stu, torch.full_like(stu, -1e20))
        tch = torch.where(valid, tch, torch.full_like(tch, -1e20))
    distillation_losses = kl_divergence(log_q=tch, log_p=stu)
    if n_pad > 0 and not _WARNED:
        _warn_padding(int(n_pad))

    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage = (-token_kl * overlap_mask).sum(dim=-1) / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    out = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }
    # This arm builds its diagnostics inline rather than through
    # _overlap_diagnostics, so the two shared panels have to be added explicitly.
    # It is the arm that most needs pi(S-bar): truncating to the teacher's top-k IS
    # its method, and the headline theorem says the error that truncation incurs is
    # governed by the student mass left outside.
    probs_on_support = student_topk_log_probs.exp()
    for width in PI_TAIL_WIDTHS:
        if width <= teacher_topk_ids.shape[-1]:
            out[f"pi_tail_k{width}"] = (1.0 - probs_on_support[..., :width].sum(dim=-1)).clamp(0.0, 1.0)
    # Same reason the panels above are repeated here: this arm does not route through
    # _overlap_diagnostics, so anything added there silently skips the one arm whose
    # method IS top-k truncation.
    out["overlap_teacher_mass"] = (teacher_topk_log_probs.exp() * overlap_mask).sum(dim=-1)
    out["overlap_student_mass"] = (student_topk_log_probs.exp() * overlap_mask).sum(dim=-1)
    t_ent_topk = -(teacher_topk_log_probs.exp() * teacher_topk_log_probs).sum(dim=-1)
    s_ent = -(student_log_probs.exp() * student_log_probs).sum(dim=-1)
    out["entropy_student"] = s_ent
    out["entropy_teacher_topk"] = t_ent_topk
    out["entropy_gap_abs"] = (s_ent - t_ent_topk).abs()
    if SHADOW_ENABLED:
        out.update(_shadow_panel(student_log_probs, teacher_topk_log_probs, teacher_topk_ids,
                                 student_topk_log_probs, student_topk_ids))
    return out


TOPK_DISPATCH = {"lsm_topk_renorm": compute_reverse_kl_topk}

_original_compute_topk_loss = None


def _dispatching_compute_topk_loss(config, distillation_config, data, student_logits, data_format):
    """Route to our kernel by loss_mode; anything else keeps verl's behaviour."""
    loss_mode = distillation_config.distillation_loss.loss_mode
    fn = TOPK_DISPATCH.get(loss_mode)
    if fn is None:
        return _original_compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    if config.strategy not in ("fsdp", "veomni"):
        raise NotImplementedError(f"SimOPD top-k arms are FSDP-only for now, got {config.strategy=}")

    outputs = fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )
    expected = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected, f"Expected {expected}, got {v.shape} for {k=}"
    return outputs


def install():
    global _original_compute_topk_loss
    import verl.trainer.distillation.losses as vl

    if vl.compute_topk_loss is _dispatching_compute_topk_loss:
        return
    _original_compute_topk_loss = vl.compute_topk_loss
    vl.compute_topk_loss = _dispatching_compute_topk_loss


# ---------------------------------------------------------------------------
# Shared helpers for arms that need the student's full distribution.
# ---------------------------------------------------------------------------


def _prepare(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled):
    """Unwrap verl's nested teacher tensors and align them with student_logits.

    With SIMOPD_KEEP_SAMPLED=1 the teacher tensors carry one extra trailing column
    holding the sampled token (see simopd.teacher_patch); `want_sampled` splits it
    off. Everything before that split is the teacher's rank-ordered top-k.
    """
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor

    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    t_lp = teacher_topk_log_probs.values().unsqueeze(0)
    t_id = teacher_topk_ids.values().unsqueeze(0)
    if get_ulysses_sequence_parallel_world_size() > 1:
        t_lp = slice_input_tensor(t_lp, dim=1)
        t_id = slice_input_tensor(t_id, dim=1)
    assert t_lp.shape[:2] == t_id.shape[:2] == student_logits.shape[:2]

    sampled_lp = sampled_id = None
    if want_sampled:
        k_cfg = config.distillation_loss.topk
        if t_lp.shape[-1] != k_cfg + 1:
            raise RuntimeError(
                f"D-axis arms need SIMOPD_KEEP_SAMPLED=1: expected teacher width {k_cfg + 1}, "
                f"got {t_lp.shape[-1]}. Without it the sampled token's teacher logprob is dropped "
                "by verl and the arm cannot keep vanilla's objective."
            )
        t_lp, sampled_lp = t_lp[..., :k_cfg], t_lp[..., k_cfg]
        t_id, sampled_id = t_id[..., :k_cfg], t_id[..., k_cfg].long()

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return student_log_probs, t_lp, t_id, sampled_lp, sampled_id


def _student_entropy(student_log_probs):
    """Full-vocabulary token entropy, computed here rather than taken from
    model_output['entropy'] -- that key only exists when calculate_entropy is on,
    and the usual way to switch it on (entropy_coeff != 0) adds an entropy bonus
    to the loss, which would contaminate any arm that used it."""
    return -(student_log_probs.exp() * student_log_probs).sum(dim=-1)


def _minmax(x, mask=None):
    """Batch-relative min-max normalisation, as TIP specifies. A micro-batch is a
    smaller population than TIP's, so the normaliser is noisier; realised
    selection rates are logged so that noise stays visible."""
    flat = x[mask] if mask is not None else x
    lo, hi = flat.min(), flat.max()
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def _weighted_sampled_token_loss(student_log_probs, sampled_lp, sampled_id, weight, loss_config):
    """vanilla's objective (sampled-token reverse KL), scaled by a D-axis weight."""
    stu_sampled = torch.gather(student_log_probs, dim=-1, index=sampled_id.unsqueeze(-1)).squeeze(-1)
    tch_sampled = sampled_lp
    if loss_config.log_prob_min_clamp is not None:
        stu_sampled = stu_sampled.clamp_min(loss_config.log_prob_min_clamp)
        tch_sampled = tch_sampled.clamp_min(loss_config.log_prob_min_clamp)
    return (stu_sampled - tch_sampled) * weight


# Widths at which to report student tail mass. Only values <= the configured topk
# are usable: the teacher's returned support is rank-ordered, so a narrower support
# is its prefix, but a wider one is simply not there.
# Shadow selection is a handful of elementwise ops on tensors the kernel already
# holds, so it is on by default; SIMOPD_SHADOW=0 turns it off.
SHADOW_ENABLED = os.environ.get("SIMOPD_SHADOW", "1") == "1"
PI_TAIL_WIDTHS = tuple(int(x) for x in os.environ.get("SIMOPD_PI_TAIL_WIDTHS", "8,16,32").split(","))


def _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids):
    """Rethinking Eq.6/Eq.7 diagnostics, kept identical across every top-k arm."""
    stu_at_teacher = torch.gather(student_log_probs, dim=-1, index=t_id)
    overlap_mask = (t_id.unsqueeze(-1) == stu_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = t_lp.exp() * (t_lp - stu_at_teacher)
    adv = (-token_kl * overlap_mask).sum(dim=-1) / overlap_count.clamp_min(1)
    out = {
        "student_mass": stu_at_teacher.exp().sum(dim=-1),
        "teacher_mass": t_lp.exp().sum(dim=-1),
        "overlap_count": overlap_count,
        "overlap_token_advantage": torch.where(overlap_count > 0, adv, torch.zeros_like(adv)),
    }
    # Rethinking App B.1's "quality" form of overlap. The count version (Eq.6) says how
    # many tokens the two top-k sets share; this says how much PROBABILITY those shared
    # tokens carry. Their claim that the intersection holds 97-99% of the mass is the
    # load-bearing step in "support size does not matter", and only the mass version can
    # confirm or refute it -- a large intersection of negligible tokens looks identical
    # by count.
    out["overlap_teacher_mass"] = (t_lp.exp() * overlap_mask).sum(dim=-1)
    out["overlap_student_mass"] = (stu_at_teacher.exp() * overlap_mask).sum(dim=-1)
    # Rethinking Eq.8, |H(q) - H(p)|. The student side is exact (full vocabulary); the
    # teacher side can only be computed on the returned top-k, which UNDERSTATES its
    # entropy by the tail it cannot see. Both sides are emitted separately so the
    # approximation stays visible instead of being buried inside a difference: teacher
    # mass below 1.0 is exactly how much of the teacher's distribution is missing.
    t_ent_topk = -(t_lp.exp() * t_lp).sum(dim=-1)
    s_ent = -(student_log_probs.exp() * student_log_probs).sum(dim=-1)
    out["entropy_student"] = s_ent
    out["entropy_teacher_topk"] = t_ent_topk
    out["entropy_gap_abs"] = (s_ent - t_ent_topk).abs()
    # pi(S-bar): student mass OUTSIDE the teacher's support. This is the quantity the
    # headline theorem is written in -- truncated reverse-KL error =
    # pi(S-bar) * KL(pi||q | S-bar) -- and the literature reports only the intersection
    # (overlap ratio), never the tail. Reported at several widths from one forward
    # pass, since a narrower support is a prefix of the rank-ordered one we already
    # have: that is the whole K sweep for free, and it answers whether two K values
    # are even materially different before anyone spends a run finding out.
    probs = stu_at_teacher.exp()
    for width in PI_TAIL_WIDTHS:
        if width <= t_id.shape[-1]:
            out[f"pi_tail_k{width}"] = (1.0 - probs[..., :width].sum(dim=-1)).clamp(0.0, 1.0)
    return out


def _shadow_panel(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids):
    """What the OTHER D-axis selectors would have picked, inside this run.

    Redundancy prediction #4 (plan §4) says TIP, Teachability and SelecTKD select
    largely the same tokens. Testing that with three training runs answers it three
    times over; the selectors are pure functions of tensors this kernel already holds,
    so every run can carry all three masks for the cost of evaluating them.

    Jaccard is not emitted directly -- the metric plumbing reports means over the
    response mask, and mean(A&B)/mean(A|B) IS |A∩B|/|A∪B| because the token count
    cancels. So the intersection and union indicators go out separately and the
    ledger divides them.

    This is also how a combination gets screened before it costs a run: two settings
    whose shadow masks agree almost everywhere are one arm wearing two names.
    """
    masks = {}
    out = {}
    for name, fn in (("tip", _tip_score), ("teach", _teachability_score),
                     ("selectkd", _selectkd_score)):
        keep, _ = fn(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids)
        masks[name] = keep
        out[f"shadow_{name}"] = keep.float()
    names = list(masks)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = masks[names[i]], masks[names[j]]
            out[f"shadow_and_{names[i]}_{names[j]}"] = (a & b).float()
            out[f"shadow_or_{names[i]}_{names[j]}"] = (a | b).float()
    return out


SHADOW_KEYS = (
    "shadow_tip", "shadow_teach", "shadow_selectkd",
    "shadow_and_tip_teach", "shadow_or_tip_teach",
    "shadow_and_tip_selectkd", "shadow_or_tip_selectkd",
    "shadow_and_teach_selectkd", "shadow_or_teach_selectkd",
)
PI_TAIL_KEYS = tuple(f"pi_tail_k{w}" for w in PI_TAIL_WIDTHS)
OVERLAP_KEYS = ("overlap_teacher_mass", "overlap_student_mass",
                "entropy_student", "entropy_teacher_topk", "entropy_gap_abs")


# Shared retention for every D-axis selector, so the three arms are supervision-budget
# matched by construction (the protocol requires equal budget within an axis).
D_RETENTION = float(os.environ.get("SIMOPD_D_RETENTION", "0.5"))
# C2: candidate pool width and the average per-token support size to aim for.
QB_TARGET_BUDGET = float(os.environ.get("SIMOPD_QB_TARGET_BUDGET", "8"))
QB_MARGIN = os.environ.get("SIMOPD_QB_MARGIN", "max")  # q | pi | max
# E1: weight on the value-KL anchor beside the rank loss.
PL_ANCHOR_COEF = float(os.environ.get("SIMOPD_PL_ANCHOR_COEF", "0.1"))


def _rescale_selection(losses, keep):
    """Keep total loss magnitude comparable to vanilla after masking tokens.

    Without this a selector arm is indistinguishable from a learning-rate cut:
    agg_loss divides by the full token count, so dropping half the tokens halves
    the update. Same correction as the G-axis gates.
    """
    kept = keep.sum().clamp_min(1).float()
    total = torch.ones_like(keep, dtype=torch.float32).sum()
    return losses * keep * (total / kept)


def _topk_by_score(score, retention):
    """Keep the top `retention` fraction of positions by score, batch-relative."""
    thresh = torch.quantile(score.float().flatten(), 1.0 - retention)
    return score >= thresh


def compute_quantile_budget_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format):
    """C axis [OURS]: per-token adaptive support by a batch-level margin quantile.

    Fixed top-k spends the same vocabulary budget on a token the student is sure
    about and on a genuine fork. Here a single batch-level threshold tau is placed
    on a per-candidate margin, so the realised support size varies per token while
    the average is pinned to QB_TARGET_BUDGET -- budget-matched against a fixed-k
    arm in expectation, but allocated where the distributions actually disagree.
    """
    from verl.trainer.distillation.fsdp.losses import kl_divergence

    student_log_probs, t_lp, t_id, _, _ = _prepare(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=False
    )
    stu_at_teacher = torch.gather(student_log_probs, dim=-1, index=t_id)

    q, pi = t_lp.exp(), stu_at_teacher.exp()
    margin = {"q": q, "pi": pi}.get(QB_MARGIN, torch.maximum(q, pi))

    k = t_lp.shape[-1]
    frac = 1.0 - min(QB_TARGET_BUDGET / k, 1.0)
    tau = torch.quantile(margin.float().flatten(), frac)
    keep = margin >= tau
    keep[..., 0] = True  # the teacher's own top-1 is never dropped, so no support is empty

    neg = torch.finfo(t_lp.dtype).min
    stu_masked = torch.where(keep, stu_at_teacher, torch.full_like(stu_at_teacher, neg))
    tch_masked = torch.where(keep, t_lp, torch.full_like(t_lp, neg))
    stu_n = stu_masked - torch.logsumexp(stu_masked, dim=-1, keepdim=True)
    tch_n = tch_masked - torch.logsumexp(tch_masked, dim=-1, keepdim=True)
    stu_n = torch.where(keep, stu_n, torch.full_like(stu_n, -1e20))
    tch_n = torch.where(keep, tch_n, torch.full_like(tch_n, -1e20))

    losses = kl_divergence(log_q=tch_n, log_p=stu_n)

    stu_topk_ids = torch.topk(student_log_probs, k=k, dim=-1).indices
    if SHADOW_ENABLED:
        out_shadow = _shadow_panel(student_log_probs, t_lp, t_id,
                                   torch.gather(student_log_probs, dim=-1, index=t_id), stu_topk_ids)
    out = _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids)
    if SHADOW_ENABLED:
        out.update(out_shadow)
    out["distillation_losses"] = losses
    out["qb_budget"] = keep.sum(dim=-1).float()
    out["qb_captured_mass"] = (q * keep).sum(dim=-1)
    return out


def compute_pl_rank_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format):
    """E axis [OURS]: Plackett-Luce rank loss over the teacher's top-k, plus a value anchor.

    Rank-only supervision asks the student to reproduce the teacher's *ordering*
    rather than its probabilities, which is the weaker and possibly more
    transferable target when the capacity gap makes exact values unreachable, and
    it lines up with greedy decoding. The value anchor is kept because sampled
    evaluation (avg@32) still rewards calibrated margins; its coefficient is the
    arm's internal ablation.

    verl returns the teacher's top-k already in rank order, so the target
    permutation is the identity along the last dimension.
    """
    from verl.trainer.distillation.fsdp.losses import kl_divergence

    student_log_probs, t_lp, t_id, _, _ = _prepare(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=False
    )
    s = torch.gather(student_log_probs, dim=-1, index=t_id)

    # PL log-likelihood of the teacher's order: sum_i [ s_i - logsumexp(s_i..s_k) ].
    # The suffix logsumexp is a reversed cumulative logsumexp.
    suffix_lse = torch.flip(torch.logcumsumexp(torch.flip(s, dims=[-1]), dim=-1), dims=[-1])
    pl_loglik = (s - suffix_lse).sum(dim=-1)
    rank_loss = -pl_loglik / s.shape[-1]

    stu_n = s - torch.logsumexp(s, dim=-1, keepdim=True)
    tch_n = t_lp - torch.logsumexp(t_lp, dim=-1, keepdim=True)
    value_anchor = kl_divergence(log_q=tch_n, log_p=stu_n)

    losses = rank_loss + PL_ANCHOR_COEF * value_anchor

    stu_topk_ids = torch.topk(student_log_probs, k=t_lp.shape[-1], dim=-1).indices
    if SHADOW_ENABLED:
        out_shadow = _shadow_panel(student_log_probs, t_lp, t_id,
                                   torch.gather(student_log_probs, dim=-1, index=t_id), stu_topk_ids)
    out = _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids)
    if SHADOW_ENABLED:
        out.update(out_shadow)
    out["distillation_losses"] = losses
    out["pl_rank_loss"] = rank_loss
    out["pl_value_anchor"] = value_anchor
    return out


def _d_axis_kernel(score_fn, extra_fn=None):
    """Build a D-axis kernel: vanilla's sampled-token objective, weighted by a selector.

    The base loss stays sampled-token reverse KL for every D-axis arm, so what
    varies between them is only *which tokens are supervised* -- the axis's actual
    claim. Computing the selector needs the teacher's top-k, and keeping vanilla's
    objective needs the sampled token's teacher logprob; both are available only
    because simopd.teacher_patch stops verl discarding the latter.
    """

    def kernel(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format):
        student_log_probs, t_lp, t_id, sampled_lp, sampled_id = _prepare(
            student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=True
        )
        stu_at_teacher = torch.gather(student_log_probs, dim=-1, index=t_id)
        stu_topk_ids = torch.topk(student_log_probs, k=t_lp.shape[-1], dim=-1).indices

        # teacher_patch writes -inf if it ever fails to find the sampled token. One
        # such token would make the loss inf and NaN every gradient in the batch, with
        # no error. Floor it at the teacher's weakest top-k entry -- a true upper bound
        # on a token that ranked below them -- and surface the rate.
        finite = torch.isfinite(sampled_lp)
        n_missing = (~finite).sum()
        if n_missing > 0:
            floor = t_lp.min(dim=-1).values
            sampled_lp = torch.where(finite, sampled_lp, floor)

        keep, diag = score_fn(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids)
        raw = _weighted_sampled_token_loss(
            student_log_probs, sampled_lp, sampled_id, torch.ones_like(sampled_lp), config.distillation_loss
        )
        losses = _rescale_selection(raw, keep)

        out = _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids)
        if SHADOW_ENABLED:
            out.update(_shadow_panel(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids))
        out["distillation_losses"] = losses
        out["d_selected_frac"] = keep.float()
        out["d_sampled_missing"] = (~finite).float()
        out.update(diag)
        return out

    return kernel


def _tip_score(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids):
    """TIP 2604.14084: soft-OR of student entropy and teacher-weighted divergence.

    s = h_hat + d_hat - h_hat*d_hat on min-max normalised inputs, with the top 2%
    of entropy clipped within the batch, then the top D_RETENTION fraction kept.
    """
    h = _student_entropy(student_log_probs)
    h_clip = torch.quantile(h.float().flatten(), 0.98)
    h = h.clamp(max=h_clip)
    delta = (t_lp.exp() * (t_lp - stu_at_teacher)).sum(dim=-1)  # forward divergence at this token
    h_n, d_n = _minmax(h), _minmax(delta)
    score = h_n + d_n - h_n * d_n
    return _topk_by_score(score, D_RETENTION), {"tip_entropy_mean": h}


def _selectkd_score(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids):
    """SelecTKD 2510.24021 (greedy Top-k): the student proposes its top-1, the
    teacher verifies by whether that token is inside its own top-k.

    Unlike the other two D-axis selectors this arm's retention is data-determined,
    not tunable, so it is not budget-matched with them by construction. The
    acceptance rate (the paper's TAR) is logged so the ledger can report the
    realised budget instead of pretending it matches.
    """
    student_top1 = stu_topk_ids[..., :1]
    accepted = (t_id == student_top1).any(dim=-1)
    return accepted, {"selectkd_tar": accepted.float()}


def _teachability_score(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids):
    """Teachability 2605.26844: s = disagreement x compatibility.

    Compatibility is the teacher's probability mass on the student's top-K. The
    teacher only reports its own top-k, so this is the mass on the intersection --
    a lower bound, exact whenever the student's top-K sits inside the teacher's,
    which Rethinking reports is the common case. Recorded as a deviation.
    """
    disagreement = (t_lp.exp() * (t_lp - stu_at_teacher)).sum(dim=-1)
    in_student_topk = (t_id.unsqueeze(-1) == stu_topk_ids.unsqueeze(-2)).any(dim=-1)
    compatibility = (t_lp.exp() * in_student_topk).sum(dim=-1)
    score = _minmax(disagreement) * _minmax(compatibility)
    return _topk_by_score(score, D_RETENTION), {"teach_compatibility": compatibility}


TOPK_DISPATCH.update(
    {
        "qb_quantile_budget": compute_quantile_budget_topk,
        "pl_rank_anchor": compute_pl_rank_topk,
        "tip_select": _d_axis_kernel(_tip_score),
        "selectkd_verify": _d_axis_kernel(_selectkd_score),
        "teachability_select": _d_axis_kernel(_teachability_score),
    }
)
