"""C-axis arms that need the full student distribution (the top-k path).

verl computes top-k losses inside the logits processor, and `compute_topk_loss`
hardcodes `compute_forward_kl_topk` regardless of loss_mode -- there is no
dispatch seam, so a registry entry alone cannot change the top-k objective. We
add that seam by rebinding `compute_topk_loss` at import time (see `install()`),
which keeps the verl checkout unmodified and rebasable. Registry entries whose
names appear in TOPK_DISPATCH get our kernel; everything else falls through to
verl's original function untouched.
"""

import math
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
    data=None,
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
    student_topk_ids = _student_topk_ids(student_log_probs, k=teacher_topk_ids.shape[-1])
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
    out["rank_kendall_tau"] = _kendall_tau(student_topk_log_probs)
    t_ent_topk = -(teacher_topk_log_probs.exp() * teacher_topk_log_probs).sum(dim=-1)
    s_ent = _student_entropy(student_log_probs)
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
        data=data,
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


def _prepare_streaming(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled):
    """_prepare without materializing [T, V] log-probs (the KEEP_SAMPLED family's
    18.6GiB killer): returns the per-token logsumexp instead. Every consumer
    derives log p(j) = logits[j] - lse from gathers; see _lse_chunked. The
    teacher-side unwrap below mirrors _prepare line for line."""
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

    lse = _lse_chunked(student_logits)
    return lse, t_lp, t_id, sampled_lp, sampled_id


def _lse_chunked(student_logits, chunk=256):
    """Per-token logsumexp over the vocab, chunked over tokens -- the flash-softmax
    idea applied at the loss layer. log p(j) = logits[j] - lse, so the fp32 [T, V]
    log_softmax output (the 18.6GiB single allocation that killed every b3 attempt
    at steps 76-84 and ground d1's relay legs to 107-119 once response lengths
    saturated the 16k cap) is never materialized. Plain autograd ops per chunk:
    values and gradients are exact, and backward touches one [chunk, V] softmax
    at a time (~150MB at chunk=256 on a 152k vocab)."""
    return torch.cat([torch.logsumexp(sl.float(), dim=-1)
                      for sl in student_logits.split(chunk, dim=-2)], dim=-1)


def _entropy_from_logits(student_logits, lse, chunk=256):
    """Full-vocab token entropy from raw logits + lse: H = lse - sum_v p_v*logit_v,
    elementwise-equal to _student_entropy(log_softmax(logits)) with no [T, V]
    intermediate. Gradient equivalence is covered by the same CPU suite as
    _lse_chunked."""
    outs = []
    for sl, sub in zip(student_logits.split(chunk, dim=-2), lse.split(chunk, dim=-1)):
        slf = sl.float()
        p = (slf - sub.unsqueeze(-1)).exp()
        outs.append(sub - (p * slf).sum(dim=-1))
    return torch.cat(outs, dim=-1)


def _stat_mask(teacher_topk_log_probs, data, total):
    """Packed-view mask of the tokens batch statistics may legitimately see.

    Kernels receive the FULL attended sequence (prompt + response + verl's dummy
    first row per sequence), but every batch-relative statistic -- min-max and
    robust normalizers, retention thresholds, qb's tau, selection rescale -- is
    defined by the papers and the registration over RESPONSE tokens. The audit
    (2026-08-07 F3/F4) measured the dummy row alone dragging TIP's divergence
    normalizer by ~30x, degenerating d1 to entropy-only selection, and prompt
    tokens contributing 6-50% of the population depending on training phase.

    Built from the two length sources that exist in the micro-batch: the nested
    teacher tensor's per-sequence row counts and response_mask's per-sequence
    true counts; each sequence's LAST resp_len rows are its response. Returns
    None (statistics fall back to the full population) when `data` is absent --
    CPU harnesses -- or the two sources disagree, loudly.
    """
    import sys as _sys

    if data is None:
        return None
    rm = data.get("response_mask", None)
    if rm is None:
        return None
    try:
        offs = teacher_topk_log_probs.offsets()
        full_lens = (offs[1:] - offs[:-1]).tolist()
        if rm.is_nested:
            resp_lens = [int(x.sum()) for x in rm.unbind()]
        else:
            resp_lens = rm.sum(dim=-1).tolist()
    except Exception as e:
        print(f"[simopd] stat_mask unavailable ({e!r}); batch statistics fall back to "
              f"the full packed population", file=_sys.stderr, flush=True)
        return None
    if len(full_lens) != len(resp_lens) or sum(full_lens) != total:
        print(f"[simopd] stat_mask length mismatch (teacher rows {sum(full_lens)} over "
              f"{len(full_lens)} seqs vs response {len(resp_lens)} seqs, packed {total}); "
              f"falling back to full population", file=_sys.stderr, flush=True)
        return None
    # ON THE KERNEL'S DEVICE. Without device= this mask was born on CPU while
    # every tensor it gates lives on cuda, and the first `keep & mask` killed
    # every stat-mask consumer (d1/d2/d3 directly, every kernel via the shadow
    # panel) on its first training step -- 2026-08-08, three runs down in the
    # opening minutes of wave 5. The CPU suite structurally cannot catch this:
    # with data=None (its documented harness path) the mask is never built.
    m = torch.zeros(total, dtype=torch.bool, device=teacher_topk_log_probs.device)
    pos = 0
    for fl, rl in zip(full_lens, resp_lens):
        rl = min(int(rl), int(fl))
        m[pos + fl - rl: pos + fl] = True
        pos += fl
    return m.unsqueeze(0)


def _student_entropy(student_log_probs, chunk=256):
    """Full-vocabulary token entropy, computed here rather than taken from
    model_output['entropy'] -- that key only exists when calculate_entropy is on,
    and the usual way to switch it on (entropy_coeff != 0) adds an entropy bonus
    to the loss, which would contaminate any arm that used it.

    Chunked over the token dimension, because the naive form materialises an
    exp(B,S,V) chain -- ~7GB transient at a 12k-token microbatch on a 152k vocab --
    and that growth of the caching allocator's reserved pool is what killed d1_tip:
    verl's sleep/wake weight sync OOM'd at step 2's wake_up, the first sync after
    the optimizer moments (12.7GB) materialise. Per-token entropy is independent
    across tokens, so slicing changes the peak and not one bit of the result;
    verified elementwise against the naive form.

    chunk=256 (2026-08-08, twice in one day: 4096 -> 1024 -> 256): the 4096-chunk
    transient (~2.5GB) OOM'd d1_tip_s0 at 17,408-token microbatches; 1024 (~0.6GB)
    then OOM'd d1_tip_s1 against 458MiB free, because the D family at 16k runs the
    whole step with under 2GB of headroom -- the chunk was never the disease, the
    margin is. 256 puts the transient near 150MB. The companion measure is
    SIMOPD_SHADOW=0 in the lane launcher: the shadow panel triples the selector
    transients for pure diagnostics, and it is fingerprint-excluded precisely
    because it never changes what the loss computes."""
    out = torch.empty(student_log_probs.shape[:-1],
                      dtype=student_log_probs.dtype, device=student_log_probs.device)
    for i in range(0, student_log_probs.shape[1], chunk):
        sl = student_log_probs[:, i:i + chunk]
        out[:, i:i + chunk] = -(sl.exp() * sl).sum(dim=-1)
    return out


def _student_topk_ids(student_log_probs, k, chunk=256):
    """Student top-k INDICES over the vocab, chunked over the token dimension.

    torch.topk on the full [*, T, V] view allocates a sort workspace on the same
    scale as its input. Stacked on the log_softmax output and the update-phase
    activation transients, that workspace is the 17-19GiB single allocation that
    killed every b3 quad attempt at steps 76-84 and capped the d1/g2 relay legs
    at ~10 steps once response lengths saturated the 16k cap (the length wall,
    2026-08-09). Per-token top-k is independent across tokens, so slicing the
    token dimension changes the peak and not one index; the consumers
    (_overlap_diagnostics, _shadow_panel) take indices only, so there is no
    autograd surface at all. Same chunk=256 rationale as _student_entropy above:
    ~150MB transient at a 152k vocab.
    """
    outs = [torch.topk(sl, k=k, dim=-1).indices
            for sl in student_log_probs.split(chunk, dim=-2)]
    return torch.cat(outs, dim=-2)


def _minmax(x, mask=None):
    """Batch-relative min-max normalisation, as TIP specifies. A micro-batch is a
    smaller population than TIP's, so the normaliser is noisier; realised
    selection rates are logged so that noise stays visible."""
    flat = x[mask] if mask is not None else x
    lo, hi = flat.min(), flat.max()
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def _weighted_sampled_token_loss(stu_sampled, sampled_lp, weight, loss_config):
    """vanilla's objective (sampled-token reverse KL), scaled by a D-axis weight.

    Takes the student's log-prob at the sampled token PRE-GATHERED (streaming
    callers derive it as gather(logits) - lse; see _lse_chunked)."""
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


def _kendall_tau(s):
    """Kendall tau-a between the student's within-support values and the teacher's
    rank order. verl delivers the top-k already rank-sorted, so the reference
    permutation is the identity and tau reduces to the mean pairwise sign (ties
    count zero): +1 = order matched, -1 = reversed. The E-axis ladder's free
    mechanism panel -- together with student_mass (support coverage, already
    logged) it dates, on EVERY top-k arm, whether order or mass converges first."""
    k = s.shape[-1]
    diff = s.unsqueeze(-1) - s.unsqueeze(-2)
    iu = torch.triu_indices(k, k, offset=1, device=s.device)
    return torch.sign(diff[..., iu[0], iu[1]]).mean(dim=-1)


def _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids, stu_at_teacher=None, s_ent=None):
    """Rethinking Eq.6/Eq.7 diagnostics, kept identical across every top-k arm.

    Streaming callers pass stu_at_teacher (= gather(logits) - lse) AND s_ent
    (from _entropy_from_logits) with None for student_log_probs -- the tail of
    this function needs the entropy panel, and forgetting that cost all six
    streaming relaunches a NoneType crash on 2026-08-09. The materialized path
    is unchanged for everyone else."""
    if stu_at_teacher is None:
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
    out["rank_kendall_tau"] = _kendall_tau(stu_at_teacher)
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
    if s_ent is None:
        s_ent = _student_entropy(student_log_probs)
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


def _shadow_panel(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat=None):
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
    _shadow_ent = lambda: _student_entropy(student_log_probs)  # noqa: E731 -- shadow is
    # always called with a materialized log-probs tensor (streaming kernels build one
    # locally before entering the panel, and shadow is off in production lanes anyway).
    for name, fn in (("tip", _tip_score), ("teach", _teachability_score),
                     ("selectkd", _selectkd_score)):
        keep, _ = fn(_shadow_ent, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat)
        # selectkd returns weights in {beta, 1} since audit r5; its SET, for
        # redundancy comparison, is the fully-supervised tokens.
        if keep.dtype.is_floating_point:
            keep = keep >= 1.0
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
                "entropy_student", "entropy_teacher_topk", "entropy_gap_abs",
                "rank_kendall_tau")


# Shared retention for every D-axis selector, so the three arms are supervision-budget
# matched by construction (the protocol requires equal budget within an axis).
D_RETENTION = float(os.environ.get("SIMOPD_D_RETENTION", "0.5"))
# C2: candidate pool width and the average per-token support size to aim for.
QB_TARGET_BUDGET = float(os.environ.get("SIMOPD_QB_TARGET_BUDGET", "8"))
QB_MARGIN = os.environ.get("SIMOPD_QB_MARGIN", "max")  # q | pi | max
# E1: weight on the value-KL anchor beside the rank loss.
PL_ANCHOR_COEF = float(os.environ.get("SIMOPD_PL_ANCHOR_COEF", "0.1"))


def _rescale_selection(losses, keep, mask=None):
    """Keep total loss magnitude comparable to vanilla after masking tokens.

    Without this a selector arm is indistinguishable from a learning-rate cut:
    agg_loss divides by the full token count, so dropping half the tokens halves
    the update. Same correction as the G-axis gates.
    """
    kept = keep.sum().clamp_min(1).float()
    total = (mask.sum() if mask is not None else torch.ones_like(keep, dtype=torch.float32).sum()).float()
    return losses * keep * (total / kept)


def _topk_by_score(score, retention, mask=None):
    """Keep the top `retention` fraction of positions by score, batch-relative.

    With a stat mask the threshold population AND the selectable set are the
    response tokens; positions outside it are never selected (their loss is
    discarded downstream anyway, but keeping them out preserves the realized
    retention the ledger reports)."""
    pop = (score[mask] if mask is not None else score).float().flatten()
    thresh = torch.quantile(pop, 1.0 - retention)
    keep = score >= thresh
    if mask is not None:
        keep = keep & mask
    return keep


def compute_quantile_budget_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
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
    stat = _stat_mask(teacher_topk_log_probs, data, t_lp.shape[1])
    stu_at_teacher = torch.gather(student_log_probs, dim=-1, index=t_id)

    q, pi = t_lp.exp(), stu_at_teacher.exp()
    margin = {"q": q, "pi": pi}.get(QB_MARGIN, torch.maximum(q, pi))

    k = t_lp.shape[-1]
    frac = 1.0 - min(QB_TARGET_BUDGET / k, 1.0)
    tau = torch.quantile((margin[stat] if stat is not None else margin).float().flatten(), frac)
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

    stu_topk_ids = _student_topk_ids(student_log_probs, k=k)
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


def compute_pl_rank_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
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

    stu_topk_ids = _student_topk_ids(student_log_probs, k=t_lp.shape[-1])
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


def compute_set_coverage_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
    """E axis [OURS]: set-membership supervision -- student mass on the teacher's top-k.

    Loosest rung of the within-support information ladder (values c1 > z-values e3 >
    order e1 > SET e2). The structural term -log sum_topk pi = -log(1 - pi(S-bar))
    trains exactly the quantity the headline theorem bounds truncation error by and
    c4 logs as a panel. Deliberately indifferent to HOW mass is distributed inside
    the support: degenerate or unstable optima are themselves the verdict. The value
    anchor is e1's, at the same coefficient, so e1<->e2 differ in exactly one term
    (rank -> set); the anchor is the axis's registered common additive.
    """
    from verl.trainer.distillation.fsdp.losses import kl_divergence

    student_log_probs, t_lp, t_id, _, _ = _prepare(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=False
    )
    s = torch.gather(student_log_probs, dim=-1, index=t_id)
    coverage_log = torch.logsumexp(s, dim=-1)
    coverage_loss = -coverage_log

    stu_n = s - torch.logsumexp(s, dim=-1, keepdim=True)
    tch_n = t_lp - torch.logsumexp(t_lp, dim=-1, keepdim=True)
    value_anchor = kl_divergence(log_q=tch_n, log_p=stu_n)

    losses = coverage_loss + PL_ANCHOR_COEF * value_anchor

    stu_topk_ids = _student_topk_ids(student_log_probs, k=t_lp.shape[-1])
    if SHADOW_ENABLED:
        out_shadow = _shadow_panel(student_log_probs, t_lp, t_id, s, stu_topk_ids)
    out = _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids)
    if SHADOW_ENABLED:
        out.update(out_shadow)
    out["distillation_losses"] = losses
    out["e2_coverage"] = coverage_log.exp()
    out["e2_value_anchor"] = value_anchor
    return out


def compute_zvalue_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
    """E axis [OURS]: affine-invariant value matching -- z-score both sides within
    the support, then the axis's KL form. z-as-adaptive-temperature pre-process from
    Logit Standardization in KD (2403.01427, CVPR24; vision/off-policy, form source
    only -- the OPD arm is ours).

    The rung between c1's raw values and e1's pure order: gap RATIOS kept, shift and
    scale discarded. z on log-probs equals z on logits exactly (the logsumexp shift
    is a per-token constant and std is shift-invariant), so the top-k log-prob
    payload suffices. NO value anchor: the structural term is itself a value matcher
    and a raw anchor would smuggle scale back into the rung (asymmetry vs e1/e2
    registered in the arm note).
    """
    from verl.trainer.distillation.fsdp.losses import kl_divergence

    student_log_probs, t_lp, t_id, _, _ = _prepare(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=False
    )
    s = torch.gather(student_log_probs, dim=-1, index=t_id)
    s_std = s.std(dim=-1, keepdim=True, correction=0)
    t_std = t_lp.std(dim=-1, keepdim=True, correction=0)
    s_z = (s - s.mean(dim=-1, keepdim=True)) / (s_std + 1e-6)
    t_z = (t_lp - t_lp.mean(dim=-1, keepdim=True)) / (t_std + 1e-6)
    stu_n = torch.log_softmax(s_z, dim=-1)
    tch_n = torch.log_softmax(t_z, dim=-1)
    losses = kl_divergence(log_q=tch_n, log_p=stu_n)

    stu_topk_ids = _student_topk_ids(student_log_probs, k=t_lp.shape[-1])
    if SHADOW_ENABLED:
        out_shadow = _shadow_panel(student_log_probs, t_lp, t_id, s, stu_topk_ids)
    out = _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids)
    if SHADOW_ENABLED:
        out.update(out_shadow)
    out["distillation_losses"] = losses
    out["e3_std_ratio"] = s_std.squeeze(-1) / (t_std.squeeze(-1) + 1e-6)
    return out


def _d_axis_kernel(score_fn, extra_fn=None):
    """Build a D-axis kernel: vanilla's sampled-token objective, weighted by a selector.

    The base loss stays sampled-token reverse KL for every D-axis arm, so what
    varies between them is only *which tokens are supervised* -- the axis's actual
    claim. Computing the selector needs the teacher's top-k, and keeping vanilla's
    objective needs the sampled token's teacher logprob; both are available only
    because simopd.teacher_patch stops verl discarding the latter.
    """

    def kernel(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
        lse, t_lp, t_id, sampled_lp, sampled_id = _prepare_streaming(
            student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=True
        )
        stu_at_teacher = torch.gather(student_logits, dim=-1, index=t_id) - lse.unsqueeze(-1)
        stu_topk_ids = _student_topk_ids(student_logits, k=t_lp.shape[-1])
        # The diagnostics tail needs the entropy panel regardless of selector, so
        # compute it eagerly once and let tip's thunk reuse the same tensor.
        s_ent = _entropy_from_logits(student_logits, lse)
        s_ent_fn = lambda: s_ent  # noqa: E731

        # teacher_patch writes -inf if it ever fails to find the sampled token. One
        # such token would make the loss inf and NaN every gradient in the batch, with
        # no error. Floor it at the teacher's weakest top-k entry -- a true upper bound
        # on a token that ranked below them -- and surface the rate.
        finite = torch.isfinite(sampled_lp)
        n_missing = (~finite).sum()
        if n_missing > 0:
            floor = t_lp.min(dim=-1).values
            sampled_lp = torch.where(finite, sampled_lp, floor)

        stat = _stat_mask(teacher_topk_log_probs, data, t_lp.shape[1])
        keep, diag = score_fn(s_ent_fn, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat)
        stu_sampled = torch.gather(student_logits, dim=-1, index=sampled_id.unsqueeze(-1)).squeeze(-1) - lse
        raw = _weighted_sampled_token_loss(
            stu_sampled, sampled_lp, torch.ones_like(sampled_lp), config.distillation_loss
        )
        # Two selector families (audit r5): boolean masks are rescaled by selected
        # count -- both TIP's and TA-OPD's losses normalize over SELECTED tokens, and
        # without the rescale a selector is indistinguishable from a learning-rate
        # cut. Float weights (SelecTKD's V_t in {beta, 1}) multiply the loss as-is:
        # their sum_t V_t * D_t keeps full-batch normalization, and rescaling would
        # un-do the very down-weighting under audit.
        if keep.dtype.is_floating_point:
            losses = raw * keep
            selected = (keep >= 1.0).float()
        else:
            losses = _rescale_selection(raw, keep, stat)
            selected = keep.float()

        out = _overlap_diagnostics(None, t_lp, t_id, stu_topk_ids,
                                   stu_at_teacher=stu_at_teacher, s_ent=s_ent)
        if SHADOW_ENABLED:
            # Shadow needs the materialized distribution; it is off in production
            # lanes (SIMOPD_SHADOW=0) and only then does this [T, V] tensor exist.
            slp = F.log_softmax(student_logits, dim=-1)
            out.update(_shadow_panel(slp, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat))
        out["distillation_losses"] = losses
        out["d_raw_k1"] = raw          # panel source: the UNWEIGHTED signed k1 (F6)
        out["d_selected_frac"] = selected
        out["d_sampled_missing"] = (~finite).float()
        out.update(diag)
        return out

    return kernel


# Audit r5, against each paper's own defaults and (where released) official code:
# SelecTKD verifies against the teacher's top-5, not the payload width, and its
# rejected tokens are DOWN-WEIGHTED (beta=0.01 default), not masked; TA-OPD
# normalizes with a Q05/Q95 robust clip (tip_compat.py: opd_metric_q_low/high
# default 0.05/0.95), reads compatibility over the student's top-16, and its paper
# recommends a 5% budget (set in the arm's env, not here).
SELECTKD_K = int(os.environ.get("SIMOPD_SELECTKD_K", "5"))
SELECTKD_BETA = float(os.environ.get("SIMOPD_SELECTKD_BETA", "0.01"))
TEACH_K = int(os.environ.get("SIMOPD_TEACH_K", "16"))
# d1's pre-registered decomposition (2026-08-07): soft_or is TIP's method; the two
# single-branch modes test the paper's own strongest claims in isolation --
# entropy_only against their "50% entropy retention matches full-token" result,
# divergence_only against "confidently-wrong <10% of tokens nearly matches"
# (pair it with SIMOPD_D_RETENTION=0.1 as registered). In the fingerprint via the
# SIMOPD_ capture, so ablation runs are automatically a distinct batch.
TIP_MODE = os.environ.get("SIMOPD_TIP_MODE", "soft_or")


def _robust_norm(x, mask=None):
    """TA-OPD's batch-relative robust scaling: clip((z - Q05)/(Q95 - Q05), 0, 1).

    Their tip_compat.py default ('batch_quantile'); plain min-max is the fallback
    they ship but not the shipped default. Same micro-batch-population caveat as
    _minmax."""
    flat = (x[mask] if mask is not None else x).float().flatten()
    lo = torch.quantile(flat, 0.05)
    hi = torch.quantile(flat, 0.95)
    return ((x - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)


def _tip_score(s_ent_fn, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat=None):
    """TIP 2604.14084: soft-OR of student entropy and teacher-student divergence.

    s = h_hat + d_hat - h_hat*d_hat on plain min-max normalised inputs, top
    D_RETENTION fraction kept (0.5 = the paper's primary configuration). An
    earlier version clipped entropy at p98 first; the paper specifies plain
    min-max and even names its outlier sensitivity as a limitation, so the clip
    was this repo quietly fixing the audited method -- removed (audit r5). The
    divergence is teacher-top-k truncated (recorded; teacher_mass quantifies)."""
    h = s_ent_fn()
    delta = (t_lp.exp() * (t_lp - stu_at_teacher)).sum(dim=-1)
    h_n, d_n = _minmax(h, stat), _minmax(delta, stat)
    if TIP_MODE == "entropy_only":
        score = h_n
    elif TIP_MODE == "divergence_only":
        score = d_n
    elif TIP_MODE == "soft_or":
        score = h_n + d_n - h_n * d_n
    else:
        raise ValueError(f"SIMOPD_TIP_MODE must be soft_or|entropy_only|divergence_only, got {TIP_MODE!r}")
    return _topk_by_score(score, D_RETENTION, stat), {"tip_entropy_mean": h}


def _selectkd_score(s_ent_fn, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat=None):
    """SelecTKD 2510.24021 (greedy Top-k variant): the student proposes its top-1,
    the teacher verifies membership in its own top-k -- k = the paper's default 5,
    checked against the FIRST five of the rank-ordered payload, not all 32 (audit
    r5: verifying against the payload width tripled the effective acceptance
    window and would have inflated TAR).

    Rejected tokens are down-weighted by beta=0.01 (their stated default), not
    masked: V_t in {beta, 1} multiplies the base loss directly, with no
    selected-count rescale -- their L = sum_t V_t * D_t keeps the full-batch
    normalization. Retention is data-determined; TAR is logged."""
    student_top1 = stu_topk_ids[..., :1]
    accepted = (t_id[..., :SELECTKD_K] == student_top1).any(dim=-1)
    weights = torch.where(accepted, 1.0, SELECTKD_BETA).to(stu_at_teacher.dtype)
    return weights, {"selectkd_tar": accepted.float()}


def _teachability_score(s_ent_fn, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat=None):
    """TA-OPD 2605.26844: s = disagreement x compatibility, Q05/Q95-normalized.

    Compatibility is the teacher's mass on the student's top-K, K=16 (their
    default; the paper sweeps 8/16/32). The teacher only reports its own top-k,
    so this is the mass on the intersection -- a lower bound, exact whenever the
    student's top-16 sits inside the teacher's top-32, which Rethinking reports
    is the common case. Recorded as a deviation. The arm's budget lives in its
    env (paper-recommended 5%), and the paper explicitly blesses the
    sampled-token base loss this kernel family uses."""
    disagreement = (t_lp.exp() * (t_lp - stu_at_teacher)).sum(dim=-1)
    stu_topK = stu_topk_ids[..., :TEACH_K]
    in_student_topk = (t_id.unsqueeze(-1) == stu_topK.unsqueeze(-2)).any(dim=-1)
    compatibility = (t_lp.exp() * in_student_topk).sum(dim=-1)
    score = _robust_norm(disagreement, stat) * _robust_norm(compatibility, stat)
    return _topk_by_score(score, D_RETENTION, stat), {"teach_compatibility": compatibility}


# B axis, adaptive member (EOPD 2603.07079, ICML 2026; pre-registered addition
# 2026-08-06 after the coverage sweep, BEFORE any of its data existed). Reverse KL's
# mode-seeking is precise where the teacher is confident and unstable where the
# teacher is spread out; EOPD routes per token on teacher entropy -- reverse KL in
# low-entropy regions, forward KL in high-entropy ones. In this registry the arm is
# exactly a per-token ROUTER between two objectives that already exist as arms:
# vanilla/b1's sampled-token reverse KL and b2's teacher-top-k forward KL (same
# clamp_min(0) as verl gives b2, so the components stay comparable arm-to-arm).
# EOPD's official defaults (WLS04/EOPD core_algos.compute_policy_loss_on_policy_distill):
# soft_kd_entropy_threshold=0.8 -- a FIXED absolute entropy in nats, not a quantile --
# and soft_kd_coef=1.0.
B3_ENT_THRESH = float(os.environ.get("SIMOPD_B3_ENT_THRESH", "0.8"))
B3_SOFT_COEF = float(os.environ.get("SIMOPD_B3_SOFT_COEF", "1.0"))


def compute_eopd_gate_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
    """B3, third form, this one against the official code (audit r5, WLS04/EOPD).

    "Augmenting" in their abstract is literal: `pg_loss = pg_loss + soft_kd_loss`.
    The reverse-KL PG base runs on EVERY token -- exactly vanilla's sampled-k1
    advantage under the clipped surrogate -- and high-teacher-entropy tokens get an
    ADDITIVE top-k forward-KL term, gated by a fixed absolute threshold. Both prior
    forms here were wrong about that: r3's where(gate, fkl, rkl) switched the RKL
    off on gated tokens, and moved the arm to the direct branch when the official
    loss is PG-based throughout. The base loss below feeds the PG advantage; the
    gated FKL is exported separately and added to the FINAL loss by b3_additive's
    wrapper, because anything returned from here would be detached into the
    advantage and lose its pathwise gradient.

    Recorded deviations: teacher entropy from top-k truncation (theirs is
    full-vocab from the ref pass; teacher_mass quantifies the unseen tail, and a
    truncated entropy can only under-shoot, so the 0.8 gate fires on fewer tokens
    -- b3_gate reports the realised fraction) and micro-batch normalization of the
    additive term."""
    lse, t_lp, t_id, sampled_lp, sampled_id = _prepare_streaming(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=True
    )
    stat = _stat_mask(teacher_topk_log_probs, data, t_lp.shape[1])
    stu_at_teacher = torch.gather(student_logits, dim=-1, index=t_id) - lse.unsqueeze(-1)
    stu_topk_ids = _student_topk_ids(student_logits, k=t_lp.shape[-1])

    finite = torch.isfinite(sampled_lp)
    if (~finite).sum() > 0:
        sampled_lp = torch.where(finite, sampled_lp, t_lp.min(dim=-1).values)
    stu_sampled = torch.gather(student_logits, dim=-1, index=sampled_id.unsqueeze(-1)).squeeze(-1) - lse
    raw = _weighted_sampled_token_loss(
        stu_sampled, sampled_lp, torch.ones_like(sampled_lp), config.distillation_loss
    )

    t_ent = -(t_lp.exp() * t_lp).sum(dim=-1)
    gate = t_ent >= B3_ENT_THRESH
    # Their exact truncated form: sum over teacher top-k of p*(log p - log q),
    # unnormalized, no clamp -- verl's clamp_min(0) is b2's choice, not theirs.
    fkl = (t_lp.exp() * (t_lp - stu_at_teacher)).sum(dim=-1)

    out = _overlap_diagnostics(None, t_lp, t_id, stu_topk_ids, stu_at_teacher=stu_at_teacher,
                               s_ent=_entropy_from_logits(student_logits, lse))
    if SHADOW_ENABLED:
        slp = F.log_softmax(student_logits, dim=-1)
        out.update(_shadow_panel(slp, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat))
    out["distillation_losses"] = raw
    out["b3_soft_kd"] = fkl * gate.float()
    out["b3_gate"] = gate.float()
    out["b3_missing"] = (~finite).float()
    return out


# FiRe Eq.5-8 needs per-token TEACHER entropy, which the estimator path cannot see,
# so g2 moves to the top-k path (audit r4: the arm had implemented only the paper's
# first stage -- the trajectory filter -- and silently skipped "Then Reweight", the
# title's second half). The kernel emits COMPONENTS; trajectory-level operations
# (filter, per-trajectory weight normalization) need sequence boundaries, which exist
# only in the padded view, so they live in the registry post-processor.
JSD_BETA = float(os.environ.get("SIMOPD_JSD_BETA", "0.5"))
PI_TAIL_EPS = float(os.environ.get("SIMOPD_PI_TAIL_EPS", "0.05"))


def compute_intersection_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
    """C axis, c3 (supplement): thunlp/OPD's `intersection` support strategy,
    audited against their dp_actor.compute_distillation_reward (2026-08-07).

    Their form: per-CANDIDATE advantages A = -(S - T) * w over the candidates in
    both top-k sets, w = student probabilities renormalized over the valid set
    (their default reward_weight_mode='student_p', normalize=True), then a 3D PPO
    surrogate summed over candidates -- which their own code reduces, at
    ppo_epochs=1, to the memory-efficient direct form L = -sum sg(A) * log pi.
    We implement exactly that reduced form on the DIRECT branch (faithful to
    their shipped path; the 3D-ratio variant differs only off-policy). The
    candidate set is symmetric (teacher-topk intersect student-topk), so our
    teacher-side payload carries every value needed -- no second forward.

    Empty-intersection tokens (overlap 0) contribute zero loss; the realised
    intersection size is on the overlap panel every top-k arm already logs."""
    student_log_probs, t_lp, t_id, _, _ = _prepare(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=False
    )
    stu_at_teacher = torch.gather(student_log_probs, dim=-1, index=t_id)
    stu_topk_ids = _student_topk_ids(student_log_probs, k=t_lp.shape[-1])
    stat = _stat_mask(teacher_topk_log_probs, data, t_lp.shape[1])

    valid = (t_id.unsqueeze(-1) == stu_topk_ids.unsqueeze(-2)).any(dim=-1)      # in both top-ks
    kl_val = (stu_at_teacher - t_lp).float()                                     # their k1 per candidate
    w_log = stu_at_teacher.float().masked_fill(~valid, torch.finfo(torch.float32).min)
    w = torch.softmax(w_log, dim=-1) * valid                                     # student_p renormalized on valid
    adv = (-kl_val * w).detach()                                                 # their rm_scores, treated constant
    losses = -(adv * stu_at_teacher.float() * valid).sum(dim=-1)                 # -sum sg(A) * log pi

    out = _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids)
    if SHADOW_ENABLED:
        out.update(_shadow_panel(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat))
    out["distillation_losses"] = losses.to(student_log_probs.dtype)
    out["c3_inter_size"] = valid.float().sum(dim=-1)
    return out


def compute_pi_tail_budget_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
    """C axis, c4 [OURS, headline-constructive]: pin the theorem's own quantity.

    The headline says truncated-RKL error = pi(S-bar) * KL(tail conditionals):
    controlled by the STUDENT's tail mass, while common practice truncates by
    teacher rank alone. This arm chooses, per token, the SMALLEST prefix of the
    teacher's rank-ordered top-k whose STUDENT mass reaches 1 - PI_TAIL_EPS --
    cutting on the student's edge inside the teacher's candidate pool -- then
    applies c1's renormalized reverse KL on that support (same objective family,
    DIRECT branch, own registration). If the full pool cannot reach the target,
    the whole pool is used and c4_eps_missed logs how often. Teacher top-1 is
    always kept. Realised budget (c4_budget) and realised student tail mass
    (c4_pi_tail) make the theorem's quantity a first-class logged series."""
    from verl.trainer.distillation.fsdp.losses import kl_divergence

    student_log_probs, t_lp, t_id, _, _ = _prepare(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=False
    )
    stu_at_teacher = torch.gather(student_log_probs, dim=-1, index=t_id)
    stu_topk_ids = _student_topk_ids(student_log_probs, k=t_lp.shape[-1])
    stat = _stat_mask(teacher_topk_log_probs, data, t_lp.shape[1])

    pi = stu_at_teacher.float().exp()
    cum = pi.cumsum(dim=-1)
    target = 1.0 - PI_TAIL_EPS
    reached = cum >= target
    # smallest prefix reaching the target; if never reached, the full pool
    first = torch.where(reached.any(dim=-1),
                        reached.float().argmax(dim=-1),
                        torch.full(reached.shape[:-1], reached.shape[-1] - 1,
                                   dtype=torch.long, device=reached.device))
    idx = torch.arange(t_lp.shape[-1], device=t_lp.device)
    keep = idx <= first.unsqueeze(-1)                                            # top-1 always in

    neg = torch.finfo(torch.float32).min
    stu_k = stu_at_teacher.float().masked_fill(~keep, neg)
    tch_k = t_lp.float().masked_fill(~keep, neg)
    stu_n = stu_k - torch.logsumexp(stu_k, dim=-1, keepdim=True)
    tch_n = tch_k - torch.logsumexp(tch_k, dim=-1, keepdim=True)
    stu_n = stu_n.masked_fill(~keep, neg)
    tch_n = tch_n.masked_fill(~keep, neg)
    losses = kl_divergence(log_q=tch_n, log_p=stu_n)

    out = _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids)
    if SHADOW_ENABLED:
        out.update(_shadow_panel(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat))
    out["distillation_losses"] = losses.to(student_log_probs.dtype)
    out["c4_budget"] = keep.float().sum(dim=-1)
    out["c4_pi_tail"] = (1.0 - (pi * keep).sum(dim=-1)).clamp(0.0, 1.0)
    out["c4_eps_missed"] = (~reached.any(dim=-1)).float()
    return out


def compute_jsd_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
    """B axis, b4 (supplement cohort): GKD's generalized JSD-beta on the teacher's
    renormalized top-k support (2306.13649; TRL's canonical interpolation, audited
    r5: mixture m = beta*teacher + (1-beta)*student, jsd = beta*KL(teacher||m) +
    (1-beta)*KL(student||m); beta=0.5 is symmetric JSD. Endpoints degenerate to
    zero; the SCALED limits jsd/beta -> renorm FKL(T||S) and jsd/(1-beta) ->
    renorm RKL(S||T) are verified numerically to 5e-4 -- the first draft stated
    them backwards and its own limit test caught it). The plan's axis-B text promised JSD
    and skew-KL displaced it -- this arm pays the debt. Distributional both sides
    (JSD's teacher-outer half is not sampled-estimable), DIRECT branch per GKD;
    support truncation is the recorded C-axis-style deviation, teacher_mass
    quantifies it. SIMOPD_JSD_BETA is the pre-registered internal ablation
    (GKD finds the optimum task-dependent across {0.1, 0.5, 0.9})."""
    from verl.trainer.distillation.fsdp.losses import kl_divergence

    student_log_probs, t_lp, t_id, _, _ = _prepare(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=False
    )
    stu_at_teacher = torch.gather(student_log_probs, dim=-1, index=t_id)
    stu_topk_ids = _student_topk_ids(student_log_probs, k=t_lp.shape[-1])
    stat = _stat_mask(teacher_topk_log_probs, data, t_lp.shape[1])

    b = JSD_BETA
    stu_n = (stu_at_teacher - torch.logsumexp(stu_at_teacher, dim=-1, keepdim=True)).float()
    tch_n = (t_lp - torch.logsumexp(t_lp, dim=-1, keepdim=True)).float()
    log_m = torch.logaddexp(tch_n + math.log(b), stu_n + math.log(1.0 - b))
    losses = b * kl_divergence(log_q=log_m, log_p=tch_n) \
        + (1.0 - b) * kl_divergence(log_q=log_m, log_p=stu_n)

    out = _overlap_diagnostics(student_log_probs, t_lp, t_id, stu_topk_ids)
    if SHADOW_ENABLED:
        out.update(_shadow_panel(student_log_probs, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat))
    out["distillation_losses"] = losses.to(student_log_probs.dtype)
    return out


FIRE_ALPHA = float(os.environ.get("SIMOPD_FIRE_ALPHA", "1.0"))
FIRE_BETA = float(os.environ.get("SIMOPD_FIRE_BETA", "1.0"))


def compute_fire_components(student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, data=None):
    lse, t_lp, t_id, sampled_lp, sampled_id = _prepare_streaming(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=True
    )
    stat = _stat_mask(teacher_topk_log_probs, data, t_lp.shape[1])
    stu_at_teacher = torch.gather(student_logits, dim=-1, index=t_id) - lse.unsqueeze(-1)
    stu_topk_ids = _student_topk_ids(student_logits, k=t_lp.shape[-1])

    finite = torch.isfinite(sampled_lp)
    if (~finite).sum() > 0:
        sampled_lp = torch.where(finite, sampled_lp, t_lp.min(dim=-1).values)

    stu_sampled = torch.gather(student_logits, dim=-1, index=sampled_id.unsqueeze(-1)).squeeze(-1) - lse
    raw = _weighted_sampled_token_loss(
        stu_sampled, sampled_lp, torch.ones_like(sampled_lp), config.distillation_loss
    )
    s_ent = _entropy_from_logits(student_logits, lse)
    out = _overlap_diagnostics(None, t_lp, t_id, stu_topk_ids, stu_at_teacher=stu_at_teacher, s_ent=s_ent)
    if SHADOW_ENABLED:
        slp = F.log_softmax(student_logits, dim=-1)
        out.update(_shadow_panel(slp, t_lp, t_id, stu_at_teacher, stu_topk_ids, stat))
    # Teacher entropy from its top-k understates by the unseen tail (teacher_mass in
    # the diagnostics quantifies it); recorded deviation, same as the entropy panel.
    out["distillation_losses"] = raw
    out["fire_t_ent"] = -(t_lp.exp() * t_lp).sum(dim=-1)
    out["fire_s_ent"] = s_ent
    out["fire_tch_lp"] = sampled_lp
    out["fire_missing"] = (~finite).float()
    return out


TOPK_DISPATCH.update(
    {
        "eopd_entropy_gate": compute_eopd_gate_topk,
        "k1_fire_gate": compute_fire_components,
        "jsd_topk": compute_jsd_topk,
        "intersection_topk": compute_intersection_topk,
        "pi_tail_budget": compute_pi_tail_budget_topk,
        "qb_quantile_budget": compute_quantile_budget_topk,
        "pl_rank_anchor": compute_pl_rank_topk,
        "set_coverage_anchor": compute_set_coverage_topk,
        "zvalue_topk": compute_zvalue_topk,
        "tip_select": _d_axis_kernel(_tip_score),
        "selectkd_verify": _d_axis_kernel(_selectkd_score),
        "teachability_select": _d_axis_kernel(_teachability_score),
    }
)
