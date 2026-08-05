"""SimOPD arm losses, registered into verl's distillation registry.

Imported by src/sitecustomize.py, which every interpreter on PYTHONPATH runs at
startup -- including verl's Ray workers, where the registry lookup actually
happens. Nothing in verl is modified.

Sign convention (verified against verl 2026-07-31):
  kl_penalty(logprob=student, ref_logprob=teacher, "k1") = log p_stu - log q_tch
and `distillation_loss` then uses `-losses` as the advantage, so the advantage is
log q_tch - log p_stu = Delta-ell_t, exactly Demystifying's per-token signal.
A loss-space transform therefore applies to -Delta-ell; for odd transforms (all
of ours) that is identical to transforming Delta-ell itself.
"""

import math
import os

import torch
from verl.trainer.distillation.losses import (
    DistillationLossSettings,
    register_distillation_loss,
)

from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding

from simopd import topk_losses

# Adds the loss_mode dispatch that verl's top-k path lacks (see topk_losses).
topk_losses.install()

_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# H-axis supervision window. verl's loss config is a fixed dataclass, so arm
# hyper-parameters that it has no field for come in by env var; both the driver
# and the Ray workers read it because verl forwards the environment it was
# launched with. Pre-registered default 512 (ESR used 100 on much shorter
# responses; 512 is the comparable fraction of our 8k cap).
FIRST_SEGMENT_K = int(os.environ.get("SIMOPD_FIRST_SEGMENT_K", "512"))

# FiRe drops the bottom 20% of trajectories by normalised teacher logprob.
FIRE_DROP_FRAC = float(os.environ.get("SIMOPD_FIRE_DROP_FRAC", "0.2"))


def _unpack(model_output, data):
    """Student/teacher sampled-token logprobs and the response mask, padded & aligned."""
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    mask = data["response_mask"]
    mask = mask.to_padded_tensor(False).bool() if mask.is_nested else mask.bool()
    assert teacher_log_probs.shape == student_log_probs.shape == mask.shape
    return student_log_probs, teacher_log_probs, mask


def _signal_quantiles(losses, mask, name):
    """Per-token signal distribution (METRICS.md section 3).

    `name` matters: only the k1 family's loss equals -Delta-ell, so only those arms
    may report under delta_ell_*. The top-k arms optimise a divergence (>=0) or a
    rank loss, and publishing those as Delta-ell would put three different
    quantities on one cross-arm panel and make the curves look comparable when they
    measure different things. They report loss_* instead.
    """
    x = losses[mask].detach().float()
    if name == "delta_ell":
        x = -x  # paper convention: positive = teacher puts more mass here than the student
    if x.numel() == 0:
        return {}
    qs = torch.quantile(x, torch.tensor(_QUANTILES, device=x.device))
    metrics = {
        f"distillation/{name}_p{int(q * 100)}": Metric(aggregation=AggregationType.MEAN, value=v)
        for q, v in zip(_QUANTILES, qs, strict=True)
    }
    metrics[f"distillation/{name}_absmean"] = Metric(aggregation=AggregationType.MEAN, value=x.abs().mean())
    return metrics


def _delta_ell_metrics(losses, mask):
    return _signal_quantiles(losses, mask, "delta_ell")


def _clip_metrics(losses, mask, distillation_config):
    """F axis: what fraction of the signal the hard clip actually touches.

    f2_hard_clip turns on verl's loss_max_clamp, which is applied downstream of the
    loss fn, so without this the arm's own mechanism would be invisible -- its
    Delta-ell panel is the pre-clip distribution, identical to vanilla's.
    """
    c = distillation_config.distillation_loss.loss_max_clamp
    if c is None:
        return {}
    hit = (losses[mask].detach().abs() > c).float().mean()
    return {"distillation/clip_hit_rate": Metric(aggregation=AggregationType.MEAN, value=hit)}


@register_distillation_loss(
    DistillationLossSettings(names=["k1_rec"], use_estimator=True)
)  # type: ignore[arg-type]
def k1_with_recorder(config, distillation_config, model_output, data):
    """Vanilla k1, bit-identical to verl's, plus the Delta-ell distribution panel.

    The arm's math is unchanged -- this exists only so every run, baseline
    included, carries the mechanistic metrics the audit promises.
    """
    student, teacher, mask = _unpack(model_output, data)
    losses = kl_penalty(logprob=student, ref_logprob=teacher, kl_penalty="k1")
    metrics = {"distillation/abs_loss": Metric(aggregation=AggregationType.MEAN, value=losses[mask].abs().mean())}
    metrics.update(_delta_ell_metrics(losses, mask))
    metrics.update(_clip_metrics(losses, mask, distillation_config))
    return losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["skew_kl_a0.1"], use_estimator=True)
)  # type: ignore[arg-type]
def skew_kl(config, distillation_config, model_output, data):
    """B axis: skew-KL with alpha=0.1 (DistiLLM 2402.03898), the pre-registered value.

    SKL_a(p||q) = KL(p || a*p + (1-a)*q). The mixture's log-density at the sampled
    token is exactly computable from the two sampled-token logprobs we already have
    -- logaddexp(log a + log p, log(1-a) + log q) -- so no top-k or full-vocab pass
    is needed and this arm stays budget-matched with vanilla on the same C axis.

    Mixing toward the student bounds the estimator when the teacher assigns near-zero
    mass, which is the mode-collapse failure DistiLLM targets.
    """
    student, teacher, mask = _unpack(model_output, data)
    alpha = 0.1
    # In fp32: the mixture is a logaddexp of two shifted log-densities, and in bf16
    # log(alpha) alone already costs ~2 decimal digits.
    s32, t32 = student.float(), teacher.float()
    log_mix = torch.logaddexp(
        s32 + math.log(alpha),
        t32 + math.log(1.0 - alpha),
    )
    losses = (s32 - log_mix).to(student.dtype)

    metrics = {"distillation/abs_loss": Metric(aggregation=AggregationType.MEAN, value=losses[mask].abs().mean())}
    metrics.update(_delta_ell_metrics(losses, mask))
    return losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["k1_firstseg"], use_estimator=True)
)  # type: ignore[arg-type]
def k1_first_segment(config, distillation_config, model_output, data):
    """H axis: supervise only the first FIRST_SEGMENT_K response tokens.

    The loss-mask form of ESR 2605.27028's claim that the teacher's signal is
    concentrated early. Generation is NOT truncated -- ESR's rollout-truncation
    form changes the sampling distribution itself, which would confound this with
    the A axis, so only the supervision window moves (SimOPD-casefile H).

    Masked tokens are rescaled away exactly as in the G-axis gates. Without that
    the arm would also shrink the update by the masked fraction, and a drop could
    be blamed on a smaller effective learning rate rather than on the late tokens
    carrying signal -- which is the only thing this arm is meant to test. The
    covered fraction is still logged so the ledger can state the budget it used.
    """
    student, teacher, mask = _unpack(model_output, data)
    losses = kl_penalty(logprob=student, ref_logprob=teacher, kl_penalty="k1")

    positions = torch.arange(losses.shape[1], device=losses.device).unsqueeze(0)
    window = positions < FIRST_SEGMENT_K
    keep = window & mask
    kept, total = keep.sum().clamp_min(1).float(), mask.sum().clamp_min(1).float()
    losses = losses * keep * (total / kept)

    metrics = {"distillation/abs_loss": Metric(aggregation=AggregationType.MEAN, value=losses[mask].abs().mean())}
    metrics.update(_delta_ell_metrics(losses, keep))
    metrics["distillation/firstseg_covered_frac"] = Metric(
        aggregation=AggregationType.MEAN, value=kept / total
    )
    return losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["k1_softlog"], use_estimator=True)
)  # type: ignore[arg-type]
def k1_soft_log_compression(config, distillation_config, model_output, data):
    """F axis: soft log compression, sign(d)*log(1+|d|) (Demystifying's winner).

    Bounds the tail of the per-token signal without the discontinuity of a hard
    clip, which is what the paper credits for suppressing length exploitation.
    Both the raw and compressed signal are recorded, so the arm's effect on the
    tail is visible rather than inferred.
    """
    student, teacher, mask = _unpack(model_output, data)
    raw = kl_penalty(logprob=student, ref_logprob=teacher, kl_penalty="k1")
    compressed = torch.sign(raw) * torch.log1p(raw.abs())

    metrics = {"distillation/abs_loss": Metric(aggregation=AggregationType.MEAN, value=compressed[mask].abs().mean())}
    metrics.update(_delta_ell_metrics(compressed, mask))
    with torch.no_grad():
        raw_abs = raw[mask].abs().mean()
        metrics["distillation/softlog_raw_absmean"] = Metric(aggregation=AggregationType.MEAN, value=raw_abs)
        # ~1.0 would mean the transform is inert on this batch
        metrics["distillation/softlog_shrink_ratio"] = Metric(
            aggregation=AggregationType.MEAN,
            value=compressed[mask].abs().mean() / raw_abs.clamp_min(1e-8),
        )
    return compressed, metrics


def _reweight_kept(losses, mask, keep_seq):
    """Zero out dropped trajectories and rescale so token-mean matches a real filter.

    Gating arms drop trajectories. Implementing that as a loss mask keeps the
    dropped tokens in agg_loss's denominator, which would silently shrink the
    effective step size in proportion to the drop rate -- the arm would then be
    measuring a learning-rate change, not gating. Rescaling by
    kept_tokens/total_tokens restores the mean a true batch filter would produce.
    """
    keep = keep_seq.unsqueeze(-1) & mask
    kept, total = keep.sum().clamp_min(1).float(), mask.sum().clamp_min(1).float()
    return losses * keep * (total / kept), keep


@register_distillation_loss(
    DistillationLossSettings(names=["k1_verified_only"], use_estimator=True)
)  # type: ignore[arg-type]
def k1_verified_only(config, distillation_config, model_output, data):
    """G axis: distil only rollouts that pass the rule verifier.

    Discipline from the plan: the verifier only filters, its answer never enters
    the training input. Implemented as a rescaled loss mask rather than a batch
    filter so it composes with the other axes in the greedy rounds; the rescale in
    _reweight_kept makes the two equivalent in expectation.

    The gate reads `advantages` rather than the raw verifier score. verl's trainer
    writes only advantages and returns back for the actor to consume -- the actual
    keys present were confirmed by an earlier run of this arm, which raised with
    the micro-batch's key list -- so token_level_scores never reaches a loss fn.
    Under this protocol (adv_estimator=grpo, use_task_rewards=False) advantages are
    derived from that verifier score alone, so a trajectory with positive total
    advantage is exactly one the verifier accepted. The assumption is asserted
    below: if the estimator ever changes, this fails loudly instead of quietly
    gating on something else.
    """
    student, teacher, mask = _unpack(model_output, data)
    losses = kl_penalty(logprob=student, ref_logprob=teacher, kl_penalty="k1")

    if "advantages" not in data.keys():
        raise KeyError(
            "k1_verified_only needs `advantages` (the verifier-derived signal) in the "
            f"actor micro-batch; got {sorted(data.keys())}"
        )
    adv = data["advantages"]
    adv = adv.to_padded_tensor(0.0) if adv.is_nested else adv
    keep_seq = (adv * mask).sum(dim=-1) > 0
    losses, keep = _reweight_kept(losses, mask, keep_seq)

    metrics = {"distillation/abs_loss": Metric(aggregation=AggregationType.MEAN, value=losses[mask].abs().mean())}
    metrics.update(_delta_ell_metrics(losses, keep))
    # A batch where nothing verified produces no update at all -- correct for this
    # arm, but it has to be visible or it looks like a silently dead step.
    metrics["distillation/gate_keep_frac"] = Metric(
        aggregation=AggregationType.MEAN, value=keep_seq.float().mean()
    )
    return losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["k1_fire_gate"], use_estimator=True)
)  # type: ignore[arg-type]
def k1_fire_likelihood_gate(config, distillation_config, model_output, data):
    """G axis: drop the bottom quantile of trajectories by teacher likelihood (FiRe 2606.02684).

    FiRe ranks trajectories by s(y) = (1/T) sum log pi*(y_t | ...) and discards the
    bottom 20%. The threshold is taken within the micro-batch, so it tracks the
    running distribution rather than a fixed cutoff -- with the caveat that a
    micro-batch is a noisier ranking population than FiRe's full batch; the
    realised drop fraction is logged so that noise is visible.
    """
    student, teacher, mask = _unpack(model_output, data)
    losses = kl_penalty(logprob=student, ref_logprob=teacher, kl_penalty="k1")

    lengths = mask.sum(dim=-1).clamp_min(1)
    s_y = (teacher * mask).sum(dim=-1) / lengths  # normalised teacher logprob per trajectory
    thresh = torch.quantile(s_y.float(), FIRE_DROP_FRAC)
    keep_seq = s_y >= thresh
    losses, keep = _reweight_kept(losses, mask, keep_seq)

    metrics = {"distillation/abs_loss": Metric(aggregation=AggregationType.MEAN, value=losses[mask].abs().mean())}
    metrics.update(_delta_ell_metrics(losses, keep))
    metrics["distillation/gate_keep_frac"] = Metric(aggregation=AggregationType.MEAN, value=keep_seq.float().mean())
    metrics["distillation/fire_s_y_thresh"] = Metric(aggregation=AggregationType.MEAN, value=thresh)
    return losses, metrics


def _topk_registry_fn(*extra_keys, signal="loss"):
    """Post-processor for our top-k arms.

    Deliberately NOT verl's compute_forward_kl_topk: that one ends with
    `clamp_min(0.0)`, correct for its unnormalised top-k forward KL but fatal for
    the D-axis arms, whose base loss is vanilla's k1 and is legitimately negative
    wherever the student already outranks the teacher. Clamping there would erase
    exactly half the signal without erroring.
    """

    def fn(config, distillation_config, model_output, data):
        losses = no_padding_2_padding(model_output["distillation_losses"], data)
        mask = data["response_mask"]
        mask = mask.to_padded_tensor(False).bool() if mask.is_nested else mask.bool()

        metrics = {"distillation/abs_loss": Metric(aggregation=AggregationType.MEAN, value=losses[mask].abs().mean())}
        metrics.update(_signal_quantiles(losses, mask, signal))

        k = distillation_config.distillation_loss.topk
        for key, label in [("student_mass", "student_mass"), ("teacher_mass", "teacher_mass")]:
            if key in model_output:
                v = no_padding_2_padding(model_output[key], data)
                metrics[f"distillation/{label}"] = Metric(aggregation=AggregationType.MEAN, value=v[mask].mean())
        if "overlap_count" in model_output:
            oc = no_padding_2_padding(model_output["overlap_count"], data)
            metrics["distillation/overlap_ratio"] = Metric(
                aggregation=AggregationType.MEAN, value=oc[mask].float().mean() / k
            )
        if "overlap_token_advantage" in model_output:
            ota = no_padding_2_padding(model_output["overlap_token_advantage"], data)
            metrics["distillation/overlap_token_advantage"] = Metric(
                aggregation=AggregationType.MEAN, value=ota[mask].mean()
            )
        for key in extra_keys:
            if key in model_output:
                v = no_padding_2_padding(model_output[key], data)
                metrics[f"distillation/{key}"] = Metric(aggregation=AggregationType.MEAN, value=v[mask].float().mean())
        return losses, metrics

    return fn


from simopd.topk_losses import OVERLAP_KEYS, PI_TAIL_KEYS, SHADOW_KEYS

# Every top-k arm reports the same two panels on top of its own keys: the shadow
# masks (what the other D-axis selectors would have chosen -- redundancy prediction
# #4, and the cheap way to tell whether two settings are actually different) and
# pi(S-bar) at several support widths (the headline theorem's quantity, and the K
# sweep for free). Both are pure functions of tensors the kernels already compute.
_PANELS = SHADOW_KEYS + PI_TAIL_KEYS + OVERLAP_KEYS

for _name, _extras in [
    # C axis. Was delegating to verl's compute_forward_kl_topk post-processor, which
    # reports the Eq.6/Eq.7 metrics but silently ignores any key it does not know --
    # so this arm, the one whose whole method is truncating to the teacher's top-k,
    # was the only one not reporting pi(S-bar), the quantity the headline theorem
    # says governs that truncation's error. _topk_registry_fn reports the same
    # Eq.6/Eq.7 metrics and the panels.
    ("lsm_topk_renorm", ()),
    ("qb_quantile_budget", ("qb_budget", "qb_captured_mass")),
    ("pl_rank_anchor", ("pl_rank_loss", "pl_value_anchor")),
    ("tip_select", ("d_selected_frac", "d_sampled_missing", "tip_entropy_mean")),
    ("selectkd_verify", ("d_selected_frac", "d_sampled_missing", "selectkd_tar")),
    ("teachability_select", ("d_selected_frac", "d_sampled_missing", "teach_compatibility")),
]:
    _extras = _extras + _PANELS
    # D-axis arms keep vanilla's k1 base loss, so -loss really is Delta-ell for them;
    # the C/E arms optimise a divergence or a rank loss and must not claim otherwise.
    _signal = "delta_ell" if _name in ("tip_select", "selectkd_verify", "teachability_select") else "loss"
    register_distillation_loss(DistillationLossSettings(names=[_name], use_topk=True))(
        _topk_registry_fn(*_extras, signal=_signal)
    )
