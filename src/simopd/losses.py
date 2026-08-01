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

import os

import torch
import verl.trainer.distillation.losses as _vl
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


def _delta_ell_metrics(losses, mask):
    """Flight-recorder panel for the per-token signal (METRICS.md section 3).

    Reported on Delta-ell = -loss so the numbers read in the paper's convention
    (positive = teacher assigns more mass than the student at this token).
    """
    delta = -losses[mask].detach().float()
    if delta.numel() == 0:
        return {}
    qs = torch.quantile(delta, torch.tensor(_QUANTILES, device=delta.device))
    metrics = {
        f"distillation/delta_ell_p{int(q * 100)}": Metric(aggregation=AggregationType.MEAN, value=v)
        for q, v in zip(_QUANTILES, qs, strict=True)
    }
    metrics["distillation/delta_ell_absmean"] = Metric(aggregation=AggregationType.MEAN, value=delta.abs().mean())
    return metrics


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
    log_mix = torch.logaddexp(
        torch.log(torch.tensor(alpha, device=student.device, dtype=student.dtype)) + student,
        torch.log(torch.tensor(1.0 - alpha, device=student.device, dtype=student.dtype)) + teacher,
    )
    losses = student - log_mix

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

    Note this arm deliberately breaks the equal-supervision budget rule: cutting
    supervised tokens IS its claim, so the ledger reports supervised_token_count
    rather than matching it.
    """
    student, teacher, mask = _unpack(model_output, data)
    losses = kl_penalty(logprob=student, ref_logprob=teacher, kl_penalty="k1")

    k = FIRST_SEGMENT_K
    positions = torch.arange(losses.shape[1], device=losses.device).unsqueeze(0)
    window = positions < k
    losses = losses * window

    metrics = {"distillation/abs_loss": Metric(aggregation=AggregationType.MEAN, value=losses[mask].abs().mean())}
    metrics.update(_delta_ell_metrics(losses, mask & window))
    metrics["distillation/firstseg_covered_frac"] = Metric(
        aggregation=AggregationType.MEAN,
        value=(mask & window).sum().float() / mask.sum().clamp_min(1).float(),
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


@register_distillation_loss(
    DistillationLossSettings(names=["lsm_topk_renorm"], use_topk=True)
)  # type: ignore[arg-type]
def lsm_truncated_reverse_kl(config, distillation_config, model_output, data):
    """C axis: truncated reverse KL on the teacher's top-k support (LSM 2603.25562).

    The kernel runs in the logits processor (simopd.topk_losses) and emits exactly
    the keys verl's top-k post-processor expects, so this delegates to it and
    inherits the overlap-ratio / mass diagnostics unchanged -- those are the
    Rethinking Eq.6/Eq.7 metrics the flight recorder already promises.
    """
    return _vl.compute_forward_kl_topk(config, distillation_config, model_output, data)


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
    filter so it composes with the other axes in the greedy rounds; the rescale
    in _reweight_kept makes the two equivalent in expectation.
    """
    student, teacher, mask = _unpack(model_output, data)
    losses = kl_penalty(logprob=student, ref_logprob=teacher, kl_penalty="k1")

    for key in ("token_level_scores", "token_level_rewards"):
        if key in data.keys():
            seq_score = data[key].sum(dim=-1)
            break
    else:
        raise KeyError(
            "k1_verified_only needs the verifier score in the actor micro-batch; "
            f"found none of token_level_scores/token_level_rewards in {sorted(data.keys())}"
        )
    keep_seq = seq_score > 0.5
    losses, keep = _reweight_kept(losses, mask, keep_seq)

    metrics = {"distillation/abs_loss": Metric(aggregation=AggregationType.MEAN, value=losses[mask].abs().mean())}
    metrics.update(_delta_ell_metrics(losses, keep))
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


def _topk_registry_fn(*extra_keys):
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
        metrics.update(_delta_ell_metrics(losses, mask))

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


for _name, _extras in [
    ("qb_quantile_budget", ("qb_budget", "qb_captured_mass")),
    ("pl_rank_anchor", ("pl_rank_loss", "pl_value_anchor")),
    ("tip_select", ("d_selected_frac", "tip_entropy_mean")),
    ("selectkd_verify", ("d_selected_frac", "selectkd_tar")),
    ("teachability_select", ("d_selected_frac", "teach_compatibility")),
]:
    register_distillation_loss(DistillationLossSettings(names=[_name], use_topk=True))(_topk_registry_fn(*_extras))
