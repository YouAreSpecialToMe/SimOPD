"""Fixed scaling recipe: quantile support, SKL, credit clip, SelecTKD, failures.

The direct update and the driver-side verifier gate are deliberately separate.
See docs/SCALING-RECIPE.md for the exact gradient and normalization contract.
No verl imports at module scope: the numerical core is CPU-testable.
"""

import functools
import math
import os

import torch

MODE = "composed_skl_selectkd"


def prediction_mask(teacher, response_mask, total):
    """Mask logits predicting responses, not input-token rows or the final dummy.

    Matches pinned verl no_padding_2_padding: [end-response_len-1:end-1].
    Refuse unsupported layouts instead of computing quantiles over prompts.
    """
    if response_mask.is_nested:
        lengths = [int(x.sum()) for x in response_mask.unbind()]
    else:
        lengths = response_mask.sum(-1).tolist()
    offsets = teacher.offsets().tolist()
    if len(offsets) != len(lengths) + 1 or offsets[-1] != total:
        raise ValueError("composed loss requires an unsliced, packed teacher payload (SP=1)")
    mask = torch.zeros((1, total), dtype=torch.bool, device=teacher.device)
    for start, end, length in zip(offsets[:-1], offsets[1:], lengths):
        length = int(length)
        if length < 0 or length >= end - start:
            raise ValueError("each sequence needs a nonempty prompt and valid response length")
        mask[:, end - length - 1:end - 1] = True
    return mask


def adaptive_support(student_lp, teacher_lp, valid, budget):
    """Original Q margin=max(p,q), response-only packed-microbatch quantile."""
    if not math.isfinite(budget) or not 1 <= budget <= teacher_lp.shape[-1]:
        raise ValueError("budget must be finite and within the candidate width")
    with torch.no_grad():
        margin = torch.maximum(student_lp.detach().exp(), teacher_lp.detach().exp())
        pop = margin[valid].flatten()
        keep = torch.zeros_like(margin, dtype=torch.bool)
        if pop.numel():
            tau = torch.quantile(pop.float(), 1.0 - budget / margin.shape[-1])
            keep = margin >= tau
        keep[..., 0] = True
    return keep


def clipped_skl(student_lp, teacher_lp, keep, alpha=0.1, clip=10.0):
    """Direct reverse SKL on renormalized support, with clipped local credit.

    p=student, q=teacher, m=alpha*p+(1-alpha)*q.
    c=log(m/p)+alpha*p/m-alpha is zero at p=q. Without clipping,
    -sum stopgrad(p*c)*log(p) has exactly the gradient of KL(p||m),
    including m's student dependence. Clipping c defines a modified update,
    not scalar-loss clamping. The returned forward value remains raw SKL;
    diagnostics separately report the clipped credit.
    """
    if not 0 <= alpha < 1 or not math.isfinite(clip) or clip <= 0:
        raise ValueError("require 0 <= alpha < 1 and finite positive credit clip")
    if student_lp.shape != teacher_lp.shape or keep.shape != student_lp.shape:
        raise ValueError("support tensors must have the same shape")
    if not keep.any(-1).all():
        raise ValueError("empty vocabulary support")
    # Preserve float64 for gradient tests; never run probability math in bf16.
    dtype = torch.float64 if student_lp.dtype == torch.float64 else torch.float32
    p0, q0 = student_lp.to(dtype), teacher_lp.detach().to(dtype)
    if not torch.isfinite(p0[keep]).all() or not torch.isfinite(q0[keep]).all():
        raise ValueError("non-finite retained log probability")
    pn = p0.masked_fill(~keep, -torch.inf)
    qn = q0.masked_fill(~keep, -torch.inf)
    pn = pn - torch.logsumexp(pn, -1, keepdim=True)
    qn = qn - torch.logsumexp(qn, -1, keepdim=True)
    # Off-support entries are zeroed before arithmetic to avoid inf-inf/0*inf.
    pn = torch.where(keep, pn, 0.0)
    qn = torch.where(keep, qn, 0.0)
    prob = pn.exp() * keep
    mix = qn if alpha == 0 else torch.logaddexp(pn + math.log(alpha), qn + math.log1p(-alpha))
    raw = (prob * (pn - mix)).sum(-1)
    # Avoid 0 * inf for the alpha=0 (ordinary reverse-KL) boundary.
    correction = alpha * (pn - mix).exp() - alpha if alpha else 0.0
    credit = (mix - pn + correction).detach()
    credit = torch.where(keep, credit, 0.0)
    bounded = credit.clamp(-clip, clip)
    surrogate = -(prob.detach() * bounded * pn).sum(-1)
    losses = raw.detach() + (surrogate - surrogate.detach())
    diagnostics = {
        "recipe_skl": raw.detach(),
        "recipe_clip_frac": ((credit.abs() > clip) & keep).sum(-1) / keep.sum(-1),
        "recipe_credit_abs": (prob.detach() * credit.abs()).sum(-1),
        "recipe_clipped_credit_abs": (prob.detach() * bounded.abs()).sum(-1),
    }
    return losses, diagnostics


def selectkd_weights(teacher_ids, student_top1, k=5, epsilon=0.01):
    if not 1 <= k <= teacher_ids.shape[-1] or not 0 < epsilon <= 1:
        raise ValueError("invalid SelecTKD k / epsilon")
    accepted = (teacher_ids[..., :k] == student_top1.unsqueeze(-1)).any(-1)
    return torch.where(accepted, 1.0, epsilon), accepted.float()


def compute_composed_topk(student_logits, teacher_topk_log_probs, teacher_topk_ids,
                          config, data_format, data=None):
    from simopd import topk_losses as T
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size

    if data is None or "response_mask" not in data:
        raise ValueError("composed loss needs response metadata")
    if get_ulysses_sequence_parallel_world_size() != 1:
        raise ValueError("composed quantile scope currently requires SP=1")
    lse, q, ids, _, _ = T._prepare_streaming(
        student_logits, teacher_topk_log_probs, teacher_topk_ids, config, want_sampled=False)
    p = student_logits.gather(-1, ids).float() - lse.unsqueeze(-1)
    valid = prediction_mask(teacher_topk_log_probs, data["response_mask"], p.shape[1])
    keep = adaptive_support(p, q, valid, float(os.environ.get("SIMOPD_QB_TARGET_BUDGET", "8")))
    loss, out = clipped_skl(p, q, keep,
                           float(os.environ.get("SIMOPD_COMPOSED_ALPHA", "0.1")),
                           float(os.environ.get("SIMOPD_COMPOSED_CLIP", "10")))
    weight, accepted = selectkd_weights(
        ids, student_logits.detach().argmax(-1),
        int(os.environ.get("SIMOPD_SELECTKD_K", "5")),
        float(os.environ.get("SIMOPD_SELECTKD_BETA", "0.01")))
    out.update(distillation_losses=loss * weight,
               recipe_selectkd_weight=weight, recipe_selectkd_accept=accepted,
               recipe_support_size=keep.sum(-1).float(),
               recipe_teacher_mass=(q.detach().exp() * keep).sum(-1),
               recipe_student_mass=(p.detach().exp() * keep).sum(-1))
    return out


def failure_metadata(scores, response_mask):
    """Read binary verifier rewards, never normalized GRPO advantages."""
    if scores.shape != response_mask.shape or not torch.isfinite(scores).all():
        raise ValueError("missing/malformed verifier scores")
    if not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("response mask must be binary")
    totals = (scores * response_mask).sum(-1)
    if not ((totals == 0) | (totals == 1)).all():
        raise ValueError("composed recipe expects binary 0/1 math-verifier outcomes")
    counts = response_mask.sum(-1)
    if not (counts > 0).all():
        raise ValueError("empty response is a rollout error, not a verifier failure")
    failed = totals == 0
    n_all, n_failed = counts.sum(), counts[failed].sum()
    scale = float(n_all / n_failed) if n_failed > 0 else 0.0
    return failed, scale, int(n_failed)


def apply_failure_gate(losses, data):
    """One global failure normalization, unchanged by DP/microbatch splitting."""
    if "simopd_failure" not in data or "simopd_failure_scale" not in data:
        raise RuntimeError("composed verifier hook missing; refusing ungated training")
    failed = data["simopd_failure"].to(losses.device)
    scale = data["simopd_failure_scale"].to(losses.device)
    if failed.shape != losses.shape[:1] or scale.shape != failed.shape:
        raise ValueError("invalid composed failure metadata shape")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("global empty batch must be skipped at the trainer")
    return losses * failed[:, None] * scale[:, None]


def install_trainer(cls=None):
    """Attach the verifier gate before DP splitting; skip globally empty updates."""
    if os.environ.get("SIMOPD_COMPOSED", "0") != "1":
        return
    if cls is None:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        cls = RayPPOTrainer
    original = cls._update_actor
    if getattr(original, "_simopd_composed", False):
        return

    @functools.wraps(original)
    def update(self, batch):
        actor = self.config.actor_rollout_ref.actor
        rollout = self.config.actor_rollout_ref.rollout
        loss_cfg = self.distillation_config.distillation_loss
        if (loss_cfg.loss_mode != MODE or loss_cfg.use_policy_gradient
                or loss_cfg.use_task_rewards or loss_cfg.loss_max_clamp is not None):
            raise ValueError("composed recipe requires its direct mode and internal credit clipping")
        if (actor.ppo_epochs != 1 or rollout.n != 1 or actor.loss_agg_mode != "token-mean"
                or len(batch.batch["response_mask"]) != actor.ppo_mini_batch_size):
            raise ValueError("composed global normalization requires one full-batch update, n=1, token-mean")
        if "token_level_scores" not in batch.batch:
            raise ValueError("verifier scores missing; cannot select failure responses")
        failed, scale, n_tokens = failure_metadata(
            batch.batch["token_level_scores"], batch.batch["response_mask"])
        if not n_tokens:
            from verl import DataProto
            return DataProto.from_single_dict(data={}, meta_info={"metrics": {
                "actor/recipe/skipped_all_success": 1.0,
                "actor/recipe/failure_fraction": 0.0,
                "actor/recipe/failure_tokens": 0,
            }})
        batch.batch["simopd_failure"] = failed
        batch.batch["simopd_failure_scale"] = torch.full_like(failed, scale, dtype=torch.float32)
        result = original(self, batch)
        result.meta_info["metrics"].update({
            "actor/recipe/skipped_all_success": 0.0,
            "actor/recipe/failure_fraction": float(failed.float().mean()),
            "actor/recipe/failure_tokens": n_tokens,
            "actor/recipe/failure_normalization": scale,
        })
        return result

    update._simopd_composed = True
    cls._update_actor = update
