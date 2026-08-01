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
import verl.trainer.distillation.losses as vl
from verl.trainer.distillation.fsdp.losses import kl_divergence
from verl.workers.config import DistillationLossConfig

# "renorm" divides both sides by their top-k mass; "tailbucket" instead appends a
# single bucket carrying the leftover mass. The casefile pre-registers these as an
# internal ablation of the same arm, not as two separate arms.
SUPPORT_MODE = os.environ.get("SIMOPD_SUPPORT_MODE", "renorm")


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
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor

    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    loss_config: DistillationLossConfig = config.distillation_loss

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)

    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    if SUPPORT_MODE == "renorm":
        # Both sides become distributions over S; the tail is dropped, not modelled.
        stu = student_topk_log_probs - torch.logsumexp(student_topk_log_probs, dim=-1, keepdim=True)
        tch = teacher_topk_log_probs - torch.logsumexp(teacher_topk_log_probs, dim=-1, keepdim=True)
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
    distillation_losses = kl_divergence(log_q=tch, log_p=stu)

    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage = (-token_kl * overlap_mask).sum(dim=-1) / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


TOPK_DISPATCH = {"lsm_topk_renorm": compute_reverse_kl_topk}

_original_compute_topk_loss = vl.compute_topk_loss


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
    vl.compute_topk_loss = _dispatching_compute_topk_loss
