#!/usr/bin/env bash
# SimOPD vanilla OPD baseline — Demystifying-aligned protocol (docs/PROTOCOL-demystifying.md)
#   loss: reverse-KL sampled-token (k1) as per-token advantage Delta-l_t, PG formulation
#   data: nvidia/Nemotron-Cascade-RL-Math (verl parquet via scripts/prep_nemotron_math.py)
#   val:  MATH500 pass@1 (greedy)
# Screening default: 0.6B-Base <- 1.7B, 8k response cap, 150 steps (see below).
# Anchor run:  STUDENT_MODEL=Qwen/Qwen3-1.7B-Base TEACHER_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
#              MAX_RESPONSE_LENGTH=16384 TOTAL_TRAINING_STEPS=-1 EXPERIMENT_NAME=anchor_1.7b_from_4b2507
#
# The 150-step horizon is calibrated on the 0.6B SCREENING tier only, and the anchor
# above must keep overriding it. Whether a 0.6B-Base student saturating by step 25 is
# an OPD property or an artifact of a small student against a 2.8x teacher is exactly
# what the anchor (1.7B-Base <- 4B) settles -- and no audited paper answers it, since
# 8 of the 10 do not report a step count at all.

set -xeuo pipefail

# verl's console logger writes the per-step metric lines to stdout, which Python
# block-buffers when it is a file rather than a tty. tqdm goes to stderr and shows up
# immediately, so a run looks healthy while its metrics sit unflushed -- and anything
# still in that buffer is lost if the process is killed. That is how the 2026-07-31
# run produced 24 GPU-hours and zero metrics, and the early-stop rule added on
# 2026-08-04 makes it far more likely to bite, since enforcing it means SIGTERM to a
# run whose numbers are the evidence for stopping it.
export PYTHONUNBUFFERED=1

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-0.6B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-1.7B}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}

# k1_rec, not stock k1: mathematically identical, but it carries the Delta-ell panel
# METRICS.md requires on EVERY run. Defaulting to stock k1 meant f2_hard_clip and
# a2_coldstart (which set other knobs, not the loss mode) silently differed from
# vanilla in two ways instead of one. Requires src/ on PYTHONPATH; it fails loudly.
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1_rec}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-32}

# Arm knobs that are absent from the vanilla protocol: only pass them when set,
# so a run's config hash records exactly the one deviation that defines its arm.
ARM_ARGS=()
[ -n "${LOSS_MAX_CLAMP:-}" ] && ARM_ARGS+=(distillation.distillation_loss.loss_max_clamp="${LOSS_MAX_CLAMP}")
[ -n "${LOG_PROB_MIN_CLAMP:-}" ] && ARM_ARGS+=(distillation.distillation_loss.log_prob_min_clamp="${LOG_PROB_MIN_CLAMP}")

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}   # = train batch: single epoch per rollout batch
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}  # v3.1 screening cap; anchor/final: 16384
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}

actor_lr=${ACTOR_LR:-1e-6}
# 150, not the v3.1 plan's 300 (pre-registration amended 2026-08-04 on measurement).
# vanilla_s0 reached its final MATH500 pass@1 by step 25 and was in Mode A from ~90;
# steps 150-300 cost 19 hours in which every rollout hits the length cap and pass@1
# does not move. 150 still contains the whole story an arm can tell -- plateau (25),
# entropy collapse (40), Mode A onset (90) and 60 steps of established Mode A --
# which the F axis needs, since preventing Mode A is the thing it is judged on.
total_training_steps=${TOTAL_TRAINING_STEPS:-150}  # -1 = run total_epochs
total_epochs=${TOTAL_EPOCHS:-3}
test_freq=${TEST_FREQ:-25}
save_freq=${SAVE_FREQ:--1}

# verl keeps every checkpoint by default (max_actor_ckpt_to_keep: null). At ~8-9GB
# per checkpoint for 0.6B (weights + fp32 optimizer moments + master weights), 300
# steps at SAVE_FREQ=50 is ~50GB per run and ~850GB for a 17-run campaign, which
# would fill a DSW workspace volume mid-flight. Keep the newest two: one to resume
# from, one as a fallback if the newest is torn.
max_ckpt_keep=${MAX_CKPT_KEEP:-2}

use_remove_padding=${USE_REMOVE_PADDING:-True}     # needs flash-attn; set False before FA build lands

project_name=${PROJECT_NAME:-simopd}
experiment_name=${EXPERIMENT_NAME:-vanilla_$(basename $STUDENT_MODEL)_from_$(basename $TEACHER_MODEL)}
logger=${LOGGER:-'["console","wandb"]'}

data_dir=${DATA_DIR:-$HOME/data/simopd_math}
train_files="['$data_dir/${TRAIN_FILE_BASENAME:-train.parquet}']"
val_files="['$data_dir/math500.parquet']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    trainer.balance_batch=True \
    trainer.logger="${logger}" \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.max_actor_ckpt_to_keep=${max_ckpt_keep} \
    trainer.default_local_dir=${CKPT_ROOT:-/scratch/zz865/simopd/ckpt}/${project_name}/${experiment_name} \
    distillation.enabled=True \
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.85 \
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens} \
    distillation.distillation_loss.loss_mode=${distillation_loss_mode} \
    distillation.distillation_loss.topk=${distillation_topk} \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient} \
    "${ARM_ARGS[@]+"${ARM_ARGS[@]}"}" \
    "$@"
