#!/usr/bin/env bash
# SimOPD vanilla OPD baseline — Demystifying-aligned protocol (docs/PROTOCOL-demystifying.md)
#   loss: reverse-KL sampled-token (k1) as per-token advantage Delta-l_t, PG formulation
#   data: nvidia/Nemotron-Cascade-RL-Math (verl parquet via scripts/prep_nemotron_math.py)
#   val:  MATH500 pass@1 (greedy)
# Screening default: 1.7B-Base <- 4B-Instruct-2507, 8k response cap, 150 steps.
# Anchor run:  MAX_RESPONSE_LENGTH=16384 TOTAL_TRAINING_STEPS=-1 \
#              EXPERIMENT_NAME=anchor_1.7b_from_4b2507
#              (same models as screening now -- the anchor cell IS the screening cell)
#
# NOTE the 150-step horizon below was calibrated on the 0.6B tier, which this file no
# longer uses. It is kept as an upper bound, not as a claim: the 50-step probe showed
# 1.7B still improving at step 50 (0.468 -> 0.604 -> 0.636) with clip_ratio only 0.27,
# so 1.7B reaches Mode A later than 0.6B did and 150 may be too short rather than too
# long. What actually decides each run is watch.py's early-stop rule, which fires on
# the run's own measured degeneration; the horizon is the ceiling it operates under.
# Re-derive it once the first full 1.7B vanilla run exists.

set -xeuo pipefail

# verl's console logger writes the per-step metric lines to stdout, which Python
# block-buffers when it is a file rather than a tty. tqdm goes to stderr and shows up
# immediately, so a run looks healthy while its metrics sit unflushed -- and anything
# still in that buffer is lost if the process is killed. That is how the 2026-07-31
# run produced 24 GPU-hours and zero metrics, and the early-stop rule added on
# 2026-08-04 makes it far more likely to bite, since enforcing it means SIGTERM to a
# run whose numbers are the evidence for stopping it.
export PYTHONUNBUFFERED=1

# Screening tier, decided 2026-08-04 on measurement rather than on the v3.1 plan's
# speed argument. Both changed:
#
#   student 0.6B-Base -> 1.7B-Base.  0.6B converged to MATH500 0.468, which is exactly
#     where 1.7B-Base STARTS untrained. Everything the 0.6B campaign would have
#     measured sits below the real student's zero point, and 0.6B saturates by step 25
#     while 1.7B is still climbing at 50 (0.468 -> 0.604 -> 0.636). 1.7B-Base is also
#     the only student both anchor papers use.
#
#   teacher 1.7B -> 4B-Instruct-2507.  Measured non-thinking MATH500 ceilings:
#     4B-Instruct-2507 0.896, 8B 0.792, 1.7B 0.702. The 2507 Instruct models are
#     Qwen's non-thinking-native line; Qwen3-8B is a hybrid whose strength is its
#     thinking mode, so our enable_thinking=False protocol costs it more than its
#     extra parameters buy. Under that constraint a bigger teacher is not a stronger
#     one. 1.7B-Base <- 4B-Instruct-2507 is also Demystifying's own off-the-shelf
#     cell, so the screening tier and the replication anchor are the same run.
#
# Off-the-shelf on purpose: 4 audited papers train a GRPO teacher, but every one of
# them also reports off-the-shelf teachers, so self-training is nobody's requirement
# -- and a teacher only we possess would be the one component of this audit that
# nobody else can reproduce.
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}

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
# 12288, not 20480: at the 1.7B tier the actor shares its GPU with the vLLM engine,
# and weights + fp32 AdamW moments + master weights + grads are ~27GB before a single
# activation. Verified on the 50-step probe at this value with no OOM.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

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

# verl keeps every checkpoint by default (max_actor_ckpt_to_keep: null). A 1.7B
# checkpoint is ~25GB (bf16 weights + fp32 optimizer moments + master weights), so
# keeping everything at SAVE_FREQ=50 is ~75GB per run and well over 1TB across a
# campaign -- it fills a DSW workspace volume mid-flight rather than at the start.
# Keep the newest two: one to resume from, one if the newest is torn.
# (The tier change from 0.6B roughly tripled this; MAX_CKPT_KEEP=1 halves it again.)
max_ckpt_keep=${MAX_CKPT_KEEP:-2}

# 'hf_model' is added to save_contents below. Without it verl writes only FSDP shards
# (model_world_size_*.pt) plus a huggingface/ directory holding the config and
# tokenizer but NO weights -- which nothing can load. Every downstream metric reads a
# checkpoint: AMC23 avg@32, the per-arm transfer column, the diversity panel, and the
# per-problem MATH500 artifact McNemar is computed on. Discovering this after a
# campaign means the training is done and not one verdict can be issued.
# Costs ~3.4GB per checkpoint at 1.7B bf16, against ~25GB of shards and optimizer
# state already being written -- about 14%, for the difference between a checkpoint
# and an artifact.

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
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.45} \
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
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
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
