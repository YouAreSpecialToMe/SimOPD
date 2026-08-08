#!/usr/bin/env bash
# SimOPD vanilla OPD baseline — Demystifying-aligned protocol (docs/PROTOCOL-demystifying.md)
#   loss: reverse-KL sampled-token (k1) as per-token advantage Delta-l_t, PG formulation
#   data: nvidia/Nemotron-Cascade-RL-Math (verl parquet via scripts/prep_nemotron_math.py)
#   val:  MATH500 pass@1 (greedy)
# Screening default: 1.7B-Base <- 4B-Instruct-2507, 16,384 response cap, 250 steps (protocol 3.8 / plan sec 4).
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

# ModelScope, resurrection three. Round one: the image exported it and ${VAR:-False}
# is a default, not an override. Round two: simopd_env.sh carries an explicit
# VLLM_USE_MODELSCOPE=True from a long-finished download errand. Round three: lanes
# died at vLLM init because SOME launch chain reached this script with the flag
# truthy and no modelscope in the venv. run_parallel force-assigns both flags -- but
# only launches that pass through run_parallel. This is the innermost shell every
# variant must traverse, so the assignment lives here too: derived from the ONE knob,
# assigned unconditionally, never defaulted.
if [ "${SIMOPD_USE_MODELSCOPE:-0}" = "1" ]; then
    export VERL_USE_MODELSCOPE=True VLLM_USE_MODELSCOPE=True
else
    export VERL_USE_MODELSCOPE=False VLLM_USE_MODELSCOPE=False
fi

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
# j1 cell (audit r5 addendum): KDRL-faithful GRPO needs real groups; every other
# arm keeps n=1. Group count rides TRAIN_BATCH_SIZE (prompts), sequences = bs*n.
rollout_n=${ROLLOUT_N:-1}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-16384} # PROTOCOL sec 3.8 (2026-08-07): campaign
                                                  # cap = 16,384 -- Demystifying (the protocol
                                                  # anchor) trains here, TM too; the old 8192
                                                  # was the v3.1 screening economy, retired.
                                                  # Runs at 8192 form the pilot batch.
# 12288, not 20480: at the 1.7B tier the actor shares its GPU with the vLLM engine,
# and weights + fp32 AdamW moments + master weights + grads are ~27GB before a single
# activation. Verified on the 50-step probe at this value with no OOM.
# 17408 = 1024 prompt + 16384 response: dynamic-bsz micro-batches must hold ONE full
# sequence or verl asserts. 12288 was the 8k-era value, memory-verified on the 50-step
# probe; 17408 is +42% activation budget on the same 1.7B statics -- re-verify with a
# single-lane probe BEFORE the fleet inherits this default.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-17408}

actor_lr=${ACTOR_LR:-1e-6}
# 150, not the v3.1 plan's 300 (pre-registration amended 2026-08-04 on measurement).
# vanilla_s0 reached its final MATH500 pass@1 by step 25 and was in Mode A from ~90;
# steps 150-300 cost 19 hours in which every rollout hits the length cap and pass@1
# does not move. 150 still contains the whole story an arm can tell -- plateau (25),
# entropy collapse (40), Mode A onset (90) and 60 steps of established Mode A --
# which the F axis needs, since preventing Mode A is the thing it is judged on.
total_training_steps=${TOTAL_TRAINING_STEPS:-250}  # protocol horizon (plan sec 4); was 150 pre-amendment  # -1 = run total_epochs
total_epochs=${TOTAL_EPOCHS:-3}
test_freq=${TEST_FREQ:-25}
save_freq=${SAVE_FREQ:-25}   # PROTOCOL 3.7 density; -1 was the pre-suite default

# verl keeps every checkpoint by default (max_actor_ckpt_to_keep: null). A 1.7B
# checkpoint is ~25GB (bf16 weights + fp32 optimizer moments + master weights), so
# keeping everything at SAVE_FREQ=50 is ~75GB per run and well over 1TB across a
# campaign -- it fills a DSW workspace volume mid-flight rather than at the start.
# Keep the newest two: one to resume from, one if the newest is torn.
# (The tier change from 0.6B roughly tripled this; MAX_CKPT_KEEP=1 halves it again.)
max_ckpt_keep=${MAX_CKPT_KEEP:--1}   # -1 = keep ALL (PROTOCOL 3.7: the suite
                                     # sweeps every saved step; keep=2 made the
                                     # 11-point curve physically impossible --
                                     # audit 2026-08-07 config-C1). Disk math:
                                     # ~11 x 28.4GB ~= 310GB/run, user-accepted.

# 'hf_model' is added to save_contents below. Without it verl writes only FSDP shards
# (model_world_size_*.pt) plus a huggingface/ directory holding the config and
# tokenizer but NO weights -- which nothing can load. Every downstream metric reads a
# checkpoint: AMC23 avg@32, the per-arm transfer column, the diversity panel, and the
# per-problem MATH500 artifact McNemar is computed on. Discovering this after a
# campaign means the training is done and not one verdict can be issued.
# Costs ~3.4GB per checkpoint at 1.7B bf16, against ~25GB of shards and optimizer
# state already being written -- about 14%, for the difference between a checkpoint
# and an artifact.

# vLLM shares this GPU with the actor -- the teacher gets its own card at 0.85, being
# pure inference with nothing to share with. gpu_memory_utilization is a fraction of
# TOTAL memory reserved at engine init, not a usage cap, so it trades directly against
# the actor's 25.3GB of static state at 1.7B (3.2 bf16 weights + 6.3 fp32 master +
# 12.7 AdamW moments + 3.2 grads) before a single activation.
#
#   0.45 -> vLLM 36.0GB, 61.3/80 used, 18.7 free
#   0.55 -> vLLM 44.0GB, 69.3/80 used, 10.7 free
#   0.65 -> vLLM 52.0GB, 77.3/80 used,  2.7 free   -- too close
#
# 0.55 from 2026-08-05. free_cache_engine=True (verl default) sleeps the engine during
# the actor update, so the two do not peak together; what stays resident regardless is
# the optimizer state and vLLM's cuda graphs, which rollout.yaml notes cannot be
# offloaded. The gain is specific: under Mode A responses grow toward the 8k cap, the
# KV cache holds fewer concurrent sequences, and throughput falls faster than length
# rises -- 8GB more cache buys concurrency back exactly where it is being lost.
#
# NOT numerically neutral, so it belongs in the fingerprint. Rollout sampling uses one
# engine-level seed (rollout.seed=42), not per-request seeds, so cache size -> batch
# composition -> both the RNG stream and the reduction order. Runs that must be
# comparable have to share it: vanilla s0/s1/s2 are the noise floor, and a config
# difference among them would inflate it and make every downstream verdict falsely
# conservative.
#
# DEFAULT 0.45 for the whole screening campaign (audit 2026-08-06). It was 0.55 here
# while campaign.sh exported 0.45 -- which protected exactly one entrance. Any launch
# not through campaign.sh (cornell's campaign.sbatch, a manual run_parallel, f1's
# +100-step resume) got 0.55: the resume is refused by the fingerprint, but a FRESH
# arm at 0.55 starts quietly and becomes an incomparable batch member that nothing
# refuses. Same lesson as ModelScope: the protocol value lives at the innermost
# layer, not at whichever launcher remembered to export it. 0.55 returns as the
# anchor-round default, where the whole batch starts together.
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.45}

# --- memory placement and the hub, both fixed at the innermost layer (2026-08-08) ---
#
# 1. FSDP offload during rollout. free_cache_engine sleeps the vLLM engine for the
#    actor update, but the reverse handoff is what broke: at 16k the actor's static
#    state plus activations leave the address space too fragmented for vLLM's
#    wake_up to re-map its cache, and the run dies mid-training with "CUDA Error:
#    out of memory at cumem_allocator.cpp" (e2/e3, steps 36-51, 08-08). Offloading
#    params and optimizer to host memory between updates gives the mapping room.
#    Purely WHERE tensors live -- the arithmetic, the RNG stream and the fingerprint
#    are untouched, so this does not fork the batch. Costs some step time.
# 2. expandable_segments fights the fragmentation itself rather than its symptom;
#    same neutrality argument.
# 3. HF_HUB_OFFLINE. Twelve lanes starting at once each ask hf-mirror.com to list the
#    teacher repo, and the mirror answers 429 -- three c4 seeds died at step 0 for a
#    rate limit that had nothing to do with the arm (08-08). The weights are already
#    on disk; going offline makes the run depend on local files instead of somebody
#    else's uptime, which is what a pre-registered protocol wants anyway. If a model
#    is genuinely missing, the failure is immediate and legible instead of a race.
fsdp_param_offload=${FSDP_PARAM_OFFLOAD:-True}
fsdp_optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD:-True}
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

use_remove_padding=${USE_REMOVE_PADDING:-True}     # needs flash-attn; set False before FA build lands

project_name=${PROJECT_NAME:-simopd}
experiment_name=${EXPERIMENT_NAME:-vanilla_$(basename $STUDENT_MODEL)_from_$(basename $TEACHER_MODEL)}
logger=${LOGGER:-'["console","wandb"]'}

data_dir=${DATA_DIR:-$HOME/data/simopd_math}
train_files="['$data_dir/${TRAIN_FILE_BASENAME:-train.parquet}']"
val_files="['$data_dir/math500.parquet']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# ---- resume ---------------------------------------------------------------
# We have no resume code of our own; verl's resume_mode defaults to "auto", which
# silently continues from the newest checkpoint in default_local_dir. That is right
# for preemption -- a requeued DLC or slurm job must resume without a human -- and
# wrong for everything else, because it cannot tell a preempted run from a relaunch
# with a changed config. The second case splices two configs into one curve and says
# nothing, which for a pre-registered audit is a contaminated arm that still looks
# like an arm. StatefulDataLoader restores the data order, so a legitimate resume is
# faithful; the risk is entirely about resuming the wrong thing.
#
# So: record what defines the run, and refuse when the record contradicts it. Only
# the fields that make two runs incomparable go in. total_training_steps does not --
# extending a horizon is a legitimate resume, and the noise floor may well ask for it.
#
# Two fields the audit (2026-08-06) found missing:
#   extra=$*      the hydra overrides _lane.sh appends (data.seed, rollout.seed, and
#                 anything future). The old seed=${SEED:-} field read a variable the
#                 caller never exports, so it hashed as empty on every run.
#   simopd=...    every SIMOPD_* env knob. These ARE arm definitions -- retention,
#                 budget, margin, segment K, drop fraction -- read by the kernels
#                 straight from env, so a changed knob otherwise resumed silently.
#                 SHADOW and PI_TAIL_WIDTHS are excluded: they alter which metrics are
#                 emitted, never what the loss computes, and blocking a resume over a
#                 diagnostics toggle would be the check crying wolf.
ckpt_dir=${CKPT_ROOT:-/scratch/zz865/simopd/ckpt}/${project_name}/${experiment_name}
resume_mode=auto
# On a fresh run this is the step-0 anchor of the arm's curve and a wiring check
# worth 75 minutes before committing 14 hours: 0.474 says the student and the
# chat template loaded correctly, 0.05 says they did not. The resume branch below
# turns it off, where it buys neither.
# Off by default (2026-08-05). A full MATH500 pass@1 is ~75 minutes and its step-0
# value is the SAME for every arm -- one base model, greedy, one val set -- so
# 15 arms spend ~19 GPU-hours re-measuring one constant. It is recorded instead:
# scripts/preflight.py STEP0_MATH500 = 0.468, the verl-path value, so every curve
# stays on one measurement pipeline (eval_offline.py reports 0.4740 for the same
# model, and mixing the two would put each arm's first point on a different one).
# The wiring check it also performed is now scripts/preflight.py, in ~20 seconds.
val_before_train=${VAL_BEFORE_TRAIN:-False}
# g3_kdrl (audit r5 addendum): KDRL's objective is J_GRPO - beta*KL, which is
# exactly verl's use_task_rewards=True combine (policy_loss + distill*coef). Off
# for every other arm; the coefficient default 1.0 is verl's and only bites when
# task rewards are on.
use_task_rewards=${USE_TASK_REWARDS:-False}
distillation_loss_coef=${DISTILLATION_LOSS_COEF:-1.0}
fingerprint=$(printf '%s\n' \
    "student=$STUDENT_MODEL" "teacher=$TEACHER_MODEL" \
    "loss=$distillation_loss_mode" "pg=$use_policy_gradient" "topk=$distillation_topk" \
    "taskrw=$use_task_rewards" "dcoef=$distillation_loss_coef" "rolloutn=$rollout_n" \
    "ngpus=${NGPUS_PER_NODE}" "tws=${TEACHER_WORLD_SIZE}" \
    "arm=${ARM_ARGS[*]-}" "extra=$*" \
    "simopd=$(env | LC_ALL=C grep '^SIMOPD_' | grep -vE '^SIMOPD_(SHADOW|PI_TAIL_WIDTHS|LANE_TAG)=' | LC_ALL=C sort | tr '\n' ' ')" \
    "bs=$train_batch_size" "mini=$ppo_mini_batch_size" "lr=$actor_lr" \
    "prompt=$max_prompt_length" "resp=$max_response_length" \
    "data=$train_files" "pad=$use_remove_padding" \
    "vllmmem=$rollout_gpu_mem_util" "tokpergpu=$ppo_max_token_len_per_gpu" \
    | sha1sum | cut -c1-12)
fp_file="$ckpt_dir/simopd_fingerprint.txt"
latest_file="$ckpt_dir/latest_checkpointed_iteration.txt"

if [ -f "$latest_file" ]; then
    _at=$(cat "$latest_file" 2>/dev/null)
    if [ "${RESUME:-}" = fresh ]; then
        echo "FATAL: RESUME=fresh, but $ckpt_dir already holds a checkpoint at step $_at." >&2
        echo "       Starting over while it sits there leaves old and new checkpoints in one" >&2
        echo "       directory, and the next auto-resume would pick whichever number is" >&2
        echo "       larger -- possibly the run you meant to discard. Move it aside first:" >&2
        echo "         mv $ckpt_dir ${ckpt_dir}.superseded" >&2
        exit 1
    fi
    if [ -f "$fp_file" ] && [ "$(cat "$fp_file")" != "$fingerprint" ]; then
        echo "FATAL: $ckpt_dir holds a checkpoint at step $_at from a DIFFERENT config." >&2
        echo "         on disk $(cat "$fp_file")   now $fingerprint" >&2
        echo "       Resuming would splice two configs into one curve with nothing in the" >&2
        echo "       logs to say so. Pick one:" >&2
        echo "         mv $ckpt_dir ${ckpt_dir}.superseded    # start this config cleanly" >&2
        echo "         RESUME=force ...                        # you know they are compatible" >&2
        [ "${RESUME:-}" = force ] || exit 1
        echo "       RESUME=force given -- continuing, and recording the new fingerprint." >&2
    fi
    # A resume has a second, independent reason to skip validation: verl runs
    # val_before_train AFTER _load_checkpoint and unconditionally (ray_trainer 1404 then
    # 1413), so it would re-measure the step it just restored. save_freq is a multiple
    # of test_freq here, so that step already has a validation logged against it.
    echo "=== resuming $experiment_name from step $_at (fingerprint $fingerprint)"
    [ -f "$fp_file" ] || echo "    NOTE: no fingerprint on disk (checkpoint predates this check)." \
        "Cannot verify the config matches; verify by hand if this arm is going in the paper."
else
    echo "=== fresh start: $experiment_name (fingerprint $fingerprint)"
fi
mkdir -p "$ckpt_dir" && printf '%s\n' "$fingerprint" > "$fp_file"

# The gate val_before_train used to be, without the 75 minutes of generation.
python3 "$(dirname "$0")/preflight.py" \
    --student "$STUDENT_MODEL" --teacher "$TEACHER_MODEL" \
    --data "$data_dir/${TRAIN_FILE_BASENAME:-train.parquet}" \
    --val "$data_dir/math500.parquet" \
    --loss "$distillation_loss_mode" \
    --max-prompt-length "$max_prompt_length"

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
    $(: "PROTOCOL sec 3.6 (2026-08-07): these five rode verl defaults unregistered. "
       "Values UNCHANGED -- pinned so a verl bump cannot drift them silently; verl "
       "is not in the campaign pin set.") \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.999]" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${fsdp_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${fsdp_optimizer_offload} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    $(: "1.0/1.0 is PROTOCOL (sec 3.5, 2026-08-07): every audited paper trains at "
       "tau=1 -- GKD gamma=1 explicitly -- and on-policy unbiasedness needs full "
       "support; the official 0.7/0.8 card values are deployment inference params") \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    trainer.balance_batch=True \
    trainer.logger="${logger}" \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.max_actor_ckpt_to_keep=${max_ckpt_keep} \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    trainer.default_local_dir=${ckpt_dir} \
    trainer.resume_mode=${resume_mode} \
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
    distillation.distillation_loss.use_task_rewards=${use_task_rewards} \
    distillation.distillation_loss.distillation_loss_coef=${distillation_loss_coef} \
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient} \
    "${ARM_ARGS[@]+"${ARM_ARGS[@]}"}" \
    "$@"
