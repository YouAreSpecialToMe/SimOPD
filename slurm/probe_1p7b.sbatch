#!/usr/bin/env bash
#SBATCH -J simopd-probe17b
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=150G
#SBATCH -t 24:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

# Does the screening tier belong at 1.7B <- 8B instead of 0.6B <- 1.7B?
#
# vanilla_s0 made 0.6B look like a poor screening vehicle: MATH500 pass@1 reached its
# final value at step 25, entropy collapsed by 40, and from step ~90 every rollout hit
# the 8192 cap. Verdicts would rest on a baseline that spends most of its life
# degenerating. 1.7B-Base is also the field's modal student (Demystifying, Rethinking,
# Teachability all use it) and Qwen3-8B is one of Demystifying's own teachers, so this
# cell is literature-anchored rather than invented.
#
# What stopped us was cost, and nobody can quote that number yet: our 465 s/it was an
# artifact of everything hitting the length cap, so the naive parameter-ratio estimate
# is meaningless. 50 steps answers all three questions at once:
#
#   s/it              -> is a 15-arm greedy campaign 140 GPU-hours or 1400?
#   val curve shape   -> is saturating at step 25 an OPD property or a 0.6B artifact?
#   clip_ratio, entropy -> does 1.7B enter Mode A too, i.e. is the 150-step horizon
#                          tier-specific?
#
# 50 is enough because on 0.6B all three signals appeared inside the first 50 steps.
# Everything else is held at the screening protocol -- same 8192 cap, same batch, lr,
# temperature and template -- so the curves are directly comparable to vanilla_s0.
# Changing the cap here would confound the one comparison this run exists to make.

set -uo pipefail
cd /home/zz865/pythonProject/SimOPD
source simopd/bin/activate

export VERL_USE_MODELSCOPE=False
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp}/ray_probe17b_$SLURM_JOB_ID
export CKPT_ROOT=${CKPT_ROOT:-/scratch/zz865/simopd/ckpt}
export WANDB_DIR=${WANDB_DIR:-/scratch/zz865/simopd/wandb}
mkdir -p "$RAY_TMPDIR" "$CKPT_ROOT" "$WANDB_DIR"

export STUDENT_MODEL=Qwen/Qwen3-1.7B-Base
export TEACHER_MODEL=Qwen/Qwen3-8B
export EXPERIMENT_NAME=probe_1p7b_from_8b
export TOTAL_TRAINING_STEPS=50
export TEST_FREQ=25            # val at 25 and 50: the two points that show the shape
export SAVE_FREQ=50
export MAX_CKPT_KEEP=1

# The 1.7B actor shares its GPU with the vLLM rollout engine. At 0.6 utilisation vLLM
# takes 48GB of an 80GB card, leaving ~32GB for weights, fp32 AdamW moments, master
# weights and gradients -- about 27GB before a single activation. 0.45 leaves headroom;
# a probe that dies on OOM answers none of the three questions.
python3 - <<'PY' || true
print("expected footprint: actor 1.7B bf16 3.4G + AdamW fp32 13.6G + master 6.8G "
      "+ grads 3.4G = 27.2G, plus activations; teacher 8B bf16 = 16G on its own GPU")
PY

nvidia-smi -L
bash scripts/run_opd_baseline.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288
echo "exit=$?"
