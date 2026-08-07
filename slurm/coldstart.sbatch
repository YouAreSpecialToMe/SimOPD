#!/usr/bin/env bash
#SBATCH -J simopd-coldstart
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=150G
#SBATCH -t 12:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

# Arm a2 stages 1-2: teacher generations -> SFT checkpoint.
# Stage 3 is an ordinary OPD run pointed at that checkpoint:
#   sbatch --export=ALL,RUNS="a2_coldstart:0" slurm/campaign.sbatch
# which the registry wires up via STUDENT_MODEL and DATA_DIR.

set -euo pipefail

PROJ_DIR=$HOME/pythonProject/SimOPD
cd "$PROJ_DIR"
# One venv, ./simopd, on every machine. SIMOPD_VENV overrides it.
VENV=${SIMOPD_VENV:-simopd}
source "$VENV/bin/activate"

export PYTHONPATH="$PROJ_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
export RAY_TMPDIR=/scratch/zz865/ray_tmp
# vLLM v1 forks its EngineCore; anything that touched CUDA in this process first --
# gen_coldstart does -- dies with "Cannot re-initialize CUDA in forked subprocess".
# The 27-second death of the first submission, in one line. Spawn is the fix vLLM's
# own error message prescribes.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
SFT_CKPT=${SFT_CKPT:-/scratch/zz865/simopd/ckpt/coldstart_sft}
mkdir -p "$RAY_TMPDIR" "$SFT_CKPT"

# Tier defaults corrected 2026-08-06: this file predates the 2026-08-04 tier change
# and still said 0.6B-Base <- 1.7B. Submitting it as-was would have generated the
# cold-start data with the WRONG teacher and SFT'd the WRONG student -- an entire
# wasted stage plus an a2 arm quietly trained off-tier. The audit habit that caught
# it: read the assembled file before the first submit, not the diff that last touched it.
TEACHER=${TEACHER:-Qwen/Qwen3-4B-Instruct-2507}
STUDENT=${STUDENT:-Qwen/Qwen3-1.7B-Base}
DATA_DIR=${DATA_DIR:-$HOME/data/simopd_math}

echo "=== stage 1: teacher generations on the reserved prompt slice ==="
python scripts/gen_coldstart_data.py \
    --teacher "$TEACHER" \
    --train-parquet "$DATA_DIR/train.parquet" \
    --out-dir "$DATA_DIR" \
    --n-prompts "${N_PROMPTS:-3000}" \
    --n-samples "${N_SAMPLES:-4}"

echo "=== stage 2: SFT the student on them ==="
# Same max length as the screening protocol so the warm-up cannot be credited to
# a different context budget.
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.sft_trainer \
    data.train_files="$DATA_DIR/coldstart_sft.parquet" \
    data.messages_key=messages \
    data.max_length="${SFT_MAX_LEN:-8192}" \
    data.train_batch_size="${SFT_BATCH:-128}" \
    model.path="$STUDENT" \
    trainer.default_local_dir="$SFT_CKPT" \
    trainer.total_epochs="${SFT_EPOCHS:-2}" \
    "$@"

echo "SFT checkpoint: $SFT_CKPT"
echo "COLDSTART_DONE"
