#!/usr/bin/env bash
#SBATCH -J simopd-base
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=150G
#SBATCH -t 72:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

# Vanilla OPD baseline (Demystifying-aligned protocol). Defaults = 300-step screen.
# Shakedown: sbatch --export=ALL,TOTAL_TRAINING_STEPS=15,TEST_FREQ=15,EXPERIMENT_NAME=shakedown slurm/baseline.sbatch

set -euo pipefail

PROJ_DIR=$HOME/pythonProject/SimOPD
cd "$PROJ_DIR"
# One venv, ./simopd, on every machine. SIMOPD_VENV overrides it.
VENV=${SIMOPD_VENV:-simopd}
source "$VENV/bin/activate"

export RAY_TMPDIR=/scratch/zz865/ray_tmp
mkdir -p "$RAY_TMPDIR" /scratch/zz865/simopd/ckpt /scratch/zz865/simopd/evals

# our arm losses register into verl's registry via sitecustomize on PYTHONPATH
export PYTHONPATH="$PROJ_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

# wandb: online mode verified from compute node 2026-07-31; credential in ~/.netrc
export WANDB_DIR=/scratch/zz865/simopd/wandb
mkdir -p "$WANDB_DIR"

nvidia-smi -L

bash scripts/run_opd_baseline.sh
