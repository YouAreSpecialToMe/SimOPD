#!/usr/bin/env bash
#SBATCH -J simopd-smoke
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=150G
#SBATCH -t 03:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

set -euo pipefail

PROJ_DIR=$HOME/pythonProject/SimOPD
cd "$PROJ_DIR"
source .venv/bin/activate

# node-local scratch for ray tmp + checkpoints (node /tmp is only 15G)
export RAY_TMPDIR=/scratch/zz865/ray_tmp
mkdir -p "$RAY_TMPDIR" /scratch/zz865/simopd/ckpt

export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

nvidia-smi -L

bash scripts/run_opd_smoke.sh
