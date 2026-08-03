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
# The venv lives at ./simopd. An existing .venv is still honoured so the Cornell
# box, which has one and is mid-campaign, keeps working; new installs get simopd.
# SIMOPD_VENV overrides either way.
VENV=${SIMOPD_VENV:-}
if [ -z "$VENV" ]; then
    for _c in simopd .venv; do [ -d "$_c" ] && { VENV=$_c; break; }; done
fi
VENV=${VENV:-simopd}
source "$VENV/bin/activate"

# node-local scratch for ray tmp + checkpoints (node /tmp is only 15G)
export RAY_TMPDIR=/scratch/zz865/ray_tmp
mkdir -p "$RAY_TMPDIR" /scratch/zz865/simopd/ckpt

export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

nvidia-smi -L

bash scripts/run_opd_smoke.sh
