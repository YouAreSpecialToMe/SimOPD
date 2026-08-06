#!/usr/bin/env bash
#SBATCH -J simopd-eval
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=60G
#SBATCH -t 6:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out
# Ledger feed for one finished arm:  MATH500 pass@1 (per-problem, verdict.py's input)
# then AMC23 avg@32. Usage:
#   sbatch --export=ALL,CKPT=<.../global_step_N/actor>,RUN_ID=<name>,STEP=<N> slurm/eval_arm.sbatch
set -euo pipefail
cd $HOME/pythonProject/SimOPD
source "${SIMOPD_VENV:-simopd}/bin/activate"
export PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
# Home is the shared filesystem here; the script's default out-dir is node-local
# /scratch, which the login node -- and verdict.py -- cannot see.
EVALS=${EVALS:-$HOME/data/simopd_evals}
[ -e "$CKPT" ] || { echo "FATAL: $CKPT not on this node (checkpoints are node-local)"; exit 1; }
python scripts/eval_offline.py --model "$CKPT" --run-id "$RUN_ID" --step "$STEP" \
    --benchmarks math500 --out-dir "$EVALS"
python scripts/eval_offline.py --model "$CKPT" --run-id "$RUN_ID" --step "$STEP" \
    --benchmarks amc23 --n 32 --temperature 0.7 --out-dir "$EVALS"
echo EVAL_ARM_DONE
