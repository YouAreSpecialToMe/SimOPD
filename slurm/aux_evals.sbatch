#!/usr/bin/env bash
#SBATCH -J simopd-aux
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=60G
#SBATCH -t 8:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out
# Transfer column for one arm, then the informativeness/AUROC backfill (g2's premise;
# Rethinking's 0.73-0.75 is the number we compare against directly).
#   sbatch --export=ALL,CKPT=...,RUN_ID=...,STEP=... slurm/aux_evals.sbatch
set -euo pipefail
cd $HOME/pythonProject/SimOPD
source "${SIMOPD_VENV:-simopd}/bin/activate"
export PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
export NLTK_DISABLE_IMPORT_SECURITY=1     # IFEval's nltk vs the venv-inside-repo layout
[ -e "$CKPT" ] || { echo "FATAL: $CKPT not on this node"; exit 1; }
# eval_transfer takes (run_id, step) and finds the checkpoint itself under
# CKPT_ROOT -- the first submission passed the ckpt PATH as run_id and died in its
# first minute. The audit rule, again: read the callee's signature, not one's memory
# of it.
CKPT_ROOT=${CKPT_ROOT:-/scratch/zz865/simopd/ckpt} bash scripts/eval_transfer.sh "$RUN_ID" "$STEP"
python scripts/informativeness.py --model Qwen/Qwen3-1.7B-Base --run-id student_base --step 0
python scripts/informativeness.py --model "$CKPT" --run-id "$RUN_ID" --step "$STEP"
echo AUX_DONE
