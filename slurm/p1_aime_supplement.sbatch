#!/usr/bin/env bash
#SBATCH -J simopd-p1aime
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=60G
#SBATCH -t 8:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out
# Annex val-set decision (design-thinking-cells.md): MATH500 likely saturates in the
# thinking regime, so the annex's primary val is AIME avg@32. These are the zero
# points that confirm (or refute) the saturation prediction and size the headroom.
set -euo pipefail
cd $HOME/pythonProject/SimOPD; source simopd/bin/activate
export PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
EVALS=$HOME/data/simopd_evals
python scripts/eval_offline.py --model Qwen/Qwen3-4B-Thinking-2507 --run-id teacher_ceiling_Qwen3-4B-Thinking-2507_think \
    --step -1 --benchmarks aime24,aime25 --think --max-tokens 16384 --n 32 --temperature 0.7 --out-dir $EVALS
python scripts/eval_offline.py --model Qwen/Qwen3-1.7B --run-id student_hybrid_1p7b_think \
    --step 0 --benchmarks aime24,aime25 --think --max-tokens 16384 --n 32 --temperature 0.7 --out-dir $EVALS
python scripts/eval_offline.py --model Qwen/Qwen3-8B --run-id teacher_ceiling_Qwen3-8B_think \
    --step -1 --benchmarks aime24,aime25 --think --max-tokens 16384 --n 32 --temperature 0.7 --out-dir $EVALS
echo P1AIME_DONE
