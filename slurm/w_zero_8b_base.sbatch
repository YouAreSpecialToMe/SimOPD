#!/usr/bin/env bash
#SBATCH -J simopd-w-zero
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=60G
#SBATCH -t 4:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out
# W-wave zero point: 8B-Base under OUR protocol (non-thinking, 8k) on the cell's val.
set -euo pipefail
cd $HOME/pythonProject/SimOPD; source simopd/bin/activate
export PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
EVALS=$HOME/data/simopd_evals
python scripts/eval_offline.py --model Qwen/Qwen3-8B-Base --run-id student_zero_8B-Base \
    --step 0 --benchmarks math500 --out-dir $EVALS
python scripts/eval_offline.py --model Qwen/Qwen3-8B-Base --run-id student_zero_8B-Base \
    --step 0 --benchmarks amc23 --n 32 --temperature 0.7 --top-p 0.95 --max-tokens 16384 --out-dir $EVALS
echo W_ZERO_DONE
