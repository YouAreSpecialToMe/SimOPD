#!/usr/bin/env bash
#SBATCH -J simopd-p1bc
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=120G
#SBATCH -t 10:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out
# P1b+P1c: the upper teacher ladder. 14B on one card; 32B needs tp=2 (64GB bf16 --
# quantization is banned, logprobs are the audited object) and doubles as its memory
# smoke test.
set -euo pipefail
cd $HOME/pythonProject/SimOPD; source simopd/bin/activate
export PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
EVALS=$HOME/data/simopd_evals
python scripts/eval_offline.py --model Qwen/Qwen3-14B --run-id teacher_ceiling_Qwen3-14B \
    --step -1 --benchmarks math500 --out-dir $EVALS
python scripts/eval_offline.py --model Qwen/Qwen3-32B --run-id teacher_ceiling_Qwen3-32B \
    --step -1 --benchmarks math500 --tp 2 --out-dir $EVALS
echo P1BC_DONE
