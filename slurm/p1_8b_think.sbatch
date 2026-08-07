#!/usr/bin/env bash
#SBATCH -J simopd-p1-8bt
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=60G
#SBATCH -t 6:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out
# Completes the 2x2 that carries a pre-registered prediction: the non-thinking
# inversion (4B-2507 > 8B) is explained by the hybrid's confiscated scratchpad, so
# 8B's think-minus-nothink delta should be large and positive, likely lifting it back
# above 0.896. If the inversion vanishes under thinking, our D6 mechanism story is
# independently confirmed; if it persists, that is its own section.
set -euo pipefail
cd $HOME/pythonProject/SimOPD; source simopd/bin/activate
export PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
python scripts/eval_offline.py --model Qwen/Qwen3-8B --run-id teacher_ceiling_Qwen3-8B_think \
    --step -1 --benchmarks math500 --think --max-tokens 16384 --out-dir $HOME/data/simopd_evals
echo P18BT_DONE
