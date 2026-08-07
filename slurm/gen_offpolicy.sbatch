#!/usr/bin/env bash
#SBATCH -J simopd-gkdgen
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=60G
#SBATCH -t 10:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out
set -euo pipefail
cd $HOME/pythonProject/SimOPD; source simopd/bin/activate
export PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING VLLM_WORKER_MULTIPROC_METHOD=spawn
python scripts/gen_offpolicy.py --out $HOME/data/simopd_math/gkd_offpolicy.parquet
echo GKDGEN_DONE
