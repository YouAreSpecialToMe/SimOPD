#!/usr/bin/env bash
#SBATCH -J simopd-owed
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 6:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

# The two GPUs here are worth more on what DSW should NOT duplicate. Both items below
# are owed W1 deliverables that need a GPU but no training, so they finish while the
# DSW campaign is still on its first wave.
#
#   1. the transfer column's end-to-end path, which has never once run. Its scoring
#      halves are verified (HumanEval+ 163/164, MBPP+ 376/378, twice, deterministic)
#      but vLLM generate -> official harness -> parquet has not been, because every
#      GPU attempt was blocked by the account's 2-GPU cap. Scored on the probe's real
#      1.7B checkpoint, so it doubles as that tier's step-50 transfer numbers.
#
#   2. AMC23 teacher ceilings. BENCHMARKS §4 marked the MATH500 ceilings done on
#      2026-08-04 and left AMC23 owed -- it is the Gap Recovery Rate denominator for
#      the secondary decision metric, and verdict.py cannot fill that column without it.
#
# Inference only: one GPU, so it leaves the second free.

set -uo pipefail
cd /home/zz865/pythonProject/SimOPD
source simopd/bin/activate

export VERL_USE_MODELSCOPE=False VLLM_USE_MODELSCOPE=False
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
OUT=${EVAL_OUT_DIR:-/home/zz865/data/simopd_evals}
mkdir -p "$OUT"

echo "=== 1. transfer column, end to end ==="
CKPT=$(ls -d /scratch/zz865/simopd/ckpt/simopd/probe_1p7b_from_8b/global_step_*/actor 2>/dev/null |
       sort -t_ -k3 -n | tail -1)
if [ -n "$CKPT" ]; then
    echo "checkpoint: $CKPT"
    # parallel 4, not the default: evalplus forks a worker per unit test and this node
    # also hosts training lanes. Under contention the same canonical solutions scored
    # 160/164 instead of 163/164 -- a trick would look like it destroyed code ability.
    time python scripts/eval_offline.py \
        --model "$CKPT" --benchmarks humanevalplus,mbppplus,ifeval \
        --run-id probe_1p7b_from_8b --step 50 \
        --temperature 0 --n 1 --max-tokens 2048 --parallel 4 --out-dir "$OUT"
else
    echo "no probe checkpoint on this node's /scratch; skipping (it is node-local)" >&2
fi

echo
echo "=== 2. AMC23 teacher ceilings (GRR denominator, secondary metric) ==="
# avg@32 at tau=0.7/top-p=0.95, the protocol's checkpoint-eval sampling, so the
# denominator is measured the same way the numerator will be.
for M in Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B Qwen/Qwen3-1.7B; do
    echo "--- $M"
    python scripts/eval_offline.py --model "$M" --benchmarks amc23 \
        --run-id "teacher_ceiling_$(basename "$M")" --step -1 \
        --n 32 --temperature 0.7 --top-p 0.95 --max-tokens 8192 --out-dir "$OUT" || true
done

echo
echo "=== summary ==="
grep -h "^\[" /home/zz865/pythonProject/SimOPD/logs/simopd-owed-"$SLURM_JOB_ID".out 2>/dev/null | sed 's|-> /.*||'
echo done
