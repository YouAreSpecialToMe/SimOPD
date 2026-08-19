#!/usr/bin/env bash
# 补充探针:tau=1.0 采样态停止行为(区分"通道被压死"vs"仅失去 argmax")
set -uo pipefail
ROOT=/mgfs/shared/Group_GY/changhao/SimOPD-exp
D=/mgfs/shared/Group_GY/changhao/simopd_data
OUT=$D/evals_v2probe
cd "$ROOT"
source /mgfs/shared/Group_GY/changhao/SimOPD/simopd/bin/activate
[ -f /mgfs/shared/Group_GY/changhao/SimOPD/simopd_env.sh ] && source /mgfs/shared/Group_GY/changhao/SimOPD/simopd_env.sh
export PYTHONUNBUFFERED=1
cell() { local g=$1 run=$2 st=$3
  CUDA_VISIBLE_DEVICES=$g python scripts/eval_offline.py \
    --model "$D/ckpt/simopd/${run}/global_step_${st}/actor/huggingface" \
    --benchmarks math500_sub100 --n 1 --temperature 1.0 --top-p 1.0 --max-tokens 16384 \
    --run-id "${run}_tau1" --step "$st" --seed 0 --out-dir "$OUT" \
    > "$OUT/cell_tau1_${run}_${st}.log" 2>&1
  echo "cell tau1 ${run}@${st} rc=$?"
}
cell 0 a1_gkd_mix0.5_s0_16k 125 &
cell 1 a3_offpolicy_s0_16k  250 &
wait
echo TAU1_DONE
