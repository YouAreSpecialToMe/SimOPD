#!/usr/bin/env bash
# v2 定性探针:贪心 math500 全文 dump(峰值 ckpt vs 最新 ckpt),6 cell 并行。
# 复刻 docs/late-training-collapse.md 的诊断协议:greedy mean@1, 16384 预算,
# stop 契约 auto -> 各 ckpt 自己的 v2 pin(151643,151645)。
set -uo pipefail
ROOT=/mgfs/shared/Group_GY/changhao/SimOPD-exp
D=/mgfs/shared/Group_GY/changhao/simopd_data
OUT=$D/evals_v2probe
mkdir -p "$OUT"
cd "$ROOT"
source /mgfs/shared/Group_GY/changhao/SimOPD/simopd/bin/activate
[ -f /mgfs/shared/Group_GY/changhao/SimOPD/simopd_env.sh ] && source /mgfs/shared/Group_GY/changhao/SimOPD/simopd_env.sh
export PYTHONUNBUFFERED=1
cell() { # gpu run step
  local g=$1 run=$2 st=$3
  CUDA_VISIBLE_DEVICES=$g python scripts/eval_offline.py \
    --model "$D/ckpt/simopd/${run}/global_step_${st}" \
    --benchmarks math500 --n 1 --temperature 0.0 --max-tokens 16384 \
    --run-id "$run" --step "$st" --seed 0 --out-dir "$OUT" \
    > "$OUT/cell_${run}_${st}.log" 2>&1
  echo "cell ${run}@${st} rc=$?"
}
cell 0 a1_gkd_mix0.5_s0_16k 50  &
cell 1 a1_gkd_mix0.5_s0_16k 125 &
cell 2 a3_offpolicy_s0_16k  50  &
cell 3 a3_offpolicy_s0_16k  250 &
cell 4 h6_gen_sched_s0_16k  25  &
cell 5 h6_gen_sched_s0_16k  175 &
wait
echo V2PROBE_DONE
ls -la "$OUT"/*.parquet 2>/dev/null | tail -8
