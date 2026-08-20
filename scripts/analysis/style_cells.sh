#!/usr/bin/env bash
# 风格四方对照 cell:教师 / 基座 / c2@250 / c4@250,贪心 math500 16k,带全文
set -uo pipefail
ROOT=/mgfs/shared/Group_GY/changhao/SimOPD-exp
D=/mgfs/shared/Group_GY/changhao/simopd_data
OUT=$D/evals_style
mkdir -p "$OUT"
cd "$ROOT"
source /mgfs/shared/Group_GY/changhao/SimOPD/simopd/bin/activate
[ -f /mgfs/shared/Group_GY/changhao/SimOPD/simopd_env.sh ] && source /mgfs/shared/Group_GY/changhao/SimOPD/simopd_env.sh
export PYTHONUNBUFFERED=1
cell() { local g=$1 model=$2 rid=$3 st=$4
  CUDA_VISIBLE_DEVICES=$g python scripts/eval_offline.py \
    --model "$model" --benchmarks math500 --n 1 --temperature 0.0 --max-tokens 16384 \
    --run-id "$rid" --step "$st" --seed 0 --out-dir "$OUT" \
    > "$OUT/cell_${rid}.log" 2>&1
  echo "cell $rid rc=$?"
}
( sleep 0;   cell 0 "Qwen/Qwen3-4B-Instruct-2507" teacher_greedy_text -1 ) &
( sleep 40;  cell 1 "Qwen/Qwen3-1.7B-Base" base_greedy_text -1 ) &
( sleep 80;  cell 2 "$D/ckpt/simopd/c2_quantile_budget_s0_16k/global_step_250/actor/huggingface" c2_greedy_text 250 ) &
( sleep 120; cell 3 "$D/ckpt/simopd/c4_pi_tail_budget_s0_16k/global_step_250/actor/huggingface" c4_greedy_text 250 ) &
wait
echo STYLE_CELLS_DONE
