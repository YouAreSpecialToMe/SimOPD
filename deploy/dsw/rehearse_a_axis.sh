#!/usr/bin/env bash
# 3-step rehearsal for one A-axis arm on a 2-GPU pair. The unlock gate每臂要求的
# 彩排入口:走与 _lane.sh 相同的 arm.py env 纪律(赋值后 eval,拒绝臂在此死掉,
# 绝不 eval 空串跑成 vanilla),同一 seed 传参形状(data.seed / rollout.seed)。
#
#   bash deploy/dsw/rehearse_a_axis.sh a4_dagger_anneal 0,1
#   bash deploy/dsw/rehearse_a_axis.sh a5_aggrevate    2,3
#
# 通过判据(人工看日志,每臂彩排清单在 configs/arms.yaml 的 note 里):
#   a1/a3/a4  "[simopd] gkd_mix armed" + "cache loaded, 14467 prompts" +
#             realised lambda 打印贴合常数/调度;侧带 jsonl 出行
#   a4 额外   lam_target 逐步下降(schedule);distillation/gkd_* 指标行出现
#   a5        "[simopd] a5_aggrevate armed" + keys loaded + teacher_registry
#             published 打印;无重复 request_id 报错(复核 NEW-ISSUE 3);
#             哨兵剥离生效(teacher 侧无递归);outcome 计数和==n_seen
#   全部      3 步跑完,无 Traceback
set -euo pipefail
ARM=${1:?arm id}
GPUS=${2:?gpu pair, e.g. 0,1}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
D=/mgfs/shared/Group_GY/changhao/simopd_data
cd "$ROOT"
source simopd/bin/activate

_arm_env=$(python scripts/arm.py env "$ARM")
eval "$_arm_env"

export EXPERIMENT_NAME="rehearsal_${ARM}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE=offline                 # 彩排不进仪表板,指标看控制台/侧带
export DATA_DIR=$D/simopd_math
export TOTAL_TRAINING_STEPS=3
export MAX_RESPONSE_LENGTH=16384
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}   # campaign 钉的值
export PYTHONUNBUFFERED=1

LOG=$D/gen_offpolicy_logs/rehearsal_${ARM}.log
echo "rehearsal $ARM on GPUs $GPUS -> $LOG"
bash scripts/run_opd_baseline.sh \
    data.seed=0 \
    actor_rollout_ref.rollout.seed=0 \
    2>&1 | tee "$LOG" | grep -E '\[simopd\]|step:|Error|Traceback|gkd_|a5_' || true
echo "=== rehearsal $ARM finished; sideband: ==="
cat "/tmp/simopd_gkd_stats_rehearsal_${ARM}.jsonl" 2>/dev/null || echo "(no sideband rows)"
