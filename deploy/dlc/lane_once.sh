#!/usr/bin/env bash
# 单条 lane 就地补起 —— 不重发整槽。
#
# 场景(2026-08-23 实发):一条 lane 三次尝试都失败退出了,而同槽另外三条正健康推进。
# 走 fleet_relaunch 会把那三条一起打回最近 ckpt(每条最多丢 25 步 ≈ 5 小时),代价远大于
# 这一条本身。这里用与 corr_wave_fleet.sh::_launch_lane 完全一致的环境把这一条单独拉起,
# 写同一个 lane 日志、存同一个 ckpt 目录、从最近 ckpt 续 —— 对盘上的产物而言毫无区别。
#
# 代价说清楚:它不在舰队监管里 —— 没有卡死看门(舰队 40 分钟无输出会杀了重试),跑满
# 250 后也不会自动把卡交给 eval。所以这是补漏工具,不是常规发车方式;常规仍走 lane 图 +
# fleet_relaunch。用完记得自己盯着它跑满。
#
# 用法(pod 上):bash deploy/dlc/lane_once.sh <arm> <gpus> [seed] [slot]
set -u
ARM=${1:?arm}; GPUS=${2:?gpus,如 6,7}; SEED=${3:-0}; SLOT=${4:-x}
ROOT=${ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
D=${SIMOPD_STORE:-/mgfs/shared/Group_GY/changhao/simopd_data}
export SIMOPD_STORE=$D   # arms.yaml 的资产路径靠它展开(arm.py env)
cd "$ROOT" || exit 1
source simopd/bin/activate
[ -f "$ROOT/simopd_env.sh" ] && source "$ROOT/simopd_env.sh"
export DATA_DIR=$D/simopd_math CKPT_ROOT=$D/ckpt
_pick=${SIMOPD_NET_IFACE:-}
if [ -z "$_pick" ]; then
    for _c in bond0 bond1 eth0; do [ -e "/sys/class/net/$_c" ] && { _pick=$_c; break; }; done
    [ -n "$_pick" ] || _pick=lo
fi
export NCCL_SOCKET_IFNAME=$_pick GLOO_SOCKET_IFNAME=$_pick NCCL_NET_PLUGIN=none
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} ROLLOUT_GPU_MEM_UTIL=0.45 PYTHONUNBUFFERED=1
_arm_env=$(python scripts/arm.py env "$ARM") || { echo "arm.py 拒绝了 $ARM"; exit 2; }
eval "$_arm_env"
export EXPERIMENT_NAME="${ARM}_s${SEED}_16k"
export CUDA_VISIBLE_DEVICES=$GPUS
export TOTAL_TRAINING_STEPS=250
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
_short() { local q="${1%/}"; q="${q%/hf}"; basename "$q"; }
export WANDB_RUN_GROUP="$(_short "${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}")__from__$(_short "${TEACHER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}")__s${SEED}"
export WANDB_TAGS="${ARM},N,seed${SEED},dlc_corr_wave1,slot${SLOT}"
echo "lane ${ARM}(单起): SIMOPD_STOP_IDS=${SIMOPD_STOP_IDS:-<unset>} SIMOPD_TERM_EVENT=${SIMOPD_TERM_EVENT:-0} GPUS=${GPUS} $(date)"
for attempt in 1 2 3; do
    bash scripts/run_opd_baseline.sh data.seed="$SEED" actor_rollout_ref.rollout.seed="$SEED" \
        && { echo "lane ${ARM}: 单起完成 $(date)"; exit 0; }
    echo "lane ${ARM}: 单起第 $attempt 次失败($(date));30s 后从 ckpt 续"
    sleep 30
done
echo "lane ${ARM}: 单起三次都失败 $(date)"; exit 1
