#!/usr/bin/env bash
# H 轴 5-lane DLC 任务:一个 8 卡 worker,四槽布局把五臂装满 —— 两条最快臂
# (h7 512 帽 / h8 2048 帽)在同一对卡上串跑,墙钟与慢臂对齐,零卡位浪费:
#
#   slot0  h6_gen_sched    GPUs 0,1   (递进帽 128->16384)
#   slot1  h9_prune_adapt -> a2_coldstart  GPUs 2,3  (自适应预算;h9 最快先完,
#          腾出的对卡链跑 A 基线 a2 —— v2 单 seed 波的搭载,见 R5 附录契约法令)
#   slot2  h10_task_subset GPUs 4,5   (50% 任务子集,全深)
#   slot3  h7_gen512 -> h8_gen2048  GPUs 6,7  (固定帽,串行链)
#
# 血统:deploy/dlc/a_axis_fleet.sh 的 11 修全部继承(NCCL/GLOO 接口按存在性
# 自选不信任平台注入、显式 pid wait(裸 wait 会等 tee 进程替换而死锁)、零存档
# 禁报成功、远程中止开关、SEED 参数化、seed!=0 只等标记)。与 A 版的差异:
#   Phase P -> 资产断言(H 轴无预计算;只需 a5 的 .dry 键文件与 h10 子集 parquet)
#   Phase R -> 只认 .OK 标记,缺失即挂起待查:彩排是 gpu193 调试节点的职责
#             (用户指定),DLC 上重彩排徒耗重启配额
#
# 提交:本机运行打印控制台卡片;容器内(rank env 存在)即载荷。

set -uo pipefail

ROOT=/mgfs/shared/Group_GY/changhao/SimOPD
D=${SIMOPD_STORE:-/mgfs/shared/Group_GY/changhao/simopd_data}
export SIMOPD_STORE=$D   # arms.yaml 的资产路径写成 $SIMOPD_STORE/...,arm.py env 在此展开
ARMS_ALL=(h6_gen_sched h9_prune_adapt h10_task_subset h7_gen512 h8_gen2048)
SEED=${SEED:-0}

_abort_check() {
    if [ -f "$D/a_axis/fleet_abort_h_s${SEED}" ]; then
        echo "abort marker $D/a_axis/fleet_abort_h_s${SEED} found; exiting 17 for AIMaster restart"
        exit 17
    fi
}

# ---------------------------------------------------------------- submitter --
if [ -z "${MLP_ROLE_INDEX:-}${MLP_WORKER_RACK_RANK_INDEX:-}${DLC_JOB_ID:-}" ]; then
    cat <<CARD
========= DLC 控制台表单(H 轴 v2 + a2 搭载,每 seed 一作业,1 worker × 8 GPU) =========
  任务名称   simopd-h-axis-v2-s${SEED}
  节点数量   1
  单节点GPU  8      CPU 64      内存 512Gi(照 A 轴成功任务规格)
  镜像/资源组/数据集挂载:照抄 simopd-a-axis-4lane 成功表单(挂载须含 /mgfs)
  执行命令:
    SEED=${SEED} bash $ROOT/deploy/dlc/h_axis_fleet.sh
  前置:$D/a_axis/ 下须有五个 rehearsal_h*.OK(彩排在调试节点做,不在 DLC 重跑);
        v2 契约冒烟由 A 舰队 Phase R 与调试节点彩排承担,建议 A-s0 彩排 PASS 后再交本卡
==========================================================================
本机不是 DLC 容器(无 rank env),以上为提交卡片;容器内执行同一脚本即载荷。
CARD
    exit 0
fi

# ------------------------------------------------------------------ payload --
_rank=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
if [ "${_rank}" != "0" ]; then
    while true; do _abort_check; echo "rank ${_rank}: single-worker job, idling ($(date))"; sleep 600; done
fi
cd "$ROOT"
LOGD=$D/a_axis
mkdir -p "$LOGD"
exec > >(tee -a "$LOGD/hfleet_$(date +%Y%m%d_%H%M%S).log") 2>&1
echo "== h_axis_fleet on $(hostname), seed ${SEED}, git $(git log --oneline -1 2>/dev/null | head -1)"
git config --global --add safe.directory "$ROOT" 2>/dev/null || true
nvidia-smi -L | head -8
source simopd/bin/activate
[ -f "$ROOT/simopd_env.sh" ] && source "$ROOT/simopd_env.sh"

export DATA_DIR=$D/simopd_math
export CKPT_ROOT=$D/ckpt
# NCCL/GLOO 接口:A 舰队 bug 11 终版原样(平台预注入 bond1 是幽灵,按存在性选)。
_ifs=$(ls /sys/class/net 2>/dev/null | tr "\n" " ")
_pick=${SIMOPD_NET_IFACE:-}
if [ -z "$_pick" ]; then
    for _c in bond0 bond1 eth0; do
        [ -e "/sys/class/net/$_c" ] && { _pick=$_c; break; }
    done
    [ -n "$_pick" ] || _pick=lo
fi
export NCCL_SOCKET_IFNAME=$_pick GLOO_SOCKET_IFNAME=$_pick
export NCCL_NET_PLUGIN=none
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
echo "== net ifaces: picked $_pick (pod has: $_ifs) NCCL_NET_PLUGIN=none"
export ROLLOUT_GPU_MEM_UTIL=0.45
export PYTHONUNBUFFERED=1

# 限域 lint(seed-0 专责,A 舰队同则)。
if [ "$SEED" = 0 ]; then
LINT_LOG=$LOGD/h_arm_lint.log
python scripts/arm_lint.py > "$LINT_LOG" 2>&1 || true
BAD=$(grep 'PROBLEM' "$LINT_LOG" \
      | grep -E 'h6_gen_sched|h7_gen512|h8_gen2048|h9_prune_adapt|h10_task_subset|a2_coldstart' \
      | grep -vE 'campaign\.tsv row|verdict\.py ARMS|appears 2x in campaign\.tsv' || true)
      # 第三个排除类(2026-08-19):a2_s0 在 campaign.tsv 有两行历史死行(wave-3
      # Cornell 占位 + wave-5 m-fleet 已烤行,均被不存在的 needs= 守卫锁死,不可
      # 认领)。DLC 舰队不读该清单,行重复是 m-fleet 侧的归档卫生问题,与
      # "缺 row"同类,不该拦发射。真正的发射安全由 env 冲突/指纹检查把关。
if [ -n "$BAD" ]; then
    while true; do
        _abort_check
        echo "H ARM_LINT (scoped) FAILED -- lanes NOT launched ($(date)):"
        echo "$BAD"
        sleep 600
    done
fi
echo "== arm_lint: scoped gate clean (full report: $LINT_LOG)"
else
echo "== arm_lint: skipped for seed ${SEED} (registry git-pinned; seed-0 gates)"
fi

# --------------------------------------------------------- Phase P' 资产断言 --
# a2 搭载:stage-1 SFT 产物(merged HF 导出)与其余量训练集也须在位。
for f in "$D/gkd_offpolicy.parquet.dry" "$D/simopd_math/train_sub50.parquet" \
         "$D/ckpt/coldstart_sft/hf/config.json" "$D/simopd_math/train_coldstart_remainder.parquet"; do
    if [ ! -f "$f" ]; then
        while true; do _abort_check; echo "ASSET MISSING: $f -- lanes NOT launched ($(date))"; sleep 600; done
    fi
done
echo "== Phase P': assets present (.dry keys + train_sub50 + coldstart_sft/hf + remainder)"

# ------------------------------------------------------------ Phase R 标记门 --
_missing=""
for ARM in "${ARMS_ALL[@]}"; do
    [ -f "$LOGD/rehearsal_${ARM}.OK" ] || _missing="$_missing $ARM"
done
if [ -n "$_missing" ]; then
    while true; do
        _abort_check
        echo "REHEARSAL MARKERS MISSING:$_missing -- rehearse on gpu193 (deploy/dsw/rehearse_a_axis.sh), then touch the .OK ($(date))"
        sleep 600
    done
fi
echo "== Phase R: 5/5 rehearsal markers found (rehearsed on the debug node)"

# -------------------------------------------------------------- Phase L 发射 --
echo "== Phase L: 5 lanes (4 slots, h7->h8 chained) x 250 steps, seed ${SEED}"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE=offline

_launch_lane() {  # ARM PAIR  -- 有界重试,run_opd_baseline 从 ckpt 自恢复
    local ARM=$1 PAIR=$2 attempt
    for attempt in 1 2 3; do
        (
            set -e
            _arm_env=$(python scripts/arm.py env "$ARM")
            eval "$_arm_env"
            export EXPERIMENT_NAME="${ARM}_s${SEED}_16k"
            export CUDA_VISIBLE_DEVICES=$PAIR
            export TOTAL_TRAINING_STEPS=250
            export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
            export WANDB_RUN_GROUP="Qwen3-1.7B-Base__from__Qwen3-4B-Instruct-2507__s${SEED}"
            # stop2 = 双停契约 v2 波次标记;a2 是搭载的 A 基线,轴标签如实标 A。
            _ax=H; [ "$ARM" = a2_coldstart ] && _ax=A
            export WANDB_TAGS="${ARM},${_ax},seed${SEED},dlc_h_axis,stop2"
            bash scripts/run_opd_baseline.sh \
                data.seed="$SEED" \
                actor_rollout_ref.rollout.seed="$SEED"
        ) && { echo "lane ${ARM}: attempt $attempt completed"; return 0; }
        echo "lane ${ARM}: attempt $attempt failed ($(date)); resume-retry"
        sleep 30
    done
    return 1
}

_lpids=()
( _launch_lane h6_gen_sched    0,1 ) > "$LOGD/lane_h6_gen_sched_s${SEED}.log"    2>&1 & _lpids+=($!)
( _launch_lane h9_prune_adapt 2,3 > "$LOGD/lane_h9_prune_adapt_s${SEED}.log" 2>&1
  _launch_lane a2_coldstart   2,3 > "$LOGD/lane_a2_coldstart_s${SEED}.log"   2>&1 ) & _lpids+=($!)
( _launch_lane h10_task_subset 4,5 ) > "$LOGD/lane_h10_task_subset_s${SEED}.log" 2>&1 & _lpids+=($!)
( _launch_lane h7_gen512 6,7 > "$LOGD/lane_h7_gen512_s${SEED}.log" 2>&1
  _launch_lane h8_gen2048 6,7 > "$LOGD/lane_h8_gen2048_s${SEED}.log" 2>&1 ) & _lpids+=($!)
echo "lanes launched: h6->0,1 h9(->a2 chain)->2,3 h10->4,5 h7(->h8 chain)->6,7 (logs $LOGD/lane_*_s${SEED}.log)"
# 显式 pid,绝不裸 wait(tee 进程替换死锁,A 舰队 bug 10)。
wait "${_lpids[@]}"
echo "== Phase L done"
ok=0
for ARM in "${ARMS_ALL[@]}" a2_coldstart; do
    ck=$(ls -d "$D/ckpt/simopd/${ARM}_s${SEED}_16k/global_step_"* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    echo "  ${ARM}_s${SEED}_16k: last checkpoint step ${ck:-NONE}"
    [ -n "$ck" ] && ok=$((ok+1))
done
if [ "$ok" -eq 0 ]; then
    while true; do
        _abort_check
        echo "ALL LANES DEAD, zero checkpoints banked -- NOT declaring success; inspect $LOGD/lane_h*_s${SEED}.log ($(date))"
        sleep 600
    done
fi
echo "H_AXIS_FLEET_DONE"
