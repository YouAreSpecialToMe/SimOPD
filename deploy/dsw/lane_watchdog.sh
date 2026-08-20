#!/usr/bin/env bash
# 单机(DSW)lane 看门狗 —— run_parallel 的 lane 死了不会自己回来。
#
#   nohup bash deploy/dsw/lane_watchdog.sh <launcher.log> > <wd.log> 2>&1 &
#   DRY=1 bash deploy/dsw/lane_watchdog.sh <launcher.log> --once
#
# 舰队脚本(deploy/dlc/corr_wave_fleet.sh)的 _launch_lane 会重试三次,DSW 这条路
# 上的 _lane.sh 一次都不重试:进程一退,那条 lane 就永久死了,GPU 空着到天亮也
# 没人知道。wave 20 四条臂跑在 8 卡机上,整夜无人看,所以补上这一层。
#
# 做法:从 launcher 日志里读 "lane N  GPUs [a,b] -> arm:seed" 的映射,逐条检查
# 对应的 lane 日志;判定为死就在它原来的 GPU 对上重新拉起 run_parallel(单 lane,
# 独立的 ray 临时目录),resume_mode=auto 会从最近的检查点接着跑。
#
# 判死的两个条件都要满足,宁可漏救不可误杀:
#   * lane 日志静默 >= STALE_MIN 分钟且步数 < STEPS;
#   * 那对 GPU 上没有任何计算进程 —— 这是关键一条:长上下文生成阶段日志可以安静
#     很久,但只要 GPU 上还有进程它就没死。(2026-08-20 我就把一次 40 分钟的
#     vLLM 初始化误判成挂死,险些白杀四条正常 lane。)
set -uo pipefail

LAUNCH_LOG=${1:?用法: lane_watchdog.sh <run_parallel 的 launcher 日志> [--once]}
ONCE=0; [ "${2:-}" = "--once" ] && ONCE=1
ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STALE_MIN=${STALE_MIN:-45}
PERIOD=${PERIOD:-3600}
STEPS=${STEPS:-250}
DRY=${DRY:-0}
VENV=${SIMOPD_VENV:-/mgfs/shared/Group_GY/changhao/SimOPD/simopd}
LOG_DIR=${LOG_DIR:-$ROOT/logs}

_say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# lane 号 -> "gpus arm:seed"
_map() { grep -aoE "lane [0-9]+  GPUs \[[0-9,]+\] -> [a-zA-Z0-9_.]+:[0-9]+" "$LAUNCH_LOG" | tail -20; }

_busy() {   # 这对 GPU 上还有计算进程吗
    local pids; pids=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
    [ -n "$pids" ]
}

_latest_lane_log() {  # 该 lane 号最新的日志(重启会生成新文件)
    ls -t "$LOG_DIR"/lane${1}_*.log 2>/dev/null | head -1
}

_check() {
    local line lane gpus spec arm f step m age
    while read -r line; do
        [ -n "$line" ] || continue
        lane=$(sed -E 's/lane ([0-9]+).*/\1/' <<< "$line")
        gpus=$(sed -E 's/.*\[([0-9,]+)\].*/\1/' <<< "$line")
        spec=$(sed -E 's/.*-> //' <<< "$line")
        arm=${spec%%:*}
        f=$(_latest_lane_log "$lane")
        if [ -z "$f" ]; then _say "lane$lane ($arm): 没有日志,跳过"; continue; fi
        step=$(grep -aoE "global_step:[0-9]+" "$f" | tail -1 | cut -d: -f2)
        m=$(stat -c %Y "$f" 2>/dev/null || echo 0); age=$(( ($(date +%s) - m) / 60 ))
        if [ "${step:-0}" -ge "$STEPS" ] 2>/dev/null; then _say "lane$lane ($arm): 跑满 $step,完成"; continue; fi
        if [ "$age" -lt "$STALE_MIN" ]; then _say "lane$lane ($arm): ok step=${step:-init} age=${age}m"; continue; fi
        if _busy "$gpus"; then
            _say "lane$lane ($arm): 静默 ${age}m 但 GPU [$gpus] 上仍有进程 —— 判为慢不判为死,不动"
            continue
        fi
        _say "lane$lane ($arm): 死了(静默 ${age}m,GPU [$gpus] 空)step=${step:-none} -> 重新拉起"
        if [ "$DRY" = 1 ]; then _say "lane$lane: DRY,不拉起"; continue; fi
        ( cd "$ROOT" && RAY_TMPDIR_TAG="wd${lane}_" LANES=1 GPU_LIST="$gpus" STEPS="$STEPS" \
            SIMOPD_VENV="$VENV" nohup bash deploy/dsw/run_parallel.sh "$spec" \
            >> "${LAUNCH_LOG%.log}_wd.log" 2>&1 & )
        _say "lane$lane ($arm): 已提交(GPU_LIST=$gpus, ray tag wd${lane}_)"
        sleep 60      # 错开,别让两条同时重建
    done <<< "$(_map)"
}

_say "单机看门狗启动:$LAUNCH_LOG STALE_MIN=$STALE_MIN PERIOD=${PERIOD}s DRY=$DRY"
while true; do
    _check
    [ "$ONCE" = 1 ] && break
    sleep "$PERIOD"
done
