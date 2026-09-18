#!/usr/bin/env bash
# 一次性:LA 00:30 若分区还有空闲节点就提前开窗,窗口末端仍是当天 LA 08:30。
#
# 用户 2026-09-16 定:"今晚洛杉矶 12 点半之后没人用就启动,跑到明天上午八点半"。
# 常规策略是 01:00–08:30(night_window.sh 的 START_MIN/END_MIN),这里只把**起点**
# 提前 30 分钟,而且附加"有空闲节点"这个条件 —— 分区被同事占满时就不抢,等 01:00
# 常规开窗后由 Slurm 排队。
#
# 做法:写 gates/force_window_until = 当天 LA 08:30 的 epoch。守卫和看门狗都每轮现读
# 这个文件,`now < fu` 即判定为窗口内,过了 08:30 自动失效回落到常规规则。
# 干完就把自己从 crontab 里摘掉 —— 这是一次性的,不是新策略。

set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:${PATH:-}
ROOT=$SIMOPD_ROOT
STORE=$SIMOPD_STORE
LOG="$STORE/scheduler/open_early.log"
TAG=open_early_once.sh

say() { echo "[$(date -u '+%m-%d %H:%M:%SUTC') / $(TZ=America/Los_Angeles date '+%H:%M %Z')] $*" >> "$LOG"; }

# 先摘 crontab:无论后面成功与否都只试这一次,免得明晚又提前开窗。
crontab -l 2>/dev/null | grep -v "$TAG" | crontab - && say "已从 crontab 摘除自己"

idle=$(sinfo ${SLURM_PARTITION:+-p "$SLURM_PARTITION"} -h -t idle -o '%D' 2>/dev/null | awk '{s+=$1} END{print s+0}')
if [ "${idle:-0}" -lt 1 ]; then
    say "分区无空闲节点(idle=$idle),不提前开窗;01:00 常规窗口照常"
    exit 0
fi

END=$(TZ=America/Los_Angeles date -d 'today 08:30' +%s)
now=$(date +%s)
if [ "$now" -ge "$END" ]; then
    say "已过当天 08:30,不动"
    exit 0
fi

mkdir -p "$STORE/gates"
echo "$END" > "$STORE/gates/force_window_until"
say "空闲 $idle 个节点 -> 提前开窗,窗口末端 $(TZ=America/Los_Angeles date -d @"$END" '+%m-%d %H:%M %Z')"

# 守卫不在就拉起(它自己会拉调度器)。flock 挡重复实例。
if ! pgrep -f 'night_window\.sh' >/dev/null 2>&1; then
    cd "$ROOT" || exit 1
    # shellcheck disable=SC1091
    . ./simopd_env.sh >/dev/null 2>&1
    setsid nohup bash "$ROOT/slurm/night_window.sh" >> "$STORE/scheduler/window.log" 2>&1 &
    say "守卫不在,已拉起"
fi
exit 0
