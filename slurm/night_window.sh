#!/usr/bin/env bash
# 夜间窗口守卫:只在洛杉矶时间 01:00–08:30 之间跑,其余时段一张卡都不占。
#
#   setsid nohup bash slurm/night_window.sh >> $SIMOPD_STORE/scheduler/window.log 2>&1 &
#
# 用户 2026-09-05 定:"洛杉矶凌晨1点之后到早上8:30,有卡空出来就用;其他时段不跑;
# 到八点半没跑完的 job 就停。"
#
# 为什么按时区名判断而不是换算成固定的 UTC 时刻:洛杉矶有夏令时,PDT 是 UTC-7、
# PST 是 UTC-8,2026-11-01 会切换。写死 08:00/15:30 UTC 的话那天起就整体偏一小时。
# `TZ=America/Los_Angeles date` 由系统 tzdata 负责,切换当天自动对。
#
# 为什么是常驻进程而不是 cron:cron 任务只在某个会话活着时才触发,而这条策略要在
# 无人值守时也成立。这个进程 setsid 脱离会话,只要登录节点不重启就一直在。
set -u
cd $SIMOPD_ROOT
. ./simopd_env.sh >/dev/null 2>&1

# 不要依赖环境里的 USER:crontab 拉起时环境是最小集,USER 可能根本没有。
# 2026-09-06 就是这么炸的 —— 看门狗查不到守卫 -> 每 5 分钟多起一个守卫 ->
# 每个守卫起一个调度器 -> 每个调度器 `squeue -u ""` 查不到作业、以为 0/4 ->
# 各自投满 4 个,一共超投到 7 个作业。id -un 不依赖环境,永远对。
ME=$(id -un)

# 单例锁。2026-09-06 的事故里同时跑起了 2 个守卫 + 3 个调度器,每个都以为自己是唯一的、
# 各自投满上限,一共超投到 7 个作业。根因(环境缺 USER)已修,但"不可能起第二个"这件事
# 不该依赖根因不复发 —— flock 是无条件的。
exec 9>"/tmp/simopd_night_window.lock" || exit 1
if ! flock -n 9; then
    echo "[$(date -u '+%m-%d %H:%M:%S')] 已有实例在跑,本次退出" >&2
    exit 0
fi
# CONTINUOUS=1:不受时间窗限制,一直跑(用户 2026-09-06 定"先一直跑着")。
# 不能靠把窗口设成 00:00-24:00 来实现:那样 _end_epoch 是"今晚午夜",越接近午夜
# 作业的 --time 越短,最后调度器因为"剩不到 10 分钟"直接不投了。连续模式下改成
# 滚动上限 —— 每个作业最多 LANE_HOURS 小时,到点被 Slurm 收掉、调度器立刻从
# ckpt 续投,既不会跑飞也不会有窗口边界。
# 连续模式的信号必须是文件,不能只靠环境变量:守卫可能由 crontab 看门狗拉起,
# 那个环境里没有我们 export 的任何东西。2026-09-06 实测踩过 —— 看门狗拉起的守卫
# 不认 CONTINUOUS,按夜间窗口把作业全停了。
CONTINUOUS=${CONTINUOUS:-0}
_continuous() { [ "$CONTINUOUS" = 1 ] || [ -f "$SIMOPD_STORE/gates/continuous" ]; }
LANE_HOURS=${LANE_HOURS:-12}
WTZ=${WTZ:-America/Los_Angeles}
START_MIN=${START_MIN:-60}     # 01:00
END_MIN=${END_MIN:-510}        # 08:30
POLL=${POLL:-120}
SCHED_POLL=${SCHED_POLL:-120}
# 窗口内"有多少节点用多少"(用户 2026-09-06 定)。
# 关键不在 MAX_JOBS 而在 ARMS_PER_LANE:每条 lane 排 2 条臂时,一个作业吃 8 条,
# 29 条臂 4 个作业就装完了,MAX_JOBS 再高也没臂可派。改成每 lane 一条 -> 一个作业
# 4 条臂 -> 25 条可派的臂铺成 7 个作业 = 7 个节点同时开工。
# 而且排在 lane 里的第二条臂本来就跑不到:前一条要 200 步、按当前 ~600 s/步 得 30 多
# 小时,一个 7.5 小时的窗口根本轮不上。所以这么改不损失任何东西,纯赚并行度。
# 用户 2026-09-06 定:最多四个节点。四个都给训练(= 16 条臂并发)。
MAX_JOBS=${MAX_JOBS:-4}
# 临时开窗期间的节点数,可以和常规窗口不同(用户 2026-09-06:现在先用 2 个,
# 到洛杉矶 01:00 回到原计划的 4 个)。切换时重起调度器 —— 它只在启动时读 MAX_JOBS,
# 不重起就一直用旧值;重起不影响已在跑的作业,台账也是持久化的。
# 同样必须能从文件读:守卫随时可能被 crontab 看门狗拉起,那个环境里没有我们 export
# 的任何东西。只靠环境变量的话,守卫一重启就悄悄回到默认节点数。
FORCE_MAX_JOBS=${FORCE_MAX_JOBS:-$(cat "${SIMOPD_STORE:-/tmp}/gates/force_max_jobs" 2>/dev/null | tr -dc 0-9)}
FORCE_MAX_JOBS=${FORCE_MAX_JOBS:-$MAX_JOBS}
# 每条 lane 排几条臂。设成 2 的理由:一条臂跑完时,lane 直接接下一条 —— 零空档、
# 也不打断同作业其他 lane。设成 1 的话 lane 会退出、2 张卡空着,要等看门狗两轮
# (约 30 分钟)确认再把**整个作业**取消重投,同作业另外 3 条臂各损失最多 25 步。
# 并发度两者相同(都是 4 lane/节点),差别只在"臂跑完之后会怎样"。
# 走文件,每轮现读:守卫可能被 crontab 拉起,环境变量不可靠。
ARMS_PER_LANE=${ARMS_PER_LANE:-$(cat "${SIMOPD_STORE:-/tmp}/gates/arms_per_lane" 2>/dev/null | tr -dc 0-9)}
ARMS_PER_LANE=${ARMS_PER_LANE:-2}
# 评测专列默认不开(用户选了"都给训练")。开了就占掉四分之一训练。要开就 WANT_EVAL=1 重起本守卫。
# 代价要知道:75 个 ckpt 一格没评,而 composite/verdict/论文表全靠评测产出。
WANT_EVAL=${WANT_EVAL:-0}
# A 轴预演门默认不跑:它要独占一个节点约半小时,而四个节点只够 16 条臂并发、
# 可派的臂还有 25 条 —— A 轴那 4 条好几个窗口都轮不到,现在解锁没意义。
# 名册跑薄了(可派 < 16 条)再 WANT_AGATE=1 开一次。
# 同样走文件:守卫随时可能被 crontab 看门狗拉起,环境变量在那里不存在。
# gates/want_agate 存在就跑一次 A 轴预演门。
WANT_AGATE=${WANT_AGATE:-0}
[ -f "${SIMOPD_STORE:-/tmp}/gates/want_agate" ] && WANT_AGATE=1

say() { echo "[$(date -u '+%m-%d %H:%M:%S')UTC / $(TZ=$WTZ date '+%H:%M %Z')] $*"; }

_sched_pid() { ps -u "$ME" -o pid= -o args= | grep -F 'campaign_schedul''er.sh' | grep -v grep | awk '{print $1}'; }
_our_jobs()  { squeue -u "$ME" -h -o "%i %j" 2>/dev/null | awk '$2 ~ /^simopd/ {print $1}'; }

# 距窗口结束还有几分钟 —— 交给调度器当作业的 --time,这样即使本守卫挂了,
# Slurm 也不会让作业跑过八点半。
_mins_left() {
    # 从 _end_epoch 推,别再独立算一遍:临时开窗时两套算法会打架(实测日志里出现过
    # "--time 设为 -260 分钟",因为这里按常规窗口末端算、而末端已经过去了)。
    echo $(( ( $(_end_epoch) - $(date +%s) ) / 60 ))
}
# 窗口结束的绝对时刻(epoch)。给调度器,让它每次投递都按"现在距这一刻还有多久"算
# --time —— 传固定分钟数的话,后半夜补投的作业会拿到启动时的旧值、跑过窗口。
_end_epoch() {
    if _continuous; then echo $(( $(date +%s) + LANE_HOURS * 3600 )); return; fi
    # 临时开窗期间,"窗口末端"就是那个 epoch —— 作业的 --time 要按它算,不然会被
    # 按常规窗口末端(可能已经过去了)算出负数或错值。
    local fu; fu=$(cat "$SIMOPD_STORE/gates/force_window_until" 2>/dev/null | tr -dc 0-9)
    if [ -n "$fu" ] && [ "$(date +%s)" -lt "$fu" ]; then echo "$fu"; return; fi
    TZ=$WTZ date -d "today $((END_MIN/60)):$(printf %02d $((END_MIN%60)))" +%s
}

_in_window() {
    _continuous && return 0
    # 临时开窗:$SIMOPD_STORE/gates/force_window_until 里放一个 epoch,在那之前一律
    # 算窗口内,过后自动回到常规的 START_MIN–END_MIN 规则,不改每日策略。
    # 用文件而不是环境变量,是因为系统看门狗(另一个进程、由 crond 拉起)也要认同一个
    # 信号 —— 否则守卫以为在跑、看门狗以为该停,两边打架。
    local fu; fu=$(cat "$SIMOPD_STORE/gates/force_window_until" 2>/dev/null | tr -dc 0-9)
    if [ -n "$fu" ] && [ "$(date +%s)" -lt "$fu" ]; then return 0; fi
    if [ -n "${FORCE_UNTIL:-}" ] && [ "$(date +%s)" -lt "$FORCE_UNTIL" ]; then return 0; fi
    local h m cur
    h=$(TZ=$WTZ date +%H); m=$(TZ=$WTZ date +%M)
    cur=$((10#$h * 60 + 10#$m))
    [ "$cur" -ge "$START_MIN" ] && [ "$cur" -lt "$END_MIN" ]
}

say "窗口守卫启动:每天 $((START_MIN/60)):$(printf %02d $((START_MIN%60)))–$((END_MIN/60)):$(printf %02d $((END_MIN%60))) $WTZ,其余时段不占卡"
state=""
while true; do
    if _in_window; then
        if [ "$state" != in ]; then say "=== 进入窗口 ==="; state=in; fi
        left=$(_mins_left)

        # 评测:75 个 ckpt 一个都没评过,而评测不依赖训练跑完。占一个节点,
        # 队列空了它自己退(EMPTY_EXIT),不会白占。
        if [ "$WANT_EVAL" = 1 ] && [ "$left" -gt 60 ] \
           && ! squeue -u "$ME" -h -o %j | grep -q evalfarm; then
            o=$(EVAL_GRID=full sbatch ${SLURM_EXCLUDE_NODES:+--exclude="$SLURM_EXCLUDE_NODES"} --time="$((left/60)):$(printf %02d $((left%60))):00" \
                slurm/retrain_eval_farm.sbatch 2>&1)
            say "评测专列: $o"
        fi

        # A 轴预演门:没过就永远铺不上那 4 条臂。每天最多试一次,失败了不刷屏重投。
        if [ "$WANT_AGATE" = 1 ] && [ ! -f "$SIMOPD_STORE/gates/a_axis_ok" ] \
           && [ "$left" -gt 120 ] && [ ! -f "$SIMOPD_STORE/gates/.tried_$(date -u +%Y%m%d)" ] \
           && ! squeue -u "$ME" -h -o %j | grep -q agate; then
            mkdir -p "$SIMOPD_STORE/gates"; touch "$SIMOPD_STORE/gates/.tried_$(date -u +%Y%m%d)"
            o=$(sbatch ${SLURM_EXCLUDE_NODES:+--exclude="$SLURM_EXCLUDE_NODES"} slurm/retrain_a_axis_gate.sbatch 2>&1)
            say "A 轴预演门: $o"
        fi

        # 现在该用几个节点
        # 两个都要**每轮现读**,不能用启动时的快照:用户会在守卫活着的时候改这些文件。
        # 今天已经因为"启动时读一次"踩了三回(LANE_TIME、WINDOW_END_EPOCH、这里)。
        want=$MAX_JOBS
        fu=$(cat "$SIMOPD_STORE/gates/force_window_until" 2>/dev/null | tr -dc 0-9)
        if [ -n "$fu" ] && [ "$(date +%s)" -lt "$fu" ]; then
            fm=$(cat "$SIMOPD_STORE/gates/force_max_jobs" 2>/dev/null | tr -dc 0-9)
            want=${fm:-$FORCE_MAX_JOBS}
        fi
        if [ -n "$(_sched_pid)" ] && [ "${cur_max:-}" != "$want" ] && [ -n "${cur_max:-}" ]; then
            say "节点数 $cur_max -> $want,重起调度器(不影响在跑的作业)"
            kill $(_sched_pid); sleep 3
        fi

        if [ -z "$(_sched_pid)" ]; then
            cur_max=$want
            apl=$(cat "$SIMOPD_STORE/gates/arms_per_lane" 2>/dev/null | tr -dc 0-9)
            ARMS_PER_LANE=${apl:-$ARMS_PER_LANE}
            say "调度器不在,拉起(节点上限 $want)(作业 --time 设为 ${left} 分钟,到点 Slurm 自己收)"
            setsid nohup env POLL=$SCHED_POLL MAX_JOBS=$want \
                ARMS_PER_LANE=$ARMS_PER_LANE WINDOW_END_EPOCH=$(_end_epoch) \
                bash slurm/campaign_scheduler.sh >> "$SIMOPD_STORE/scheduler/sched.log" 2>&1 &
        fi
    else
        if [ "$state" != out ]; then say "=== 离开窗口,停机 ==="; state=out; fi
        P=$(_sched_pid); [ -n "$P" ] && { kill $P; say "调度器 $P 已停"; sleep 3; }
        J=$(_our_jobs | tr '\n' ' ')
        [ -n "$J" ] && { scancel $J; say "已取消: $J"; }
    fi
    sleep "$POLL"
done
