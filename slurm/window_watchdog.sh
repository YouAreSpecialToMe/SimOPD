#!/usr/bin/env bash
# 夜间窗口的第二层保险,由系统 crontab 每 5 分钟拉起一次。幂等,可与 night_window.sh 并存。
#
# 为什么要有它:night_window.sh 是个常驻进程 —— 进程会被 OOM、被误杀、随登录节点重启
# 而消失。crontab 由 crond 负责,登录节点重启后自动恢复,不依赖任何会话或长驻进程。
#
# 三件事,按重要性排:
#   1. 窗口外一律停干净(准点结束的硬保证,不依赖守卫进程还活着)
#   2. 把任何会跑过窗口末端的作业时限**削**到窗口末端(Slurm 允许降不允许升)
#   3. 窗口内守卫进程不在就拉起来(它自己再去管调度器,这里不直接碰调度器,免得双启)
set -u
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
ROOT=$SIMOPD_ROOT
cd "$ROOT" || exit 1
. ./simopd_env.sh >/dev/null 2>&1

# 不要依赖环境里的 USER:crontab 拉起时环境是最小集,USER 可能根本没有。
# 2026-09-06 就是这么炸的 —— 看门狗查不到守卫 -> 每 5 分钟多起一个守卫 ->
# 每个守卫起一个调度器 -> 每个调度器 `squeue -u ""` 查不到作业、以为 0/4 ->
# 各自投满 4 个,一共超投到 7 个作业。id -un 不依赖环境,永远对。
ME=$(id -un)
WTZ=America/Los_Angeles
START_MIN=60; END_MIN=510
LOG=$SIMOPD_STORE/scheduler/watchdog.log
say() { echo "[$(date -u '+%m-%d %H:%M:%S')UTC / $(TZ=$WTZ date '+%H:%M %Z')] $*" >> "$LOG"; }

h=$(TZ=$WTZ date +%H); m=$(TZ=$WTZ date +%M)
cur=$((10#$h * 60 + 10#$m))          # 10#:08/09 会被当非法八进制,正好是收尾那一小时
END_EPOCH=$(TZ=$WTZ date -d "today $((END_MIN/60)):$(printf %02d $((END_MIN%60)))" +%s)
# 临时开窗(用户临时说"现在跑 N 小时"):守卫和本看门狗必须认同一个信号,否则一个
# 以为在跑、一个以为该停,作业会被反复杀掉重投。到点后文件自动失效,回到常规窗口。
# 连续模式:$SIMOPD_STORE/gates/continuous 存在就一直算窗口内。和守卫认同一个信号,
# 否则守卫在跑、看门狗按夜间窗口判断该停,作业会被反复杀掉重投。
if [ -f "$SIMOPD_STORE/gates/continuous" ]; then
    cur=$START_MIN; END_EPOCH=$(( $(date +%s) + 86400 ))
fi
FORCE=$(cat "$SIMOPD_STORE/gates/force_window_until" 2>/dev/null | tr -dc 0-9)
if [ -n "$FORCE" ] && [ "$(date +%s)" -lt "$FORCE" ]; then
    cur=$START_MIN; END_EPOCH=$FORCE      # 强制判定为"窗口内",末端用强制截止时刻
fi

_jobs() { squeue -u "$ME" -h -o "%i %j" 2>/dev/null | awk '$2 ~ /^simopd/ {print $1}'; }
_daemon() { ps -u "$ME" -o pid= -o args= | grep -F 'night_win''dow.sh' | grep -v grep | awk '{print $1}'; }
_sched()  { ps -u "$ME" -o pid= -o args= | grep -F 'campaign_schedul''er.sh' | grep -v grep | awk '{print $1}'; }

if [ "$cur" -ge "$START_MIN" ] && [ "$cur" -lt "$END_MIN" ]; then
    # --- 窗口内 ---
    if [ -z "$(_daemon)" ]; then
        say "!! 守卫进程不在,拉起"
        setsid nohup bash "$ROOT/slurm/night_window.sh" >> "$SIMOPD_STORE/scheduler/window.log" 2>&1 &
    fi
    # --- 掉队检测:作业还活着但卡没跑满 ---
    # 今天两次都是静默的:一个节点因为 LANES_B 没传只用了 4 张卡,另一个因为端口撞车
    # 死了一条 lane 空出 2 张。作业本身不会退,squeue 看着一切正常,卡就这么空烧一整夜。
    # 判据保守:跑够 20 分钟(过了拉起期)、连续两轮(约 10 分钟)都有 >=2 张空卡才动手,
    # 取消后调度器会重投并从 ckpt 续跑,代价上限 25 步 —— 远小于空烧几小时。
    STRIKE=$SIMOPD_STORE/scheduler/strikes
    mkdir -p "$STRIKE"
    for j in $(_jobs); do
        # 掉队检测**只适用于 lane 作业**。预演门(simopd-agate)和评测专列(simopd-evalfarm)
        # 有自己的生命周期:预演门是四个彩排并行,一条条跑完就一条条放卡,天然会"空卡越来
        # 越多" —— 2026-09-08 06:50 就因此把跑到 3/4 PASS 的预演门杀了,白费半小时。
        nm=$(squeue -j "$j" -h -o %j 2>/dev/null)
        case "$nm" in simopd-lane*) : ;; *) continue ;; esac
        # 收尾期不再取消。取消-重投这件事只有在"重投还来得及干活"时才划算:光是
        # vLLM 起模型就 10-15 分钟(2026-09-09 实测 7218 用了 15 分钟才到 8/8),
        # 再加回退到上一个 25 步存档,窗口末尾的一次误杀是纯损失、没有挽回时间。
        # 而 lane 换臂(前一条跑完、后一条正在加载)期间本来就会短暂放卡,恰好长得
        # 像掉队 —— 两者叠加,末段最容易把健康作业杀掉。剩余 < 45 分钟一律不判。
        if [ "$(( (END_EPOCH - $(date +%s)) / 60 ))" -lt 45 ]; then
            rm -f "$STRIKE/$j"; continue
        fi
        age=$(squeue -j "$j" -h -o %M 2>/dev/null | awk -F: '{n=NF; s=$n; if(n>1) s+=$(n-1)*60; if(n>2) s+=$(n-2)*3600; print int(s/60)}')
        [ "${age:-0}" -lt 20 ] && continue
        used=$(timeout 60 srun --jobid="$j" --overlap nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>2000{c++} END{print c+0}')
        [ -z "$used" ] && continue                 # 查不到就跳过,不猜
        # 该占几张卡要看作业**实际派了几条 lane**,不能假定 8 张。名册见底时调度器只
        # 派得出 1-3 条 lane,那种作业本来就用不满一个节点 —— 2026-09-07 00:05 就因此
        # 把只有 2 条 lane 的作业当成掉队取消,重投后还是 2 条 lane、还是被判掉队,
        # h3/h4 每轮退回 @75,形成永不结束的取消-重投循环。
        # 而且要看**还没干完的 lane**,不是派了几条。一条 lane 把分给它的臂全跑完就
        # 打印 LANE_DONE 并退出,它那两张卡是正常释放的 —— 2026-09-08 10:35 就因此
        # 把 6697 杀了:lane1 只分到 vanilla_te 一条臂,跑完报 OK 走人,剩 6/8 卡,
        # 被判"空 2 张",另外三条正在训练的 lane(含 c5_union_rkl,每步 18 分钟)
        # 陪葬回退到上一个 25 步存档。
        jout="$OPD_LOGS/simopd-lane-auto-$j.out"
        lanes=$(grep -cE "^  lane [0-9]+ +GPUs" "$jout" 2>/dev/null)
        case "${lanes:-0}" in 0) lanes=4 ;; esac    # 读不到就按满编算
        active=0
        for _ll in $(grep -ohE "/home/[^ ]*/logs/lanes/lane[0-9]+_[0-9_]+\.log" "$jout" 2>/dev/null | sort -u); do
            grep -q 'LANE_DONE' "$_ll" 2>/dev/null || active=$((active + 1))
        done
        [ "$active" -eq 0 ] && active=$lanes            # 读不到 lane 日志就退回旧口径
        idle=$(( active * 2 - used ))
        if [ "$idle" -ge 2 ]; then
            n=$(( $(cat "$STRIKE/$j" 2>/dev/null || echo 0) + 1 ))
            echo "$n" > "$STRIKE/$j"
            say "作业 $j: $lanes 条 lane / $active 条还在干(应占 $((active*2)) 卡),实占 $used,空 $idle(第 $n 次)"
            if [ "$n" -ge 2 ]; then
                scancel "$j" && say "!! 作业 $j 连续两轮掉队($idle 张空卡),已取消,交给调度器重投"
                rm -f "$STRIKE/$j"
            fi
        else
            rm -f "$STRIKE/$j"
        fi
    done
    find "$STRIKE" -type f -mmin +180 -delete 2>/dev/null   # 清掉早已结束的作业的计数

    # 削掉会跑过窗口的作业时限。Slurm 只允许降不允许升,所以这一步只会更严、不会更松。
    # **只削 RUNNING 的**。PENDING 作业的 StartTime 是 Slurm 的预估开跑时刻,不是事实:
    # 2026-09-08 10:10,排队中的 6735 被预估到 15:28 才开跑,于是算出 left = 15:30-15:28
    # = 1 分钟,时限被削成 0:01:00;它 10:35 真正起来,80 秒后就 CANCELLED DUE TO TIME
    # LIMIT。排队的不用现在削 —— 时限从实际开跑起算,等它跑起来下一轮(最多 5 分钟后)
    # 再按真实 StartTime 削一次就是精确的,而且窗口外那条分支会无差别 scancel 兜底。
    for j in $(_jobs); do
        info=$(scontrol show job "$j" 2>/dev/null)
        case "$info" in *JobState=RUNNING*) : ;; *) continue ;; esac
        et=$(echo "$info" | grep -oE "EndTime=[0-9T:-]+" | cut -d= -f2)
        [ -z "$et" ] && continue
        if [ "$(date -d "$et" +%s 2>/dev/null || echo 0)" -gt "$((END_EPOCH + 120))" ]; then
            st=$(echo "$info" | grep -oE "StartTime=[0-9T:-]+" | cut -d= -f2)
            left=$(( (END_EPOCH - $(date -d "$st" +%s)) / 60 ))
            [ "$left" -lt 1 ] && left=1
            scontrol update jobid="$j" TimeLimit=$((left/60)):$(printf %02d $((left%60))):00 \
                && say "!! 作业 $j 原本会跑到 $et,已削到窗口末端"
        fi
    done
else
    # --- 窗口外:停干净。这是"准点结束"的最后一道保证 ---
    P=$(_sched); [ -n "$P" ] && { kill $P; say "窗口外,停调度器 $P"; sleep 2; }
    J=$(_jobs | tr '\n' ' ')
    [ -n "$J" ] && { scancel $J; say "窗口外,取消作业: $J"; }
fi

# 显式 exit 0:上面两条 [ -n ... ] && {...} 在"没东西要停"的正常静默路径上返回 1,
# 脚本就以 1 收尾。没有 set -e,功能不受影响,但兜底脚本平时一直非零退出会让任何
# 按退出码判死活的监控永远报错、真出事时反而看不出来。
exit 0
