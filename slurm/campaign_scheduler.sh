#!/usr/bin/env bash
# 重训调度器:保证可用节点上一直有活,臂跑完自动补下一条。
#
#   bash slurm/campaign_scheduler.sh              # 前台,看着它跑
#   setsid nohup bash slurm/campaign_scheduler.sh >> $SIMOPD_STORE/scheduler/sched.log 2>&1 &
#   DRY=1 bash slurm/campaign_scheduler.sh        # 只说要投什么,不真投
#
# 为什么需要它:一个节点 = 一个 sbatch = 4 条 lane,lane 把分给它的臂顺序跑完就退,
# 最后一条 lane 一退整个作业就释放节点。没人盯着的话节点就这么空在那儿 —— 而集群里
# 没有 pending 队列,空出来也不会有人替我们用掉,纯浪费。这个循环把"节点空了就投下一批"
# 变成机器的事。
#
# 派工原则:
#   * 台账 $STATE/dispatched.tsv 记 arm -> jobid。作业还活着 = 这条臂有人管,不重投;
#     作业没了而臂又没跑完 = 放回队列(ckpt 每 25 步一档 + 指纹校验会自动续跑,
#     所以重投最多损失 25 步,不是从头)。
#   * 一次 sbatch 只能有一个 STEPS,所以每批只从**同一个步数组**里取臂。
#   * A 轴(P3)有前置:a1/a3/a4 要 gkd 缓存,四条都要过预演门(gates/a_axis_ok)。
#   * 节点排除名单由 SLURM_EXCLUDE_NODES 管(见 simopd_env.example.sh),别写死在 sbatch 里。
set -u
# 不要依赖环境里的 USER:crontab 拉起时环境是最小集,USER 可能根本没有。
# 2026-09-06 就是这么炸的 —— 看门狗查不到守卫 -> 每 5 分钟多起一个守卫 ->
# 每个守卫起一个调度器 -> 每个调度器 `squeue -u ""` 查不到作业、以为 0/4 ->
# 各自投满 4 个,一共超投到 7 个作业。id -un 不依赖环境,永远对。
ME=$(id -un)

# 单例锁。2026-09-06 的事故里同时跑起了 2 个守卫 + 3 个调度器,每个都以为自己是唯一的、
# 各自投满上限,一共超投到 7 个作业。根因(环境缺 USER)已修,但"不可能起第二个"这件事
# 不该依赖根因不复发 —— flock 是无条件的。
exec 9>"/tmp/simopd_scheduler.lock" || exit 1
if ! flock -n 9; then
    echo "[$(date -u '+%m-%d %H:%M:%S')] 已有实例在跑,本次退出" >&2
    exit 0
fi
cd $SIMOPD_ROOT
. ./simopd_env.sh >/dev/null 2>&1

ROSTER=${ROSTER:-configs/retrain_roster.tsv}
CK=${CKPT_ROOT:-$SIMOPD_STORE/ckpt}/simopd
STATE=${STATE:-$SIMOPD_STORE/scheduler}
LEDGER=$STATE/dispatched.tsv
MAX_JOBS=${MAX_JOBS:-3}            # 可同时占用的节点数;被排除的和 drained 的不算
LANES=${LANES:-4}                  # 每节点 4 条 lane x 2 卡
ARMS_PER_LANE=${ARMS_PER_LANE:-2}  # 一条 lane 排两条臂,跑完一条不放节点
POLL=${POLL:-300}
DRY=${DRY:-0}
GKD=${DATA_DIR:?}/gkd_offpolicy.parquet
A_GATE=$SIMOPD_STORE/gates/a_axis_ok
mkdir -p "$STATE" "$(dirname "$A_GATE")"
touch "$LEDGER"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 作业还活着吗(任何状态都算,PENDING 也占着一个名额)
_alive() { squeue -h -j "$1" -o %i 2>/dev/null | grep -q .; }

_live_jobs() { squeue -u "$ME" -h -o "%i %j" 2>/dev/null | awk '$2 ~ /^simopd-lane/ {print $1}'; }

say "scheduler up: MAX_JOBS=$MAX_JOBS LANES=$LANES ARMS_PER_LANE=$ARMS_PER_LANE roster=$ROSTER dry=$DRY"

while true; do
    # ---- 1. 台账收尸:作业没了的条目撤销,臂回队列 --------------------------------
    if [ -s "$LEDGER" ]; then
        tmp=$(mktemp)
        while IFS=$'\t' read -r arm job ts; do
            [ -z "${arm:-}" ] && continue
            if _alive "$job"; then printf '%s\t%s\t%s\n' "$arm" "$job" "$ts" >> "$tmp"; fi
        done < "$LEDGER"
        mv "$tmp" "$LEDGER"
    fi

    # ---- 2. 算每条臂的状态 -------------------------------------------------------
    todo=""   # "prio:steps:arm"
    done_n=0; held_n=0
    while IFS=$'\t' read -r arm steps prio note; do
        case "$arm" in ''|'#'*) continue ;; esac
        if [ -d "$CK/${arm}_s0/global_step_$steps" ]; then done_n=$((done_n+1)); continue; fi
        if cut -f1 "$LEDGER" | grep -qx "$arm"; then held_n=$((held_n+1)); continue; fi
        # A 轴前置:缺缓存或没过预演门就先不排
        if [ "$prio" = 3 ]; then
            [ -f "$A_GATE" ] || continue
            case "$arm" in a5_*) : ;; *) [ -f "$GKD" ] || continue ;; esac
        fi
        todo="$todo $prio:$steps:$arm"
    done < "$ROSTER"

    submitted=0
    # squeue 查不到和真的没有,在代码里长得一样 —— 上次就是这么超投的。
    # squeue 本身失败就跳过这一轮,宁可不投,不可乱投。
    if ! squeue -u "$ME" -h -o %i >/dev/null 2>&1; then
        say "!! squeue 查询失败,本轮不投任何作业"
        sleep "$POLL"; continue
    fi
    n_live=$(_live_jobs | wc -l)
    n_todo=$(echo $todo | wc -w)
    say "作业 $n_live/$MAX_JOBS  已完成 $done_n  在管 $held_n  可排 $n_todo"

    # ---- 3. 有空位就投一批 -------------------------------------------------------
    if [ "$n_live" -lt "$MAX_JOBS" ] && [ "$n_todo" -gt 0 ]; then
        # 按优先级排序,第一条决定这一批的步数组;整批只取同一步数的臂
        first=$(echo $todo | tr ' ' '\n' | sort -t: -k1,1n -k3,3 | head -1)
        grp_steps=${first#*:}; grp_steps=${grp_steps%%:*}
        # 装箱不能贪心。每作业硬塞满 LANES*ARMS_PER_LANE 会把臂堆进前几个节点、
        # 后面的节点无工可派 —— 2026-09-08 就是 21 条臂占满 3 个节点 12 条 lane,
        # 9 条在 lane 里排队,而第 4 个节点整夜空着。按**还空着的节点数摊平**:
        # 先让所有 lane 都有活干,臂数超过总 lane 数时才 2 层堆叠。
        free_jobs=$((MAX_JOBS - n_live)); [ "$free_jobs" -lt 1 ] && free_jobs=1
        cap=$(( (n_todo + free_jobs - 1) / free_jobs ))          # ceil(待派 / 空节点)
        [ "$cap" -lt 1 ] && cap=1
        [ "$cap" -gt $((LANES * ARMS_PER_LANE)) ] && cap=$((LANES * ARMS_PER_LANE))
        batch=$(echo $todo | tr ' ' '\n' | sort -t: -k1,1n -k3,3 \
                | awk -F: -v s="$grp_steps" '$2==s {print $3}' | head -$cap)
        n_batch=$(echo $batch | wc -w)
        runs=$(echo $batch | tr ' ' '\n' | sed 's/$/:0/' | tr '\n' ' ')
        # lane 数别超过臂数,否则空 lane 直接退,连带整个作业提前释放节点
        lanes=$LANES; [ "$n_batch" -lt "$lanes" ] && lanes=$n_batch

        # A 组填不满这个节点就用另一个步数组补上剩下的 lane。不这么做的话,某个步数组
        # 快派完时最后那一两条臂会独占一个节点 —— 名册里 250 步只有 5 条,正好会撞上。
        # sbatch 支持第二组(两次 run_parallel.sh,GPU_LIST 分卡 + RAY_TMPDIR_TAG 分目录)。
        runs_b=""; lanes_b=0; steps_b=""
        if [ "$lanes" -lt "$LANES" ]; then
            lanes_b=$((LANES - lanes))
            other=$(echo $todo | tr ' ' '\n' | sort -t: -k1,1n -k3,3 \
                    | awk -F: -v s="$grp_steps" '$2!=s {print $2":"$3}')
            steps_b=$(echo "$other" | head -1 | cut -d: -f1)
            if [ -n "$steps_b" ]; then
                bb=$(echo "$other" | awk -F: -v s="$steps_b" '$1==s {print $2}' | head -$((lanes_b * ARMS_PER_LANE)))
                n_bb=$(echo $bb | wc -w)
                [ "$n_bb" -lt "$lanes_b" ] && lanes_b=$n_bb
                runs_b=$(echo $bb | tr ' ' '\n' | sed 's/$/:0/' | tr '\n' ' ')
            fi
            [ -z "$runs_b" ] && lanes_b=0
        fi

        say "投一批:A=$n_batch 条 / $lanes lane / STEPS=$grp_steps -> $runs"
        [ "$lanes_b" -gt 0 ] && say "        B=$(echo $runs_b | wc -w) 条 / $lanes_b lane / STEPS=$steps_b -> $runs_b"
        if [ "$DRY" = 1 ]; then
            say "  (DRY=1,没真投)"
        else
            # 作业的 --time:让 Slurm 自己在窗口末端收工,守卫进程万一挂了也不会跑过头。
            # 必须用**绝对截止时刻**每次现算,不能用固定的分钟数:调度器要活一整夜、
            # 期间不断补投,拿启动那一刻算出的 449 分钟去投早上 07:00 的作业,那作业会
            # 一直跑到下午 —— 超窗六小时。(2026-09-06 实测发现,改前就是这个行为。)
            # 截止时刻每次现读,不能用启动时的快照:用户可以在调度器活着的时候改窗口
            # (gates/force_window_until),快照会让之后所有作业拿到过期的 --time。
            # 这和早先 LANE_TIME 那个 bug 是同一类 —— 凡是"会变的东西"都不能只在
            # 启动时读一次。
            # force_window_until 是**临时**开窗的覆盖值,过期就必须当它不存在 ——
            # night_window.sh 的 _end_epoch/_in_window 都有 `now < fu` 这道保护,
            # 这里当初漏了。后果:2026-09-09 01:00 守卫正常开窗,而调度器读到昨天
            # 那个已过期的截止时刻,每轮都算出"距窗口结束只剩 -990 分钟",连拒 38 轮、
            # 整整 76 分钟一个作业没投。不修的话,每个强制窗口过期后的夜晚都会整晚空转。
            _wend=$(cat "$SIMOPD_STORE/gates/force_window_until" 2>/dev/null | tr -dc 0-9)
            [ -n "$_wend" ] && [ "$(date +%s)" -ge "$_wend" ] && _wend=""   # 过期即忽略
            # 回落顺序:未过期的强制值 -> 启动时的快照 -> 今天的常规窗口末端(与守卫同源)
            [ -z "$_wend" ] && _wend=${WINDOW_END_EPOCH:-}
            [ -z "$_wend" ] && _wend=$(TZ=${WTZ:-America/Los_Angeles} date -d "today ${WEND_HHMM:-8:30}" +%s)
            tflag=""
            if [ -n "$_wend" ]; then
                _left=$(( ( _wend - $(date +%s) ) / 60 ))
                if [ "$_left" -gt 10 ]; then
                    tflag="--time=$((_left / 60)):$(printf %02d $((_left % 60))):00"
                else
                    say "距窗口结束只剩 ${_left} 分钟,不再投新作业"
                    sleep "$POLL"; continue
                fi
            fi
            out=$(RUNS="$runs" LANES=$lanes STEPS=$grp_steps \
                  RUNS_B="$runs_b" LANES_B=$lanes_b STEPS_B="${steps_b:-200}" \
                  sbatch $tflag ${SLURM_EXCLUDE_NODES:+--exclude="$SLURM_EXCLUDE_NODES"} --job-name=simopd-lane-auto slurm/retrain_lane_node.sbatch 2>&1)
            say "  $out"
            job=$(echo "$out" | grep -oE '[0-9]+$')
            if [ -n "$job" ]; then
                submitted=1
                for a in $batch $(echo $runs_b | tr ' ' '\n' | sed 's/:0$//'); do
                    printf '%s\t%s\t%s\n' "$a" "$job" "$(date -Is)" >> "$LEDGER"
                done
            else
                say "  !! sbatch 没给出 job id,这一批没记台账,下一轮会重试"
            fi
        fi
    fi

    if [ "$submitted" = 1 ]; then
        sleep 10          # 让 squeue 反映出新作业,然后立刻再算一轮
    else
        sleep "$POLL"
    fi
done
