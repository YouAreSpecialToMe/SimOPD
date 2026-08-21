#!/usr/bin/env bash
# 永不退出的 DLC 载体。提交一次,之后换任务 = 在跳板机上改共享树里的 payload 脚本。
#
# 为什么要它:corr 波这套是"一次 dlc submit 跑一份 lane 表",换实验就得重投 DLC 或者
# 从 DSW 上重新 launch,而 DLC 的凭据只在 pod 里、跳板机上没有(设计如此)。于是每次
# 换任务都卡在人身上。这个载体把 DLC 作业变成一台常驻机器:它唯一的职责是反复执行
# 共享盘上的 payload,谁改了 payload,下一轮就跑谁。
#
# 提交(在 DLC 控制台;推荐 7 个单节点作业,k=0..6,故障互相隔离):
#   SLOT=<k> bash /mgfs/shared/Group_GY/changhao/SimOPD-exp/deploy/dlc/forever.sh
# (也可以一单 7 节点 + SLOT=auto 按 rank 推槽,但任一节点故障平台会整单重启,
#  7 个槽一起回退到最近检查点;单节点作业把故障半径限制在一个槽。)
#
# 之后在跳板机上换任务(不碰 DLC;都有 task.sh 子命令,直接摸标记文件也行):
#   vi  $D/forever/payload_slot3.sh      # 只给 slot3 换
#   vi  $D/forever/payload.sh            # 给所有槽换(每槽 payload 优先)
#   touch $D/forever/swap_slot3          # 立刻换:终止 slot3 正在跑的 payload 并重读
#   touch $D/forever/stop_slot3          # 立刻停:终止并空转待命,直到删掉标记
#   touch $D/forever/reload_slot3        # 温和:不杀正在跑的,只把空转/退避的等待掐短
#
# swap 和 stop 是"立刻"的,因为 payload 可能永远不结束 —— corr_wave_fleet.sh 的臂全部
# 跑满后会进它自己的 holding 死循环,温和的 reload 就永远等不到那一刻,人还得回去重投
# DLC,那正是这个载体要消灭的事。代价说清楚:正在训练时 swap/stop 会让那些 lane 回退
# 到最近检查点(<=25 步),所以它是明确的人为动作,不是自动行为。
#
# 调试通道(像 ssh 那样敲一条看一眼,不用等下一轮):
#   跳板机:  bash deploy/dlc/task.sh sh 3 'nvidia-smi; ls $D/corr_wave | tail'
#   原理:命令写进 $D/forever/cmd/slot3/inbox/,pod 收到后放【后台】执行 —— 一条长命令
#   不会挡住 swap/stop 的轮询,也不属于 payload 进程组,payload 被杀它照常跑完。
#   单条默认 300s 超时;跳板机用 CMD_T=秒 放宽(task.sh 把它作为 <id>.t 随件送来)。
#   要真终端(vim/top/python -i/cd 保留)用 task.sh tty <槽>:文件 PTY 桥,见 ptybridge.py。
#
# payload 契约:一个普通 bash 脚本,能拿到 SLOT / SEED / ROOT / D / LOGD;正常返回即
# 本轮结束,载体歇 IDLE_S 秒后重读重跑 —— 所以 payload 要么自己可续跑/幂等,要么用
# exit 42 声明"做完了":载体把该槽挂起(parked)不再重跑,task.sh go/set/swap 解除。
# 不设 payload 时跑 corr_wave_fleet.sh(即现状)。
#
# 跳板机怎么知道它活没活:载体把全部输出 tee 到 $D/forever/log/slot<k>.log,并每
# HB_EVERY_S 秒把心跳写进 $D/forever/hb_slot<k>;task.sh status 两样都显示。启动先认领
# 槽位:心跳还新鲜(<HB_STALE_S 秒)说明槽上已有活载体(双载体会把同一份检查点写坏),
# 就等它变陈 —— 平台自动重启作业时旧 pod 若真死了,至多等一个陈化期,不用人来。
#
# 三条保命规则,都是被真事故逼出来的:
#  1 快照后再跑。bash 是边读边执行的,你在跳板机上改到一半、pod 正好读到半个文件,
#    行为无法预测。所以每轮先把 payload 复制成 running 副本再执行 —— 跑到一半的改动
#    只影响下一轮。
#  2 先 bash -n 再跑,不过就退回上一版 last_good。一个语法错误如果直接执行,载体会
#    进入"秒退 -> 重跑 -> 秒退"的热循环,把日志刷爆而看上去像在工作。
#  3 秒退要退避。payload 若在 MIN_RUN_S 内返回,视为异常,退避时间翻倍(上限 15 分钟);
#    正常跑完则重置。
# 外加一条收尾规则:swap/stop 杀完进程组后,ray/vLLM 里自行 setsid 的守护(EngineCore
# 那一课)会逃出组,所以真 pod 里再按 /proc cmdline 挥一遍镰刀;非 pod 环境(电池、
# 跳板机)绝不挥 —— 那是共享机器,FOREVER_SCYTHE=0 也可硬关。
set -uo pipefail

ROOT=${ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
D=${SIMOPD_STORE:-/mgfs/shared/Group_GY/changhao/simopd_data}
SEED=${SEED:-0}
FDIR=$D/forever
IDLE_S=${IDLE_S:-60}          # 一轮正常结束后歇多久再重读
MIN_RUN_S=${MIN_RUN_S:-120}   # 短于这个算秒退
BACKOFF_S0=${BACKOFF_S0:-60}  # 首次退避;之后翻倍。可配置是为了能被电池验证 ——
BACKOFF_MAX=${BACKOFF_MAX:-900}  # 写死 60 的话,只要上限低于它就永远看不出有没有在翻倍
HB_EVERY_S=${HB_EVERY_S:-30}  # 心跳间隔;HB_STALE_S 秒没跳视为死。可配置同样是给电池的
HB_STALE_S=${HB_STALE_S:-90}
LOG_MAX_MB=${LOG_MAX_MB:-50}

_rank=${RANK:-${MLP_ROLE_INDEX:-${MLP_WORKER_RACK_RANK_INDEX:-0}}}
SLOT_BASE=${SLOT_BASE:-0}
SLOT=${SLOT:-auto}
[ "$SLOT" = auto ] && SLOT=$(( SLOT_BASE + _rank ))
export SLOT SEED ROOT D
LOGD=$D/corr_wave                       # payload 沿用舰队的日志根
export LOGD
RUND=$FDIR/run/slot${SLOT}
mkdir -p "$FDIR" "$RUND" 2>/dev/null || true

_say() { echo "[$(date '+%F %T')] slot${SLOT}: $*"; }
_fsize() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0; }
_mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0; }

# ---- 心跳 --------------------------------------------------------------------
# 认领成功前绝不跳(_claimed 门):正在等旧载体变陈的新载体如果自己也写 hb,
# 等的就是自己的心跳,永远等不到陈。
_tok="$(hostname 2>/dev/null || echo unknown)-$$-${RANDOM}"
_claimed=""
_hb_last=0
_beat() {
    [ -n "$_claimed" ] || return 0
    local now; now=$(date +%s)
    [ $(( now - _hb_last )) -ge "$HB_EVERY_S" ] || return 0
    _hb_last=$now
    printf '%s\n' "$_tok" > "$FDIR/hb_slot${SLOT}" 2>/dev/null || true
}

# ---- 调试通道 ----------------------------------------------------------------
# 请求 = inbox 里的一个 .sh(可带同名 .t 随件指定超时);完成 = out/<id>.rc 出现
# (最后写,所以读到 .rc 时 .out 一定完整)。命令放后台执行:长命令不挡 swap/stop
# 轮询,且不在 payload 进程组里,payload 被杀它照常跑完。
CMDD=$FDIR/cmd/slot${SLOT}
mkdir -p "$CMDD/inbox" "$CMDD/out" 2>/dev/null || true
_serve_cmds() {
    local f id t
    for f in "$CMDD/inbox"/*.sh; do
        [ -e "$f" ] || continue
        id=$(basename "$f" .sh)
        mv "$f" "$CMDD/out/$id.sh" 2>/dev/null || continue   # 原子取走,避免重复执行
        [ -f "$CMDD/inbox/$id.t" ] && mv "$CMDD/inbox/$id.t" "$CMDD/out/$id.t" 2>/dev/null
        t=${CMD_TIMEOUT:-300}
        if [ -s "$CMDD/out/$id.t" ]; then
            t=$(head -1 "$CMDD/out/$id.t" | tr -dc 0-9)
            [ -n "$t" ] || t=${CMD_TIMEOUT:-300}
        fi
        _say "调试命令 $id 开始(timeout ${t}s,后台执行)"
        (
            if command -v timeout >/dev/null 2>&1; then
                ( cd "$ROOT" 2>/dev/null || cd /; \
                  SLOT=$SLOT SEED=$SEED ROOT=$ROOT D=$D LOGD=$LOGD \
                  timeout "$t" bash "$CMDD/out/$id.sh" ) > "$CMDD/out/$id.out" 2>&1
            else   # macOS 电池机可能没有 timeout;pod 一定有
                ( cd "$ROOT" 2>/dev/null || cd /; \
                  SLOT=$SLOT SEED=$SEED ROOT=$ROOT D=$D LOGD=$LOGD \
                  bash "$CMDD/out/$id.sh" ) > "$CMDD/out/$id.out" 2>&1
            fi
            rc_=$?
            echo "$rc_" > "$CMDD/out/$id.rc"                 # 最后写 = 完成信号
            _say "调试命令 $id 结束 rc=$rc_"
        ) &
    done
}
_sleep_serving() {   # 秒数 [r=允许被 reload/swap 掐短]
    local left=$1 cut=${2:-}
    while [ "$left" -gt 0 ]; do
        if [ -n "$cut" ] && { [ -f "$FDIR/reload_slot${SLOT}" ] || [ -f "$FDIR/swap_slot${SLOT}" ]; }; then
            _say "等待被 reload/swap 掐短,立刻进下一轮"
            return 0
        fi
        _serve_cmds
        _beat
        sleep $(( left < ${POLL_S:-3} ? left : ${POLL_S:-3} ))
        left=$(( left - ${POLL_S:-3} ))
    done
}

# ---- 镰刀 --------------------------------------------------------------------
# 进程组连坐杀不干净:ray/vLLM 里自行 setsid 的守护(EngineCore)会逃出组,继续占卡。
# 所以 swap/stop 杀完组之后按 /proc cmdline 再扫一遍。候选(只列不杀)与执刀分开,
# 前者电池可测;执刀有双保险:非 pod 环境(没有 DLC/MLP rank env)不挥,FOREVER_SCYTHE=0
# 硬关 —— 电池跑在跳板机/本机这种共享机器上,在那里挥镰刀会杀到别人的进程。
_scythe_candidates() {
    local me=$$ pid c
    for pid in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
        [ "$pid" = "$me" ] && continue
        c=$(tr "\0" " " < "/proc/$pid/cmdline" 2>/dev/null) || continue
        case "$c" in
            *VLLM::*|*EngineCore*|*vllm*|*ray::*|*raylet*|*verl.trainer.main_ppo*|*eval_offline.py*|*eval_suite.py*)
                echo "$pid" ;;
        esac
    done
}
_scythe() {
    if [ "${FOREVER_SCYTHE:-1}" != 1 ]; then
        _say "FOREVER_SCYTHE=0,跳过镰刀"
        return 0
    fi
    if [ -z "${MLP_ROLE_INDEX:-}${MLP_WORKER_RACK_RANK_INDEX:-}${DLC_JOB_ID:-}" ]; then
        _say "非 pod 环境,跳过镰刀(共享机器上不能按名字杀进程)"
        return 0
    fi
    local n=0 pid
    for pid in $(_scythe_candidates); do
        kill -9 "$pid" 2>/dev/null && n=$((n+1))
    done
    _say "镰刀:清掉 $n 个逃出进程组的 GPU 残留(vllm/ray/EngineCore)"
    [ "$n" -gt 0 ] && sleep 2
    return 0
}
# 电池用的只读入口:只打印候选 pid,不杀不跑
if [ -n "${FOREVER_LIST_SCYTHE:-}" ]; then _scythe_candidates; exit 0; fi

# 提交端(没有 DLC rank env)只打控制台卡片,不跑
if [ -z "${MLP_ROLE_INDEX:-}${MLP_WORKER_RACK_RANK_INDEX:-}${DLC_JOB_ID:-}" ] && [ -z "${FOREVER_FORCE:-}" ]; then
    cat <<CARD
========= 永续载体:DLC 控制台卡片(提交一次,之后不用再碰 DLC)=========
  推荐:提交 7 个单节点作业(故障互相隔离;k=0..6):
    任务名称   simopd-forever-s${SEED}-slot<k>
    节点数量   1            单节点GPU 8   CPU 64   内存 512Gi
    镜像/资源组/挂载:照抄 simopd-corr-wave1 的成功表单(挂载须含 /mgfs)
    执行命令:
      SLOT=<k> SEED=${SEED} bash $ROOT/deploy/dlc/forever.sh
  (不推荐但可行:一单 7 节点 + SLOT=auto 按 rank 推槽 —— 任一节点故障整单重启,
   7 个槽一起回退到最近检查点)

  提交后在跳板机上全用 bash deploy/dlc/task.sh(或直接摸 \$D/forever/ 下的标记):
    status / set / swap / stop / go / reload / clear / sh / watch
    \$D/forever/payload_slot<k>.sh 给某槽换任务(优先);payload.sh 给所有槽
  不放 payload 时,每轮跑 deploy/dlc/corr_wave_fleet.sh(= 现在的行为)
  活没活:task.sh status 显示心跳与最后一行日志(\$D/forever/hb_slot<k>、log/slot<k>.log)
CARD
    exit 0
fi

# 从这里起是真载体:全部输出 tee 进共享盘日志 —— 跳板机没有 DLC 凭据看不了控制台,
# 载体自己的叙述(第几轮/语法门/退避)不落 /mgfs 的话,出问题时人在跳板机上是瞎的。
LOGF=$FDIR/log/slot${SLOT}.log
mkdir -p "$FDIR/log" 2>/dev/null || true
exec > >(tee -a "$LOGF") 2>&1

# 槽位认领:防双载体(平台重启作业但旧 pod 没死透 / 人重复提交)。写入自己的 token,
# 略等后读回验证,被别人覆盖就退让。HB_CLAIM=0 给电池跳过用。
_claim() {
    if [ "${HB_CLAIM:-1}" != 1 ]; then
        _say "HB_CLAIM=0:跳过槽位认领(电池模式)"
        _claimed=1
        return 0
    fi
    local f=$FDIR/hb_slot${SLOT} age
    while :; do
        if [ -f "$f" ]; then
            age=$(( $(date +%s) - $(_mtime "$f") ))
            if [ "$age" -lt "$HB_STALE_S" ]; then
                _say "hb_slot${SLOT} 心跳 ${age}s 前还在跳:疑似双载体(旧 pod 未死?),等它变陈再接管"
                _sleep_serving "${HB_WAIT_S:-30}"
                continue
            fi
        fi
        printf '%s\n' "$_tok" > "$f" 2>/dev/null || true
        sleep "${HB_VERIFY_S:-5}"
        if [ "$(head -1 "$f" 2>/dev/null)" = "$_tok" ]; then
            _say "槽位认领成功(token $_tok)"
            _claimed=1
            _beat
            return 0
        fi
        _say "认领被别的载体覆盖,退让重试"
        sleep $(( 5 + RANDOM % 10 ))
    done
}
_claim

_pick_payload() {   # 打印本轮该跑的脚本路径
    local p
    for p in "$FDIR/payload_slot${SLOT}.sh" "$FDIR/payload.sh" \
             "$ROOT/deploy/dlc/corr_wave_fleet.sh"; do
        [ -s "$p" ] && { printf '%s' "$p"; return 0; }
    done
    return 1
}

_backoff=0
_round=0
while true; do
    _round=$((_round + 1))
    _beat
    # 日志与调试产物的体积闸(tee -a 是 O_APPEND,truncate 后续写自动回到文件头)
    if [ "$(_fsize "$LOGF")" -gt $(( LOG_MAX_MB * 1024 * 1024 )) ]; then
        tail -c 1048576 "$LOGF" > "$LOGF.1" 2>/dev/null || true
        : > "$LOGF"
        _say "日志超 ${LOG_MAX_MB}MB,截断(尾部 1MB 存在 $LOGF.1)"
    fi
    find "$CMDD/out" -type f -mmin +10080 -delete 2>/dev/null || true

    if [ -f "$FDIR/stop_slot${SLOT}" ]; then
        _say "stop_slot${SLOT} 在:空转待命(删掉它即恢复;调试通道照常服务)"
        _sleep_serving 60
        continue
    fi
    if [ -f "$FDIR/parked_slot${SLOT}" ]; then
        _say "payload 曾声明完成(rc=42):挂起待命(task.sh go / set / swap 解除;调试通道照常)"
        _sleep_serving 60
        continue
    fi

    # 空转/退避期攒下的 swap、reload 到这里就算生效了(马上重读 payload),消费掉。
    # 千万不能把 swap 留给监督循环:它会把刚按你要求换上的新 payload 当场误杀 ——
    # 这正是审计里那个"空转期发 swap"的竞态。
    rm -f "$FDIR/reload_slot${SLOT}" "$FDIR/swap_slot${SLOT}"

    src=$(_pick_payload) || { _say "没有可用 payload,60s 后重试"; _sleep_serving 60 r; continue; }
    run=$RUND/payload.running.sh
    # 规则 1:快照。cp 到 running 副本再执行,跑到一半的改动只影响下一轮。
    if ! cp "$src" "$run.tmp" 2>/dev/null; then
        _say "payload 复制失败($src);60s 后重试"; _sleep_serving 60 r; continue
    fi
    # 规则 2:语法门。不过就退回上一版 last_good;没有 last_good 就等人修。
    if bash -n "$run.tmp" 2>"$RUND/syntax.err"; then
        mv "$run.tmp" "$run"
        cp "$run" "$RUND/last_good.sh"
    else
        _say "payload 语法不过($src):$(head -2 "$RUND/syntax.err" | tr '\n' ' ')"
        rm -f "$run.tmp"
        if [ -s "$RUND/last_good.sh" ]; then
            _say "退回上一版 last_good.sh"
            cp "$RUND/last_good.sh" "$run"
        else
            _say "也没有 last_good,120s 后重试(去跳板机修 $src)"; _sleep_serving 120 r; continue
        fi
    fi

    _say "第 ${_round} 轮:payload=$src  sha=$(md5sum "$run" 2>/dev/null | cut -c1-8)  $(git -C "$ROOT" log --oneline -1 2>/dev/null | cut -c1-40)"

    # setsid:payload 自成进程组,这样 swap/stop 能连它拉起的 lane 子进程一起收掉。
    # 不这么做的话 kill 只打到 payload 自己,verl/ray 会变成孤儿继续占着卡。
    t0=$(date +%s)
    # setsid 在 pod(Linux)上一定有;macOS 上没有,电池在那里跑时退化为普通后台任务 ——
    # 那时 kill -TERM -$pid 打不到组,只能打到 payload 自己。功能上够用,但孤儿清理弱,
    # 所以退化路径要说出来而不是静默生效。
    if command -v setsid >/dev/null 2>&1; then
        setsid bash "$run" &
    else
        _say "本机无 setsid,payload 不独立成组(孤儿清理会弱一些)"
        bash "$run" &
    fi
    _pg=$!
    _killed=""
    while kill -0 "$_pg" 2>/dev/null; do
        if [ -f "$FDIR/swap_slot${SLOT}" ] || [ -f "$FDIR/stop_slot${SLOT}" ]; then
            _killed=$([ -f "$FDIR/stop_slot${SLOT}" ] && echo stop || echo swap)
            _say "收到 ${_killed}:终止当前 payload 进程组(正在训练的 lane 会回退到最近检查点)"
            kill -TERM -"$_pg" 2>/dev/null || kill -TERM "$_pg" 2>/dev/null
            pkill -TERM -P "$_pg" 2>/dev/null
            for _i in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$_pg" 2>/dev/null || break; sleep 1; done
            kill -KILL -"$_pg" 2>/dev/null || kill -KILL "$_pg" 2>/dev/null
            pkill -KILL -P "$_pg" 2>/dev/null
            break
        fi
        _serve_cmds
        _beat
        sleep "${POLL_S:-3}"
    done
    wait "$_pg" 2>/dev/null; rc=$?
    rm -f "$FDIR/swap_slot${SLOT}"        # swap 是一次性的;stop 要人手删,它是持续状态
    [ -n "$_killed" ] && _scythe          # 组杀后补刀:清逃出组的 GPU 守护(真 pod 里才挥)
    dt=$(( $(date +%s) - t0 ))
    _say "第 ${_round} 轮结束 rc=$rc,历时 $((dt/60)) 分钟${_killed:+(被 ${_killed} 终止)}"
    # 被人为终止的不算"秒退",不该触发退避
    [ -n "$_killed" ] && { _backoff=0; continue; }

    # rc=42 = payload 声明"做完了":挂起该槽,不重跑也不算异常(一次性任务的正门)
    if [ "$rc" -eq 42 ]; then
        touch "$FDIR/parked_slot${SLOT}"
        _say "payload 返回 42=完成:挂起待命,不再重跑(task.sh go / 换 payload 解除)"
        _backoff=0
        continue
    fi

    # 规则 3:秒退退避。语法没问题但一跑就退(缺文件、缺环境变量……)同样会热循环。
    if [ "$dt" -lt "$MIN_RUN_S" ]; then
        _backoff=$(( _backoff == 0 ? BACKOFF_S0 : _backoff * 2 ))
        [ "$_backoff" -gt "$BACKOFF_MAX" ] && _backoff=$BACKOFF_MAX
        _say "不足 ${MIN_RUN_S}s 就返回,判为异常;退避 ${_backoff}s"
        _sleep_serving "$_backoff" r
    else
        _backoff=0
        _sleep_serving "$IDLE_S" r
    fi
done
