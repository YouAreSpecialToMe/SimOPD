#!/usr/bin/env bash
# 永不退出的 DLC 载体。提交一次,之后换任务 = 在跳板机上改共享树里的 payload 脚本。
#
# 为什么要它:corr 波这套是"一次 dlc submit 跑一份 lane 表",换实验就得重投 DLC 或者
# 从 DSW 上重新 launch,而 DLC 的凭据只在 pod 里、跳板机上没有(设计如此)。于是每次
# 换任务都卡在人身上。这个载体把 DLC 作业变成一台常驻机器:它唯一的职责是反复执行
# 共享盘上的 payload,谁改了 payload,下一轮就跑谁。
#
# 提交(在 DLC 控制台,一次):
#   SLOT=auto bash /mgfs/shared/Group_GY/changhao/SimOPD-exp/deploy/dlc/forever.sh
#
# 之后在跳板机上换任务(不碰 DLC):
#   vi  $D/forever/payload_slot3.sh      # 只给 slot3 换
#   vi  $D/forever/payload.sh            # 给所有槽换(每槽 payload 优先)
#   touch $D/forever/swap_slot3          # 立刻换:终止 slot3 正在跑的 payload 并重读
#   touch $D/forever/stop_slot3          # 立刻停:终止并空转待命,直到删掉标记
#   touch $D/forever/reload_slot3        # 温和换:等这一轮自然结束再重读
#
# swap 和 stop 是"立刻"的,因为 payload 可能永远不结束 —— corr_wave_fleet.sh 的臂全部
# 跑满后会进它自己的 holding 死循环,温和的 reload 就永远等不到那一刻,人还得回去重投
# DLC,那正是这个载体要消灭的事。代价说清楚:正在训练时 swap/stop 会让那些 lane 回退
# 到最近检查点(<=25 步),所以它是明确的人为动作,不是自动行为。
#
# 调试通道(像 ssh 那样敲一条看一眼,不用等下一轮):
#   跳板机:  bash deploy/dlc/task.sh sh 3 'nvidia-smi; ls $D/corr_wave | tail'
#   原理:命令写进 $D/forever/cmd/slot3/inbox/,pod 在轮询间隙执行,输出回写 out/。
#   载体在跑 payload 时、stop 空转时、退避时都服务命令 —— 想调试的时刻恰恰是这些时刻。
#
# payload 契约:一个普通 bash 脚本,能拿到 SLOT / SEED / ROOT / D / LOGD;正常返回即
# 本轮结束,载体休息 IDLE_S 秒后重读。不设 payload 时跑 corr_wave_fleet.sh(即现状)。
#
# 三条保命规则,都是被真事故逼出来的:
#  1 快照后再跑。bash 是边读边执行的,你在跳板机上改到一半、pod 正好读到半个文件,
#    行为无法预测。所以每轮先把 payload 复制成 running 副本再执行 —— 跑到一半的改动
#    只影响下一轮。
#  2 先 bash -n 再跑,不过就退回上一版 last_good。一个语法错误如果直接执行,载体会
#    进入"秒退 -> 重跑 -> 秒退"的热循环,把日志刷爆而看上去像在工作。
#  3 秒退要退避。payload 若在 MIN_RUN_S 内返回,视为异常,退避时间翻倍(上限 15 分钟);
#    正常跑完则重置。
set -uo pipefail

ROOT=${ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
D=${SIMOPD_STORE:-/mgfs/shared/Group_GY/changhao/simopd_data}
SEED=${SEED:-0}
FDIR=$D/forever
IDLE_S=${IDLE_S:-60}          # 一轮正常结束后歇多久再重读
MIN_RUN_S=${MIN_RUN_S:-120}   # 短于这个算秒退
BACKOFF_S0=${BACKOFF_S0:-60}  # 首次退避;之后翻倍。可配置是为了能被电池验证 ——
BACKOFF_MAX=${BACKOFF_MAX:-900}  # 写死 60 的话,只要上限低于它就永远看不出有没有在翻倍

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

# ---- 调试通道 ----------------------------------------------------------------
# 请求 = inbox 里的一个 .sh;完成 = out/<id>.rc 出现(最后写,所以读到 .rc 时 .out 一定完整)。
# 每条命令都套 timeout:一条 hang 住的命令不能把整个轮询循环卡死,否则 swap/stop 也失灵。
CMDD=$FDIR/cmd/slot${SLOT}
mkdir -p "$CMDD/inbox" "$CMDD/out" 2>/dev/null || true
_serve_cmds() {
    local f id
    for f in "$CMDD/inbox"/*.sh; do
        [ -e "$f" ] || continue
        id=$(basename "$f" .sh)
        mv "$f" "$CMDD/out/$id.sh" 2>/dev/null || continue   # 原子取走,避免重复执行
        _say "调试命令 $id 开始"
        ( cd "$ROOT" 2>/dev/null || cd /; \
          SLOT=$SLOT SEED=$SEED ROOT=$ROOT D=$D LOGD=$LOGD \
          timeout "${CMD_TIMEOUT:-300}" bash "$CMDD/out/$id.sh" ) \
            > "$CMDD/out/$id.out" 2>&1
        echo $? > "$CMDD/out/$id.rc"                          # 最后写 = 完成信号
        _say "调试命令 $id 结束 rc=$(cat "$CMDD/out/$id.rc")"
    done
}
_sleep_serving() {   # 边睡边服务命令 —— 想调试的时刻正是载体在等的时刻
    local left=$1
    while [ "$left" -gt 0 ]; do
        _serve_cmds
        sleep $(( left < ${POLL_S:-3} ? left : ${POLL_S:-3} ))
        left=$(( left - ${POLL_S:-3} ))
    done
}

# 提交端(没有 DLC rank env)只打控制台卡片,不跑
if [ -z "${MLP_ROLE_INDEX:-}${MLP_WORKER_RACK_RANK_INDEX:-}${DLC_JOB_ID:-}" ] && [ -z "${FOREVER_FORCE:-}" ]; then
    cat <<CARD
========= 永续载体:DLC 控制台卡片(提交一次,之后不用再碰 DLC)=========
  任务名称   simopd-forever-s${SEED}
  节点数量   7            单节点GPU 8   CPU 64   内存 512Gi
  镜像/资源组/挂载:照抄 simopd-corr-wave1 的成功表单(挂载须含 /mgfs)
  执行命令:
    SLOT=auto SEED=${SEED} bash $ROOT/deploy/dlc/forever.sh

  提交后在跳板机上换任务(不碰 DLC):
    \$D/forever/payload_slot<k>.sh   给某个槽换任务(优先)
    \$D/forever/payload.sh           给所有槽换任务
    touch \$D/forever/reload_slot<k>  本轮跑完立刻重读
    touch \$D/forever/stop_slot<k>    该槽空转待命
  不放 payload 时,每轮跑 deploy/dlc/corr_wave_fleet.sh(= 现在的行为)
CARD
    exit 0
fi

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

    if [ -f "$FDIR/stop_slot${SLOT}" ]; then
        _say "stop_slot${SLOT} 在:空转待命(删掉它即恢复;调试通道照常服务)"
        _sleep_serving 60
        continue
    fi

    src=$(_pick_payload) || { _say "没有可用 payload,60s 后重试"; sleep 60; continue; }
    run=$RUND/payload.running.sh
    # 规则 1:快照。cp 到 running 副本再执行,跑到一半的改动只影响下一轮。
    if ! cp "$src" "$run.tmp" 2>/dev/null; then
        _say "payload 复制失败($src);60s 后重试"; sleep 60; continue
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
            _say "也没有 last_good,120s 后重试(去跳板机修 $src)"; sleep 120; continue
        fi
    fi

    rm -f "$FDIR/reload_slot${SLOT}"     # 本轮已经重读过了
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
        sleep "${POLL_S:-3}"
    done
    wait "$_pg" 2>/dev/null; rc=$?
    rm -f "$FDIR/swap_slot${SLOT}"        # swap 是一次性的;stop 要人手删,它是持续状态
    dt=$(( $(date +%s) - t0 ))
    _say "第 ${_round} 轮结束 rc=$rc,历时 $((dt/60)) 分钟${_killed:+(被 ${_killed} 终止)}"
    # 被人为终止的不算"秒退",不该触发退避
    [ -n "$_killed" ] && { _backoff=0; continue; }

    # 规则 3:秒退退避。语法没问题但一跑就退(缺文件、缺环境变量……)同样会热循环。
    if [ "$dt" -lt "$MIN_RUN_S" ]; then
        _backoff=$(( _backoff == 0 ? BACKOFF_S0 : _backoff * 2 ))
        [ "$_backoff" -gt "$BACKOFF_MAX" ] && _backoff=$BACKOFF_MAX
        _say "不足 ${MIN_RUN_S}s 就返回,判为异常;退避 ${_backoff}s"
        _sleep_serving "$_backoff"
    else
        _backoff=0
        _sleep_serving "$IDLE_S"
    fi
done
