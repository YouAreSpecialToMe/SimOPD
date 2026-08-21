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
#   touch $D/forever/reload_slot3        # 让 slot3 当前这轮结束后立刻重读(不加也行,跑完自然重读)
#   touch $D/forever/stop_slot3          # 让 slot3 空转待命(不杀正在跑的那轮)
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
        _say "stop_slot${SLOT} 在:空转待命(删掉它即恢复)"
        sleep 60
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

    t0=$(date +%s)
    bash "$run"
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    _say "第 ${_round} 轮结束 rc=$rc,历时 $((dt/60)) 分钟"

    # 规则 3:秒退退避。语法没问题但一跑就退(缺文件、缺环境变量……)同样会热循环。
    if [ "$dt" -lt "$MIN_RUN_S" ]; then
        _backoff=$(( _backoff == 0 ? BACKOFF_S0 : _backoff * 2 ))
        [ "$_backoff" -gt "$BACKOFF_MAX" ] && _backoff=$BACKOFF_MAX
        _say "不足 ${MIN_RUN_S}s 就返回,判为异常;退避 ${_backoff}s"
        sleep "$_backoff"
    else
        _backoff=0
        sleep "$IDLE_S"
    fi
done
