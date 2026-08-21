#!/usr/bin/env bash
# 在跳板机上给永续载体(deploy/dlc/forever.sh)换任务。不碰 DLC。
#
#   task.sh status                 各槽在跑什么、活没活(心跳 + 最后一行日志)
#   task.sh set   <槽> <脚本>      装上新 payload(仅装,下一轮生效)
#   task.sh swap  <槽> <脚本>      装上并立刻切(杀掉当前 payload)
#   task.sh stop  <槽>             立刻停,空转待命
#   task.sh go    <槽>             解除 stop / parked(rc=42 挂起)
#   task.sh reload <槽>            不杀正在跑的,只把空转/退避的等待掐短
#   task.sh clear <槽>             卸掉 payload,回到默认(corr_wave_fleet.sh)
#   task.sh sh    <槽> '<命令>'    在那个 pod 上执行并回显(像 ssh 一样调试)
#   task.sh sh    <槽> -f <脚本>   同上,但送一个脚本文件
#   task.sh watch <槽> <文件>      跟随文件(/mgfs 上的直接 tail -f;pod 本地的走通道)
#   task.sh tty   <槽>             真交互 shell(文件 PTY 桥):vim/top/python -i/cd 都行,Ctrl-] 断开
#   槽可以写 all(sh 除外)
#
# 单条调试命令 pod 侧默认 300s 超时;长命令用 CMD_T 放宽,如:
#   CMD_T=1800 bash task.sh sh 3 'python scripts/xxx.py'
#
# 为什么要这个而不是直接 vi:载体虽然有语法门和 last_good 兜底,但那是最后一道防线。
# 这里先 bash -n 再原子 mv 上去,让"写到一半被读到"根本不发生。
set -uo pipefail
D=${SIMOPD_STORE:-/mgfs/shared/Group_GY/changhao/simopd_data}
F=$D/forever
SLOTS_ALL="0 1 2 3 4 5 6"
mkdir -p "$F" 2>/dev/null || true

_slots() { [ "$1" = all ] && echo "$SLOTS_ALL" || echo "$1"; }
_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0; }
_mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0; }
# 边跑边把新产生的字节吐出来。命令的输出直接重定向进 .out,所以它是随写随长的;
# 等 .rc 出现才打印的话,一条跑十分钟的命令就十分钟不吭声,那不叫调试。
_stream_until() {   # 输出文件 完成标志文件 超时秒
    local out=$1 doneflag=$2 limit=$3 seen=0 sz i=0
    while [ ! -f "$doneflag" ] && [ "$i" -lt "$limit" ]; do
        if [ -f "$out" ]; then
            sz=$(_size "$out")
            [ "$sz" -gt "$seen" ] && { tail -c +$((seen + 1)) "$out"; seen=$sz; }
        fi
        sleep 1; i=$((i + 1))
    done
    [ -f "$out" ] && { sz=$(_size "$out"); [ "$sz" -gt "$seen" ] && tail -c +$((seen + 1)) "$out"; }
    [ -f "$doneflag" ]
}

_install() {   # 槽 脚本
    local k=$1 src=$2 dst="$F/payload_slot$1.sh"
    [ -s "$src" ] || { echo "!! $src 不存在或为空"; return 1; }
    bash -n "$src" || { echo "!! $src 语法不过,没装"; return 1; }
    cp "$src" "$dst.tmp" && mv "$dst.tmp" "$dst"     # 原子:载体永远读不到半截
    rm -f "$F/parked_slot$k"                          # 换了任务,rc=42 的挂起自然解除
    echo "slot$k <- $src  ($(md5sum "$dst" | cut -c1-8))"
}

case "${1:-status}" in
  status)
    echo "载体目录 $F"
    now=$(date +%s)
    for k in $SLOTS_ALL; do
        p="(默认 corr_wave_fleet.sh)"
        [ -s "$F/payload.sh" ] && p="(全局 payload.sh)"
        [ -s "$F/payload_slot$k.sh" ] && p="payload_slot$k.sh $(md5sum "$F/payload_slot$k.sh" | cut -c1-8)"
        st=""
        [ -f "$F/stop_slot$k" ] && st=" [STOP]"
        [ -f "$F/parked_slot$k" ] && st="$st [PARKED rc=42 完成,go 解除]"
        [ -f "$F/swap_slot$k" ] && st="$st [swap 待生效]"
        hb="心跳:无记录(载体大概没起来过)"
        if [ -f "$F/hb_slot$k" ]; then
            a=$(( now - $(_mtime "$F/hb_slot$k") ))
            if [ "$a" -le "${HB_STALE_S:-90}" ]; then hb="活着(心跳 ${a}s 前)"
            else hb="疑似死了(心跳 ${a}s 前)"; fi
        fi
        age=""
        run="$F/run/slot$k/payload.running.sh"
        [ -f "$run" ] && age=" 本轮 $(( (now - $(_mtime "$run")) / 60 ))m 前起"
        echo "  slot$k  $hb  $p$st$age"
        lg="$F/log/slot$k.log"
        [ -s "$lg" ] && echo "         └ $(tail -c 4096 "$lg" | tail -1 | cut -c1-150)"
    done
    ;;
  set)   for k in $(_slots "${2:?槽}"); do _install "$k" "${3:?脚本}"; done ;;
  swap)  for k in $(_slots "${2:?槽}"); do _install "$k" "${3:?脚本}" && touch "$F/swap_slot$k" && echo "  -> swap_slot$k 已触发(当前 payload 会被终止;训练中的 lane 回退到最近检查点)"; done ;;
  stop)  for k in $(_slots "${2:?槽}"); do touch "$F/stop_slot$k"; echo "slot$k 停(空转待命),恢复用: $0 go $k"; done ;;
  go)    for k in $(_slots "${2:?槽}"); do rm -f "$F/stop_slot$k" "$F/parked_slot$k"; echo "slot$k 恢复(stop/parked 都已解除)"; done ;;
  reload) for k in $(_slots "${2:?槽}"); do touch "$F/reload_slot$k"; echo "slot$k reload:正在跑的不动,空转/退避的立刻进下一轮"; done ;;
  clear) for k in $(_slots "${2:?槽}"); do rm -f "$F/payload_slot$k.sh" "$F/parked_slot$k"; echo "slot$k 卸掉 payload,下一轮回到默认"; done ;;
  sh)
    # 同步调试:写进 inbox,pod 后台执行,这边流式回显,等 .rc 出现(它最后写,
    # 所以读到时 .out 一定完整)。载体跑 payload / stop 空转 / 退避时都服务命令。
    k=${2:?槽}; shift 2
    [ "$k" = all ] && { echo "!! sh 一次只对一个槽"; exit 2; }
    C=$F/cmd/slot$k; mkdir -p "$C/inbox" "$C/out"
    id=$(date +%s)-$$-$RANDOM
    if [ "${1:-}" = -f ]; then
        [ -s "${2:?脚本}" ] || { echo "!! $2 不存在或为空"; exit 1; }
        bash -n "$2" || { echo "!! 语法不过,没发"; exit 1; }
        cp "$2" "$C/inbox/.$id.sh"
    else
        printf '%s\n' "${*:?命令}" > "$C/inbox/.$id.sh"
    fi
    # 超时随件先落地(.t 不触发执行),再原子亮出 .sh —— pod 取件时随件一定已就位
    [ -n "${CMD_T:-}" ] && printf '%s\n' "$CMD_T" > "$C/inbox/$id.t"
    mv "$C/inbox/.$id.sh" "$C/inbox/$id.sh"      # 原子出现,pod 不会取到半截
    if _stream_until "$C/out/$id.out" "$C/out/$id.rc" "${WAIT_S:-300}"; then
        rc=$(cat "$C/out/$id.rc")
        [ "$rc" = 0 ] || echo "[rc=$rc]"
        rm -f "$C/out/$id".{sh,out,rc,t}
        exit "$rc"
    fi
    echo "!! 等了 ${WAIT_S:-300}s 没等到结束 —— 命令还在跑,或 slot$k 的载体没起来(task.sh status)"
    echo "   命令仍在 pod 上跑完;输出留在 $C/out/$id.out"
    exit 124
    ;;
  watch)
    # 跟随一个文件。/mgfs 上的东西跳板机本来就能直接读 —— 舰队日志全在那儿,
    # 所以那种情况根本不用走通道,直接 tail -f,零延迟。只有 pod 本地的路径
    # (/tmp、/root 之类)才需要通过命令通道去 tail。走通道时把 pod 侧超时一起
    # 放宽(CMD_T),否则 pod 默认 300s 会把承诺 600s 的跟随拦腰砍断。
    k=${2:?槽}; path=${3:?文件}
    if [ -e "$path" ]; then
        echo "== $path 在共享盘上,直接跟随(Ctrl-C 停)"
        exec tail -f "$path"
    fi
    echo "== $path 是 slot$k 的 pod 本地路径,经命令通道跟随 ${WATCH_S:-600}s(Ctrl-C 停)"
    WAIT_S=$(( ${WATCH_S:-600} + 30 )) CMD_T=$(( ${WATCH_S:-600} + 20 )) exec "$0" sh "$k" \
        "timeout ${WATCH_S:-600} tail -n ${WATCH_N:-40} -F '$path'"
    ;;
  tty)
    # 真交互 shell(实验性):文件 PTY 桥 —— pod 在 PTY 里 fork bash,输入输出走共享盘
    # 两个字节流文件;不要网络出站、不要凭据。经调试通道 setsid 拉起 server(脱离单条
    # 命令的 timeout),然后本地 attach。一场会话一个新 bash,断开即弃(不是 tmux)。
    k=${2:?槽}
    [ "$k" = all ] && { echo "!! tty 一次只对一个槽"; exit 2; }
    ROOT=${ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
    S=$F/tty/slot$k/$(date +%s)-$$
    mkdir -p "$S"
    find "$F/tty" -mindepth 2 -maxdepth 2 -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    bash "$0" sh "$k" "setsid python3 $ROOT/deploy/dlc/ptybridge.py serve '$S' >/dev/null 2>&1 </dev/null & echo TTY-SPAWN-OK" \
        | grep -q TTY-SPAWN-OK || { echo "!! 没能经调试通道拉起 ptybridge(task.sh status 看载体活没活)"; exit 1; }
    for i in $(seq 40); do [ -f "$S/shb" ] && break; sleep 0.5; done
    [ -f "$S/shb" ] || { echo "!! server 没起来(pod 上 python3 不可用?看 task.sh sh $k 'command -v python3')"; exit 1; }
    exec python3 "$(cd "$(dirname "$0")" && pwd)/ptybridge.py" attach "$S"
    ;;
  *) sed -n '2,23p' "$0"; exit 2 ;;
esac
