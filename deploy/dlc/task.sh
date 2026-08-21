#!/usr/bin/env bash
# 在跳板机上给永续载体(deploy/dlc/forever.sh)换任务。不碰 DLC。
#
#   task.sh status                 各槽在跑什么、活没活
#   task.sh set   <槽> <脚本>      装上新 payload(仅装,下一轮生效)
#   task.sh swap  <槽> <脚本>      装上并立刻切(杀掉当前 payload)
#   task.sh stop  <槽>             立刻停,空转待命
#   task.sh go    <槽>             解除 stop
#   task.sh clear <槽>             卸掉 payload,回到默认(corr_wave_fleet.sh)
#   task.sh sh    <槽> '<命令>'    在那个 pod 上执行并回显(像 ssh 一样调试)
#   task.sh sh    <槽> -f <脚本>   同上,但送一个脚本文件
#   task.sh watch <槽> <文件>      跟随文件(/mgfs 上的直接 tail -f;pod 本地的走通道)
#   槽可以写 all
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
    echo "slot$k <- $src  ($(md5sum "$dst" | cut -c1-8))"
}

case "${1:-status}" in
  status)
    echo "载体目录 $F"
    for k in $SLOTS_ALL; do
        p="(默认 corr_wave_fleet.sh)"
        [ -s "$F/payload.sh" ] && p="(全局 payload.sh)"
        [ -s "$F/payload_slot$k.sh" ] && p="payload_slot$k.sh $(md5sum "$F/payload_slot$k.sh" | cut -c1-8)"
        st=""
        [ -f "$F/stop_slot$k" ] && st=" [STOP]"
        [ -f "$F/swap_slot$k" ] && st="$st [swap 待生效]"
        run="$F/run/slot$k/payload.running.sh"
        age=""
        [ -f "$run" ] && age=" 本轮 $(( ($(date +%s) - $(stat -c %Y "$run")) / 60 ))m 前起"
        echo "  slot$k  $p$st$age"
    done
    ;;
  set)   for k in $(_slots "${2:?槽}"); do _install "$k" "${3:?脚本}"; done ;;
  swap)  for k in $(_slots "${2:?槽}"); do _install "$k" "${3:?脚本}" && touch "$F/swap_slot$k" && echo "  -> swap_slot$k 已触发(当前 payload 会被终止;训练中的 lane 回退到最近检查点)"; done ;;
  stop)  for k in $(_slots "${2:?槽}"); do touch "$F/stop_slot$k"; echo "slot$k 停(空转待命),恢复用: $0 go $k"; done ;;
  go)    for k in $(_slots "${2:?槽}"); do rm -f "$F/stop_slot$k"; echo "slot$k 恢复"; done ;;
  clear) for k in $(_slots "${2:?槽}"); do rm -f "$F/payload_slot$k.sh"; echo "slot$k 卸掉 payload,下一轮回到默认"; done ;;
  sh)
    # 同步调试:写进 inbox,等 .rc 出现(它最后写,所以读到时 .out 一定完整),打印输出。
    # 载体在跑 payload / stop 空转 / 退避时都服务命令 —— 想调试的时刻正是这些时刻。
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
    mv "$C/inbox/.$id.sh" "$C/inbox/$id.sh"      # 原子出现,pod 不会取到半截
    if _stream_until "$C/out/$id.out" "$C/out/$id.rc" "${WAIT_S:-300}"; then
        rc=$(cat "$C/out/$id.rc")
        [ "$rc" = 0 ] || echo "[rc=$rc]"
        rm -f "$C/out/$id".{sh,out,rc}
        exit "$rc"
    fi
    echo "!! 等了 ${WAIT_S:-300}s 没等到结束 —— 命令还在跑,或 slot$k 的载体没起来(task.sh status)"
    echo "   命令仍在 pod 上跑完;输出留在 $C/out/$id.out"
    exit 124
    ;;
  watch)
    # 跟随一个文件。/mgfs 上的东西跳板机本来就能直接读 —— 舰队日志全在那儿,
    # 所以那种情况根本不用走通道,直接 tail -f,零延迟。只有 pod 本地的路径
    # (/tmp、/root 之类)才需要通过命令通道去 tail。
    k=${2:?槽}; path=${3:?文件}
    if [ -e "$path" ]; then
        echo "== $path 在共享盘上,直接跟随(Ctrl-C 停)"
        exec tail -f "$path"
    fi
    echo "== $path 是 slot$k 的 pod 本地路径,经命令通道跟随 ${WATCH_S:-600}s(Ctrl-C 停)"
    WAIT_S=$(( ${WATCH_S:-600} + 30 )) exec "$0" sh "$k" \
        "timeout ${WATCH_S:-600} tail -n ${WATCH_N:-40} -F '$path'"
    ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
