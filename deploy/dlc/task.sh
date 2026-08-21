#!/usr/bin/env bash
# 在跳板机上给永续载体(deploy/dlc/forever.sh)换任务。不碰 DLC。
#
#   task.sh status                 各槽在跑什么、活没活
#   task.sh set   <槽> <脚本>      装上新 payload(仅装,下一轮生效)
#   task.sh swap  <槽> <脚本>      装上并立刻切(杀掉当前 payload)
#   task.sh stop  <槽>             立刻停,空转待命
#   task.sh go    <槽>             解除 stop
#   task.sh clear <槽>             卸掉 payload,回到默认(corr_wave_fleet.sh)
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
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
