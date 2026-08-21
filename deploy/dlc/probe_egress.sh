#!/usr/bin/env bash
# 从 pod 里探网络出站:反向 SSH 隧道(真 shell 的 A 路)可行性的第一问。
# B 路(文件 PTY 桥,task.sh tty)不依赖这个;这只决定要不要再上隧道换更低延迟。
#
# 用法(载体上线后):bash deploy/dlc/task.sh sh 3 -f deploy/dlc/probe_egress.sh
# 目标列表:$D/forever/probe_targets.txt,每行 host:port(# 开头为注释)
set -uo pipefail
D=${SIMOPD_STORE:-/mgfs/shared/Group_GY/changhao/simopd_data}
L=$D/forever/probe_targets.txt
echo "== pod $(hostname 2>/dev/null) 出站探测 $(date '+%F %T')"
echo "-- 本机地址: $(hostname -I 2>/dev/null || echo '?')"
[ -s "$L" ] || { echo "!! 没有 $L(host:port 每行一个)"; exit 1; }
while IFS=: read -r h p; do
    case "$h" in ''|\#*) continue ;; esac
    if timeout 4 bash -c "exec 3<>/dev/tcp/$h/$p" 2>/dev/null; then
        echo "  通   $h:$p"
    else
        echo "  不通 $h:$p"
    fi
done < "$L"
command -v ssh >/dev/null 2>&1 && echo "-- pod 里有 ssh 客户端" || echo "-- pod 里没有 ssh 客户端(A 路要装)"
