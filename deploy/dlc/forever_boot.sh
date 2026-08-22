#!/usr/bin/env bash
# DLC 执行命令建议用这个,不直接用 forever.sh:外面多一圈进程级看门 ——
# 载体 bash 万一被打死(payload 误伤 pkill、OOM 波及),只要 pod 还活着就地重拉;
# 每次重拉都重新读 forever.sh,载体代码升级随重启自动生效。
#
# 它管不了 pod 整个没了(节点故障/平台时限/控制台停止)—— 那一层要靠 DLC 提交表单里
# 的「自动重启/容错策略」,以及断后重投:状态全在共享盘,重投同一条命令即无缝接管。
set -u
ROOT=${ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
n=0
while true; do
    n=$((n + 1))
    echo "[forever_boot] 第 ${n} 次拉起载体 $(date '+%F %T')"
    rc=0; bash "$ROOT/deploy/dlc/forever.sh" || rc=$?
    # rc=0 只有卡片模式(非 pod 环境)会出现 —— 别在跳板机上无限刷卡片
    [ "$rc" -eq 0 ] && { echo "[forever_boot] 载体 rc=0(卡片模式),不重拉"; exit 0; }
    echo "[forever_boot] 载体退出 rc=$rc,${BOOT_RETRY_S:-30}s 后重拉(pod 未死就不放弃)"
    sleep "${BOOT_RETRY_S:-30}"
done
