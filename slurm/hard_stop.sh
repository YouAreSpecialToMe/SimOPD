#!/usr/bin/env bash
# 到点无条件停:杀调度器 + scancel 所有 simopd 训练作业。
# 用户 2026-09-05 指令:"跑8个小时一定要停,无论跑的怎么样了"。
#   DEADLINE=<epoch> setsid nohup bash slurm/hard_stop.sh >> <log> 2>&1 &
# 做成独立进程而不是 cron/会话任务:cron 只在 REPL 空闲时触发、会话结束就没了,
# 而"无论如何"要求这个停机不能依赖任何一个会话还活着。
set -u
cd $SIMOPD_ROOT
. ./simopd_env.sh >/dev/null 2>&1
# 不要依赖环境里的 USER:crontab 拉起时环境是最小集,USER 可能根本没有。
# 2026-09-06 就是这么炸的 —— 看门狗查不到守卫 -> 每 5 分钟多起一个守卫 ->
# 每个守卫起一个调度器 -> 每个调度器 `squeue -u ""` 查不到作业、以为 0/4 ->
# 各自投满 4 个,一共超投到 7 个作业。id -un 不依赖环境,永远对。
ME=$(id -un)
DEADLINE=${DEADLINE:?need epoch seconds}
say() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }

say "hard stop 定时器启动,截止 $(date -u -d "@$DEADLINE" '+%F %H:%M UTC'),还有 $(( (DEADLINE - $(date +%s)) / 60 )) 分钟"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    left=$(( DEADLINE - $(date +%s) ))
    [ "$left" -gt 600 ] && sleep 300 || sleep 10
done

say "=== 到点,开始停机 ==="
# 先杀调度器,否则它三分钟内把作业重投回来
S=$(ps -u "$ME" -o pid= -o args= | grep -F 'campaign_schedul''er.sh' | grep -v grep | awk '{print $1}')
[ -n "$S" ] && { kill $S; say "调度器 $S 已停"; } || say "调度器本来就没在跑"
sleep 3

J=$(squeue -u "$ME" -h -o "%i %j" | awk '$2 ~ /^simopd/ {print $1}' | tr '\n' ' ')
if [ -n "$J" ]; then scancel $J; say "已取消作业: $J"; else say "没有在跑的 simopd 作业"; fi
sleep 30

say "=== 停机后状态 ==="
squeue -u "$ME" -o "%.7i %.20j %.9T %R" | sed 's/^/  /'
for d in "${CKPT_ROOT:-$SIMOPD_STORE/ckpt}"/simopd/*_s0; do
    [ -d "$d" ] || continue
    say "  $(basename "$d") 续跑点 @$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 无)"
done
say "=== 完 ==="
