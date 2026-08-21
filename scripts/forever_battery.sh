#!/usr/bin/env bash
# 永续载体的离线电池 —— 不碰 GPU、不碰 DLC,只验载体语义。
#
#   bash scripts/forever_battery.sh
#
# 载体无人值守,失败模式不是"不工作"而是"看起来在工作":语法错的 payload 秒退→重跑→
# 秒退,日志刷爆而什么都没跑。所以逐条钉:快照隔离、语法门、退回 last_good、秒退退避、
# stop/reload 标记、每槽 payload 优先。
set -uo pipefail
PASS=0
ok() { if [ "$1" = 1 ]; then PASS=$((PASS+1)); echo "  ok  $2"; else echo "FAIL: $2"; exit 1; fi; }

SRC=$(cd "$(dirname "$0")/.." && pwd)/deploy/dlc/forever.sh
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
export ROOT=$T/tree SIMOPD_STORE=$T/data SEED=0 SLOT=3 FOREVER_FORCE=1
export IDLE_S=1 MIN_RUN_S=3 BACKOFF_S0=1 BACKOFF_MAX=4 POLL_S=1
F=$T/data/forever; R=$F/run/slot3
mkdir -p "$ROOT/deploy/dlc" "$T/data/corr_wave" "$F"
# 兜底 payload(载体在没有 payload 时会跑它)
printf '#!/usr/bin/env bash\necho FALLBACK-RAN\nsleep 4\n' > "$ROOT/deploy/dlc/corr_wave_fleet.sh"

# 载体是死循环,所以每次都在后台跑一小段再收掉
run_for() {  # 秒数 -> 输出到 $T/out
    : > "$T/out"
    bash "$SRC" > "$T/out" 2>&1 &
    local pid=$!
    sleep "$1"
    kill -TERM $pid 2>/dev/null; pkill -P $pid 2>/dev/null; sleep 0.3; kill -KILL $pid 2>/dev/null
    wait $pid 2>/dev/null || true
}

# --- 1. 没有 payload 时跑兜底
run_for 6
ok $(grep -q "FALLBACK-RAN" "$T/out" && echo 1 || echo 0) "无 payload 时跑 corr_wave_fleet.sh 兜底"

# --- 2. 每槽 payload 优先于全局 payload
printf '#!/usr/bin/env bash\necho GLOBAL-RAN\nsleep 4\n' > "$F/payload.sh"
printf '#!/usr/bin/env bash\necho SLOT3-RAN\nsleep 4\n' > "$F/payload_slot3.sh"
run_for 6
ok $(grep -q "SLOT3-RAN" "$T/out" && echo 1 || echo 0) "payload_slot3.sh 优先于 payload.sh"
ok $(grep -q "GLOBAL-RAN" "$T/out" && echo 0 || echo 1) "全局 payload 未被执行"

# --- 3. 快照隔离:执行的是副本,不是原件
ok $([ -s "$R/payload.running.sh" ] && echo 1 || echo 0) "跑的是快照副本 payload.running.sh"
ok $(cmp -s "$R/payload.running.sh" "$F/payload_slot3.sh" && echo 1 || echo 0) "快照内容与原件一致"
ok $([ -s "$R/last_good.sh" ] && echo 1 || echo 0) "通过语法门的版本存为 last_good.sh"

# --- 4. 语法门 + 退回 last_good:半截文件不该被执行
printf '#!/usr/bin/env bash\nif [ 1 = 1 ; then echo BROKEN-RAN\n' > "$F/payload_slot3.sh"
run_for 6
ok $(grep -q "BROKEN-RAN" "$T/out" && echo 0 || echo 1) "语法错的 payload 从未被执行"
ok $(grep -q "语法不过" "$T/out" && echo 1 || echo 0) "语法不过有明确日志,不是静默跳过"
ok $(grep -q "退回上一版" "$T/out" && echo 1 || echo 0) "退回 last_good.sh"
ok $(grep -q "SLOT3-RAN" "$T/out" && echo 1 || echo 0) "退回后真的跑了上一版"

# --- 5. 秒退退避:不能热循环
printf '#!/usr/bin/env bash\necho QUICK-EXIT\nexit 1\n' > "$F/payload_slot3.sh"
rm -f "$R/last_good.sh"
run_for 7
n=$(grep -c "QUICK-EXIT" "$T/out" || true)
ok $(grep -q "判为异常;退避" "$T/out" && echo 1 || echo 0) "秒退被判为异常并退避"
ok $([ "$n" -le 3 ] && echo 1 || echo 0) "7 秒内最多跑 3 轮(退避生效,得 $n 轮)"
seq=$(grep -o "退避 [0-9]*s" "$T/out" | grep -o "[0-9]*" | tr '\n' ' ')
ok $(echo "$seq" | grep -q "^1 2" && echo 1 || echo 0) "退避在翻倍(得 [$seq],期望 1 2 4 ...)"
ok $(echo "$seq" | tr ' ' '\n' | awk '$1>4{f=1} END{exit f?1:0}' && echo 1 || echo 0) "退避不超过 BACKOFF_MAX=4"

# --- 6. stop 标记:空转待命,不执行 payload
printf '#!/usr/bin/env bash\necho SHOULD-NOT-RUN\nsleep 4\n' > "$F/payload_slot3.sh"
touch "$F/stop_slot3"
run_for 4
ok $(grep -q "SHOULD-NOT-RUN" "$T/out" && echo 0 || echo 1) "stop 标记在时不执行 payload"
ok $(grep -q "空转待命" "$T/out" && echo 1 || echo 0) "空转有日志"
rm -f "$F/stop_slot3"

# --- 7. reload 标记被消费(否则会被误认为一直有新任务)
touch "$F/reload_slot3"
run_for 5
ok $([ ! -f "$F/reload_slot3" ] && echo 1 || echo 0) "reload 标记跑完一轮后被消费"

# --- 8. 槽号从 rank 推导
unset SLOT; export RANK=5
printf '#!/usr/bin/env bash\necho RANK5\nsleep 4\n' > "$F/payload_slot5.sh"
mkdir -p "$F/run/slot5"
run_for 6
ok $(grep -q "RANK5" "$T/out" && echo 1 || echo 0) "SLOT=auto 时按 RANK 推导槽号(RANK=5 -> slot5)"
ok $(grep -q "slot5:" "$T/out" && echo 1 || echo 0) "日志带槽号前缀"

# --- 9. swap:payload 陷在死循环里也能被换掉(本次改动的全部意义)
# 现实原型:corr_wave_fleet.sh 的臂全跑满后进 "wave complete, holding" 死循环,
# 温和的 reload 永远等不到那一刻 —— 不能换就还得回去重投 DLC。
rm -rf "$F"; mkdir -p "$F/run/slot3"
unset RANK; export SLOT=3
printf '#!/usr/bin/env bash\necho LOOPER-STARTED\nwhile true; do sleep 1; done\n' > "$F/payload_slot3.sh"
: > "$T/out"
bash "$SRC" > "$T/out" 2>&1 & BG=$!
sleep 4
ok $(grep -q "LOOPER-STARTED" "$T/out" && echo 1 || echo 0) "死循环 payload 已起来"
printf '#!/usr/bin/env bash\necho NEWTASK-RAN\nsleep 30\n' > "$F/payload_slot3.sh"
touch "$F/swap_slot3"
sleep 6
ok $(grep -q "收到 swap" "$T/out" && echo 1 || echo 0) "swap 被识别"
ok $(grep -q "NEWTASK-RAN" "$T/out" && echo 1 || echo 0) "死循环被终止后跑起了新 payload"
ok $([ ! -f "$F/swap_slot3" ] && echo 1 || echo 0) "swap 是一次性的,已被消费"
ok $(grep -q "判为异常;退避" "$T/out" && echo 0 || echo 1) "人为终止不算秒退,未触发退避"
kill -TERM $BG 2>/dev/null; pkill -P $BG 2>/dev/null; sleep 0.3; kill -KILL $BG 2>/dev/null; wait $BG 2>/dev/null || true

# --- 10. 进程组连坐:payload 拉起的子进程也要被收掉(否则 verl/ray 变孤儿占着卡)
rm -rf "$F"; mkdir -p "$F/run/slot3"
MARK=$T/child_alive
printf '#!/usr/bin/env bash\n( while true; do touch %s; sleep 1; done ) &\necho PARENT-UP\nwhile true; do sleep 1; done\n' "$MARK" > "$F/payload_slot3.sh"
: > "$T/out"
bash "$SRC" > "$T/out" 2>&1 & BG=$!
sleep 4
ok $([ -f "$MARK" ] && echo 1 || echo 0) "payload 的子进程在跑(留下心跳文件)"
touch "$F/stop_slot3"; sleep 6
rm -f "$MARK"; sleep 3
ok $([ ! -f "$MARK" ] && echo 1 || echo 0) "stop 之后子进程也死了(进程组连坐,不留孤儿)"
ok $(grep -q "空转待命" "$T/out" && echo 1 || echo 0) "stop 后进入空转待命"
kill -TERM $BG 2>/dev/null; pkill -P $BG 2>/dev/null; sleep 0.3; kill -KILL $BG 2>/dev/null; wait $BG 2>/dev/null || true

# --- 11. 调试通道端到端:payload 在跑的同时,敲命令要能回显
rm -rf "$F"; mkdir -p "$F/run/slot3"
unset RANK; export SLOT=3
printf '#!/usr/bin/env bash\necho BUSY-PAYLOAD\nwhile true; do sleep 1; done\n' > "$F/payload_slot3.sh"
: > "$T/out"
bash "$SRC" > "$T/out" 2>&1 & BG=$!
sleep 4
ok $(grep -q "BUSY-PAYLOAD" "$T/out" && echo 1 || echo 0) "长跑 payload 已起来"

TASK=$(cd "$(dirname "$0")/.." && pwd)/deploy/dlc/task.sh
export SIMOPD_STORE=$T/data WAIT_S=20
r=$(bash "$TASK" sh 3 'echo HELLO-FROM-POD; echo $SLOT' 2>&1)
ok $(echo "$r" | grep -q "HELLO-FROM-POD" && echo 1 || echo 0) "payload 在跑时命令仍被执行并回显(得 '$(echo "$r" | head -1)')"
ok $(echo "$r" | grep -q "^3$" && echo 1 || echo 0) "命令能拿到 SLOT 等环境变量"

r=$(bash "$TASK" sh 3 'exit 7' 2>&1); rc=$?
ok $([ "$rc" = 7 ] && echo 1 || echo 0) "退出码原样透传(得 $rc)"

ok $([ -z "$(ls "$F/cmd/slot3/out" 2>/dev/null)" ] && echo 1 || echo 0) "取回后清理干净,不留垃圾"

# stop 空转时也要能调试 —— 那正是最需要调试的时候
touch "$F/stop_slot3"; sleep 4
r=$(bash "$TASK" sh 3 'echo DEBUG-WHILE-STOPPED' 2>&1)
ok $(echo "$r" | grep -q "DEBUG-WHILE-STOPPED" && echo 1 || echo 0) "stop 空转时调试通道照常服务"
rm -f "$F/stop_slot3"

kill -TERM $BG 2>/dev/null; pkill -P $BG 2>/dev/null; sleep 0.3; kill -KILL $BG 2>/dev/null; wait $BG 2>/dev/null || true

echo "forever battery ${PASS}/${PASS} pass"
