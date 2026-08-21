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
export IDLE_S=1 MIN_RUN_S=3 BACKOFF_S0=1 BACKOFF_MAX=4
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

echo "forever battery ${PASS}/${PASS} pass"
