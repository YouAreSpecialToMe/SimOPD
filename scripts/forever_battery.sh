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
export HB_EVERY_S=1 HB_CLAIM=0 FOREVER_SCYTHE=0   # 认领与镰刀各有专属 case;镰刀在共享机器上永不真挥
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
if command -v setsid >/dev/null 2>&1; then
    ok $([ ! -f "$MARK" ] && echo 1 || echo 0) "stop 之后子进程也死了(进程组连坐,不留孤儿)"
else
    echo "  skip 本机无 setsid(macOS):进程组连坐只在 Linux 可测,集群电池覆盖"
fi
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

# --- 12. 流式输出:长命令要边跑边吐,不能等跑完才一次性出来
rm -rf "$F"; mkdir -p "$F/run/slot3"
unset RANK; export SLOT=3
printf '#!/usr/bin/env bash\nwhile true; do sleep 1; done\n' > "$F/payload_slot3.sh"
: > "$T/out"
bash "$SRC" > "$T/out" 2>&1 & BG=$!
sleep 4
TASK=$(cd "$(dirname "$0")/.." && pwd)/deploy/dlc/task.sh
export SIMOPD_STORE=$T/data WAIT_S=30
# 一条跑 6 秒、每秒吐一行的命令:第 3 秒时就该已经看到前几行
bash "$TASK" sh 3 'for i in 1 2 3 4 5 6; do echo LINE-$i; sleep 1; done' > "$T/stream" 2>&1 &
SP=$!
sleep 4
early=$(grep -c "^LINE-" "$T/stream" 2>/dev/null || echo 0)
ok $([ "$early" -ge 1 ] && [ "$early" -le 5 ] && echo 1 || echo 0) \
   "命令跑到一半时已经吐出部分输出(4 秒时 $early 行,既非 0 也非全部 6)"
wait $SP 2>/dev/null || true
total=$(grep -c "^LINE-" "$T/stream" 2>/dev/null || echo 0)
ok $([ "$total" = 6 ] && echo 1 || echo 0) "结束后 6 行齐全,无重复无丢失(得 $total)"
ok $(grep -q "LINE-1" "$T/stream" && grep -q "LINE-6" "$T/stream" && echo 1 || echo 0) "首尾行都在"

# watch:共享盘上的文件直接 tail,不走通道(要用 timeout 掐掉 tail -f,macOS 没有则跳过)
if command -v timeout >/dev/null 2>&1; then
    echo "SHARED-LINE" > "$T/shared.log"
    r=$(timeout 3 bash "$TASK" watch 3 "$T/shared.log" 2>&1 | head -3)
    ok $(echo "$r" | grep -q "直接跟随" && echo 1 || echo 0) "共享盘路径识别为直接 tail(不绕通道)"
    ok $(echo "$r" | grep -q "SHARED-LINE" && echo 1 || echo 0) "直接跟随能读到内容"
else
    echo "  skip 本机无 timeout:watch 直连测试留给集群电池"
fi

kill -TERM $BG 2>/dev/null; pkill -P $BG 2>/dev/null; sleep 0.3; kill -KILL $BG 2>/dev/null; wait $BG 2>/dev/null || true

# --- 13. 空转期发 swap 不误杀下一轮(竞态回归):标记必须在 launch 前被消费
rm -rf "$F"; mkdir -p "$F/run/slot3"
unset RANK; export SLOT=3
printf '#!/usr/bin/env bash\necho P1-RAN\nsleep 4\n' > "$F/payload_slot3.sh"
: > "$T/out"
IDLE_S=30 bash "$SRC" > "$T/out" 2>&1 & BG=$!
for i in $(seq 40); do grep -q "第 1 轮结束" "$T/out" && break; sleep 0.5; done
ok $(grep -q "第 1 轮结束" "$T/out" && echo 1 || echo 0) "第一轮自然跑完,载体进入长空转(IDLE_S=30)"
printf '#!/usr/bin/env bash\necho P2-START\nsleep 4\necho P2-END\n' > "$F/payload_slot3.sh"
touch "$F/swap_slot3"
for i in $(seq 30); do grep -q "P2-END" "$T/out" && break; sleep 0.5; done
ok $(grep -q "P2-END" "$T/out" && echo 1 || echo 0) "空转被 swap 掐短,新 payload 完整跑完(没被误杀)"
ok $(grep -q "收到 swap" "$T/out" && echo 0 || echo 1) "swap 在 launch 前就被消费,监督循环没把它当杀令"
ok $([ ! -f "$F/swap_slot3" ] && echo 1 || echo 0) "swap 标记已消费"
ok $([ -s "$F/log/slot3.log" ] && grep -q "第 1 轮" "$F/log/slot3.log" && echo 1 || echo 0) "载体叙述 tee 进共享盘 log/slot3.log(跳板机可读)"
kill -TERM $BG 2>/dev/null; pkill -P $BG 2>/dev/null; sleep 0.3; kill -KILL $BG 2>/dev/null; wait $BG 2>/dev/null || true

# --- 14. reload 掐短退避:修完 payload 不用干等退避跑满
rm -rf "$F"; mkdir -p "$F/run/slot3"
printf '#!/usr/bin/env bash\necho QER\nexit 1\n' > "$F/payload_slot3.sh"
: > "$T/out"
BACKOFF_S0=20 BACKOFF_MAX=40 bash "$SRC" > "$T/out" 2>&1 & BG=$!
for i in $(seq 20); do grep -q "退避 20s" "$T/out" && break; sleep 0.5; done
ok $(grep -q "退避 20s" "$T/out" && echo 1 || echo 0) "秒退进入 20s 退避"
touch "$F/reload_slot3"
for i in $(seq 10); do grep -q "第 2 轮" "$T/out" && break; sleep 0.5; done
ok $(grep -q "掐短" "$T/out" && echo 1 || echo 0) "reload 把退避等待掐短(有日志)"
ok $(grep -q "第 2 轮" "$T/out" && echo 1 || echo 0) "没等满 20s 就进了第 2 轮"
ok $([ ! -f "$F/reload_slot3" ] && echo 1 || echo 0) "reload 标记被消费"
kill -TERM $BG 2>/dev/null; pkill -P $BG 2>/dev/null; sleep 0.3; kill -KILL $BG 2>/dev/null; wait $BG 2>/dev/null || true

# --- 15. rc=42 挂起:一次性任务的正门 —— 跑一次就停,不算异常,go/换 payload 解除
rm -rf "$F"; mkdir -p "$F/run/slot3"
printf '#!/usr/bin/env bash\necho DONE-42\nexit 42\n' > "$F/payload_slot3.sh"
run_for 6
n=$(grep -c "DONE-42" "$T/out" || true)
ok $([ "$n" = 1 ] && echo 1 || echo 0) "exit 42 只跑一次就挂起(得 $n 次)"
ok $([ -f "$F/parked_slot3" ] && echo 1 || echo 0) "parked 标记已写"
ok $(grep -q "挂起待命" "$T/out" && echo 1 || echo 0) "挂起有日志"
ok $(grep -q "判为异常" "$T/out" && echo 0 || echo 1) "rc=42 的秒退不算异常,不触发退避"
TASK=$(cd "$(dirname "$0")/.." && pwd)/deploy/dlc/task.sh
bash "$TASK" go 3 >/dev/null
ok $([ ! -f "$F/parked_slot3" ] && echo 1 || echo 0) "task.sh go 解除 parked"
run_for 5
n=$(grep -c "DONE-42" "$T/out" || true)
ok $([ "$n" = 1 ] && echo 1 || echo 0) "解除后再跑一次又自行挂起(新一段跑了 $n 次)"
touch "$F/parked_slot3"
printf '#!/usr/bin/env bash\necho X\nsleep 4\n' > "$T/np.sh"
bash "$TASK" set 3 "$T/np.sh" >/dev/null
ok $([ ! -f "$F/parked_slot3" ] && echo 1 || echo 0) "set 换 payload 也解除 parked"

# --- 16. 心跳 + 槽位认领:活没活跳板机可判;双载体不抢同一个槽
rm -rf "$F"; mkdir -p "$F/run/slot3"
printf '#!/usr/bin/env bash\nwhile true; do sleep 1; done\n' > "$F/payload_slot3.sh"
: > "$T/out"
bash "$SRC" > "$T/out" 2>&1 & BG=$!
sleep 3
ok $([ -f "$F/hb_slot3" ] && echo 1 || echo 0) "心跳文件在写(hb_slot3)"
a=$(( $(date +%s) - $(stat -c %Y "$F/hb_slot3" 2>/dev/null || stat -f %m "$F/hb_slot3") ))
ok $([ "$a" -le 2 ] && echo 1 || echo 0) "心跳新鲜(${a}s 前;HB_EVERY_S=1)"
kill -TERM $BG 2>/dev/null; pkill -P $BG 2>/dev/null; sleep 0.3; kill -KILL $BG 2>/dev/null; wait $BG 2>/dev/null || true
rm -rf "$F"; mkdir -p "$F/run/slot3"
printf '#!/usr/bin/env bash\necho CLAIM-RAN\nsleep 4\n' > "$F/payload_slot3.sh"
echo other-carrier-token > "$F/hb_slot3"     # 假装槽上已有活载体
: > "$T/out"
HB_CLAIM=1 HB_STALE_S=5 HB_WAIT_S=1 HB_VERIFY_S=0 bash "$SRC" > "$T/out" 2>&1 & BG=$!
sleep 2
ok $(grep -q "疑似双载体" "$T/out" && echo 1 || echo 0) "心跳新鲜时识别为疑似双载体并等待"
ok $(grep -q "CLAIM-RAN" "$T/out" && echo 0 || echo 1) "等待期间不跑 payload(不会两份写一个检查点)"
for i in $(seq 24); do grep -q "CLAIM-RAN" "$T/out" && break; sleep 0.5; done
ok $(grep -q "槽位认领成功" "$T/out" && echo 1 || echo 0) "心跳变陈(HB_STALE_S=5)后认领接管"
ok $(grep -q "CLAIM-RAN" "$T/out" && echo 1 || echo 0) "接管后正常跑 payload"
kill -TERM $BG 2>/dev/null; pkill -P $BG 2>/dev/null; sleep 0.3; kill -KILL $BG 2>/dev/null; wait $BG 2>/dev/null || true

# --- 17. 调试命令后台化:长命令不挡 swap、不陪葬;CMD_T 随件放宽/收紧超时
rm -rf "$F"; mkdir -p "$F/run/slot3"
printf '#!/usr/bin/env bash\necho LOOP-UP\nwhile true; do sleep 1; done\n' > "$F/payload_slot3.sh"
: > "$T/out"
bash "$SRC" > "$T/out" 2>&1 & BG=$!
sleep 3
export SIMOPD_STORE=$T/data WAIT_S=30
if command -v timeout >/dev/null 2>&1; then
    t0=$(date +%s)
    r=$(CMD_T=2 bash "$TASK" sh 3 'sleep 10; echo NEVER' 2>&1); rc=$?
    el=$(( $(date +%s) - t0 ))
    ok $([ "$rc" = 124 ] && echo 1 || echo 0) "CMD_T=2 随件生效,长命令被掐(rc=$rc)"
    ok $([ "$el" -le 7 ] && echo 1 || echo 0) "掐在 ~2s 而非默认 300s(实测 ${el}s)"
    ok $(echo "$r" | grep -q NEVER && echo 0 || echo 1) "被掐的命令没跑到最后"
else
    echo "  skip 本机无 timeout:CMD_T 掐断只在集群电池可测"
fi
bash "$TASK" sh 3 'sleep 5; echo SLOWCMD-DONE' > "$T/slow" 2>&1 & SP=$!
sleep 1
printf '#!/usr/bin/env bash\necho AFTER-SWAP\nsleep 4\n' > "$T/np2.sh"
bash "$TASK" swap 3 "$T/np2.sh" >/dev/null
sleep 3
ok $(grep -q "收到 swap" "$T/out" && echo 1 || echo 0) "长命令在跑时 swap 3s 内被响应(命令不再垫背轮询)"
wait $SP 2>/dev/null || true
ok $(grep -q "SLOWCMD-DONE" "$T/slow" && echo 1 || echo 0) "调试命令不在 payload 进程组:payload 被杀它照常跑完"
ok $(grep -q "跳过镰刀" "$T/out" && echo 1 || echo 0) "非 pod 环境/FOREVER_SCYTHE=0 不挥镰刀,只留说明"
kill -TERM $BG 2>/dev/null; pkill -P $BG 2>/dev/null; sleep 0.3; kill -KILL $BG 2>/dev/null; wait $BG 2>/dev/null || true

# --- 18. 镰刀候选(只列不杀,Linux 才有 /proc):按 cmdline 认人,与执刀分离
if [ -d /proc ]; then
    bash -c 'exec -a EngineCoreDecoy sleep 30' & DEC=$!
    sleep 0.5
    lst=$(FOREVER_LIST_SCYTHE=1 bash "$SRC" 2>/dev/null)
    ok $(echo "$lst" | grep -qx "$DEC" && echo 1 || echo 0) "镰刀候选认出化名 EngineCore 的进程(/proc cmdline 扫描)"
    kill -9 "$DEC" 2>/dev/null; wait "$DEC" 2>/dev/null || true
else
    echo "  skip 无 /proc(macOS):镰刀候选扫描留给集群电池"
fi

# --- 19. PTY 桥 server 协议(真交互 shell 的 pod 侧):字节流文件里进出一个真 bash。
# attach 端要真终端(raw 模式)没法无头测,由集群上手工验;这里钉 server 协议本身:
# cd 跨命令保留(调试通道做不到的核心诉求)、python -i、窗口尺寸下发、退出码回传。
PB=$(cd "$(dirname "$0")/.." && pwd)/deploy/dlc/ptybridge.py
SD=$T/ttysess
mkdir -p "$SD"
PTY_CHB_GRACE_S=60 python3 "$PB" serve "$SD" & PS=$!
for i in $(seq 20); do [ -f "$SD/shb" ] && break; sleep 0.5; done
ok $([ -f "$SD/shb" ] && echo 1 || echo 0) "PTY server 起来并写心跳 shb"
echo 1 > "$SD/chb"
printf 'cd /tmp\n' >> "$SD/i"; sleep 1
printf 'echo XPWDX-$PWD\n' >> "$SD/i"
for i in $(seq 20); do grep -q "XPWDX-/tmp" "$SD/o" 2>/dev/null && break; sleep 0.5; done
ok $(grep -q "XPWDX-/tmp" "$SD/o" && echo 1 || echo 0) "cd 跨命令保留(调试通道做不到,PTY 桥做到了)"
printf 'python3 -q\n' >> "$SD/i"; sleep 2
printf 'print(40+2)\n' >> "$SD/i"; sleep 1
printf 'exit()\n' >> "$SD/i"
for i in $(seq 20); do grep -q "^42" "$SD/o" 2>/dev/null && break; sleep 0.5; done
ok $(grep -q "^42" "$SD/o" && echo 1 || echo 0) "python -i 交互可用(REPL 里算出 42)"
printf '37 91' > "$SD/winsz"; sleep 1
printf 'stty size\n' >> "$SD/i"
for i in $(seq 20); do grep -q "37 91" "$SD/o" 2>/dev/null && break; sleep 0.5; done
ok $(grep -q "37 91" "$SD/o" && echo 1 || echo 0) "窗口尺寸经 winsz 下发到 PTY(vim/top 的排版前提)"
printf 'exit 5\n' >> "$SD/i"
for i in $(seq 20); do [ -f "$SD/rc" ] && break; sleep 0.5; done
ok $([ -f "$SD/rc" ] && [ "$(cat "$SD/rc")" = 5 ] && echo 1 || echo 0) "shell 退出码回传 rc 文件(得 $(cat "$SD/rc" 2>/dev/null))"
wait $PS 2>/dev/null || true

# --- 20. forever_boot 外壳:载体 bash 被打死(payload 误伤/OOM 波及),pod 未死就地重拉
rm -rf "$F"; mkdir -p "$F/run/slot3"
BOOTSRC=$(cd "$(dirname "$0")/.." && pwd)/deploy/dlc/forever_boot.sh
cp "$SRC" "$ROOT/deploy/dlc/forever.sh"      # boot 从 $ROOT 里读载体
printf '#!/usr/bin/env bash\necho BOOTED-PAYLOAD\nfor i in $(seq 300); do sleep 1; done\n' > "$F/payload_slot3.sh"
: > "$T/out"
BOOT_RETRY_S=2 bash "$BOOTSRC" > "$T/out" 2>&1 & BB=$!
for i in $(seq 30); do grep -q "BOOTED-PAYLOAD" "$T/out" && break; sleep 0.5; done
ok $(grep -q "第 1 次拉起载体" "$T/out" && grep -q "BOOTED-PAYLOAD" "$T/out" && echo 1 || echo 0) "boot 壳第 1 次拉起载体并跑起 payload"
pkill -f "$ROOT/deploy/dlc/forever.sh" 2>/dev/null   # 模拟载体进程被误杀(不动 boot 壳)
for i in $(seq 30); do grep -q "第 2 次拉起载体" "$T/out" && break; sleep 0.5; done
ok $(grep -q "载体退出 rc=" "$T/out" && echo 1 || echo 0) "载体死亡被 boot 壳看见"
ok $(grep -q "第 2 次拉起载体" "$T/out" && echo 1 || echo 0) "BOOT_RETRY_S=2 秒后就地重拉"
for i in $(seq 30); do [ "$(grep -c "BOOTED-PAYLOAD" "$T/out" || true)" -ge 2 ] && break; sleep 0.5; done
n=$(grep -c "BOOTED-PAYLOAD" "$T/out" || true)
ok $([ "$n" -ge 2 ] && echo 1 || echo 0) "重拉后的载体接着跑 payload(共起 $n 次)"
kill -TERM $BB 2>/dev/null; pkill -P $BB 2>/dev/null
pkill -f "$ROOT/deploy/dlc/forever.sh" 2>/dev/null; pkill -f "payload.running.sh" 2>/dev/null
sleep 0.3; kill -KILL $BB 2>/dev/null; wait $BB 2>/dev/null || true

# --- 21. task.sh alive 巡检:心跳断了点名 + 退出码,健康时一行 OK
rm -rf "$F"; mkdir -p "$F"
echo tok > "$F/hb_slot5"; sleep 2
r=$(HB_STALE_S=1 bash "$TASK" alive); rc=$?
ok $([ "$rc" = 1 ] && echo 1 || echo 0) "1 个断心跳 -> 退出码 1(得 $rc)"
ok $(echo "$r" | grep -q "slot5 心跳断了" && echo 1 || echo 0) "点名断的是 slot5"
echo tok > "$F/hb_slot5"
r=$(HB_STALE_S=90 bash "$TASK" alive); rc=$?
ok $([ "$rc" = 0 ] && echo "$r" | grep -q "^OK" && echo 1 || echo 0) "心跳新鲜 -> OK 退出码 0"

echo "forever battery ${PASS}/${PASS} pass"
