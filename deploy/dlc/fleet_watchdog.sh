#!/usr/bin/env bash
# 舰队夜间看门狗 —— 每小时救活舰队自己放弃的 lane。跑在跳板机上,只写标记不杀进程。
#
#   nohup bash deploy/dlc/fleet_watchdog.sh > $D/corr_wave/watchdog.log 2>&1 &
#   DRY=1 bash deploy/dlc/fleet_watchdog.sh --once     # 只报不动
#
# 为什么需要它。corr_wave_fleet.sh 的 _launch_lane 只重试三次;三次都失败后那条
# lane 就永久死了,而整槽会一直空转 —— 2026-08-20 夜里 20 条 corr lane 死了 10 条,
# 其中 slot4 整槽(含 d3/g2 两条头号臂)"wave complete, holding" 空转一小时,8 张
# H100 一步没跑,没有任何东西会发现。舰队自带的 40 分钟挂起看门狗只处理"卡住不退
# 出"的 lane;"退出了且重试用尽"的这一类,此前只能靠人。
#
# 规则,故意保守:
#   * 死:日志静默 >= STALE_MIN 分钟、步数 < 250、且尾部有 "attempt N failed"。
#     只是静默、没有 failed 行的算"卡",不碰 —— 那是舰队自己的看门狗的活,
#     两个东西同时去救同一条 lane 只会互相打架。
#   * 一个槽被 relaunch 的代价是同槽健康 lane 回退到最近检查点(<= 25 步),
#     所以每槽 COOLDOWN_H 小时内最多一次。
#   * 槽在 "wave complete, holding" 而它的臂没跑满 250:直接 relaunch,零代价
#     (没有健康 lane 可损失)。
#   * 只 touch fleet_relaunch_slot<k>_s<seed>。杀进程、扫 GPU、重跑 Phase R/L
#     全部交给舰队脚本自己那条路径,这里不重复实现,也就不会与它抢。
set -uo pipefail

D=${SIMOPD_STORE:-/mgfs/shared/Group_GY/changhao/simopd_data}
LOGD=$D/corr_wave
SEED=${SEED:-0}
STALE_MIN=${STALE_MIN:-45}
COOLDOWN_H=${COOLDOWN_H:-4}
PERIOD=${PERIOD:-3600}
DRY=${DRY:-0}
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

_ts() { date "+%Y-%m-%d %H:%M:%S"; }
_say() { echo "[$(_ts)] $*"; }

# 槽 -> lane 表。取该槽最新一份日志里最后一行 "Phase L: slot K lanes: ..."。
_slot_arms() {
    local k=$1 f
    f=$(ls -t "$LOGD"/fleet_slot${k}_s${SEED}_*.log 2>/dev/null | head -1) || return 0
    [ -n "$f" ] || return 0
    grep -a "Phase L: slot ${k} lanes:" "$f" | tail -1 |
        sed "s/.*lanes: //; s/ (250.*//" | tr " " "\n" | sed "s/:.*//" | grep -v '^$'
}

_slot_holding() {   # 该槽是不是停在 DONE 后的空转循环里
    local k=$1 f
    f=$(ls -t "$LOGD"/fleet_slot${k}_s${SEED}_*.log 2>/dev/null | head -1) || return 1
    [ -n "$f" ] && tail -3 "$f" | grep -q "wave complete, holding"
}

_lane_state() {     # arm -> "step age_min dead"
    local arm=$1 f step age dead=0 m now
    f=$LOGD/lane_${arm}_s${SEED}.log
    [ -f "$f" ] || { echo "- - missing"; return; }
    step=$(grep -aoE "global_step:[0-9]+" "$f" | tail -1 | cut -d: -f2)
    m=$(stat -c %Y "$f" 2>/dev/null || echo 0); now=$(date +%s)
    age=$(( (now - m) / 60 ))
    if [ "${step:-0}" -ge 250 ] 2>/dev/null; then
        echo "${step:-?} $age done"; return
    fi
    if [ "$age" -ge "$STALE_MIN" ] && tail -40 "$f" | grep -qE "attempt [0-9]+ failed"; then
        dead=1
    fi
    echo "${step:-?} $age $([ "$dead" = 1 ] && echo dead || echo live)"
}

_cooled() {         # 距上次动这个槽是否已超过冷却期
    local k=$1 st=$LOGD/.watchdog_slot${k}_s${SEED} last now
    [ -f "$st" ] || return 0
    last=$(cat "$st" 2>/dev/null || echo 0); now=$(date +%s)
    [ $(( (now - last) / 3600 )) -ge "$COOLDOWN_H" ]
}

_sweep() {
    local k arms arm state step age st acted
    for k in 0 1 2 3 4 5 6; do
        arms=$(_slot_arms "$k"); [ -n "$arms" ] || continue
        local dead=() live=0 done_=0
        for arm in $arms; do
            read -r step age st <<< "$(_lane_state "$arm")"
            case "$st" in
                dead) dead+=("$arm@$step") ;;
                done) done_=$((done_+1)) ;;
                *)    live=$((live+1)) ;;
            esac
        done
        acted=""
        if [ "${#dead[@]}" -gt 0 ]; then
            if _cooled "$k"; then
                acted="relaunch (dead: ${dead[*]}; 同槽 $live 条健康 lane 会回退到最近检查点)"
            else
                acted="SKIP -- 冷却中(${COOLDOWN_H}h),dead: ${dead[*]}"
            fi
        elif _slot_holding "$k" && [ "$done_" -lt "$(echo "$arms" | wc -w)" ]; then
            acted="relaunch (整槽空转但臂未跑满 250,零代价)"
        fi
        if [ -n "$acted" ]; then
            _say "slot$k: $acted"
            case "$acted" in relaunch*)
                if [ "$DRY" = 1 ]; then
                    _say "slot$k: DRY,不写标记"
                else
                    touch "$LOGD/fleet_relaunch_slot${k}_s${SEED}"
                    date +%s > "$LOGD/.watchdog_slot${k}_s${SEED}"
                    _say "slot$k: 已写 fleet_relaunch_slot${k}_s${SEED}"
                fi ;;
            esac
        else
            _say "slot$k: ok (${live} live, ${done_} done)"
        fi
    done
}

# 无家可归的臂:有检查点、没跑满 250、却不在任何槽的 lane 表里。
# 这是上面那套按槽扫描structurally 看不见的一类 —— 一个槽用 .next 换了新住户之后,
# 被顶掉的旧臂就从所有 lane 表里消失了,既不算"死"也不算"活",没有任何东西会再提它。
# 2026-08-21:f3_power@139 和 g1_verified_only@143 就这样在换图后无声停摆,
# 是人翻日志翻出来的。这里只报,不动手 —— 给它们找卡是要挤掉别人的,那是人的决定。
_homeless() {
    local mapped arm ck d n=0
    mapped=" $(for k in 0 1 2 3 4 5 6; do _slot_arms "$k"; done | tr '\n' ' ') "
    for d in "$D"/ckpt/simopd/*_corr_s${SEED}_16k "$D"/ckpt/simopd/*_n0_s${SEED}_16k; do
        [ -d "$d" ] || continue
        arm=$(basename "$d"); arm=${arm%_s${SEED}_16k}
        case "$mapped" in *" $arm "*) continue ;; esac
        ck=$(ls -d "$d/global_step_"* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
        [ -n "$ck" ] || continue
        [ "$ck" -ge "${FLEET_TOTAL_STEPS:-250}" ] 2>/dev/null && continue
        _say "无家可归: $arm 停在 ${ck} 步,不在任何槽的 lane 表里(换图时被顶掉,没有任何东西会救它)"
        n=$((n+1))
    done
    [ "$n" = 0 ] && _say "无家可归的臂:无"
}

_say "看门狗启动:STALE_MIN=$STALE_MIN COOLDOWN_H=$COOLDOWN_H PERIOD=${PERIOD}s DRY=$DRY"
while true; do
    _sweep
    _homeless
    [ "$ONCE" = 1 ] && break
    sleep "$PERIOD"
done
