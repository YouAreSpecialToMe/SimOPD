#!/usr/bin/env bash
# Corrected-rerun wave 1 on DLC: 40 cards = 5 workers x 8 GPUs = 20 two-GPU lanes -- ONE 5-worker job
# (worker rank r drives SLOT r; SLOT=k explicit keeps the one-job-per-slot form).
#
# Every cell is "the banked arm + SIMOPD_TERM_EVENT=1" (docs/RESULTS-GAPS.md, corrected-rerun
# roster; docs/MECHANISMS.md M-I). One seed (0), seed-paired against the banked arms; the
# legacy single-eos rollout contract is pinned OFF inside every *_corr arm env, so nothing
# but the knob moves. Run names follow the campaign convention <arm>_s<seed>_16k, so
# eval_suite / extract_post_eval / verdict pick them up unchanged (eval contract = the
# run's own pin, i.e. legacy -- eval_offline --stop-token-ids auto).
#
# Slot map -- 20 two-GPU lanes = 5 workers x 4 lanes = 40 cards, the user's 20-lane plan exactly.
# (The banked d/g2 rows ran in the 4-GPU family only because their kernels pre-dated the
# streaming port; the corrected cells' kernels are streaming, so they take 2 GPUs like
# vanilla_corr -- see the arms' notes; fall back to the 4-GPU family only if the rehearsal
# shows student-side pressure at 16k.)
#
#   SLOT 0  vanilla_corr(0,1)     n2_corr(2,3)              b1_skew_kl_corr(4,5)       b5_k2_corr(6,7)
#   SLOT 1  f1_soft_log_corr      f2_hard_clip_corr         f3_power_corr              g1_verified_only_corr
#   SLOT 2  g4_failure_only_corr  g6_seqmean_corr           h2_last_segment_corr       h3_random_segment_corr
#   SLOT 3  h4_random_scatter_corr b2_forward_kl_corr       e2_set_coverage_a0_corr    c3_intersection_corr
#   SLOT 4  g2_fire_likelihood_corr g5_rgopd_gate_corr      d2_selectkd_corr           d3_teachability_corr
#   SLOT 5  (backlog fill, 6th worker if any) n2_termcal   d1_tip_corr               f2_clip2.3_corr   h1_first_segment_corr
#
# Lineage: deploy/dlc/h_axis_fleet.sh (its 11 fixes inherited: NCCL/GLOO iface by presence,
# explicit pid wait, zero-checkpoint never reports success, remote abort marker, bounded
# resume-retry). Differences: ROOT is the expansion tree (SimOPD-exp -- the corrected code
# lives there); Phase R rehearses missing arms ON THE POD (3-step machine-verdicted
# rehearsals in parallel on its own GPU pairs) instead of idling for gpu193 markers; the
# abort marker is a one-shot RELOAD request: an idle pod re-execs this script from the tree
# in place (same container; no AIMaster restart, no dlc submit); consumed when acted on.
#
# Submit: run on any machine WITHOUT the DLC rank env -> prints the console cards and a
# `dlc submit` CLI template; inside the container the same script is the payload.
#
#   SLOT=0 SEED=0 bash /mgfs/shared/Group_GY/changhao/SimOPD-exp/deploy/dlc/corr_wave_fleet.sh

set -uo pipefail

ROOT=${ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
D=/mgfs/shared/Group_GY/changhao/simopd_data
SEED=${SEED:-0}
LOGD=$D/corr_wave
# ONE job for the whole wave (the 2026-08-19 afternoon submission): a 5-worker pytorchjob,
# every worker an 8-GPU pod, worker rank r drives SLOT r (+ SLOT_BASE). SLOT=auto (the
# default) means "derive from my rank"; an explicit SLOT=k keeps the old one-job-per-slot
# form (then only rank 0 works and other ranks idle). Ranks past the slot table idle.
# Worker index inside a multi-worker pytorchjob. RANK is the PyTorchJob-standard per-worker
# index (0..workers-1) and is preferred; the MLP_* names are PAI/DLC's own and are kept as
# fallbacks (the single-worker lineage never had to tell them apart). All candidates are
# printed at start so a mis-mapping is a grep, not a mystery: a collision (two workers
# deriving the same SLOT) shows as one of them idling as a DUPLICATE POD.
_rank=${RANK:-${MLP_ROLE_INDEX:-${MLP_WORKER_RACK_RANK_INDEX:-0}}}
SLOT_BASE=${SLOT_BASE:-0}
SLOT=${SLOT:-auto}
if [ "$SLOT" = auto ]; then
    SLOT=$(( SLOT_BASE + _rank ))
    _slot_from_rank=1
else
    _slot_from_rank=0
fi
export SLOT SEED SLOT_BASE

# lane spec: "arm:gpus" (2 or 4 comma-separated GPU ids). Plain case (bash 3 on the
# submitter's mac has no associative arrays).
_lanes_for() {
    case "$1" in
        0) echo "vanilla_corr:0,1 n2_corr:2,3 b1_skew_kl_corr:4,5 b5_k2_corr:6,7" ;;
        1) echo "f1_soft_log_corr:0,1 f2_hard_clip_corr:2,3 f3_power_corr:4,5 g1_verified_only_corr:6,7" ;;
        2) echo "g4_failure_only_corr:0,1 g6_seqmean_corr:2,3 h2_last_segment_corr:4,5 h3_random_segment_corr:6,7" ;;
        3) echo "h4_random_scatter_corr:0,1 b2_forward_kl_corr:2,3 e2_set_coverage_a0_corr:4,5 c3_intersection_corr:6,7" ;;
        4) echo "g2_fire_likelihood_corr:0,1 g5_rgopd_gate_corr:2,3 d2_selectkd_corr:4,5 d3_teachability_corr:6,7" ;;
        5) echo "n2_termcal:0,1 d1_tip_corr:2,3 f2_clip2.3_corr:4,5 h1_first_segment_corr:6,7" ;;
        *) echo "" ;;
    esac
}
# WAVE QUEUE: an override file replaces the built-in slot map. Written by the operator
# (hop pod) as $LOGD/slot<k>_s<seed>_lanes; the DONE idle loop below promotes a staged
# .next file into it and re-execs, so successive waves chain with NO dlc action at all.
_OVR="$LOGD/slot${SLOT}_s${SEED}_lanes"
if [ -f "$_OVR" ]; then
    LANES=$(cat "$_OVR")
    echo "== lane map OVERRIDE from $_OVR: $LANES"
else
    LANES=$(_lanes_for "$SLOT")
fi
if [ -z "$LANES" ]; then
    if [ "${_slot_from_rank:-0}" = 1 ]; then
        while true; do echo "rank ${_rank} -> SLOT ${SLOT}: no lanes in the slot table (0-5); idling ($(date))"; sleep 600; done
    fi
    echo "unknown SLOT=$SLOT (0-5)"; exit 2
fi
ARMS=(); for spec in $LANES; do ARMS+=("${spec%%:*}"); done

# The abort marker is a RELOAD request. Idle loops (no lanes running) honour it by
# re-exec'ing this script from the shared tree IN PLACE -- the same container, the same
# DLC job, no AIMaster restart policy involved and no `dlc submit`: whatever the tree
# holds at that moment (bundle-synced) is what runs next. Only the bounded fallback
# (reload budget spent) still exits 17 for AIMaster. Lanes already running cannot take
# new code (a Python process imports once); Phase L does not poll the marker.
_abort_check() {
    if [ -f "$LOGD/fleet_abort_slot${SLOT}_s${SEED}" ]; then
        rm -f "$LOGD/fleet_abort_slot${SLOT}_s${SEED}"
        export SLOT SEED CORR_FLEET_RELOADS=$(( ${CORR_FLEET_RELOADS:-0} + 1 ))
        if [ "$CORR_FLEET_RELOADS" -gt 20 ]; then
            echo "abort marker found; reload budget spent (${CORR_FLEET_RELOADS}); exiting 17 for AIMaster restart"
            exit 17
        fi
        echo "abort marker found ($(date)): reloading slot ${SLOT} IN PLACE (#${CORR_FLEET_RELOADS}) from $ROOT @ $(git -C "$ROOT" log --oneline -1 2>/dev/null | cut -c1-70) -- same container, no AIMaster, no dlc submit"
        [ -n "${_hb_pid:-}" ] && kill "$_hb_pid" 2>/dev/null
        [ -n "${LOCK:-}" ] && rm -rf "$LOCK"      # exec skips the EXIT trap; release explicitly
        # keep the SLOT mode across the reload: a rank-derived worker must re-derive (an
        # explicit numeric SLOT would flip it into single-slot mode, where rank != 0 idles)
        [ "${_slot_from_rank:-0}" = 1 ] && export SLOT=auto
        exec bash "$ROOT/deploy/dlc/corr_wave_fleet.sh"
    fi
}

# ---------------------------------------------------------------- submitter --
if [ -z "${MLP_ROLE_INDEX:-}${MLP_WORKER_RACK_RANK_INDEX:-}${DLC_JOB_ID:-}" ]; then
    echo "========= corrected-rerun wave 1: DLC console cards (one job per SLOT, 1 worker x 8 GPU) ========="
    for s in 0 1 2 3 4 5; do
        cat <<CARD
  --- SLOT $s ---------------------------------------------------------------
  任务名称   simopd-corr-wave1-slot${s}-s${SEED}
  节点数量   1
  单节点GPU  8      CPU 64      内存 512Gi(照 A/H 轴成功任务规格)
  镜像/资源组/数据集挂载:照抄 simopd-a-axis-4lane 成功表单(挂载须含 /mgfs)
  执行命令:
    SLOT=${s} SEED=${SEED} bash $ROOT/deploy/dlc/corr_wave_fleet.sh
  lanes: $(_lanes_for "$s")
CARD
    done
    cat <<TAIL
  40 cards = SLOT 0-4. EITHER 5 jobs (one per SLOT, SLOT=k explicit) OR ONE 5-worker job (SLOT unset/auto:
  worker rank r drives SLOT r; add a 6th worker to fill SLOT 5 = backlog raw N2, d1, f2_2.3, h1).
  Add-on form: SLOT_BASE=1 with 4 workers drives SLOT 1-4 next to an already-running SLOT 0 job.
  前置:rehearsal_vanilla_corr.OK(载体彩排,gpu193;rehearse_n2.sh 写在 $D/n2/,这里也认)+ 各臂 .OK,或批量标记 rehearsal_corr_wave.OK。
  重载(空转中的 pod 就地重读舰队树,不重提 job):touch $LOGD/fleet_abort_slot<k>_s${SEED}   (Phase L 跑 lane 时不轮询)
  重发(Phase L 期间就地杀掉本槽 lane 并重走 R/L,用于补一条漏掉的 lane):touch $LOGD/fleet_relaunch_slot<k>_s${SEED}
========= dlc CLI:一次性 40 卡(一个 job,5 worker x 8 GPU,rank r -> SLOT r)=========
    dlc submit pytorchjob --name=simopd-corr-wave1-40cards-s${SEED} --workers=5 --worker_gpu=8 --worker_cpu=64 \\
      --worker_memory=512Gi --worker_image="\$IMAGE" --data_sources="\$DATA_SOURCES" \\
      --workspace_id="\$WORKSPACE_ID" --resource_id="\$RESOURCE_ID" --priority=5 --job_max_running_time_minutes=0 \\
      --command="bash -lc 'SEED=${SEED} bash $ROOT/deploy/dlc/corr_wave_fleet.sh'"
========= 或补齐形式:SLOT 0 已在跑,一个 job 补 SLOT 1-4(4 worker,32 卡)=========
    ... --workers=4 ... --command="bash -lc 'SLOT_BASE=1 SEED=${SEED} bash $ROOT/deploy/dlc/corr_wave_fleet.sh'"
========= 或每 SLOT 一个 job(旧形式)=========
  for k in 0 1 2 3 4; do  ... --workers=1 ... --command="bash -lc 'SLOT=\${k} SEED=${SEED} bash $ROOT/deploy/dlc/corr_wave_fleet.sh'"; done
TAIL
    echo "本机不是 DLC 容器(无 rank env),以上为提交卡片;容器内执行同一脚本即载荷。"
    exit 0
fi

# ------------------------------------------------------------------ payload --
# 单槽作业的重复保护:一张 workers=N 的单里若把 SLOT 写死,rank>0 的 worker 全是同一槽
# 的副本,让它们空转。CORR_SLOT_OWNED=1 是永续载体(deploy/dlc/forever.sh)的豁免:
# 载体给每个 pod 各分一个槽(SLOT_BASE+rank)再把 SLOT 显式导出,rank 1/2/3 拿的是
# 12/13/14 而不是副本 —— 这条守卫会把它们全按死。2026-08-23 实锤:迁移接棒后
# slot12/13/14 空转 19.5 小时,9 条 lane 一步没跑。重复保护并不因此失守:槽锁
# (slot<k>_s<seed>.lock,带心跳与 DUPLICATE POD 判定)才是真正的互斥,载体自己的
# 心跳认领是第二道。
if [ "$_slot_from_rank" = 0 ] && [ "${_rank}" != "0" ] && [ "${CORR_SLOT_OWNED:-0}" != "1" ]; then
    while true; do _abort_check; echo "rank ${_rank}: single-slot job (SLOT=$SLOT given), idling ($(date))"; sleep 600; done
fi
cd "$ROOT"
mkdir -p "$LOGD"
# an abort marker is a ONE-SHOT restart request: consumed here so the restarted worker
# does not exit again at its first idle loop
rm -f "$LOGD/fleet_abort_slot${SLOT}_s${SEED}"
exec > >(tee -a "$LOGD/fleet_slot${SLOT}_s${SEED}_$(date +%Y%m%d_%H%M%S).log") 2>&1
echo "== worker rank ${_rank} -> SLOT ${SLOT} (SLOT_BASE=${SLOT_BASE}, mode=$([ "$_slot_from_rank" = 1 ] && echo rank-derived || echo explicit); rank env: RANK=${RANK:-<unset>} WORLD_SIZE=${WORLD_SIZE:-<unset>} MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-<unset>} MLP_WORKER_RACK_RANK_INDEX=${MLP_WORKER_RACK_RANK_INDEX:-<unset>} host=$(hostname))"

# ------------------------------------------------------------ slot lock ------
# Exactly ONE pod may drive a slot: a duplicate (a resubmitted job next to an auto-restarted
# one -- it happened to slot 0 on 2026-08-19: two pods rehearsed the same arms into the same
# dirs and would have trained the same run names into the same checkpoint dirs) idles here.
# The lock is a directory (atomic mkdir) with a heartbeat file; a lock whose heartbeat is
# older than 20 min is stale (owner pod gone) and is taken over.
LOCK=$LOGD/slot${SLOT}_s${SEED}.lock
# mkdir -p first: the previous owner's EXIT trap REMOVES the lock dir, so a duplicate that
# is idling when the owner is stopped would otherwise write its owner file into a directory
# that no longer exists -- no heartbeat, and the next pod would see a free lock and drive
# the same slot in parallel (the 2026-08-19 double-pod failure, one layer down).
_take_lock() { mkdir -p "$LOCK"; echo "$(hostname) pid=$$ $(date -u +%FT%TZ)" > "$LOCK/owner"; }
if mkdir "$LOCK" 2>/dev/null; then
    _take_lock
else
    _age=$(( $(date +%s) - $(stat -c %Y "$LOCK/owner" 2>/dev/null || echo 0) ))
    if [ "$_age" -lt 1200 ]; then
        # NOTE: no _abort_check here -- the marker is the OWNER's reload request; a duplicate
        # consuming it would steal it. A duplicate leaves via DLC stop or takes over a stale lock.
        while true; do
            _age=$(( $(date +%s) - $(stat -c %Y "$LOCK/owner" 2>/dev/null || echo 0) ))
            [ "$_age" -ge 1200 ] && { echo "lock owner heartbeat stale (${_age}s); taking over"; _take_lock; break; }
            echo "DUPLICATE POD for slot ${SLOT}: lock held by [$(cat "$LOCK/owner" 2>/dev/null)] heartbeat ${_age}s ago -- idling ($(date))"
            sleep 300
        done
    else
        echo "stale lock (heartbeat ${_age}s ago) -- taking over"; _take_lock
    fi
fi
# heartbeat while this pod lives
( while true; do sleep 300; [ -d "$LOCK" ] && touch "$LOCK/owner"; done ) & _hb_pid=$!
trap "kill $_hb_pid 2>/dev/null; rm -rf \"$LOCK\"" EXIT
echo "== slot lock taken: $(cat "$LOCK/owner")"
echo "== corr_wave_fleet slot ${SLOT} on $(hostname), seed ${SEED}, tree $ROOT @ $(git log --oneline -1 2>/dev/null | head -1)"
git config --global --add safe.directory "$ROOT" 2>/dev/null || true
nvidia-smi -L | head -8
source simopd/bin/activate
[ -f "$ROOT/simopd_env.sh" ] && source "$ROOT/simopd_env.sh"

export DATA_DIR=$D/simopd_math
export CKPT_ROOT=$D/ckpt
_ifs=$(ls /sys/class/net 2>/dev/null | tr "\n" " ")
_pick=${SIMOPD_NET_IFACE:-}
if [ -z "$_pick" ]; then
    for _c in bond0 bond1 eth0; do
        [ -e "/sys/class/net/$_c" ] && { _pick=$_c; break; }
    done
    [ -n "$_pick" ] || _pick=lo
fi
export NCCL_SOCKET_IFNAME=$_pick GLOO_SOCKET_IFNAME=$_pick
export NCCL_NET_PLUGIN=none
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
echo "== net ifaces: picked $_pick (pod has: $_ifs) NCCL_NET_PLUGIN=none"
export ROLLOUT_GPU_MEM_UTIL=0.45
export PYTHONUNBUFFERED=1

# The corrected code must actually be in this tree (a stale checkout would silently run
# the legacy carrier under a *_corr name).
python - <<'PY' || { while true; do _abort_check; echo "TREE STALE: no SIMOPD_TERM_EVENT support in src/ -- lanes NOT launched ($(date))"; sleep 120; done; }
import sys
sys.path.insert(0, "src")
from simopd import topk_losses as T
assert hasattr(T, "TERM_EVENT_FAMILY") and hasattr(T, "_collapse_terminator_support") and "k1_termfix" in T.TOPK_DISPATCH
print("tree check: corrected carrier present")
PY

# scoped lint gate (seed 0)
if [ "$SEED" = 0 ]; then
    LINT_LOG=$LOGD/corr_arm_lint_slot${SLOT}.log
    python scripts/arm_lint.py > "$LINT_LOG" 2>&1 || true
    _pat=$(IFS='|'; echo "${ARMS[*]}")
    BAD=$(grep 'PROBLEM' "$LINT_LOG" | grep -E "\[($_pat)\]" | grep -vE 'campaign\.tsv row|verdict\.py ARMS' || true)
    if [ -n "$BAD" ]; then
        while true; do _abort_check; echo "ARM_LINT (scoped) FAILED -- lanes NOT launched ($(date)):"; echo "$BAD"; sleep 120; done
    fi
    echo "== arm_lint: scoped gate clean for slot ${SLOT} (full report: $LINT_LOG)"
fi

# --------------------------------------------------- eval worker 竞态防护 --
# 2026-08-21:h7 跑满后它的 GPU 对交给了 eval worker;之后这个槽 relaunch 换图,
# 清卡和 worker 重新抓卡之间只差几十秒 —— c5_union_fkl 三次 OOM 在被 aime avg@32
# 占住的卡上(彩排还 PASS 了:Phase R 打包在别的 GPU 对上跑,遮住了冲突)。
# bringup 窗口(此处 -> lanes launched)挂 PAUSE.slot<k>;worker 每轮扫描先看它,
# 有就不领新单(正在跑的单不受影响 —— 清卡会杀掉它们,claim 走陈旧过期)。
# worker 侧对 >60 分钟的陈旧 PAUSE 免疫,所以槽脚本死在窗口里也不会永久饿死 eval。
_EVALQ_EXP=$D/evalq_exp
touch "$_EVALQ_EXP/PAUSE.slot${SLOT}" 2>/dev/null || true
_eval_unpause() { rm -f "$_EVALQ_EXP/PAUSE.slot${SLOT}" 2>/dev/null || true; }

# Bringup scythe (2026-08-21). _gpu_sweep kills by nvidia-smi pid, and inside this
# container nvidia-smi can report HOST pids -- at slot6's relaunch it printed
# "leftover compute pids [3969741 322737]" and 80GB stayed allocated: two eval-vLLM
# EngineCore orphans it could not signal, and c5_union_fkl OOMed 3x on cards that
# looked swept. Same disease as the zzx14 EngineCore orphans; same cure: ignore
# nvidia-smi's pid column and scan our own /proc by cmdline. Pod-wide, so it is
# ONLY safe here at bringup, before any lane of ours is running -- mid-wave sweeps
# must keep using _gpu_sweep's pair-scoped kill. Eval workers' bash loops hold no
# GPU and do not match; their killed inference children re-queue via stale claims.
_pod_scythe() {
    local me=$$ pid c n=0
    for pid in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
        [ "$pid" = "$me" ] && continue
        c=$(tr "\0" " " < "/proc/$pid/cmdline" 2>/dev/null) || continue
        case "$c" in
            *VLLM::*|*EngineCore*|*vllm*|*ray::*|*raylet*|*verl.trainer.main_ppo*|*eval_offline.py*|*eval_suite.py*|*eval_worker_exp.sh*)
                kill -9 "$pid" 2>/dev/null && n=$((n+1)) ;;
        esac
    done
    # eval_worker_exp.sh 也在名单里(2026-08-23 加):此前只杀引擎不杀外壳 —— 外壳是个
    # 循环,engine 被杀它下一轮立刻再认领一单、把卡重新占满。当天 g4/g6/a3 三条新 lane
    # 在三个 pod 上同样地三次 create_device_mesh OOM 就是这么来的:上一轮 lane 跑满后
    # _eval_handoff 把那对卡交给了 eval worker,重发时镰刀清了显存(Phase L 打印的
    # "used MiB: 1 1 1 ..." 是真的),_eval_unpause 一解锁 worker 又抢了回去。
    # 只在 bringup 跑,且只在训练 pod 上(评测 pod 跑的是 eval_farm.sh,不走这个脚本)。
    echo "== bringup scythe: SIGKILLed $n GPU-resident leftover(s) by /proc cmdline scan"
    sleep 5
}
_pod_scythe

# ------------------------------------------------------------ Phase R 彩排门 --
# The carrier proof (rehearsal_vanilla_corr.OK) + each slot arm's own marker, or the batch
# marker rehearsal_corr_wave.OK. Missing markers are NOT waited on: this pod has 8 idle
# H100s, so it rehearses the missing arms itself -- deploy/dsw/rehearse_n2.sh, 3 steps,
# machine-verdicted, one lane per GPU pair in parallel -- writes the .OK on PASS and skips
# (does not launch) any lane whose rehearsal FAILED. The debug-node markers ($D/n2/) are
# honored too. A slot that has no vanilla_corr rehearses vanilla_corr on its first pair as
# the carrier proof if that marker is missing.
# GPU HYGIENE. Killing a rehearsal's shell does NOT free its HBM: the driver's Ray
# workers and vLLM engine/worker subprocesses outlive it and keep their allocations, so
# the next attempt dies with "Free memory on device cuda:0 (9.04/79.1 GiB) on startup is
# less than desired GPU memory utilization" (2026-08-19, after the watchdog's
# false-positive kills). Nothing of ours may be on the GPUs before Phase L, so sweep by
# compute-app pid -- the only reliable handle, since the leftovers are not children of
# this shell -- and wait for the memory to actually come back.
_gpu_procs() { nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " " | tr "\n" " "; }
_gpu_used() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr "\n" " "; }
_gpu_sweep() {  # GPUS
    local pids; pids=$(_gpu_procs "$1")
    [ -n "${pids// /}" ] || return 0
    echo "  GPU sweep on [$1]: leftover compute pids [$pids] -- used MiB: $(_gpu_used "$1")"
    kill -TERM $pids 2>/dev/null; sleep 8
    pids=$(_gpu_procs "$1")
    [ -n "${pids// /}" ] && { echo "  GPU sweep on [$1]: SIGKILL [$pids]"; kill -KILL $pids 2>/dev/null; sleep 5; }
    echo "  GPU sweep on [$1] done -- used MiB now: $(_gpu_used "$1")"
}
_wait_gpu_free() {  # GPUS -- up to ~2 min for the driver to release
    local i used busy
    for i in $(seq 1 24); do
        busy=0
        for used in $(_gpu_used "$1"); do [ "${used:-0}" -gt 4096 ] && busy=1; done
        [ "$busy" = 0 ] && return 0
        sleep 5
    done
    echo "  WARNING: GPUs [$1] still hold $(_gpu_used "$1") MiB after 2 min; starting anyway"
}

# Ray cold-start warm-up (called before Phase R and again before Phase L, since a pod whose
# arms all carry .OK markers skips Phase R entirely and its FIRST ray.init would then be a
# lane's). ray.init() gives the raylet a HARD-CODED 30 s to register with the GCS
# (ray/_private/node.py: raylet_start_wait_time_s = 30, no env override); on a fresh pod
# the 34 MB raylet + 30 MB gcs_server + _raylet.so (~150 MB) come off /mgfs cold. 2026-08-19
# slot 1 f1 -- the pod's first ray.init, 3 min after boot -- died on exactly that; f2/f3
# minutes later were fine: purely a page-cache effect. Every big file in the ray package:
# blunt beats a curated list that misses the one file that was slow (warm: <1 s).
_warm_ray() {
    local t0w rayd; t0w=$(date +%s)
    rayd=$(python -c 'import os, ray; print(os.path.dirname(ray.__file__))' 2>/dev/null)
    [ -n "$rayd" ] && find "$rayd" -type f -size +1M -exec cat {} + >/dev/null 2>&1
    echo "== $1: ray binaries warmed in $(( $(date +%s) - t0w ))s (raylet must register within a hard-coded 30s)"
}
# Kill EVERY ray process on this pod. Only for whole-slot resets (Phase R reload, Phase L
# relaunch): the drivers die on SIGTERM without cleanup (python has no default handler),
# leaving raylet/gcs/dashboard/workers orphaned -- GPU holders are swept by pid, but the
# CPU-side raylets would accumulate across resets. Never per-lane (other lanes' rays share
# the pod).
_kill_all_ray() { pkill -f "site-packages/ray/" 2>/dev/null; sleep 2; pkill -KILL -f "site-packages/ray/" 2>/dev/null; }

# 跑满的 lane 把卡交给 eval,而不是让它空到整槽结束。
# 一条 lane 到 250 步后,它那对 H100 会一直闲着等同槽最慢的一条 —— 一夜就是几十
# GPU-小时,而 evalq_exp 里同时压着几百个待评检查点(eval_worker_exp.sh 的抬头写过
# 同一个病:训练收尾时 192 张卡里 94 张闲着,后面堵着 647 个检查点)。这里在 lane
# 退出的那一刻就把它的卡接过去,每张卡一个单卡 worker。队列用共享 claim 目录
# (mkdir 原子),多开一个 worker 只会多认领,不会重复评同一项。
# 只在真的跑满时交卡:没跑满就退出的 lane 属于崩溃,那对卡要留给它自己的重试。
_eval_handoff() {   # ARM GPUS
    local arm=$1 gpus=$2 ck q g
    [ "${FLEET_EVAL_HANDOFF:-1}" = 1 ] || return 0
    ck=$(ls -d "$D/ckpt/simopd/${arm}_s${SEED}_16k/global_step_"* 2>/dev/null |
         sed 's/.*global_step_//' | sort -n | tail -1)
    if ! [ "${ck:-0}" -ge "${FLEET_TOTAL_STEPS:-250}" ] 2>/dev/null; then
        echo "lane ${arm}: 退出时只到 ${ck:-0} 步,不交卡(留给重试/重启)"
        return 0
    fi
    q=${EVAL_QUEUE:-$D/evalq_exp}
    if [ ! -s "$q/pending.txt" ]; then
        echo "lane ${arm}: 跑满 ${ck} 步,但 eval 队列空,卡先闲着"
        return 0
    fi
    if [ ! -x "$D/eval_worker_exp.sh" ] && [ ! -f "$D/eval_worker_exp.sh" ]; then
        echo "lane ${arm}: 跑满 ${ck} 步,但找不到 $D/eval_worker_exp.sh,卡先闲着"
        return 0
    fi
    _gpu_sweep "$gpus"
    for g in ${gpus//,/ }; do
        ( cd "$ROOT" && nohup bash "$D/eval_worker_exp.sh" "$g" "$q" \
            > "$LOGD/evalw_slot${SLOT}_gpu${g}.log" 2>&1 & )
        echo "lane ${arm}: 跑满 ${ck} 步 -> GPU ${g} 交给 eval worker(队列 $q,日志 $LOGD/evalw_slot${SLOT}_gpu${g}.log)"
    done
}

_has_ok() { [ -f "$LOGD/rehearsal_$1.OK" ] || [ -f "$D/n2/rehearsal_$1.OK" ]; }
declare -a _todo=()
if ! _has_ok corr_wave; then
    for ARM in "${ARMS[@]}"; do _has_ok "$ARM" || _todo+=("$ARM"); done
fi
_need_carrier=0
_has_ok vanilla_corr || { case " ${ARMS[*]} " in *" vanilla_corr "*) ;; *) _need_carrier=1 ;; esac; }
if [ "${#_todo[@]}" -gt 0 ] || [ "$_need_carrier" = 1 ]; then
    echo "== Phase R: rehearsing on this pod: carrier=${_need_carrier} arms: ${_todo[*]:-none} ($(date))"
    _rpids=(); _rarms=(); _pairs=(0,1 2,3 4,5 6,7); _i=0
    # STAGGER: the venv AND both models live on /mgfs, so four simultaneous cold bringups
    # are ~40 processes importing torch+vllm and reading ~12 GB of weights through one
    # mount at once. On a cold node that thrashes: 2026-08-19 slot 0 on M16-221 sat 20+ min
    # after "Started a local Ray instance" with zero output (the same node needed 5 min for
    # a lint that takes 30 s warm). Starting them REHEARSE_STAGGER seconds apart lets the
    # first one warm the page cache for the rest; on a warm node the cost is ~4 min total.
    # operator knobs, env or a one-line file on shared disk (survives a reload, and I can
    # retune them from the hop pod without touching DLC): $LOGD/REHEARSE_STAGGER, .../REHEARSE_STALL_MIN
    _STAG=${REHEARSE_STAGGER:-$(cat "$LOGD/REHEARSE_STAGGER" 2>/dev/null || echo 180)}
    echo "== Phase R: GPU state before sweep -- used MiB: $(_gpu_used 0,1,2,3,4,5,6,7)"
    _gpu_sweep 0,1,2,3,4,5,6,7
    # Ray cold-start warm-up. ray.init() gives the raylet a HARD-CODED 30 s to register with
    # the GCS (ray/_private/node.py: `raylet_start_wait_time_s = 30`, no env override), and on
    # a fresh pod the 34 MB raylet + 30 MB gcs_server + their .so files come off /mgfs cold.
    # 2026-08-19 slot 1: f1 -- the pod's FIRST ray.init, 3 min after boot -- died on exactly
    # that; f2/f3 on the same pod minutes later were fine, i.e. it is purely a page-cache
    # effect. Pull the binaries in first (~66 MB, seconds) so nobody races that timer.
    _warm_ray "Phase R"
    _rehearse_one() {  # ARM PAIR DELAY
        [ "${3:-0}" -gt 0 ] && sleep "$3"
        # Sweep THIS pair right before starting, not just at Phase R entry: an arm killed at
        # the instant it was starting (2026-08-19 b5_k2_corr, killed 1 s after launch) is not
        # yet a compute app, survives the entry sweep, and only shows up on the GPUs minutes
        # later -- exactly when its replacement wants them.
        _gpu_sweep "$2"
        _wait_gpu_free "$2"
        echo "rehearsal $1: starting on GPUs $2 ($(date))"
        : > "$D/n2/rehearsal_$1.log"    # the watchdog measures silence off this mtime; a log
                                        # left by an earlier pod must not read as "stalled"
        ARM=$1 bash deploy/dsw/rehearse_n2.sh "$2" > "$LOGD/rehearse_${1}_s${SEED}.log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then touch "$LOGD/rehearsal_$1.OK"; echo "rehearsal $1: PASS"; else echo "rehearsal $1: FAIL (rc=$rc) see $LOGD/rehearse_${1}_s${SEED}.log"; fi
        return $rc
    }
    _pstart=$(date +%s); _rstart=(); _rgpus=(); _rretry=()   # per-arm: scheduled start epoch, GPU pair, retried?
    if [ "$_need_carrier" = 1 ]; then
        ( _rehearse_one vanilla_corr "${_pairs[$_i]}" 0 ) & _rpids+=($!); _rarms+=(vanilla_corr); _rstart+=("$_pstart"); _rgpus+=("${_pairs[$_i]}"); _rretry+=(0); _i=$((_i+1))
    fi
    for ARM in "${_todo[@]}"; do
        if [ $_i -ge 4 ]; then wait "${_rpids[@]}"; _rpids=(); _rarms=(); _rstart=(); _rgpus=(); _rretry=(); _i=0; fi   # >4 -> second round
        _d=$(( _i * _STAG ))
        ( _rehearse_one "$ARM" "${_pairs[$_i]}" "$_d" ) & _rpids+=($!); _rarms+=("$ARM"); _rstart+=($(( _pstart + _d ))); _rgpus+=("${_pairs[$_i]}"); _rretry+=(0); _i=$((_i+1))
    done
    echo "== Phase R: ${#_rpids[@]} rehearsals launched, staggered ${_STAG}s apart (carrier first)"
    # WATCHDOG. A plain `wait` here is a lock-out: rehearse_n2.sh has no timeout, so a
    # bringup that never produces a line (2026-08-19, M16-221) parks the fleet inside wait
    # forever -- no idle loop, so the abort/reload marker is never seen and only a DLC
    # stop+resubmit can move it. Poll instead: honour the marker, and kill a rehearsal whose
    # log has been silent for REHEARSE_STALL_MIN minutes (default 25; a normal cold bringup
    # on a slow node is ~10, a step is seconds) so the slot falls through to the idle loop
    # with a FAIL rather than hanging.
    _stall=$(( ${REHEARSE_STALL_MIN:-$(cat "$LOGD/REHEARSE_STALL_MIN" 2>/dev/null || echo 25)} * 60 ))
    _kill_rehearsals() {
        for p in "${_rpids[@]}"; do kill -TERM "$p" 2>/dev/null; done
        pkill -f "verl.trainer.main_ppo" 2>/dev/null
        sleep 5
        for p in "${_rpids[@]}"; do kill -KILL "$p" 2>/dev/null; done
        _kill_all_ray
        _gpu_sweep 0,1,2,3,4,5,6,7
    }
    while [ "${#_rpids[@]}" -gt 0 ]; do
        if [ -f "$LOGD/fleet_abort_slot${SLOT}_s${SEED}" ]; then
            echo "== Phase R: reload requested mid-rehearsal; killing rehearsals ($(date))"
            _kill_rehearsals
            _abort_check
        fi
        _alive=(); _alive_arms=(); _alive_start=(); _alive_gpus=(); _alive_retry=(); _idx=0; _now=$(date +%s)
        for p in "${_rpids[@]}"; do
            _arm=${_rarms[$_idx]}; _t0=${_rstart[$_idx]}; _gp=${_rgpus[$_idx]}; _rt=${_rretry[$_idx]}; _idx=$((_idx+1))
            if ! kill -0 "$p" 2>/dev/null; then
                # exited: its own PASS/FAIL line stands -- except a TRANSIENT bringup failure
                # (Ray's cold-start "node timed out during startup / GCS overloaded / raylet failed",
                # and a DataLoader worker killed by the OOM killer -- h3 on slot 2, same afternoon): a cold
                # pod's first Ray start; 2026-08-19 slot 1 f1 died 2 min in on exactly this and
                # would have lost its lane for the whole wave). Retry ONCE, same pair, no delay;
                # the per-arm sweep in _rehearse_one clears the dead Ray's leftovers first.
                if [ "$_rt" = 0 ] && ! _has_ok "$_arm" && grep -qaE "timed out during startup|GCS has become overloaded|raylet failed to start|Failed to connect to GCS|DataLoader worker \(pid [0-9]+\) is killed by signal|BrokenPipeError" \
                        "$LOGD/rehearse_${_arm}_s${SEED}.log" "$D/n2/rehearsal_${_arm}.log" 2>/dev/null; then
                    echo "rehearsal ${_arm}: transient Ray-startup failure -- retrying once on GPUs ${_gp} ($(date))"
                    ( _rehearse_one "$_arm" "$_gp" 0 ) & _alive+=($!); _alive_arms+=("$_arm"); _alive_start+=("$_now"); _alive_gpus+=("$_gp"); _alive_retry+=(1)
                fi
                continue
            fi
            _rlog=$D/n2/rehearsal_${_arm}.log
            # Silence is measured from the LATER of (its own scheduled start, its log's last
            # write). Measuring off the log alone killed all four arms at the first poll on
            # 2026-08-19: the files still carried the previous pod's mtime, and three of the
            # arms had not even left their stagger sleep yet.
            _ref=$_t0
            if [ -f "$_rlog" ]; then
                _m=$(stat -c %Y "$_rlog" 2>/dev/null || echo 0)
                [ "$_m" -gt "$_ref" ] && _ref=$_m
            fi
            _age=$(( _now - _ref ))
            if [ "$_age" -ge "$_stall" ]; then
                echo "rehearsal ${_arm}: FAIL (STALLED ${_age}s with no log output) -- killing; see $_rlog"
                kill -TERM "$p" 2>/dev/null; sleep 2; kill -KILL "$p" 2>/dev/null
                _gpu_sweep "${_rgpus[$(( _idx - 1 ))]}"      # its engines are not our children
                continue
            fi
            _alive+=("$p"); _alive_arms+=("$_arm"); _alive_start+=("$_t0"); _alive_gpus+=("$_gp"); _alive_retry+=("$_rt")
        done
        _rpids=("${_alive[@]+"${_alive[@]}"}"); _rarms=("${_alive_arms[@]+"${_alive_arms[@]}"}")
        _rstart=("${_alive_start[@]+"${_alive_start[@]}"}"); _rgpus=("${_alive_gpus[@]+"${_alive_gpus[@]}"}")
        _rretry=("${_alive_retry[@]+"${_alive_retry[@]}"}")
        [ "${#_rpids[@]}" -gt 0 ] && sleep 60
    done
    echo "== Phase R: rehearsals finished ($(date))"
fi
_missing=""
_has_ok vanilla_corr || _missing="$_missing vanilla_corr(carrier)"
if [ -n "$_missing" ]; then
    while true; do
        _abort_check
        # honor a carrier marker that appears later (slot 0's pod passing, or an operator
        # touch) without a restart cycle -- the message always promised this; now it is true
        if _has_ok vanilla_corr; then echo "== carrier marker appeared ($(date)); continuing to Phase L"; break; fi
        echo "CARRIER REHEARSAL FAILED/MISSING:$_missing -- lanes NOT launched; inspect $LOGD/rehearse_vanilla_corr_s${SEED}.log; fix, then touch the .OK (picked up within 2 min) or the abort marker to restart ($(date))"
        sleep 120
    done
fi
# lanes whose own rehearsal failed are dropped from this launch (logged), the rest go
_launch=""
for spec in $LANES; do
    ARM=${spec%%:*}
    if _has_ok corr_wave || _has_ok "$ARM"; then _launch="$_launch $spec"; else echo "lane $ARM: rehearsal not PASS -- skipped"; fi
done
LANES=$_launch
ARMS=(); for spec in $LANES; do ARMS+=("${spec%%:*}"); done
[ -n "$LANES" ] || { while true; do _abort_check; echo "no lane passed rehearsal in slot ${SLOT} ($(date))"; sleep 120; done; }
echo "== Phase R: markers present for: ${ARMS[*]}"

echo "== Phase L: GPU state -- used MiB: $(_gpu_used 0,1,2,3,4,5,6,7)"
_gpu_sweep 0,1,2,3,4,5,6,7
_warm_ray "Phase L"
echo "== Phase L: slot ${SLOT} lanes: $LANES (250 steps, seed ${SEED})"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE=offline

_launch_lane() {  # ARM GPUS
    local ARM=$1 GPUS=$2 attempt
    for attempt in 1 2 3; do
        (
            set -e
            _arm_env=$(python scripts/arm.py env "$ARM")   # a refused arm dies here, never silent vanilla
            eval "$_arm_env"
            export EXPERIMENT_NAME="${ARM}_s${SEED}_16k"
            export CUDA_VISIBLE_DEVICES=$GPUS
            export TOTAL_TRAINING_STEPS=250
            export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
            # Derived, not hardcoded (2026-08-21): arm.py's env above is exactly the step
            # that can swap the pair (a2_coldstart replaces STUDENT_MODEL, the I-axis and
            # any future pair cell replaces TEACHER_MODEL). A literal group name files
            # those runs under the pair they are NOT -- the one label whose whole job is
            # to say which pair a curve belongs to. Same derivation as deploy/dsw/_lane.sh;
            # defaults mirror run_opd_baseline.sh, which is where they live.
            _short() { local q="${1%/}"; q="${q%/hf}"; basename "$q"; }
            export WANDB_RUN_GROUP="$(_short "${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}")__from__$(_short "${TEACHER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}")__s${SEED}"
            export WANDB_TAGS="${ARM},N,seed${SEED},dlc_corr_wave1,slot${SLOT}"
            # every *_corr arm env pins SIMOPD_STOP_IDS=off; n2_termcal (backlog fill) does too
            echo "lane ${ARM}: SIMOPD_STOP_IDS=${SIMOPD_STOP_IDS:-<unset>} SIMOPD_TERM_EVENT=${SIMOPD_TERM_EVENT:-0} GPUS=${GPUS}"
            bash scripts/run_opd_baseline.sh \
                data.seed="$SEED" \
                actor_rollout_ref.rollout.seed="$SEED"
        ) && { echo "lane ${ARM}: attempt $attempt completed"; return 0; }
        echo "lane ${ARM}: attempt $attempt failed ($(date)); resume-retry"
        sleep 30
    done
    return 1
}

# Stagger, for the same reason Phase R does: four lanes bringing up two vLLM servers each,
# all at once, is ~40 processes racing for the same mount and the same host RAM. 2026-08-19
# 14:18: slots 0 and 3 launched their four lanes simultaneously and THREE of each four went
# silent right after "Started a local Ray instance" -- 45 min, no vLLM line, no traceback --
# while slots whose herd was smaller (3 lanes) all survived. One lane at a time warms
# everything the next one needs.
_LSTAG=${LANE_STAGGER:-$(cat "$LOGD/LANE_STAGGER" 2>/dev/null || echo 150)}
_lpids=(); _larms=(); _lgpus=(); _lstart=(); _lretry=(); _ldelay=0; _now0=$(date +%s)
for spec in $LANES; do
    ARM=${spec%%:*}; GPUS=${spec##*:}
    # 重启时,已经跑满的臂不再起 lane —— 直接把那对卡给 eval(slot6 的 h7 就是这种)
    _ck=$(ls -d "$D/ckpt/simopd/${ARM}_s${SEED}_16k/global_step_"* 2>/dev/null |
          sed 's/.*global_step_//' | sort -n | tail -1)
    if [ "${_ck:-0}" -ge "${FLEET_TOTAL_STEPS:-250}" ] 2>/dev/null; then
        echo "lane ${ARM}: 已跑满 ${_ck} 步,跳过训练"
        _eval_handoff "$ARM" "$GPUS"
        continue
    fi
    ( [ "$_ldelay" -gt 0 ] && sleep "$_ldelay"; _gpu_sweep "$GPUS"; _wait_gpu_free "$GPUS"
      _launch_lane "$ARM" "$GPUS" ) > "$LOGD/lane_${ARM}_s${SEED}.log" 2>&1 & _lpids+=($!)
    _larms+=("$ARM"); _lgpus+=("$GPUS"); _lstart+=($(( _now0 + _ldelay ))); _lretry+=(0)
    _ldelay=$(( _ldelay + _LSTAG ))
done
echo "lanes launched, staggered ${_LSTAG}s apart (logs $LOGD/lane_<arm>_s${SEED}.log)"
echo "   to add a lane later WITHOUT touching DLC: touch $LOGD/fleet_relaunch_slot${SLOT}_s${SEED}"
# Lanes hold their GPU memory from init on; the eval workers' own gpu_has_room check keeps
# them off busy cards, so the pause is only needed for the bringup window that just closed.
# (Residual known gap: a lane ATTEMPT that dies frees its pair for the seconds before the
# retry -- a worker sweep can steal it. Rare; the retry then fails loudly, not silently.)
_eval_unpause
# A plain `wait` here is the last lock-out left: with lanes running, the reload marker is
# deliberately ignored (it must never kill training), so a slot that launched only 3 of its 4
# lanes -- one arm's rehearsal having failed on something since fixed -- could not pick the
# fourth up without stopping the whole DLC job (2026-08-19: f1 and h3, 4 idle GPUs, and the
# 40 cards now live in ONE job so stopping costs every other lane too). So poll for a SECOND,
# differently-named marker that means exactly "kill my lanes and start over": deliberate,
# never confusable with the reload marker, and cheap because a lane that has banked
# checkpoints resumes from them.
while [ "${#_lpids[@]}" -gt 0 ]; do
    if [ -f "$LOGD/fleet_relaunch_slot${SLOT}_s${SEED}" ]; then
        rm -f "$LOGD/fleet_relaunch_slot${SLOT}_s${SEED}"
        echo "== Phase L: RELAUNCH requested ($(date)) -- killing this slot's lanes and re-running Phase R/L"
        for p in "${_lpids[@]}"; do kill -TERM "$p" 2>/dev/null; done
        pkill -f "verl.trainer.main_ppo" 2>/dev/null
        sleep 10
        for p in "${_lpids[@]}"; do kill -KILL "$p" 2>/dev/null; done
        _kill_all_ray
        _gpu_sweep 0,1,2,3,4,5,6,7
        [ -n "${_hb_pid:-}" ] && kill "$_hb_pid" 2>/dev/null
        [ -n "${LOCK:-}" ] && rm -rf "$LOCK"
        [ "${_slot_from_rank:-0}" = 1 ] && export SLOT=auto
        exec bash "$ROOT/deploy/dlc/corr_wave_fleet.sh"
    fi
    # Lane hang watchdog. _launch_lane already retries three times, but only when the run
    # EXITS; a lane that hangs in bringup (see above) never exits and never retries. Kill the
    # hung run itself -- not the subshell -- so its own retry loop takes the next attempt,
    # with the pair swept first.
    # 40 min: a healthy lane is never that quiet -- the agent loop prints a pending/running/
    # finished heartbeat every minute during rollout AND validation, a 16k step is ~3 min, a
    # checkpoint save a few -- while today's hangs were 45+ min of nothing after Ray init. A
    # false kill costs the steps since the last checkpoint (<= 25), a late one 40 idle min.
    _lstall=$(( ${LANE_STALL_MIN:-$(cat "$LOGD/LANE_STALL_MIN" 2>/dev/null || echo 40)} * 60 ))
    _alive=(); _alive_arms=(); _alive_gpus=(); _alive_start=(); _alive_retry=(); _idx=0; _now=$(date +%s)
    for p in "${_lpids[@]}"; do
        _arm=${_larms[$_idx]}; _gp=${_lgpus[$_idx]}; _t0=${_lstart[$_idx]}; _rt=${_lretry[$_idx]}; _idx=$((_idx+1))
        if ! kill -0 "$p" 2>/dev/null; then
            _eval_handoff "$_arm" "$_gp"     # 跑满的就地转 eval;没跑满的原样丢弃
            continue
        fi
        _llog=$LOGD/lane_${_arm}_s${SEED}.log
        _ref=$_t0
        if [ -f "$_llog" ]; then
            _m=$(stat -c %Y "$_llog" 2>/dev/null || echo 0)
            [ "$_m" -gt "$_ref" ] && _ref=$_m
        fi
        if [ $(( _now - _ref )) -ge "$_lstall" ] && [ "$_rt" -lt 2 ]; then
            echo "lane ${_arm}: no output for $(( (_now - _ref) / 60 )) min -- killing the run so its retry loop takes over ($(date))"
            pkill -f "trainer.experiment_name=${_arm}_s${SEED}_16k" 2>/dev/null
            sleep 5
            pkill -KILL -f "trainer.experiment_name=${_arm}_s${SEED}_16k" 2>/dev/null
            _gpu_sweep "$_gp"
            _rt=$(( _rt + 1 ))
        fi
        _alive+=("$p"); _alive_arms+=("$_arm"); _alive_gpus+=("$_gp"); _alive_start+=("$_t0"); _alive_retry+=("$_rt")
    done
    _lpids=("${_alive[@]+"${_alive[@]}"}"); _larms=("${_alive_arms[@]+"${_alive_arms[@]}"}")
    _lgpus=("${_alive_gpus[@]+"${_alive_gpus[@]}"}"); _lstart=("${_alive_start[@]+"${_alive_start[@]}"}")
    _lretry=("${_alive_retry[@]+"${_alive_retry[@]}"}")
    [ "${#_lpids[@]}" -gt 0 ] && sleep 60
done
echo "== Phase L done"
ok=0
for ARM in "${ARMS[@]}"; do
    ck=$(ls -d "$D/ckpt/simopd/${ARM}_s${SEED}_16k/global_step_"* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    echo "  ${ARM}_s${SEED}_16k: last checkpoint step ${ck:-NONE}"
    [ -n "$ck" ] && ok=$((ok+1))
done
if [ "$ok" -eq 0 ]; then
    while true; do
        _abort_check
        echo "ALL LANES DEAD, zero checkpoints banked -- NOT declaring success; inspect $LOGD/lane_*_s${SEED}.log ($(date))"
        sleep 120
    done
fi
echo "CORR_WAVE_FLEET_SLOT${SLOT}_DONE"
# Do NOT exit: a finished rank-0 script completes the whole pytorchjob and takes every
# other slot's worker down with it (and a completed job cannot be reloaded -- the exact
# trap the a/h fleets fell into on 2026-08-20). Hold the worker, poll the markers, and
# promote a staged next-wave lane map if one appears.
while true; do
    _abort_check
    if [ -f "$LOGD/fleet_relaunch_slot${SLOT}_s${SEED}" ]; then
        rm -f "$LOGD/fleet_relaunch_slot${SLOT}_s${SEED}"
        echo "== post-DONE relaunch requested ($(date)); re-exec"
        [ -n "${_hb_pid:-}" ] && kill "$_hb_pid" 2>/dev/null
        [ -n "${LOCK:-}" ] && rm -rf "$LOCK"
        [ "${_slot_from_rank:-0}" = 1 ] && export SLOT=auto
        exec bash "$ROOT/deploy/dlc/corr_wave_fleet.sh"
    fi
    if [ -f "$LOGD/slot${SLOT}_s${SEED}_lanes.next" ]; then
        mv "$LOGD/slot${SLOT}_s${SEED}_lanes.next" "$LOGD/slot${SLOT}_s${SEED}_lanes"
        echo "== next wave staged -> promoted: $(cat "$LOGD/slot${SLOT}_s${SEED}_lanes") ($(date)); re-exec"
        [ -n "${_hb_pid:-}" ] && kill "$_hb_pid" 2>/dev/null
        [ -n "${LOCK:-}" ] && rm -rf "$LOCK"
        [ "${_slot_from_rank:-0}" = 1 ] && export SLOT=auto
        exec bash "$ROOT/deploy/dlc/corr_wave_fleet.sh"
    fi
    echo "slot ${SLOT}: wave complete, holding for next wave / markers ($(date))"
    sleep 120
done
