#!/usr/bin/env bash
# Corrected-rerun wave 1 on DLC: 40 cards = 5 workers x 8 GPUs = 20 two-GPU lanes, one job per SLOT.
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
# abort marker is a one-shot restart request (consumed at start).
#
# Submit: run on any machine WITHOUT the DLC rank env -> prints the console cards and a
# `dlc submit` CLI template; inside the container the same script is the payload.
#
#   SLOT=0 SEED=0 bash /mgfs/shared/Group_GY/changhao/SimOPD-exp/deploy/dlc/corr_wave_fleet.sh

set -uo pipefail

ROOT=${ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
D=/mgfs/shared/Group_GY/changhao/simopd_data
SEED=${SEED:-0}
SLOT=${SLOT:-0}
LOGD=$D/corr_wave

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
LANES=$(_lanes_for "$SLOT")
[ -n "$LANES" ] || { echo "unknown SLOT=$SLOT (0-5)"; exit 2; }
ARMS=(); for spec in $LANES; do ARMS+=("${spec%%:*}"); done

_abort_check() {
    if [ -f "$LOGD/fleet_abort_slot${SLOT}_s${SEED}" ]; then
        echo "abort marker $LOGD/fleet_abort_slot${SLOT}_s${SEED} found; exiting 17 for AIMaster restart"
        exit 17
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
  40 cards = SLOT 0-4 (5 jobs, the 20-lane plan). SLOT 5 = backlog fill (raw N2, d1, f2_2.3, h1) for a 6th worker or a freed slot.
  前置:rehearsal_vanilla_corr.OK(载体彩排,gpu193;rehearse_n2.sh 写在 $D/n2/,这里也认)+ 各臂 .OK,或批量标记 rehearsal_corr_wave.OK。
  中止:touch $LOGD/fleet_abort_slot<k>_s${SEED}
========= dlc CLI 等价形式(填你们工作区的 WORKSPACE_ID/RESOURCE_ID/IMAGE/DATA_SOURCES;每个 SLOT 一条)=========
  for k in 0 1 2 3 4; do
    dlc submit pytorchjob --name=simopd-corr-wave1-slot\${k}-s${SEED} --workers=1 --worker_gpu=8 --worker_cpu=64 \\
      --worker_memory=512Gi --worker_image="\$IMAGE" --data_sources="\$DATA_SOURCES" \\
      --workspace_id="\$WORKSPACE_ID" --resource_id="\$RESOURCE_ID" --priority=5 --job_max_running_time_minutes=0 \\
      --command="bash -lc 'SLOT=\${k} SEED=${SEED} bash $ROOT/deploy/dlc/corr_wave_fleet.sh'"
  done
TAIL
    echo "本机不是 DLC 容器(无 rank env),以上为提交卡片;容器内执行同一脚本即载荷。"
    exit 0
fi

# ------------------------------------------------------------------ payload --
_rank=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
if [ "${_rank}" != "0" ]; then
    while true; do _abort_check; echo "rank ${_rank}: single-worker job, idling ($(date))"; sleep 600; done
fi
cd "$ROOT"
mkdir -p "$LOGD"
# an abort marker is a ONE-SHOT restart request: consumed here so the restarted worker
# does not exit again at its first idle loop
rm -f "$LOGD/fleet_abort_slot${SLOT}_s${SEED}"
exec > >(tee -a "$LOGD/fleet_slot${SLOT}_s${SEED}_$(date +%Y%m%d_%H%M%S).log") 2>&1

# ------------------------------------------------------------ slot lock ------
# Exactly ONE pod may drive a slot: a duplicate (a resubmitted job next to an auto-restarted
# one -- it happened to slot 0 on 2026-08-19: two pods rehearsed the same arms into the same
# dirs and would have trained the same run names into the same checkpoint dirs) idles here.
# The lock is a directory (atomic mkdir) with a heartbeat file; a lock whose heartbeat is
# older than 20 min is stale (owner pod gone) and is taken over.
LOCK=$LOGD/slot${SLOT}_s${SEED}.lock
_take_lock() { echo "$(hostname) pid=$$ $(date -u +%FT%TZ)" > "$LOCK/owner"; }
if mkdir "$LOCK" 2>/dev/null; then
    _take_lock
else
    _age=$(( $(date +%s) - $(stat -c %Y "$LOCK/owner" 2>/dev/null || echo 0) ))
    if [ "$_age" -lt 1200 ]; then
        while true; do
            _abort_check
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
python - <<'PY' || { while true; do _abort_check; echo "TREE STALE: no SIMOPD_TERM_EVENT support in src/ -- lanes NOT launched ($(date))"; sleep 600; done; }
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
        while true; do _abort_check; echo "ARM_LINT (scoped) FAILED -- lanes NOT launched ($(date)):"; echo "$BAD"; sleep 600; done
    fi
    echo "== arm_lint: scoped gate clean for slot ${SLOT} (full report: $LINT_LOG)"
fi

# ------------------------------------------------------------ Phase R 彩排门 --
# The carrier proof (rehearsal_vanilla_corr.OK) + each slot arm's own marker, or the batch
# marker rehearsal_corr_wave.OK. Missing markers are NOT waited on: this pod has 8 idle
# H100s, so it rehearses the missing arms itself -- deploy/dsw/rehearse_n2.sh, 3 steps,
# machine-verdicted, one lane per GPU pair in parallel -- writes the .OK on PASS and skips
# (does not launch) any lane whose rehearsal FAILED. The debug-node markers ($D/n2/) are
# honored too. A slot that has no vanilla_corr rehearses vanilla_corr on its first pair as
# the carrier proof if that marker is missing.
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
    _rehearse_one() {  # ARM PAIR
        ARM=$1 bash deploy/dsw/rehearse_n2.sh "$2" > "$LOGD/rehearse_${1}_s${SEED}.log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then touch "$LOGD/rehearsal_$1.OK"; echo "rehearsal $1: PASS"; else echo "rehearsal $1: FAIL (rc=$rc) see $LOGD/rehearse_${1}_s${SEED}.log"; fi
        return $rc
    }
    if [ "$_need_carrier" = 1 ]; then
        ( _rehearse_one vanilla_corr "${_pairs[$_i]}" ) & _rpids+=($!); _rarms+=(vanilla_corr); _i=$((_i+1))
    fi
    for ARM in "${_todo[@]}"; do
        if [ $_i -ge 4 ]; then wait "${_rpids[@]}"; _rpids=(); _i=0; fi     # more than 4 -> second round
        ( _rehearse_one "$ARM" "${_pairs[$_i]}" ) & _rpids+=($!); _rarms+=("$ARM"); _i=$((_i+1))
    done
    [ "${#_rpids[@]}" -gt 0 ] && wait "${_rpids[@]}"
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
        echo "CARRIER REHEARSAL FAILED/MISSING:$_missing -- lanes NOT launched; inspect $LOGD/rehearse_vanilla_corr_s${SEED}.log; fix, then touch the .OK (picked up within 10 min) or the abort marker to restart ($(date))"
        sleep 600
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
[ -n "$LANES" ] || { while true; do _abort_check; echo "no lane passed rehearsal in slot ${SLOT} ($(date))"; sleep 600; done; }
echo "== Phase R: markers present for: ${ARMS[*]}"

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
            export WANDB_RUN_GROUP="Qwen3-1.7B-Base__from__Qwen3-4B-Instruct-2507__s${SEED}"
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

_lpids=()
for spec in $LANES; do
    ARM=${spec%%:*}; GPUS=${spec##*:}
    ( _launch_lane "$ARM" "$GPUS" ) > "$LOGD/lane_${ARM}_s${SEED}.log" 2>&1 & _lpids+=($!)
done
echo "lanes launched (logs $LOGD/lane_<arm>_s${SEED}.log)"
wait "${_lpids[@]}"
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
        sleep 600
    done
fi
echo "CORR_WAVE_FLEET_SLOT${SLOT}_DONE"
