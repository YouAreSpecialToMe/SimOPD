#!/usr/bin/env bash
# Corrected-rerun wave 1 on DLC: 40 cards = 5 workers x 8 GPUs, one job per SLOT.
#
# Every cell is "the banked arm + SIMOPD_TERM_EVENT=1" (docs/RESULTS-GAPS.md, corrected-rerun
# roster; docs/MECHANISMS.md M-I). One seed (0), seed-paired against the banked arms; the
# legacy single-eos rollout contract is pinned OFF inside every *_corr arm env, so nothing
# but the knob moves. Run names follow the campaign convention <arm>_s<seed>_16k, so
# eval_suite / extract_post_eval / verdict pick them up unchanged (eval contract = the
# run's own pin, i.e. legacy -- eval_offline --stop-token-ids auto).
#
# Slot map (2-GPU lanes unless noted; g2/d2/d3 are the KEEP_SAMPLED [T,V] family = 4 GPUs):
#
#   SLOT 0  vanilla_corr(0,1)   n2_corr(2,3)          b1_skew_kl_corr(4,5)      b5_k2_corr(6,7)
#   SLOT 1  f1_soft_log_corr    f2_hard_clip_corr     f3_power_corr             g1_verified_only_corr
#   SLOT 2  g4_failure_only_corr g6_seqmean_corr      h2_last_segment_corr      h3_random_segment_corr
#   SLOT 3  h4_random_scatter_corr b2_forward_kl_corr e2_set_coverage_a0_corr   c3_intersection_corr
#   SLOT 4  g5_rgopd_gate_corr(0,1)  g2_fire_likelihood_corr(2,3,4,5)  n2_termcal(6,7 -- backlog fill: raw N2)
#   SLOT 5  d2_selectkd_corr(0,1,2,3)  d3_teachability_corr(4,5,6,7)   -- batch 2 (a 6th worker, or after any slot frees)
#
# SLOT 0-4 = the user's 20-lane plan minus d2/d3 (4-card lanes do not fit 40 with the other 18);
# 38 cards busy + 2 filled by the backlog's raw N2. SLOT 5 completes the plan on 8 more cards.
#
# Lineage: deploy/dlc/h_axis_fleet.sh (its 11 fixes inherited: NCCL/GLOO iface by presence,
# explicit pid wait, zero-checkpoint never reports success, remote abort marker, bounded
# resume-retry). Differences: ROOT is the expansion tree (SimOPD-exp -- the corrected code
# lives there), Phase R wants the carrier rehearsal marker (rehearsal_vanilla_corr.OK) plus
# each slot arm's own marker unless the batch marker rehearsal_corr_wave.OK exists.
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
        4) echo "g5_rgopd_gate_corr:0,1 g2_fire_likelihood_corr:2,3,4,5 n2_termcal:6,7" ;;
        5) echo "d2_selectkd_corr:0,1,2,3 d3_teachability_corr:4,5,6,7" ;;
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
  40 cards = SLOT 0-4 (5 jobs). SLOT 5 (d2/d3, 8 cards) = batch 2: a 6th job now, or when a slot frees.
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
exec > >(tee -a "$LOGD/fleet_slot${SLOT}_s${SEED}_$(date +%Y%m%d_%H%M%S).log") 2>&1
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

# rehearsal markers: the carrier proof + each arm's own, unless the batch marker exists
_has_ok() { [ -f "$LOGD/rehearsal_$1.OK" ] || [ -f "$D/n2/rehearsal_$1.OK" ]; }   # rehearse_n2.sh writes under $D/n2
_missing=""
_has_ok vanilla_corr || _missing="$_missing vanilla_corr(carrier)"
if ! _has_ok corr_wave; then
    for ARM in "${ARMS[@]}"; do _has_ok "$ARM" || _missing="$_missing $ARM"; done
fi
if [ -n "$_missing" ]; then
    while true; do
        _abort_check
        echo "REHEARSAL MARKERS MISSING:$_missing -- rehearse on gpu193 (ARM=<arm> bash deploy/dsw/rehearse_n2.sh 0,1), touch $LOGD/rehearsal_<arm>.OK or $LOGD/rehearsal_corr_wave.OK ($(date))"
        sleep 600
    done
fi
echo "== Phase R: rehearsal markers present"

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
