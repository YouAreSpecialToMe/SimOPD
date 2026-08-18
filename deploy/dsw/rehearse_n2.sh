#!/usr/bin/env bash
# 3-step rehearsal for n2_eos_aux on a 2-GPU pair, machine-verdicted (same shape as
# rehearse_a_axis.sh; exit 0 = every criterion passed, .OK is yours to touch).
#
#   bash deploy/dsw/rehearse_n2.sh 0,1
#
# What it proves that the CPU battery cannot: the sitecustomize hook fires inside
# vLLM's engine-core process of the TEACHER server (Sampler.gather_logprobs
# rebound), the N2 reader in the server actor turns real vLLM dicts into
# [top-64 | 151643 151645 | sampled] rows, the kernel accepts them at real widths
# and lengths, and the eos_* panels come out. The teacher job's Ray actors do NOT
# stream to this log (separate Ray job), so no banner grep on their side --
# instead the kernel is the guard: it REFUSES a payload without the exact block
# (wrong ids / -inf), so 3 green steps imply the whole chain worked.
set -euo pipefail
GPUS=${1:?gpu pair, e.g. 0,1}
ARM=n2_eos_aux
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
D=/mgfs/shared/Group_GY/changhao/simopd_data
LOGD=$D/n2
mkdir -p "$LOGD"
cd "$ROOT"
source simopd/bin/activate
[ -f "$ROOT/simopd_env.sh" ] && source "$ROOT/simopd_env.sh"

_arm_env=$(python scripts/arm.py env "$ARM")
eval "$_arm_env"

export EXPERIMENT_NAME="rehearsal_${ARM}"
export REHEARSAL=1
export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE=offline
export DATA_DIR=$D/simopd_math
export CKPT_ROOT=$D/ckpt
export TOTAL_TRAINING_STEPS=3
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
export PYTHONUNBUFFERED=1

LOG=$LOGD/rehearsal_${ARM}.log
if [ "${VERDICT_ONLY:-0}" != 1 ]; then
    rm -rf "$CKPT_ROOT/simopd/rehearsal_${ARM}"
    echo "rehearsal $ARM on GPUs $GPUS -> $LOG"
    rc=0
    bash scripts/run_opd_baseline.sh \
        data.seed=0 \
        actor_rollout_ref.rollout.seed=0 \
        trainer.resume_mode=disable \
        > "$LOG" 2>&1 || rc=$?
else
    echo "verdict-only re-judge of $LOG"
    rc=0
fi

fail() { echo "REHEARSAL FAIL [$ARM]: $1"; tail -5 "$LOG" | sed 's/^/    /'; exit 1; }

[ "$rc" -eq 0 ] || fail "run exited $rc"
steps=$(grep -c "step:" "$LOG" || true)
[ "${steps:-0}" -ge 3 ] || fail "only ${steps:-0}/3 training steps in log"
last_step=$(grep -n "step:" "$LOG" | tail -1 | cut -d: -f1)
first_tb=$(grep -n "Traceback (most recent call last)" "$LOG" | head -1 | cut -d: -f1 || true)
if [ -n "$first_tb" ] && [ "$first_tb" -lt "$last_step" ]; then
    fail "Traceback during training (line $first_tb < last step line $last_step)"
elif [ -n "$first_tb" ]; then
    echo "note [$ARM]: teardown-phase traceback tolerated (line $first_tb > last step $last_step, exit 0)"
fi
# The trainer-side install banner is visible (same Ray job as the driver).
grep -q 'b3_additive armed' "$LOG" || fail "additive-term wrapper never armed (N2's EOS term would be dropped)"
# The kernel refuses a payload without the exact block, so a green run is the proof;
# the panels are the second witness: every eos_* key must be present, eos_missing 0.
python - "$LOG" <<'PY' || exit 1
import re, sys
log = open(sys.argv[1], errors="replace").read()
steps = [l for l in log.splitlines() if "step:" in l and "eos_aux_term" in l]
if not steps:
    sys.exit("REHEARSAL FAIL [n2_eos_aux]: no step line carries distillation/eos_aux_term")
last = steps[-1]
def get(k):
    m = re.search(r"actor/distillation/%s:([-+0-9.eE]+)" % re.escape(k), last)
    return float(m.group(1)) if m else None
need = ["eos_aux_term", "eos_missing", "eos_sampled_is_stop", "eos_q_mean", "eos_p_mean",
        "eos_ravail_500up", "eos_frac_rep", "eos_diag_q_0",
        "eos_q_pos0_100", "eos_dstop_pos0_100", "eos_q_pos2k_up", "eos_dstop_pos2k_up"]
missing = [k for k in need if get(k) is None]
if missing:
    sys.exit(f"REHEARSAL FAIL [n2_eos_aux]: panels missing from step line: {missing}")
if get("eos_missing") != 0.0:
    sys.exit(f"REHEARSAL FAIL [n2_eos_aux]: eos_missing = {get('eos_missing')} (sampled column had -inf)")
q, p = get("eos_q_mean"), get("eos_p_mean")
if not (0.0 <= q <= 1.0 and 0.0 <= p <= 1.0):
    sys.exit(f"REHEARSAL FAIL [n2_eos_aux]: q/p out of [0,1]: {q} {p}")
print(f"panels ok: eos_aux_term={get('eos_aux_term'):.4g} q_mean={q:.4g} p_mean={p:.4g} "
      f"ravail_500up={get('eos_ravail_500up'):.4g} diag_q0(im_end)={get('eos_diag_q_0'):.4g} "
      f"sampled_is_stop={get('eos_sampled_is_stop'):.4g} frac_rep={get('eos_frac_rep'):.3g}")
PY
echo "REHEARSAL PASS [$ARM]  ->  touch $LOGD/rehearsal_${ARM}.OK"
