#!/usr/bin/env bash
# 3-step rehearsal for the termination family on a 2-GPU pair, machine-verdicted
# (same shape as rehearse_a_axis.sh; exit 0 = every criterion passed, .OK is yours
# to touch).
#
#   bash deploy/dsw/rehearse_n2.sh 0,1                        # ARM=vanilla_corr (default; the new vanilla)
#   ARM=n2_corr bash deploy/dsw/rehearse_n2.sh 0,1             # N2 on the corrected base
#   ARM=h2_last_segment_corr bash deploy/dsw/rehearse_n2.sh 0,1  # any *_corr cell (same carrier)
#
# What it proves that the CPU battery cannot: the sitecustomize hook fires inside
# vLLM's engine-core process of the TEACHER server (Sampler.gather_logprobs
# rebound), the N2 reader in the server actor turns real vLLM dicts into
# [top-64 | 151643 151645 | sampled] rows (E_S={151643}, E_T={151643,151645}), the kernel accepts them at real widths
# and lengths, and the eos_* panels come out. The teacher job's Ray actors do NOT
# stream to this log (separate Ray job), so no banner grep on their side --
# instead the kernel is the guard: it REFUSES a payload without the exact block
# (wrong ids / -inf), so 3 green steps imply the whole chain worked.
set -euo pipefail
GPUS=${1:?gpu pair, e.g. 0,1}
ARM=${ARM:-vanilla_corr}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
D=${SIMOPD_STORE:-/mgfs/shared/Group_GY/changhao/simopd_data}
export SIMOPD_STORE=$D          # arms.yaml 的资产路径靠它展开(arm.py env)
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

# The rollout stop contract this rehearsal ran under (run_opd_baseline.sh pins it per run):
# the termination family reads E_S off the same set, so print it next to the verdict.
_pin="$CKPT_ROOT/simopd/rehearsal_${ARM}/simopd_stop_contract.txt"
[ -f "$_pin" ] && echo "stop contract pinned for this rehearsal: $(cat "$_pin")  (E_S must equal model eos + this set)"

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
# what the arm's registry function emits: 'cal' (k1_termcal: BCE term + panels), 'family'
# (termfix / the sampled-k1 family / D / FiRe under SIMOPD_TERM_EVENT=1: panels), 'none'
# (verl-side or distributional registry fns: b2/e2/c3 corrected -- green run + 3 steps only)
PANEL_MODE=none
case "${DISTILLATION_LOSS_MODE:-k1_rec}" in
    k1_termcal) PANEL_MODE=cal ;;
    k1_termfix|k1_rec|k1_softlog|k1_power|k1_posclip|k1_tanh|k1_verified_only|k1_failure_only|k1_firstseg|k1_lastseg|k1_randseg|k1_randscatter|skew_kl_a0.1|k2_kdrl|k1_rgopd_gate|k1_fire_gate|tip_select|selectkd_verify|teachability_select)
        [ "${SIMOPD_TERM_EVENT:-0}" = 1 ] || [ "${DISTILLATION_LOSS_MODE:-}" = k1_termfix ] && PANEL_MODE=family ;;
esac
if [ "$PANEL_MODE" = cal ]; then
    grep -q 'b3_additive armed' "$LOG" || fail "additive-term wrapper never armed (N2's stop term would be dropped)"
fi
echo "panel mode for $ARM: $PANEL_MODE"
# The kernel refuses a payload without the exact block, so a green run is the proof;
# the panels are the second witness: every eos_* key must be present, eos_missing 0.
if [ "$PANEL_MODE" = none ]; then
    echo "REHEARSAL PASS [$ARM]  (green run, $steps steps; this registry function emits no eos_* panels)  ->  touch $LOGD/rehearsal_${ARM}.OK"
    exit 0
fi
python - "$LOG" "$ARM" "$PANEL_MODE" "${SIMOPD_TERM_EVENT:-0}" <<'PY' || exit 1
import re, sys
log = open(sys.argv[1], errors="replace").read()
ARM, MODE, TE = sys.argv[2], sys.argv[3], sys.argv[4] == "1"
key0 = "eos_aux_term" if MODE == "cal" else "eos_dl_at_stop"
steps = [l for l in log.splitlines() if "step:" in l and key0 in l]
if not steps:
    sys.exit(f"REHEARSAL FAIL [{ARM}]: no step line carries distillation/{key0}")
last = steps[-1]
def get(k):
    m = re.search(r"actor/distillation/%s:([-+0-9.eE]+)" % re.escape(k), last)
    return float(m.group(1)) if m else None
need = ["eos_missing", "eos_sampled_is_stop", "eos_q_mean", "eos_p_mean",
        "eos_ravail_500up", "eos_frac_rep", "eos_qm_0", "eos_qm_1", "eos_pm_0", "eos_pm_1",
        "eos_n_stop", "eos_dl_at_stop", "eos_q_at_stop", "eos_p_at_stop",
        "eos_q_pos0_100", "eos_dstop_pos0_100", "eos_q_pos2k_up", "eos_dstop_pos2k_up"]
need += ["eos_aux_term"] if MODE == "cal" else []
need += ["eos_dl_at_stop_raw"]
missing = [k for k in need if get(k) is None]
if missing:
    sys.exit(f"REHEARSAL FAIL [{ARM}]: panels missing from step line: {missing}")
if get("eos_missing") != 0.0:
    sys.exit(f"REHEARSAL FAIL [{ARM}]: eos_missing = {get('eos_missing')} (sampled column had -inf)")
q, p = get("eos_q_mean"), get("eos_p_mean")
if not (0.0 <= q <= 1.0 and 0.0 <= p <= 1.0):
    sys.exit(f"REHEARSAL FAIL [{ARM}]: q/p out of [0,1]: {q} {p}")
head = f"eos_aux_term={get('eos_aux_term'):.4g}" if MODE == "cal" else f"dl_at_stop_raw={get('eos_dl_at_stop_raw'):.3g}"
print(f"panels ok [{ARM}]: {head} q_mean={q:.4g} p_mean={p:.4g} "
      f"ravail_500up={get('eos_ravail_500up'):.4g} q(eot)={get('eos_qm_0'):.3g} q(im_end)={get('eos_qm_1'):.3g} "
      f"p(eot)={get('eos_pm_0'):.3g} p(im_end)={get('eos_pm_1'):.3g} | at student stops: n={get('eos_n_stop'):.3g} "
      f"dl={get('eos_dl_at_stop'):.3g} q={get('eos_q_at_stop'):.3g} p={get('eos_p_at_stop'):.3g} | "
      f"sampled_is_stop={get('eos_sampled_is_stop'):.4g} frac_rep={get('eos_frac_rep'):.3g}")
# the audit's prediction, checked live on 3 steps of real rollouts: with stops present, vanilla's
# token-level Delta-ell at the student's own stop tokens is deeply negative (teacher's terminator is
# <|im_end|>). N2 applies it (eos_dl_at_stop); N0 shows it in eos_dl_at_stop_raw and applies the
# event-level value instead.
raw_key = "eos_dl_at_stop_raw"
if get("eos_n_stop") and get("eos_n_stop") > 0:
    if get(raw_key) > -5.0:
        print(f"NOTE: {raw_key}={get(raw_key):.3g} is not the audit's ~-25 -- inspect before launching wave 17")
    # Did the event fix ENGAGE? The question is applied-vs-raw, not an absolute floor: the
    # event-level value is whatever the teacher's termination mass says at those stops, and it
    # is legitimately several nats negative early in training (2026-08-19 GPU: vanilla_corr
    # -3.3/-3.6 vs raw -16.6; n2_corr -4.4/-5.2 vs raw -12.5/-19.0; f1 -5.65 vs raw -16.5 --
    # the old absolute -5 threshold failed f1 for a value the carrier itself produces, and
    # would have failed N2 on the next seed). Engaged == applied is clearly ABOVE raw.
    if (TE or MODE == "family"):
        dl, raw = get("eos_dl_at_stop"), get(raw_key)
        if raw is not None and dl is not None and raw < -5.0 and dl - raw < 3.0:
            sys.exit(f"REHEARSAL FAIL [{ARM}]: applied eos_dl_at_stop={dl:.3g} vs raw {raw:.3g} "
                     f"(gain {dl - raw:.3g} nats) -- still token-level, the fix did not engage")
        if dl is not None and dl < -12.0:
            sys.exit(f"REHEARSAL FAIL [{ARM}]: applied eos_dl_at_stop={dl:.3g} is token-level deep "
                     "even though it differs from raw -- inspect before launching")
PY
echo "REHEARSAL PASS [$ARM]  ->  touch $LOGD/rehearsal_${ARM}.OK"
