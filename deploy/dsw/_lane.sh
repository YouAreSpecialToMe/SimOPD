#!/usr/bin/env bash
# One lane of deploy/dsw/run_parallel.sh: chew through LANE_RUNS on the GPUs this
# process can already see (the launcher set CUDA_VISIBLE_DEVICES). Not meant to be
# invoked directly.

set -uo pipefail

SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SIMOPD_ROOT"
# One venv, ./simopd, on every machine. SIMOPD_VENV overrides it.
VENV=${SIMOPD_VENV:-simopd}
source "$VENV/bin/activate"
export PYTHONPATH="$SNAP/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RAY_TMPDIR"

echo "lane on GPUs [${CUDA_VISIBLE_DEVICES}] : ${LANE_RUNS}"

declare -A RESULT
for entry in $LANE_RUNS; do
    ARM=${entry%%:*}
    SEED=${entry##*:}
    NAME="${ARM}_s${SEED}${LANE_TAG:+_$LANE_TAG}"
    echo ""
    echo "################ RUN: $NAME ################"
    date +"start %F %T"
    (
        set -e
        eval "$(python "$SNAP"/scripts/arm.py env "$ARM")"
        export EXPERIMENT_NAME="$NAME"
        export TOTAL_TRAINING_STEPS=$LANE_STEPS
        export TEST_FREQ=$LANE_TEST_FREQ
        export SAVE_FREQ=$LANE_SAVE_FREQ
        bash "$SNAP"/scripts/run_opd_baseline.sh \
            data.seed="$SEED" \
            actor_rollout_ref.rollout.seed="$SEED"
    )
    rc=$?   # capture before anything else, including the substitution below
    RESULT[$NAME]=$([ $rc -eq 0 ] && echo OK || echo FAIL)
    echo "################ $NAME -> ${RESULT[$NAME]} ################"
    date +"end %F %T"
    # NOT `ray stop --force`: that is machine-wide and would kill the other three
    # lanes' clusters along with this one. Ray shuts itself down when the driver
    # exits; this only sweeps orphans, matched by this lane's private temp dir.
    pkill -f "$RAY_TMPDIR" >/dev/null 2>&1 || true
    sleep 5
done

echo ""
echo "================ LANE SUMMARY (GPUs ${CUDA_VISIBLE_DEVICES}) ================"
for name in "${!RESULT[@]}"; do printf '%-44s %s\n' "$name" "${RESULT[$name]}"; done
echo "LANE_DONE"
