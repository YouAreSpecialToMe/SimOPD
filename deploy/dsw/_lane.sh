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

# Kill what this lane leaked, and only what this lane leaked.
#
# This used to be `pkill -f "$RAY_TMPDIR"` alone, which silently missed most of it:
# Ray renames its Python workers to `ray::IDLE` and friends via setproctitle, so the
# temp dir is gone from their command line the moment they are up. Every run leaked a
# few, and the next ray.init() then ATTACHES to the survivors and inherits the
# environment THEY were started with -- which is how a variable you have since fixed
# in your shell can appear not to take effect at all.
#
# setproctitle also rules out matching on the environment, which was the obvious
# replacement. argv and environ are adjacent memory, and a longer title is written
# straight over the environ block: measured on this stack, two identical launches
# differing only in whether setproctitle ran leave /proc/<pid>/environ with the
# variable (plain) and without it (renamed). Same root cause as the argv problem.
#
# What a process cannot rewrite is kernel bookkeeping, so all three nets below rest
# on that, and each covers what the others structurally cannot. `ray stop --force`
# remains off limits: it is machine-wide and would take the other lanes with it.
LANE_GPUS=${CUDA_VISIBLE_DEVICES:-}

sweep_lane() {   # sweep_lane [pgid-of-the-run]
    local pgid=${1:-} sig pid
    for sig in TERM KILL; do
        # 1. the run's own process group -- everything still parented under it.
        [ -n "$pgid" ] && kill -"$sig" -- -"$pgid" 2>/dev/null

        # 2. raylet / gcs_server / dashboard. C++ binaries that do not rename
        #    themselves and still carry --temp_dir in argv, so a pattern match does
        #    find them -- and these are the ones a later ray.init() would attach to.
        #    Anchored so that lane1 cannot also match lane10.
        pkill -"$sig" -f "${RAY_TMPDIR}(/|$)" 2>/dev/null

        # 3. the renamed Python workers, which nets 1 and 2 both miss once Ray has
        #    put them in their own session. They cannot hide holding memory on a GPU,
        #    and lanes own disjoint GPUs, so this stays lane-private. Best-effort:
        #    inside some containers nvidia-smi reports no PIDs at all.
        #    Verified 2026-08-05 on an A100 node: nvidia-smi ignores
        #    CUDA_VISIBLE_DEVICES, so `-i "$LANE_GPUS"` really does mean this lane's
        #    physical GPUs. But inside a container nvidia-smi can report HOST pids,
        #    which either do not exist here or -- worse -- name an unrelated local
        #    process. So each pid is confirmed to be ours and to look like a
        #    training process before it is signalled.
        if [ -n "$LANE_GPUS" ]; then
            for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
                             -i "$LANE_GPUS" 2>/dev/null); do
                [ -r "/proc/$pid/cmdline" ] || continue
                case "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" in
                    *python*|*ray::*|*vllm*|*VLLM*|*raylet*) ;;
                    *) continue ;;
                esac
                kill -"$sig" "$pid" 2>/dev/null
            done
        fi

        [ "$sig" = TERM ] && sleep 5
    done
}

# Before anything starts, not just after each run finishes: a crashed previous
# invocation leaves survivors that this lane would otherwise silently join.
sweep_lane

declare -A RESULT
for entry in $LANE_RUNS; do
    ARM=${entry%%:*}
    SEED=${entry##*:}
    NAME="${ARM}_s${SEED}${LANE_TAG:+_$LANE_TAG}"
    echo ""
    echo "################ RUN: $NAME ################"
    date +"start %F %T"
    # Job control only around the launch, so the run becomes its own process group
    # leader -- $! is then both the PID to wait on and the PGID for net 1. Switched
    # back off immediately: leaving it on would put the sweep's own subshells into
    # fresh groups too, for no benefit.
    set -m
    (
        set -e
        # Assignment-then-eval, NOT eval "$(...)": a command substitution used as an
        # argument discards its exit status, so a refused arm (shelved/needs/typo)
        # would eval "" and run VANILLA under the arm's name, recorded OK forever --
        # the false-OK class this campaign treats as worse than failure.
        _arm_env=$(python "$SNAP"/scripts/arm.py env "$ARM")
        eval "$_arm_env"
        export EXPERIMENT_NAME="$NAME"
        export TOTAL_TRAINING_STEPS=$LANE_STEPS
        export TEST_FREQ=$LANE_TEST_FREQ
        export SAVE_FREQ=$LANE_SAVE_FREQ
        bash "$SNAP"/scripts/run_opd_baseline.sh \
            data.seed="$SEED" \
            actor_rollout_ref.rollout.seed="$SEED"
    ) &
    RUN_PGID=$!
    set +m
    wait "$RUN_PGID"
    rc=$?   # capture before anything else, including the substitution below
    RESULT[$NAME]=$([ $rc -eq 0 ] && echo OK || echo FAIL)
    echo "################ $NAME -> ${RESULT[$NAME]} ################"
    date +"end %F %T"
    sweep_lane "$RUN_PGID"
    left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    echo "swept lane; ${left} compute process(es) still on this machine (all lanes)"
done

sweep_lane   # leave nothing behind for the next invocation to attach to

echo ""
echo "================ LANE SUMMARY (GPUs ${CUDA_VISIBLE_DEVICES}) ================"
for name in "${!RESULT[@]}"; do printf '%-44s %s\n' "$name" "${RESULT[$name]}"; done
echo "LANE_DONE"
