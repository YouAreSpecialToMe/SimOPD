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
        eval "$(python "$SNAP"/scripts/arm.py env "$ARM")"
        export EXPERIMENT_NAME="$NAME"
        # The wandb GROUP is the cell a run belongs to: model pair x seed. Runs
        # inside one group are the OPD methods being compared; groups are what you
        # compare across. verl's tracking.py calls wandb.init() with no group=
        # argument, but wandb reads WANDB_RUN_GROUP from the environment -- verified
        # end to end on 0.28.1, so no verl patch is needed.
        #
        # Set AFTER arm.py's env eval, because that is the step that can swap the
        # pair for a given arm (a2_coldstart replaces STUDENT_MODEL with its SFT
        # checkpoint, and the I-axis arms replace TEACHER_MODEL).
        #
        # The two defaults are MIRRORED from run_opd_baseline.sh:65-66, which is
        # where they actually live. If they are changed there and not here, this
        # label goes stale -- visibly, since the group name would stop matching the
        # run's own config, but stale all the same. Duplicated rather than sourced
        # because run_opd_baseline.sh runs from the snapshot and resolves them long
        # after this point.
        # basename alone is wrong for a local checkpoint: a2's STUDENT_MODEL ends
        # in .../ckpt/coldstart_sft/hf, and "hf" as a group name says nothing. Strip
        # a trailing /hf (the merger's output dir) before taking the basename, so the
        # label is coldstart_sft. Hub ids like Qwen/Qwen3-1.7B-Base are unaffected.
        _short() { local q="${1%/}"; q="${q%/hf}"; basename "$q"; }
        _stu=$(_short "${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}")
        _tch=$(_short "${TEACHER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}")
        export WANDB_RUN_GROUP="${_stu}__from__${_tch}__s${SEED}"
        # job_type and tags are the SECOND grouping axis, free: wandb lets you
        # regroup by either in the UI without touching a run. job_type is the arm,
        # so "group by job_type" gives seed-spread per method -- the stability view,
        # which the group above cannot show (its members are different methods, so
        # its aggregate band would be spread across methods, not noise).
        #
        # None of this makes data queryable that was not already: ray_trainer.py:1398
        # hands wandb the whole resolved hydra config, so data.seed, the model paths
        # and distillation.distillation_loss.loss_mode are filterable regardless.
        # These three fields only make the DEFAULT view usable without building a
        # query first.
        export WANDB_JOB_TYPE="$ARM"
        # Axis letter from the run_id prefix -- the registry's naming is a1/b2/c1/...
        # by construction. vanilla has no axis and says so rather than guessing.
        case "$ARM" in
            vanilla) _axis=baseline ;;
            *)       _axis="axis$(printf '%s' "${ARM:0:1}" | tr '[:lower:]' '[:upper:]')" ;;
        esac
        export WANDB_TAGS="${ARM},${_axis},seed${SEED},${_stu}__from__${_tch}"
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
