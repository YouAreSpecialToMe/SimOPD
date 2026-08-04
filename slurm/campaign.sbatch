#!/usr/bin/env bash
#SBATCH -J simopd-campaign
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=150G
#SBATCH -t 96:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

# One job = one worker that chews through a run list, so the whole campaign is a
# single submission per available 2-GPU slot. Parallelism comes from submitting
# this several times with disjoint RUNS (on PAI, where the account cap does not
# pin us to gpu=2), not from anything inside the job.
#
#   # everything implemented, default protocol:
#   sbatch slurm/campaign.sbatch
#   # a slice, for a second slot:
#   sbatch --export=ALL,RUNS="f1_soft_log:0 h1_first_segment:0" slurm/campaign.sbatch
#   # a quick end-to-end rehearsal before committing 40h/run:
#   sbatch --export=ALL,STEPS=3,TEST_FREQ=-1,TAG=rehearsal slurm/campaign.sbatch
#
# Each entry is arm:seed. Every run is resumable (SAVE_FREQ) and reports to wandb
# live, because a preempted or timed-out run that leaves no artifact is the one
# failure mode that already cost this project 24 GPU-hours (INFRA-NOTES).

set -uo pipefail

PROJ_DIR=$HOME/pythonProject/SimOPD
cd "$PROJ_DIR"
# One venv, ./simopd, on every machine. SIMOPD_VENV overrides it.
VENV=${SIMOPD_VENV:-simopd}
source "$VENV/bin/activate"

export RAY_TMPDIR=/scratch/zz865/ray_tmp
export CKPT_ROOT=${CKPT_ROOT:-/scratch/zz865/simopd/ckpt}
export WANDB_DIR=/scratch/zz865/simopd/wandb
mkdir -p "$RAY_TMPDIR" "$CKPT_ROOT" "$WANDB_DIR"

# Snapshot the code this job will run. bash executes a script by byte offset, so
# editing scripts/run_opd_baseline.sh while a campaign is running makes bash resume
# at a stale offset and execute garbage -- that is exactly how c1_lsm "failed" in the
# 2026-08-01 rehearsal, with a perfectly good kernel. Running from an immutable copy
# means the working tree stays editable during a 40-hour campaign.
SNAP="${SNAP_ROOT:-/scratch/zz865/simopd/snapshots}/${SLURM_JOB_ID:-$$}"
mkdir -p "$SNAP"
cp -r scripts configs src "$SNAP"/
export PYTHONPATH="$SNAP/src${PYTHONPATH:+:$PYTHONPATH}"
echo "running from snapshot: $SNAP ($(git rev-parse --short HEAD 2>/dev/null || echo nogit))"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

STEPS=${STEPS:-150}   # see run_opd_baseline.sh: 300 was 19h/arm of Mode A
TEST_FREQ=${TEST_FREQ:-25}
SAVE_FREQ=${SAVE_FREQ:-50}
TAG=${TAG:-}

# vanilla x3 seeds first: seed 0 is the reference every verdict is measured
# against, and the spread across the three IS the noise floor that gives
# "|delta| < noise floor -> tie" an operational meaning. Without it no arm can
# ever be judged 无用. Other arms are single-seed at the screening stage.
if [ -z "${RUNS:-}" ]; then
    RUNS="vanilla:0 vanilla:1 vanilla:2"
    for arm in $(python "$SNAP"/scripts/arm.py list --status stock); do
        [ "$arm" = "vanilla" ] && continue
        RUNS="$RUNS ${arm}:0"
    done
fi

echo "=== campaign: $(echo $RUNS | wc -w) runs, ${STEPS} steps each ==="
echo "$RUNS" | tr ' ' '\n' | sed 's/^/  /'
nvidia-smi -L

declare -A RESULT
for entry in $RUNS; do
    ARM=${entry%%:*}
    SEED=${entry##*:}
    NAME="${ARM}_s${SEED}${TAG:+_$TAG}"
    echo ""
    echo "################ RUN: $NAME ################"
    date +"start %F %T"
    (
        set -e
        eval "$(python "$SNAP"/scripts/arm.py env "$ARM")"
        export EXPERIMENT_NAME="$NAME"
        export PROJECT_NAME=${PROJECT_NAME:-simopd}
        export TOTAL_TRAINING_STEPS=$STEPS
        export TEST_FREQ=$TEST_FREQ
        export SAVE_FREQ=$SAVE_FREQ
        bash "$SNAP"/scripts/run_opd_baseline.sh \
            data.seed=$SEED \
            actor_rollout_ref.rollout.seed=$SEED
    )
    rc=$?   # must be captured before any other command, including the substitution below
    RESULT[$NAME]=$([ $rc -eq 0 ] && echo OK || echo FAIL)
    echo "################ $NAME -> ${RESULT[$NAME]} ################"
    date +"end %F %T"
    ray stop --force >/dev/null 2>&1 || true
    sleep 10
done

echo ""
echo "================ CAMPAIGN SUMMARY ================"
for name in "${!RESULT[@]}"; do printf '%-40s %s\n' "$name" "${RESULT[$name]}"; done
echo "=================================================="
echo "CAMPAIGN_DONE"
