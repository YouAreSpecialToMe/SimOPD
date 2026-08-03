#!/usr/bin/env bash
# Run the SimOPD campaign across the 8 GPUs of a DSW instance as 4 concurrent lanes.
#
#   bash deploy/dsw/run_parallel.sh --rehearsal          # 3 steps/arm, ~30 min/lane
#   bash deploy/dsw/run_parallel.sh                      # 300 steps/arm, the real campaign
#   LANES=2 bash deploy/dsw/run_parallel.sh "vanilla:0 vanilla:1"
#
# One run = 2 GPUs (actor + teacher pool); verl registers the teacher pool as a
# separate Ray resource pool and will not share the actor's GPU, so 2 is the floor
# and 8 GPUs means 4 lanes. Each lane gets its own CUDA_VISIBLE_DEVICES, its own Ray
# temp dir, and a lane-local log; nothing is shared between them but the filesystem.
#
# DSW is interactive, so lanes run under nohup and survive a dropped terminal --
# but NOT a DSW instance stop. Checkpoints land in $CKPT_ROOT (workspace volume)
# every SAVE_FREQ steps so a stopped instance resumes rather than restarts.

set -uo pipefail

SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SIMOPD_ROOT"
# One venv, ./simopd, on every machine. SIMOPD_VENV overrides it.
VENV=${SIMOPD_VENV:-simopd}
source "$VENV/bin/activate"
export PYTHONPATH="$SIMOPD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

REHEARSAL=0
[ "${1:-}" = "--rehearsal" ] && { REHEARSAL=1; shift; }

LANES=${LANES:-4}
GPUS_PER_RUN=${GPUS_PER_RUN:-2}
STEPS=${STEPS:-300}
TEST_FREQ=${TEST_FREQ:-25}
SAVE_FREQ=${SAVE_FREQ:-50}
TAG=${TAG:-}
if [ "$REHEARSAL" = 1 ]; then
    STEPS=3; TEST_FREQ=-1; SAVE_FREQ=-1; TAG=rehearsal
    export PROJECT_NAME=${PROJECT_NAME:-simopd_rehearsal}
fi

# Default work list: vanilla x3 seeds first (seed 0 is the reference every verdict is
# measured against, and the spread across the three IS the noise floor that gives
# "|delta| < noise floor -> tie" an operational meaning), then one seed per arm.
# a2_coldstart is excluded: it needs the SFT checkpoint from slurm/coldstart.sbatch.
RUNS=${1:-}
if [ -z "$RUNS" ]; then
    if [ "$REHEARSAL" = 1 ]; then RUNS="vanilla:0"; else RUNS="vanilla:0 vanilla:1 vanilla:2"; fi
    for a in $(python scripts/arm.py list --status stock); do
        case "$a" in vanilla|a2_coldstart) continue ;; esac
        RUNS="$RUNS ${a}:0"
    done
fi

# One immutable snapshot shared by every lane; see the note in slurm/campaign.sbatch.
SNAP="${SNAP_ROOT:-$SIMOPD_ROOT/../simopd_data/snapshots}/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$SNAP"
cp -r scripts configs src "$SNAP"/
export SNAP
echo "running from snapshot: $SNAP ($(git rev-parse --short HEAD 2>/dev/null || echo nogit))"

read -r -a ALL <<< "$RUNS"
LOG_DIR=${LOG_DIR:-$SIMOPD_ROOT/logs}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== ${#ALL[@]} runs over $LANES lanes x ${GPUS_PER_RUN} GPU, ${STEPS} steps each ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for lane in $(seq 0 $((LANES - 1))); do
    # Round-robin so a lane that draws several slow arms does not stall the others.
    lane_runs=""
    for i in "${!ALL[@]}"; do
        [ $((i % LANES)) -eq "$lane" ] && lane_runs="$lane_runs ${ALL[$i]}"
    done
    [ -z "$lane_runs" ] && continue

    first=$((lane * GPUS_PER_RUN))
    devices=$(seq -s, "$first" $((first + GPUS_PER_RUN - 1)))
    log="$LOG_DIR/lane${lane}_${STAMP}.log"
    echo "  lane $lane  GPUs [$devices] ->$lane_runs"
    echo "            log: $log"

    CUDA_VISIBLE_DEVICES="$devices" \
    RAY_TMPDIR="${RAY_TMPDIR:-/tmp}/ray_lane${lane}" \
    LANE_RUNS="$lane_runs" LANE_STEPS="$STEPS" LANE_TEST_FREQ="$TEST_FREQ" \
    LANE_SAVE_FREQ="$SAVE_FREQ" LANE_TAG="$TAG" SNAP="$SNAP" \
    nohup bash deploy/dsw/_lane.sh > "$log" 2>&1 &
done

wait_msg="lanes launched; follow with:  tail -f $LOG_DIR/lane0_${STAMP}.log"
echo ""
echo "$wait_msg"
echo "summary when done:  grep -h '\-> ' $LOG_DIR/lane*_${STAMP}.log"
