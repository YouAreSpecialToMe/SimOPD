#!/usr/bin/env bash
# Run the SimOPD campaign across the 8 GPUs of a DSW instance as 4 concurrent lanes.
#
#   bash deploy/dsw/run_parallel.sh --rehearsal          # 3 steps/arm, ~30 min/lane
#   bash deploy/dsw/run_parallel.sh                      # the real campaign (STEPS default)
#   STEPS=250 bash deploy/dsw/run_parallel.sh "vanilla:0 vanilla:1 vanilla:2 f1_soft_log:0"
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

# Everything below used to be inherited from whatever shell launched this, which is
# a silent dependency on someone having sourced simopd_env.sh in THIS terminal. Under
# nohup, a campaign started from a fresh shell would run for days with the image's
# defaults -- straight back through ModelScope -- and nobody would know until the end.
# envtest.sh sets these itself, so a green envtest was never evidence about this path.
[ -f simopd_env.sh ] && source simopd_env.sh
# ONE project-level knob. The two package flags are DERIVED from it and set
# unconditionally -- never `${VAR:-False}`, which is a default and not an override, so
# a value the image exported survives it. That is precisely how ModelScope came back
# after being fixed twice: envtest.sh assigned False outright and passed, while this
# path defaulted and inherited the image's true.
#   SIMOPD_USE_MODELSCOPE=1  to genuinely run on ModelScope (fetch_assets follows too)
if [ "${SIMOPD_USE_MODELSCOPE:-0}" = "1" ]; then
    export VERL_USE_MODELSCOPE=True VLLM_USE_MODELSCOPE=True
else
    export VERL_USE_MODELSCOPE=False VLLM_USE_MODELSCOPE=False
fi
# verl's console logger block-buffers to a file; anything unflushed dies with the
# process, which is how a 24 GPU-hour run once produced no metrics at all.
export PYTHONUNBUFFERED=1

# wandb: default to offline unless there are real credentials. Lanes run under nohup
# with no tty, so an un-authenticated wandb.init() cannot prompt -- it fails, and it
# fails in every lane at once. Offline still records everything for a later
# `wandb sync`, and watch.py reads the console logger rather than wandb anyway.
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    export WANDB_MODE=${WANDB_MODE:-offline}
    echo "wandb: no credentials found -> WANDB_MODE=offline (sync later with: wandb sync $WANDB_DIR)"
fi

REHEARSAL=0
[ "${1:-}" = "--rehearsal" ] && { REHEARSAL=1; shift; }

LANES=${LANES:-4}
GPUS_PER_RUN=${GPUS_PER_RUN:-2}
STEPS=${STEPS:-150}   # see run_opd_baseline.sh: 300 was 19h/arm of Mode A
TEST_FREQ=${TEST_FREQ:-25}
SAVE_FREQ=${SAVE_FREQ:-50}
TAG=${TAG:-}

# Ray sizes its object store at DEFAULT_OBJECT_STORE_MEMORY_PROPORTION (0.3) of
# AVAILABLE host memory -- PER INSTANCE. Each lane starts its own Ray, each sees a
# machine with plenty free, and each claims 30%: four lanes ask for 120% of the box.
# Measured: 28.3GB apiece on a 94GB host. The store is lazily backed by /dev/shm, so
# nothing fails at startup -- it fails hours later, as a lane hanging with no
# traceback while its neighbours keep running, which is exactly what happened to
# lanes 1 and 3 at step 5 while 0 and 2 reached step 126.
# Divide by the lane count so all lanes together ask for what one would have.
if [ -z "${RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION:-}" ]; then
    export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=$(awk -v n="$LANES" 'BEGIN{printf "%.4f", 0.30/(n<1?1:n)}')
fi
echo "ray object store: ${RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION} of available RAM per lane ($LANES lanes)"
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

# ---------------------------------------------------------------------------
# Pre-flight. Every one of these has cost this project real time, and each is
# cheaper to check now than to discover on lane 3 of wave 4.
# ---------------------------------------------------------------------------
_fail=0
[ "${GPUS_PER_RUN:-0}" -ge 1 ] || { echo "FATAL: GPUS_PER_RUN must be >=1 (a run needs an actor GPU and a teacher GPU)" >&2; exit 1; }
[ "${LANES:-0}" -ge 1 ] || { echo "FATAL: LANES must be >=1" >&2; exit 1; }
_need_gpus=$((LANES * GPUS_PER_RUN))
_have_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "$_have_gpus" -lt "$_need_gpus" ]; then
    echo "FATAL: $LANES lanes x $GPUS_PER_RUN GPUs needs $_need_gpus, found $_have_gpus" >&2
    echo "       LANES=$((_have_gpus / GPUS_PER_RUN)) bash $0 ..." >&2
    _fail=1
fi
# A 1.7B checkpoint is ~25GB and MAX_CKPT_KEEP defaults to 2. Filling the workspace
# volume mid-campaign loses the runs that were writing at the time, not just the space.
_runs=$(set -- $RUNS; echo $#)
_need_gb=$(( _runs * ${MAX_CKPT_KEEP:-2} * 25 ))
_free_gb=$(df -BG --output=avail "${CKPT_ROOT:-.}" 2>/dev/null | tail -1 | tr -dc '0-9')
_free_gb=${_free_gb:-$(df -BG --output=avail . | tail -1 | tr -dc '0-9')}
if [ "${_free_gb:-0}" -lt "$_need_gb" ]; then
    echo "WARNING: $_runs runs x ${MAX_CKPT_KEEP:-2} checkpoints x ~25GB = ${_need_gb}G needed, ${_free_gb}G free" >&2
    echo "         MAX_CKPT_KEEP=1 bash $0 ...   # halves it" >&2
fi
for _v in VERL_USE_MODELSCOPE VLLM_USE_MODELSCOPE; do
    case "$(eval echo \"\${$_v}\" | tr '[:upper:]' '[:lower:]')" in
        true|1|yes) echo "note: $_v is ON -- models resolve through ModelScope, and" >&2
                    echo "      fetch_assets.py checks that cache too. Consistent, just be sure." >&2 ;;
    esac
done
# A Ray head from a crashed run is not inert: a new ray.init() attaches to it and
# inherits the environment IT started with, so everything set above is bypassed.
#
# But that is only true of a cluster under a tmpdir THIS launch will use. Lanes each
# get their own RAY_TMPDIR and never share a cluster, so a healthy lane running
# elsewhere is not a reason to refuse -- and the earlier version refused anyway,
# telling the operator to `ray stop --force`, which would have killed two runs at
# step 126 with their checkpoints on disk. A check that blocks the legitimate case and
# recommends destroying live work is worse than no check.
_busy=""
for _lane in $(seq 0 $((LANES - 1))); do
    _t="${RAY_TMPDIR:-/tmp}/ray_lane${RAY_TMPDIR_TAG:-}${_lane}"
    pgrep -f "${_t}(/|$)" >/dev/null 2>&1 && _busy="$_busy $_lane"
done
if [ -n "$_busy" ]; then
    echo "FATAL: lane(s)$_busy already have a live Ray cluster under" >&2
    echo "       ${RAY_TMPDIR:-/tmp}/ray_lane${RAY_TMPDIR_TAG:-}<n>; new lanes would attach to it" >&2
    echo "       and inherit its environment." >&2
    echo "       bash deploy/dsw/sweep_lane.sh$_busy    # reclaims those only" >&2
    echo "       (NOT ray stop --force -- that is machine-wide and kills healthy lanes)" >&2
    _fail=1
elif pgrep -f "raylet|gcs_server" >/dev/null 2>&1; then
    echo "note: other Ray cluster(s) are running -- expected if lanes are already going." >&2
    echo "      They use different temp dirs, so these lanes will not attach to them." >&2
fi
python scripts/fetch_assets.py --check --essential >/dev/null 2>&1 || {
    echo "FATAL: student/teacher assets missing or corrupt." >&2
    echo "       python scripts/fetch_assets.py --check --essential   # what is wrong" >&2
    echo "       python scripts/fetch_assets.py --repair              # refetch" >&2
    _fail=1
}
python scripts/arm.py check >/dev/null 2>&1 || {
    echo "FATAL: the arm registry does not load; every lane would fail identically." >&2
    echo "       PYTHONPATH=$SIMOPD_ROOT/src python scripts/arm.py check" >&2
    _fail=1
}
[ "$_fail" = 0 ] || { echo "pre-flight failed; nothing launched." >&2; exit 1; }
echo "pre-flight ok: ${_have_gpus} GPUs, ${_free_gb}G free (need ~${_need_gb}G), assets present, no stale Ray"

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

    # Contiguous by default. GPU_LIST overrides with one space-separated group per
    # lane -- needed when a half-dead campaign frees non-adjacent GPUs and the healthy
    # lanes must not be disturbed:  GPU_LIST="2,3 6,7"
    if [ -n "${GPU_LIST:-}" ]; then
        devices=$(set -- $GPU_LIST; eval echo "\${$((lane + 1))}")
        [ -n "$devices" ] || { echo "GPU_LIST has no entry for lane $lane" >&2; continue; }
    else
        first=$((lane * GPUS_PER_RUN))
        devices=$(seq -s, "$first" $((first + GPUS_PER_RUN - 1)))
    fi
    log="$LOG_DIR/lane${lane}_${STAMP}.log"
    echo "  lane $lane  GPUs [$devices] ->$lane_runs"
    echo "            log: $log"

    CUDA_VISIBLE_DEVICES="$devices" \
    RAY_TMPDIR="${RAY_TMPDIR:-/tmp}/ray_lane${RAY_TMPDIR_TAG:-}${lane}" \
    LANE_RUNS="$lane_runs" LANE_STEPS="$STEPS" LANE_TEST_FREQ="$TEST_FREQ" \
    LANE_SAVE_FREQ="$SAVE_FREQ" LANE_TAG="$TAG" SNAP="$SNAP" \
    nohup bash deploy/dsw/_lane.sh > "$log" 2>&1 &
done

wait_msg="lanes launched; follow with:  tail -f $LOG_DIR/lane0_${STAMP}.log"
echo ""
echo "$wait_msg"
echo "summary when done:  grep -h '\-> ' $LOG_DIR/lane*_${STAMP}.log"
