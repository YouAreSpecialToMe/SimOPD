#!/usr/bin/env bash
# One arm, three steps, no lanes -- then hand the log to triage.
#
#   bash deploy/dsw/envtest.sh
#
# This is the step to run FIRST on a new machine, and the one that was skipped.
# `run_parallel.sh --rehearsal` ignites 14 arms across 4 lanes at once: on a fresh
# environment every problem surfaces four times over, interleaved, and the real
# cause is somewhere in ~19k lines. Here one run's output goes to one file, and
# triage.py reads it whatever happens.
#
# Each level below adds exactly one variable, so a failure names its own cause:
#   1. this script          -- does verl + vLLM + the teacher pool work at all
#   2. LANES=1 ... --rehearsal  -- does the lane wrapper work
#   3. run_parallel.sh --rehearsal  -- do 4 lanes coexist
set -uo pipefail

SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SIMOPD_ROOT"
[ -f simopd_env.sh ] && source simopd_env.sh
[ -d simopd ] && [ -z "${VIRTUAL_ENV:-}" ] && source simopd/bin/activate

LOG=${ENVTEST_LOG:-/tmp/simopd_envtest_$(date +%H%M%S).log}
STEPS=${STEPS:-3}

# The image exports this and verl acts on it at import: modelscope's patch_hub()
# reroutes every huggingface_hub call, and ModelScope's default branch is 'master',
# so asking for HF's 'main' fails on models already on disk.
export VERL_USE_MODELSCOPE=False
export VLLM_USE_MODELSCOPE=False   # vLLM has its own; setting one is not enough
export PYTHONUNBUFFERED=1
# wandb: default to offline unless there are real credentials. Lanes run under nohup
# with no tty, so an un-authenticated wandb.init() cannot prompt -- it fails, and it
# fails in every lane at once. Offline still records everything for a later
# `wandb sync`, and watch.py reads the console logger rather than wandb anyway.
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    export WANDB_MODE=${WANDB_MODE:-offline}
    echo "wandb: no credentials found -> WANDB_MODE=offline (sync later with: wandb sync $WANDB_DIR)"
fi

echo "=== pre-flight ==="
python scripts/fetch_assets.py --check 2>&1 | tail -5 || {
    echo "assets incomplete -- fix that first:" >&2
    echo "  python scripts/fetch_assets.py --repair" >&2
    exit 1
}
SHM=$(df -m /dev/shm 2>/dev/null | tail -1 | awk '{print $2}')
[ "${SHM:-0}" -lt 8192 ] && echo "WARNING: /dev/shm is ${SHM}M; vLLM workers die at startup below ~8G" >&2

# A Ray head left by a crashed run is not inert: the next ray.init() attaches to it,
# and its workers carry the environment from when IT started -- so the fix above
# would not reach them. Safe here because this script is the single-lane path.
if pgrep -f "raylet|gcs_server" >/dev/null 2>&1; then
    echo "stopping a leftover Ray cluster (single-lane debugging only)"
    ray stop --force >/dev/null 2>&1 || true
    sleep 3
fi

echo
echo "=== $STEPS steps, output -> $LOG ==="
TOTAL_TRAINING_STEPS=$STEPS TEST_FREQ=-1 SAVE_FREQ=-1 \
EXPERIMENT_NAME=envtest LOGGER='["console"]' \
    bash scripts/run_opd_baseline.sh > "$LOG" 2>&1
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "=== PASS (rc=0) ==="
    python scripts/triage.py "$LOG" 2>/dev/null | head -6
    echo
    echo "next:  LANES=1 bash deploy/dsw/run_parallel.sh --rehearsal \"vanilla:0\""
else
    echo "=== FAIL (rc=$rc) -- triage below; full log at $LOG ==="
    echo
    python scripts/triage.py "$LOG" --ray
fi
exit $rc
