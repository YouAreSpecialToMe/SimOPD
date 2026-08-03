#!/usr/bin/env bash
# The cross-domain transfer column for one arm's final checkpoint.
#
#   bash scripts/eval_transfer.sh vanilla_s0            # latest checkpoint
#   bash scripts/eval_transfer.sh vanilla_s0 300        # a specific step
#   bash scripts/eval_transfer.sh --all                 # every run that has one
#
# math-trained model, evaluated on code/IF: a SIDE-EFFECT measurement (METRICS §2,
# "域间迁移损益"), reported as Delta vs vanilla. It does not select -- selection stays
# in math (plan §4) -- and it says nothing about whether a trick works when distilling
# IN those domains. 1083 prompts greedy, about 0.25 GPU-hours.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -d simopd ] && source simopd/bin/activate

CKPT_ROOT=${CKPT_ROOT:-/scratch/zz865/simopd/ckpt}
PROJECT=${PROJECT_NAME:-simopd}
OUT_DIR=${EVAL_OUT_DIR:-/scratch/zz865/simopd/evals}
BENCHES=${BENCHES:-humanevalplus,mbppplus,ifeval}
# Long enough for IFEval's "at least 800 words" prompts; truncation rate is printed
# per benchmark, so if this ever binds it is visible rather than silent.
MAX_TOKENS=${MAX_TOKENS:-2048}
# evalplus forks a worker per test. Leave the training lanes their cores -- and see
# the load-sensitivity note in transfer_eval.py for what happens when you do not.
PARALLEL=${PARALLEL:-8}

run_one() {   # run_one <run_id> [step]
    local run_id=$1 step=${2:-} dir
    local base="$CKPT_ROOT/$PROJECT/$run_id"
    if [ -n "$step" ]; then
        dir="$base/global_step_$step/actor"
    else
        dir=$(ls -d "$base"/global_step_*/actor 2>/dev/null |
              sort -t_ -k3 -n | tail -1)
        step=$(basename "$(dirname "$dir")" | sed 's/global_step_//')
    fi
    if [ ! -d "$dir" ]; then
        echo "  SKIP $run_id: no checkpoint under $base" >&2
        return 1
    fi
    echo "=== $run_id @ step $step ==="
    python scripts/eval_offline.py \
        --model "$dir" --benchmarks "$BENCHES" \
        --run-id "$run_id" --step "$step" \
        --temperature 0 --n 1 --max-tokens "$MAX_TOKENS" \
        --parallel "$PARALLEL" --out-dir "$OUT_DIR"
}

if [ "${1:-}" = "--all" ]; then
    rc=0
    for base in "$CKPT_ROOT/$PROJECT"/*/; do
        [ -d "$base" ] || continue
        run_one "$(basename "$base")" || rc=1
    done
    exit $rc
fi

[ $# -ge 1 ] || { sed -n '2,9p' "$0"; exit 2; }
run_one "$@"
