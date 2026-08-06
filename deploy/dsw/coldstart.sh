#!/usr/bin/env bash
# Arm a2 stages 1-2 on a DSW box: teacher generations -> SFT checkpoint.
#
#   bash deploy/dsw/coldstart.sh                 # both stages, GPUs 0,1
#   STAGE=1 bash deploy/dsw/coldstart.sh         # generations only
#   STAGE=2 bash deploy/dsw/coldstart.sh         # SFT only (data must exist)
#
# Stage 3 is an ordinary OPD run pointed at the checkpoint this writes, wired up
# through arms.yaml's STUDENT_MODEL/DATA_DIR for a2_coldstart -- which is a
# run-defining file, so re-pointing it is a pin move with a REASON, done only once
# the checkpoint actually exists.
#
# The slurm sibling (slurm/coldstart.sbatch) cannot be used here: its paths are
# Cornell's, and more importantly its model defaults are the ABANDONED 0.6B tier
# (STUDENT=Qwen3-0.6B-Base, TEACHER=Qwen3-1.7B; gen_coldstart_data.py's --teacher
# default is 1.7B too). Running it as-is would silently produce a warm start at a
# tier the protocol no longer uses -- no error, just an arm that is not comparable
# to the other fifteen. So the tier is pinned here, from PROTOCOL-unified §1, and
# echoed before anything runs.
set -euo pipefail

SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SIMOPD_ROOT"
source simopd_env.sh

# The locked screening tier. NOT the sbatch's defaults.
TEACHER=${TEACHER:-Qwen/Qwen3-4B-Instruct-2507}
STUDENT=${STUDENT:-Qwen/Qwen3-1.7B-Base}
SFT_CKPT=${SFT_CKPT:-$CKPT_ROOT/coldstart_sft}
GPUS=${GPUS:-0,1}
STAGE=${STAGE:-all}

# Same 8192 response cap as the screening protocol: a warm start built at a
# different context budget would confound the arm with a length effect.
MAX_TOKENS=${MAX_TOKENS:-8192}
N_PROMPTS=${N_PROMPTS:-3000}
N_SAMPLES=${N_SAMPLES:-4}

export VLLM_LOGGING_LEVEL=WARNING
export PYTHONUNBUFFERED=1
# vLLM forks its EngineCore by default. gen_coldstart_data.py imports
# verl.utils.reward_score (for the verifier) before constructing the LLM, and that
# leaves a CUDA context in the parent, so the fork dies with
#   RuntimeError: Cannot re-initialize CUDA in forked subprocess
# The campaign never hits this because verl sets the method itself -- a script that
# drives vLLM directly has to.
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_coldstart}
mkdir -p "$RAY_TMPDIR" "$(dirname "$SFT_CKPT")"

echo "=== arm a2 cold start on $(hostname) ==="
echo "  teacher   $TEACHER"
echo "  student   $STUDENT"
echo "  data      $DATA_DIR"
echo "  ckpt      $SFT_CKPT"
echo "  reserved  $N_PROMPTS prompts x $N_SAMPLES samples, cap $MAX_TOKENS"
echo "  GPUs      $GPUS"

if [ "$STAGE" = all ] || [ "$STAGE" = 1 ]; then
    echo
    echo "=== stage 1: teacher generations on the reserved prompt slice ==="
    # One GPU: this is pure vLLM inference. Stage 2 wants two.
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python scripts/gen_coldstart_data.py \
        --teacher "$TEACHER" \
        --train-parquet "$DATA_DIR/train.parquet" \
        --out-dir "$DATA_DIR" \
        --n-prompts "$N_PROMPTS" \
        --n-samples "$N_SAMPLES" \
        --max-tokens "$MAX_TOKENS"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = 2 ]; then
    echo
    echo "=== stage 2: SFT the student on them ==="
    [ -f "$DATA_DIR/coldstart_sft.parquet" ] || {
        echo "FATAL: $DATA_DIR/coldstart_sft.parquet missing; run STAGE=1 first." >&2; exit 1; }
    _n=$(( $(echo "$GPUS" | tr ',' '\n' | wc -l) ))
    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nnodes=1 --nproc_per_node="$_n" \
        -m verl.trainer.sft_trainer \
        data.train_files="$DATA_DIR/coldstart_sft.parquet" \
        data.messages_key=messages \
        data.max_length="${SFT_MAX_LEN:-8192}" \
        data.train_batch_size="${SFT_BATCH:-128}" \
        model.path="$STUDENT" \
        trainer.default_local_dir="$SFT_CKPT" \
        trainer.total_epochs="${SFT_EPOCHS:-2}" \
        "$@"
    echo "SFT checkpoint: $SFT_CKPT"
fi

echo "COLDSTART_DONE"
