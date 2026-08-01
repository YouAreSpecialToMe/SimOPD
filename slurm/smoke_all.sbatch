#!/usr/bin/env bash
#SBATCH -J simopd-smokeall
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=150G
#SBATCH -t 06:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

# Smoke every implemented arm back to back in ONE 2xA100 allocation.
# The account cap is gpu=2, so sequential-in-one-job beats N queued jobs: one
# allocation, no re-queueing between arms.
#
# Deliberately uses the REAL protocol batch/length (128 / 8192) and the real
# 80G memory utilisations -- the point is to prove each arm runs under the exact
# config the screening runs will use, not merely that its loss code executes.
# Only the step count is cut (3 steps, no validation).
#
#   sbatch slurm/smoke_all.sbatch                 # every stock arm
#   sbatch --export=ALL,ARMS="vanilla f1_soft_log" slurm/smoke_all.sbatch

set -uo pipefail

PROJ_DIR=$HOME/pythonProject/SimOPD
cd "$PROJ_DIR"
source .venv/bin/activate

export RAY_TMPDIR=/scratch/zz865/ray_tmp
export CKPT_ROOT=/scratch/zz865/simopd/ckpt_smoke
export WANDB_DIR=/scratch/zz865/simopd/wandb
mkdir -p "$RAY_TMPDIR" "$CKPT_ROOT" "$WANDB_DIR"

# our arm losses register into verl's registry via sitecustomize on PYTHONPATH
export PYTHONPATH="$PROJ_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

ARMS=${ARMS:-$(python scripts/arm.py list --status stock | tr '\n' ' ')}
echo "=== smoking arms: $ARMS ==="
nvidia-smi -L

declare -A RESULT
for ARM in $ARMS; do
    echo ""
    echo "################ ARM: $ARM ################"
    (
        set -e
        eval "$(python scripts/arm.py env "$ARM")"
        export EXPERIMENT_NAME="smoke_${ARM}"
        export PROJECT_NAME=simopd_smoke
        export TOTAL_TRAINING_STEPS=3
        export TEST_FREQ=-1
        export SAVE_FREQ=-1
        env | grep -E "^(DISTILLATION_|LOSS_MAX|LOG_PROB_MIN|USE_POLICY)" | sort || true
        bash scripts/run_opd_baseline.sh trainer.val_before_train=False
    )
    if [ $? -eq 0 ]; then RESULT[$ARM]=OK; else RESULT[$ARM]=FAIL; fi
    echo "################ $ARM -> ${RESULT[$ARM]} ################"
    # Ray leaves the cluster up between arms otherwise, and the next arm's
    # resource-pool creation would collide with the previous allocation.
    ray stop --force >/dev/null 2>&1 || true
    sleep 10
done

echo ""
echo "================ SMOKE SUMMARY ================"
fails=0
for ARM in $ARMS; do
    printf '%-28s %s\n' "$ARM" "${RESULT[$ARM]}"
    [ "${RESULT[$ARM]}" = "FAIL" ] && fails=$((fails + 1))
done
echo "=============================================="
echo "SMOKE_ALL_DONE fails=$fails"
exit 0
