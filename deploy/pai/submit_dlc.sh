#!/usr/bin/env bash
# Submit one slice of the SimOPD campaign to Alibaba PAI-DLC.
#
# Same run-list interface as slurm/campaign.sbatch, and both drive the same
# scripts/run_opd_baseline.sh, so an arm is defined once and behaves identically
# on either cluster -- protocol parity is an audit requirement, not a nicety.
#
# Parallelism = one job per 2-GPU slot with a disjoint RUNS slice. That is the
# whole reason for going to PAI: locally the account is capped at gpu=2, so the
# 15 screening arms can only run one at a time.
#
#   export WORKSPACE_ID=... RESOURCE_ID=... IMAGE=... DATA_SOURCES=...
#   ./deploy/pai/submit_dlc.sh "vanilla:0 vanilla:1 vanilla:2"
#   ./deploy/pai/submit_dlc.sh "d1_tip:0 d2_selectkd:0 d3_teachability:0"
#   STEPS=3 ./deploy/pai/submit_dlc.sh "vanilla:0"        # rehearsal first

set -euo pipefail

# ---------------- ACCOUNT SETTINGS (from the PAI console) ----------------
WORKSPACE_ID=${WORKSPACE_ID:?set WORKSPACE_ID}
RESOURCE_ID=${RESOURCE_ID:?set RESOURCE_ID (专有资源组 quota id)}
IMAGE=${IMAGE:?set IMAGE (e.g. registry-vpc.cn-hangzhou.aliyuncs.com/<ns>/simopd:v1)}
DATA_SOURCES=${DATA_SOURCES:?set DATA_SOURCES (NAS dataset id, e.g. d-xxxxxxxx)}
NAS_ROOT=${NAS_ROOT:-/mnt/data}
# -------------------------------------------------------------------------

RUNS=${1:-${RUNS:-}}
[ -n "$RUNS" ] || { echo "usage: $0 \"arm:seed arm:seed ...\"  (see: python scripts/arm.py list)" >&2; exit 1; }

JOB_NAME=${JOB_NAME:-simopd-$(echo "$RUNS" | tr ' :' '--' | cut -c1-40)}
WORKER_GPU=${WORKER_GPU:-2}      # 1 actor + 1 teacher; verl's teacher pool cannot share the actor's GPU
WORKER_CPU=${WORKER_CPU:-16}
WORKER_MEMORY=${WORKER_MEMORY:-200Gi}
PRIORITY=${PRIORITY:-5}
# 0 = unlimited. Mode A length inflation pushed a 300-step run past 24h locally,
# and a run killed mid-flight leaves nothing behind, so do not cap this tightly.
MAX_MINUTES=${MAX_MINUTES:-0}

STEPS=${STEPS:-300}
TEST_FREQ=${TEST_FREQ:-25}
SAVE_FREQ=${SAVE_FREQ:-50}   # resumable: a preempted run with no checkpoint already cost this project 24 GPU-hours

# Everything stateful lives on NAS so a preempted job resumes instead of restarting.
RUN_CMD=$(cat <<EOF
set -uo pipefail
export HF_HOME=${NAS_ROOT}/hf_cache HF_HUB_OFFLINE=1
export DATA_DIR=${NAS_ROOT}/simopd_math
export CKPT_ROOT=${NAS_ROOT}/ckpt
export WANDB_DIR=${NAS_ROOT}/wandb
export RAY_TMPDIR=/root/ray_tmp
mkdir -p \\\$RAY_TMPDIR ${NAS_ROOT}/wandb ${NAS_ROOT}/ckpt
cd /opt/simopd
nvidia-smi -L
for entry in ${RUNS}; do
  ARM=\\\${entry%%:*}; SEED=\\\${entry##*:}
  echo "################ RUN: \\\${ARM}_s\\\${SEED} ################"
  (
    set -e
    eval "\\\$(python scripts/arm.py env \\\$ARM)"
    export EXPERIMENT_NAME="\\\${ARM}_s\\\${SEED}"
    export TOTAL_TRAINING_STEPS=${STEPS} TEST_FREQ=${TEST_FREQ} SAVE_FREQ=${SAVE_FREQ}
    bash scripts/run_opd_baseline.sh data.seed=\\\$SEED actor_rollout_ref.rollout.seed=\\\$SEED
  )
  echo "################ \\\${ARM}_s\\\${SEED} -> exit \\\$? ################"
  ray stop --force >/dev/null 2>&1 || true
done
echo CAMPAIGN_DONE
EOF
)

set -x
dlc submit pytorchjob \
    --name="${JOB_NAME}" \
    --workers=1 \
    --worker_gpu="${WORKER_GPU}" \
    --worker_cpu="${WORKER_CPU}" \
    --worker_memory="${WORKER_MEMORY}" \
    --worker_image="${IMAGE}" \
    --data_sources="${DATA_SOURCES}" \
    --workspace_id="${WORKSPACE_ID}" \
    --resource_id="${RESOURCE_ID}" \
    --priority="${PRIORITY}" \
    --job_max_running_time_minutes="${MAX_MINUTES}" \
    --command="bash -lc '${RUN_CMD}'"
