#!/usr/bin/env bash
# Submit one SimOPD arm to Alibaba PAI-DLC.
# Same env-var interface as slurm/baseline.sbatch, so an arm is defined once and
# runs identically on either cluster (protocol parity is an audit requirement).
#
#   EXPERIMENT_NAME=vanilla_s0 ./deploy/pai/submit_dlc.sh
#   EXPERIMENT_NAME=lsm_topk32 DISTILLATION_LOSS_MODE=forward_kl_topk ./deploy/pai/submit_dlc.sh
#
# Fill the ACCOUNT SETTINGS block once (values come from the PAI console).

set -euo pipefail

# ---------------- ACCOUNT SETTINGS (fill these in) ----------------
WORKSPACE_ID=${WORKSPACE_ID:?set WORKSPACE_ID (PAI console > 工作空间)}
RESOURCE_ID=${RESOURCE_ID:?set RESOURCE_ID (专有资源组 quota id, e.g. quotaxxxxxxxx)}
IMAGE=${IMAGE:?set IMAGE (e.g. registry-vpc.cn-hangzhou.aliyuncs.com/<ns>/simopd:v1)}
# Dataset ids: NAS holding models + data + checkpoints. Comma-separated, mounted per
# the mount path configured on each dataset (default /mnt/data).
DATA_SOURCES=${DATA_SOURCES:?set DATA_SOURCES (e.g. d-xxxxxxxxxxxx)}
NAS_ROOT=${NAS_ROOT:-/mnt/data}
# ------------------------------------------------------------------

EXPERIMENT_NAME=${EXPERIMENT_NAME:?set EXPERIMENT_NAME (= arm name; also the wandb run name)}
WORKER_GPU=${WORKER_GPU:-2}          # 1 actor + 1 teacher, our standard slot
WORKER_CPU=${WORKER_CPU:-16}
WORKER_MEMORY=${WORKER_MEMORY:-200Gi}
PRIORITY=${PRIORITY:-5}
# 0 = no limit. Mode A length inflation made a 300-step screen exceed 24h locally,
# so never set this below ~48h for a full screening run.
MAX_MINUTES=${MAX_MINUTES:-0}

# Protocol knobs forwarded to run_opd_baseline.sh (defaults live in that script)
FORWARD_ENVS="EXPERIMENT_NAME=${EXPERIMENT_NAME}"
for v in STUDENT_MODEL TEACHER_MODEL DISTILLATION_LOSS_MODE USE_POLICY_GRADIENT \
         TRAIN_BATCH_SIZE MAX_RESPONSE_LENGTH ACTOR_LR TOTAL_TRAINING_STEPS \
         TEST_FREQ SAVE_FREQ PROJECT_NAME; do
    [ -n "${!v:-}" ] && FORWARD_ENVS="${FORWARD_ENVS},${v}=${!v}"
done

# Everything stateful lives on NAS so a preempted job can resume.
RUN_CMD="export HF_HOME=${NAS_ROOT}/hf_cache HF_HUB_OFFLINE=1; \
export WANDB_DIR=${NAS_ROOT}/wandb; \
export RAY_TMPDIR=/root/ray_tmp; mkdir -p \$RAY_TMPDIR ${NAS_ROOT}/wandb; \
export DATA_DIR=${NAS_ROOT}/simopd_math; \
export CKPT_ROOT=${NAS_ROOT}/ckpt; \
cd /opt/simopd && bash scripts/run_opd_baseline.sh"

set -x
dlc submit pytorchjob \
    --name="simopd-${EXPERIMENT_NAME}" \
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
    --envs="${FORWARD_ENVS}" \
    --command="${RUN_CMD}"
