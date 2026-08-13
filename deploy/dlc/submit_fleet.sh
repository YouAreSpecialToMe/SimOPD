#!/usr/bin/env bash
# Submit the SimOPD fleet to PAI-DLC as ONE elastic pytorchjob: N workers x 8 GPU,
# every worker running deploy/dlc/worker.sh's supervisor loop. Written 2026-08-13
# for the ~500-card allocation; the exact total is not known until the quota
# lands, so sizing is derived, not assumed (see TOTAL_GPUS below).
#
# Why one job and not one-job-per-run (the older deploy/pai/submit_dlc.sh shape):
# at 512 cards the campaign is ~40-250 independent rows PLUS an eval backlog that
# must absorb every idle card. Per-run jobs would re-implement, badly, what the
# manifest pool + mkdir claims already do across nodes on /mgfs -- and they cannot
# flip a finished run's cards to eval. One job of identical supervisors can:
# training drains -> cards become eval workers within a pass; new waves land in
# campaign.tsv -> workers evict eval and claim them. Utilization is a property of
# the loop, not of the submission.
#
# This cluster's DLC pods mount /mgfs directly (tools/dlc/exp*.sh use /mgfs paths
# throughout), so there is NO data staging and NO image baking: the venv, both
# code trees, datasets and checkpoints are the same files DSW used. The image
# only needs CUDA userspace + python; the venv on /mgfs supplies the rest.
#
#   export WORKSPACE_ID=... RESOURCE_ID=... IMAGE=...
#   bash deploy/dlc/submit_fleet.sh 500             # card count as the argument -> 62 workers = 496
#   TOTAL_GPUS=512 bash deploy/dlc/submit_fleet.sh  # same thing as env; default is 500
#   WORKERS=8 bash deploy/dlc/submit_fleet.sh       # smoke at 64 GPUs first (explicit count wins)
#
# Governance notes, so the submission stays inside the campaign's rules:
#   * workers run the EXP tree (BATCH_TAG=16k, BATCH_MIN_WAVE=9) -- they cannot
#     touch waves 1-8 even if the manifest still lists them;
#   * machine identity is rank-keyed (d0..dN-1) and survives pod restarts; the
#     MACHINE_MAP upsert lives in worker.sh;
#   * preemption is priced in: SAVE_FREQ=25 banks mean a preempted row loses
#     <=24 steps, and the fingerprint decides resume-vs-refuse, same as DSW.
set -euo pipefail

# Foolproofing, from the wild (job dlc1mk5l3etxjzed, 2026-08-14): this
# SUBMITTER was pasted as the DLC job command -- all 15 pods printed the
# console card and exited 0 within 4 seconds. Inside a DLC container (rank/job
# env present) the only sensible meaning of running this script is "run the
# payload": exec worker.sh, EVAL_ONLY passes through, the card-count argument
# is meaningless there and dropped.
if [ -n "${MLP_ROLE_INDEX:-}${MLP_WORKER_RACK_RANK_INDEX:-}${DLC_JOB_ID:-}" ]; then
    echo "detected DLC container (rank/job env present): this script is the SUBMITTER,"
    echo "exec'ing the payload instead: worker.sh (EVAL_ONLY=${EVAL_ONLY:-0})"
    exec bash /mgfs/shared/Group_GY/changhao/SimOPD-exp/deploy/dlc/worker.sh
fi

# Probed facts (tools/pai/FINDINGS.md, 2026-07-30, account changhao.li):
#   * workspace is ws1741o20vr72qfb -- defaulted below, override if the quota
#     lands elsewhere;
#   * the dlc CLI must be configured with endpoint
#     dlc-gateway.gy.pai-eflops.aliyuncs.com -- the console-advertised
#     pai-proxy endpoint answers "InnerUrl invalid" to everything;
#   * credentials live pod-local (/root/.dlc/config, 600) ON PURPOSE: the
#     shared 448T mount is group-readable, keys never go there.
# RESOURCE_ID (the quota id of the ~500-card allocation) and IMAGE could not
# be discovered from this account (empty listings) -- they come with the quota
# grant / from a colleague's successful job in the console.
WORKSPACE_ID=${WORKSPACE_ID:-ws1741o20vr72qfb}
# RESOURCE_ID is only needed for CLI direct-submit; the console path (the
# default, and how the colleagues submit) picks the quota from the form's
# dropdown -- no id string required.
RESOURCE_ID=${RESOURCE_ID:-}
# The exact image the proven DSW pods run (read from a live pod's PID1 env,
# DOCKER_IMAGE_URL, 2026-08-13): ubuntu22.04 + cuda12.4 + py311 + torch2.6.0.
# Every campaign run to date executed inside this environment, so the fleet
# defaults to it; override only if the grant forces a different registry.
IMAGE=${IMAGE:-dashscope-edge-registry-vpc.ap-southeast-5.cr.aliyuncs.com/gy-pai/modelscope-ubuntu22.04-cuda12.4.0-py311-torch2.6.0:gpu}

# Sizing: workers are whole 8-GPU pods, so the fleet is WORKERS = TOTAL_GPUS/8
# rounded DOWN -- asking for more than the quota holds queues forever, stranding
# a remainder (500 -> 62 workers = 496, 4 unused) merely wastes the remainder.
# An explicitly set WORKERS always wins (smoke runs, odd quotas).
# card count: positional arg beats env beats the 500 default
TOTAL_GPUS=${1:-${TOTAL_GPUS:-500}}
case "$TOTAL_GPUS" in (''|*[!0-9]*)
    echo "FATAL: card count must be a number, got '$TOTAL_GPUS'" >&2; exit 1 ;;
esac
WORKER_GPU=${WORKER_GPU:-8}
if [ -z "${WORKERS:-}" ]; then
    WORKERS=$(( TOTAL_GPUS / WORKER_GPU ))
    SIZING="target TOTAL_GPUS=$TOTAL_GPUS, remainder $((TOTAL_GPUS - WORKERS * WORKER_GPU)) unused"
else
    SIZING="WORKERS set explicitly"
fi
if [ "$WORKERS" -lt 1 ]; then
    echo "FATAL: TOTAL_GPUS=$TOTAL_GPUS < one $WORKER_GPU-GPU worker" >&2; exit 1
fi
echo "fleet sizing: $WORKERS workers x $WORKER_GPU GPU = $((WORKERS * WORKER_GPU)) GPUs ($SIZING)"
WORKER_CPU=${WORKER_CPU:-96}
WORKER_MEMORY=${WORKER_MEMORY:-800Gi}
PRIORITY=${PRIORITY:-5}
# EVAL_ONLY=1 submits the same worker in pure eval-drainer mode (no training,
# no eviction, no identity registration) -- the "run every checkpoint through
# the offline suite first" job:  EVAL_ONLY=1 bash deploy/dlc/submit_fleet.sh 200
[ "${EVAL_ONLY:-0}" = 1 ] && _kind=eval || _kind=fleet
JOB_NAME=${JOB_NAME:-simopd-$_kind-$(date +%m%d%H%M)}
DLC=${DLC:-/mgfs/shared/Group_GY/changhao/tools/pai/bin/dlc}
# The whole design assumes /mgfs is visible inside the job. On this cluster the
# existing DLC jobs get it via their dataset attachment -- if your quota needs an
# explicit --data_sources id for the mgfs dataset, set DATA_SOURCES; if the
# workspace mounts it implicitly, leave unset. VERIFY ON THE SMOKE JOB: worker
# logs print the tree path at boot, and a job without /mgfs dies in seconds.
EXTRA_ARGS=()
[ -n "${DATA_SOURCES:-}" ] && EXTRA_ARGS+=(--data_sources="$DATA_SOURCES")

# The worker script is read from /mgfs at boot, so iterating on it needs no
# resubmission for RESTARTED pods -- but live pods only re-read it on their next
# incarnation. Whole-fleet behavior changes still deserve a fresh job.
CMD="bash /mgfs/shared/Group_GY/changhao/SimOPD-exp/deploy/dlc/worker.sh"
[ "${EVAL_ONLY:-0}" = 1 ] && CMD="EVAL_ONLY=1 $CMD"

# Two ways to hand this to DLC, same job either way. The colleagues' pattern
# is console + a payload script on /mgfs -- worker.sh IS our payload, so the
# DEFAULT here is the console card: every form value computed and printed for
# copy-paste, quota picked from the form's dropdown. CLI direct-submit only
# runs when it actually can (config present + RESOURCE_ID given).
if [ -e "$HOME/.dlc/config" ] && [ -n "$RESOURCE_ID" ] && [ "${CONSOLE:-0}" != 1 ]; then
    set -x
    "$DLC" submit pytorchjob \
        --name="$JOB_NAME" \
        --workers="$WORKERS" \
        --worker_gpu="$WORKER_GPU" \
        --worker_cpu="$WORKER_CPU" \
        --worker_memory="$WORKER_MEMORY" \
        --worker_image="$IMAGE" \
        --workspace_id="$WORKSPACE_ID" \
        --resource_id="$RESOURCE_ID" \
        --priority="$PRIORITY" \
        --job_max_running_time_minutes=0 \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        --command="$CMD"
else
    cat <<CARD
== DLC 控制台提交单(照抄进网页表单)=====================================
  任务名称        $JOB_NAME
  任务类型        PyTorch 训练 (PyTorchJob)
  节点数量        $WORKERS
  单节点资源      GPU $WORKER_GPU / CPU $WORKER_CPU / 内存 $WORKER_MEMORY
  节点镜像        $IMAGE
  工作空间        $WORKSPACE_ID
  资源配额        ${RESOURCE_ID:-(表单下拉里选新批的那份配额)}
  最长运行时长    不限
  数据集/挂载     确认 /mgfs 在容器内可见(工作空间隐式挂载则无需额外配置)
  启动命令        $CMD
==========================================================================
(可选 CLI 直提: 配好 ~/.dlc/config 且传 RESOURCE_ID=<配额id> 后重跑本脚本)
CARD
fi
