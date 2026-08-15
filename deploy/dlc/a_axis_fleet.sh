#!/usr/bin/env bash
# A 轴 4-lane DLC 任务:一个 8 卡 worker,自门控三阶段,幂等可重入。
#
#   预计算 P  gkd_offpolicy.parquet 缺失 → 8 卡分片生成 + 合并(~1h;
#             per-request seed ⇒ 重跑逐 token 一致;分片各自幂等跳过)
#   彩排  R  每臂 3 步 rehearse_a_axis.sh,机器判据自动裁定(判据见该脚本头);
#             任一臂 FAIL → 挂起待查,绝不带病发射
#   发射  L  a1/a3/a4/a5 × s0,每臂 2 卡一条 lane,250 步,ckpt 落
#             $D/ckpt/simopd/{arm}_s0_16k → post-eval 流水线的 *_16k glob 自动接管
#
# 为什么走 DLC:DSW pod 说没就没(2026-08-15 gpu141 死于预计算 20 分钟处),
# DLC 任务被回收会重启 → 本脚本重入:P 见 parquet 即跳,R 见 .OK 标记即跳,
# L 靠 run_opd_baseline 的 checkpoint 自恢复(SAVE_FREQ=25,最多亏 24 步)。
#
# 提交(与 SimOPD-exp/deploy/dlc/submit_fleet.sh 同款双模):
#   本机运行 → 打印控制台表单卡片,照抄进 DLC console(1 worker × 8 GPU);
#   DLC 容器内(存在 rank/job env)→ 本脚本即载荷,直接执行三阶段。
#
# 治理:不走 campaign.tsv(4 个 supplement 臂,单 seed 烟测规模),与
# deploy/pai/submit_dlc.sh 同一血统 -- arm.py env 赋值后 eval + 同款 seed
# 传参驱动同一 run_opd_baseline.sh,协议一致性由入口唯一性保证;git rev 开头
# 记档;wandb group 沿用 _lane.sh 的 pair×seed 格式。

set -uo pipefail

ROOT=/mgfs/shared/Group_GY/changhao/SimOPD
D=/mgfs/shared/Group_GY/changhao/simopd_data
ARMS=(a1_gkd_mix0.5 a3_offpolicy a4_dagger_anneal a5_aggrevate)
PAIRS=(0,1 2,3 4,5 6,7)

# ---------------------------------------------------------------- submitter --
if [ -z "${MLP_ROLE_INDEX:-}${MLP_WORKER_RACK_RANK_INDEX:-}${DLC_JOB_ID:-}" ]; then
    cat <<CARD
================ DLC 控制台表单(1 worker × 8 GPU) ================
  任务名称   simopd-a-axis-4lane
  节点数量   1
  单节点GPU  8      CPU 64      内存 512Gi(照 eval 舰队成功任务的规格填)
  镜像/资源组/数据集挂载:照抄上一个成功的 EVAL_ONLY 任务表单(挂载须含 /mgfs)
  执行命令:
    bash $ROOT/deploy/dlc/a_axis_fleet.sh
====================================================================
本机不是 DLC 容器(无 rank env),以上为提交卡片;容器内执行同一脚本即运行载荷。
CARD
    exit 0
fi

# ------------------------------------------------------------------ payload --
# 单 worker 任务:表单误填多节点时,rank!=0 的 pod 闲置守望而不是把同一组
# lane/分片再跑一遍(ckpt 目录撞车)。rank 取值沿用 exp worker.sh 的优先序。
_rank=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
if [ "${_rank}" != "0" ]; then
    while true; do echo "rank ${_rank}: single-worker job, idling ($(date))"; sleep 600; done
fi
cd "$ROOT"
LOGD=$D/a_axis
mkdir -p "$LOGD"
exec > >(tee -a "$LOGD/fleet_$(date +%Y%m%d_%H%M%S).log") 2>&1
echo "== a_axis_fleet on $(hostname), git $(git log --oneline -1 2>/dev/null | head -1)"
git config --global --add safe.directory "$ROOT" 2>/dev/null || true
nvidia-smi -L | head -8
source simopd/bin/activate

export DATA_DIR=$D/simopd_math
export CKPT_ROOT=$D/ckpt                 # -> $D/ckpt/simopd/<EXPERIMENT_NAME>,post-eval glob 范围内
export MAX_RESPONSE_LENGTH=16384
export ROLLOUT_GPU_MEM_UTIL=0.45         # campaign 钉值(见 deploy/campaign.sh 的论证)
export PYTHONUNBUFFERED=1

# 注册表卫生:verl 在场,跑全量 lint(EXPECT_PG 分支、旋钮被消费、双旋钮拒绝)。
# lint 不过 = 注册与代码漂移,发射必须停在这里。
python scripts/arm_lint.py || { echo "ARM_LINT FAILED -- refusing to proceed"; exit 1; }

# ------------------------------------------------------------ Phase P 预计算 --
if [ ! -f "$D/gkd_offpolicy.parquet" ]; then
    echo "== Phase P: teacher precompute (8-way shard)"
    python - <<'PY'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-1.7B-Base"):
    print("cached:", snapshot_download(m))
PY
    for g in 0 1 2 3 4 5 6 7; do
        [ -f "$D/gkd_offpolicy.parquet.shard${g}of8" ] && { echo "shard$g exists, skip"; continue; }
        CUDA_VISIBLE_DEVICES=$g python scripts/gen_offpolicy.py --shard $g/8 \
            --train-parquet "$D/simopd_math/train.parquet" \
            --out "$D/gkd_offpolicy.parquet" > "$LOGD/shard$g.log" 2>&1 &
    done
    wait
    python scripts/gen_offpolicy.py --merge 8 \
        --train-parquet "$D/simopd_math/train.parquet" \
        --out "$D/gkd_offpolicy.parquet" || { echo "MERGE FAILED"; tail -3 "$LOGD"/shard*.log; exit 1; }
fi
[ -f "$D/gkd_offpolicy.parquet.dry" ] || python scripts/gen_offpolicy.py --dry \
    --train-parquet "$D/simopd_math/train.parquet" --out "$D/gkd_offpolicy.parquet"
echo "== Phase P done: $(ls -la "$D/gkd_offpolicy.parquet" | awk '{print $5}') bytes"

# -------------------------------------------------------------- Phase R 彩排 --
echo "== Phase R: 3-step rehearsals (4 arms parallel, one 2-GPU pair each)"
for i in 0 1 2 3; do
    ARM=${ARMS[$i]}
    [ -f "$LOGD/rehearsal_${ARM}.OK" ] && { echo "rehearsal $ARM already OK, skip"; continue; }
    ( bash deploy/dsw/rehearse_a_axis.sh "$ARM" "${PAIRS[$i]}" \
        && touch "$LOGD/rehearsal_${ARM}.OK" ) &
done
wait
FAILED=""
for ARM in "${ARMS[@]}"; do
    [ -f "$LOGD/rehearsal_${ARM}.OK" ] || FAILED="$FAILED $ARM"
done
if [ -n "$FAILED" ]; then
    # 挂起待查而非退出:DLC 对非零退出可能重启风暴,而彩排失败需要人看日志。
    while true; do
        echo "REHEARSAL FAILED:$FAILED -- lanes NOT launched. logs: $LOGD/rehearsal_*.log ($(date))"
        sleep 600
    done
fi
echo "== Phase R done: 4/4 rehearsals OK"

# -------------------------------------------------------------- Phase L 发射 --
echo "== Phase L: 4 lanes x 250 steps (TEST/SAVE freq 25, auto-resume from ckpt)"
# 无 wandb 凭证时降级 offline:DLC pod 的 /root 是易失的,凭证只可能来自任务
# 环境面板或镜像;缺了它 wandb.init 会在 step 0 前杀死 lane。offline 下步级
# 指标仍在控制台日志里(gkd_* 走同一通道),post-eval 流水线不受影响。
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE=offline

for i in 0 1 2 3; do
    ARM=${ARMS[$i]}
    (
        # 有界重试:run_opd_baseline 从 checkpoint 自恢复,所以 lane 内崩溃
        # (OOM 抖动、vLLM 偶发)重试的代价只是回到上个 SAVE_FREQ=25 存档;
        # 三次仍死则留日志退出,不拖累其余 lane。
        for attempt in 1 2 3; do
            (
                set -e
                _arm_env=$(python scripts/arm.py env "$ARM")   # 拒绝臂在此死,绝不静默 vanilla
                eval "$_arm_env"
                export EXPERIMENT_NAME="${ARM}_s0_16k"
                export CUDA_VISIBLE_DEVICES=${PAIRS[$i]}
                export TOTAL_TRAINING_STEPS=250
                export WANDB_RUN_GROUP="Qwen3-1.7B-Base__from__Qwen3-4B-Instruct-2507__s0"
                export WANDB_TAGS="${ARM},A,seed0,dlc_a_axis"
                bash scripts/run_opd_baseline.sh \
                    data.seed=0 \
                    actor_rollout_ref.rollout.seed=0
            ) && { echo "lane ${ARM}: attempt $attempt completed"; break; }
            echo "lane ${ARM}: attempt $attempt failed ($(date)); resume-retry"
            sleep 30
        done
    ) > "$LOGD/lane_${ARM}.log" 2>&1 &
    echo "lane ${ARM} -> GPUs ${PAIRS[$i]}, log $LOGD/lane_${ARM}.log"
done
wait
echo "== Phase L done"
for ARM in "${ARMS[@]}"; do
    ck=$(ls -d "$D/ckpt/simopd/${ARM}_s0_16k/global_step_"* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    echo "  ${ARM}_s0_16k: last checkpoint step ${ck:-NONE}"
done
echo "A_AXIS_FLEET_DONE"
