#!/usr/bin/env bash
# A 轴 4-lane DLC 任务:一个 8 卡 worker,自门控三阶段,幂等可重入。
#
#   预计算 P  gkd_offpolicy.parquet 缺失 → 8 卡分片生成 + 合并(~1h;
#             per-request seed ⇒ 重跑逐 token 一致;分片各自幂等跳过)
#   彩排  R  每臂 3 步 rehearse_a_axis.sh,机器判据自动裁定(判据见该脚本头);
#             任一臂 FAIL → 挂起待查,绝不带病发射
#   发射  L  a1/a3/a4/a5 × s${SEED},每臂 2 卡一条 lane,250 步,ckpt 落
#             $D/ckpt/simopd/{arm}_s${SEED}_16k → post-eval 流水线的 *_16k glob 自动接管
#
# SEED 参数化(2026-08-18,全量 4 臂 × 3 seed):一个 seed 一个 DLC 作业(各
# 1 worker × 8 GPU),三次提交 —— 故障域隔离(单作业 AIMaster 重启不殃及别的
# seed),排 8 卡也比一次排 24 卡容易。SEED 从执行命令环境进载荷,缺省 0 时与
# 旧单 seed 行为逐字节一致。Phase P/R 皆 seed 无关:P 的教师缓存全 seed 共享
# (混合币本就按 (prompt,step) 确定),R 彩排验证机制不验证 seed —— seed!=0
# 的作业只等 .OK 标记,绝不自己彩排(两作业并发彩排会撞 rehearsal_* 日志与
# ckpt 命名空间)。
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
SEED=${SEED:-0}

# ---------------------------------------------------------------- submitter --
if [ -z "${MLP_ROLE_INDEX:-}${MLP_WORKER_RACK_RANK_INDEX:-}${DLC_JOB_ID:-}" ]; then
    cat <<CARD
======== DLC 控制台表单(每个 seed 一个作业,各 1 worker × 8 GPU) ========
  任务名称   simopd-a-axis-4lane-s<N>      (N = 0 / 1 / 2,共三次提交)
  节点数量   1
  单节点GPU  8      CPU 64      内存 512Gi(照 eval 舰队成功任务的规格填)
  镜像/资源组/数据集挂载:照抄上一个成功的 EVAL_ONLY 任务表单(挂载须含 /mgfs)
  执行命令(三个作业各填一条):
    SEED=0 bash $ROOT/deploy/dlc/a_axis_fleet.sh
    SEED=1 bash $ROOT/deploy/dlc/a_axis_fleet.sh
    SEED=2 bash $ROOT/deploy/dlc/a_axis_fleet.sh
  前置:$D/a_axis/ 下须已有四个 rehearsal_*.OK(seed!=0 只等标记,不自彩排)
==========================================================================
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
echo "== a_axis_fleet on $(hostname), seed ${SEED}, git $(git log --oneline -1 2>/dev/null | head -1)"
git config --global --add safe.directory "$ROOT" 2>/dev/null || true
nvidia-smi -L | head -8
source simopd/bin/activate
# 战役环境(deploy/dsw/setup.sh 生成,可重复 source):共享 HF 缓存 + 默认
# HF_HUB_OFFLINE=1 —— 429 疫苗:全舰队共享一个出口 IP,未认证配额 500 请求
# /300s,并发引擎各自 list 仓库树数秒耗尽配额、曾一波杀光 16 条 wave-1 lane;
# 本任务 8+8 个引擎并发启动,同样暴露。另携 WANDB_DIR 落 /mgfs + 手工补进的
# WANDB_API_KEY(在则 lane 直接 online;缺则文件内建 offline 降级接管)。
[ -f "$ROOT/simopd_env.sh" ] && source "$ROOT/simopd_env.sh"

export DATA_DIR=$D/simopd_math
export CKPT_ROOT=$D/ckpt                 # -> $D/ckpt/simopd/<EXPERIMENT_NAME>,post-eval glob 范围内
# NCCL/GLOO 接口按 pod 形态自选(bug 11 终版):首个 DLC 训练作业死于 FSDP 初
# 始化 "Bootstrap : no socket interface found"——MDC pod 的网卡是绑定接口
# bond0,NCCL 默认发现在此镜像上枚举不到它。实证配置来自 tools/dlc/exp*.sh
# (在这批 pod 上真实跑通过多卡训练的前辈脚本):NCCL/GLOO 双 pin bond0 +
# NET_PLUGIN=none(镜像自带网络插件是枚举失败的另一嫌疑;我们的单节点 lane
# 也用不上任何 fabric 插件)。DSW pod 无 bond0 → lo(单节点 bootstrap 足够,
# 数据面走 SHM/P2P)。eval 舰队从未撞上是因为单卡 vLLM 不碰 NCCL。全部可
# 环境覆写;NCCL_DEBUG=WARN 常开,若仍有意外,日志直接列出枚举结果。
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
    if [ -e /sys/class/net/bond0 ]; then
        export NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
    else
        export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    fi
fi
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
echo "== net ifaces: NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-unset} NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN"
export MAX_RESPONSE_LENGTH=16384
export ROLLOUT_GPU_MEM_UTIL=0.45         # campaign 钉值(见 deploy/campaign.sh 的论证)
export PYTHONUNBUFFERED=1

# 注册表卫生,限域裁决(教训 dlc51chl6xe1vawa:全量 lint 硬门撞上 campaign.tsv
# 的既有重复行 + supplement 臂设计上无清单行,12 分钟烧光 10 次重启)。全量
# 输出留档;只有"关于本任务四臂、且非战役簿记类"的问题才拦发射——env/旋钮/
# 分支漂移是 lane 安全问题,清单行覆盖与 verdict 名单是簿记,重复行属协作者
# 名册状态。失败改挂起待查:重启永远修不了注册表,不再烧重启配额。
LINT_LOG=$LOGD/arm_lint.log
python scripts/arm_lint.py > "$LINT_LOG" 2>&1 || true
BAD=$(grep 'PROBLEM' "$LINT_LOG" \
      | grep -E 'a1_gkd_mix0\.5|a3_offpolicy|a4_dagger_anneal|a5_aggrevate' \
      | grep -vE 'campaign\.tsv row|verdict\.py ARMS' || true)
if [ -n "$BAD" ]; then
    while true; do
        echo "ARM_LINT (scoped) FAILED -- lanes NOT launched ($(date)):"
        echo "$BAD"
        sleep 600
    done
fi
echo "== arm_lint: scoped gate clean (full report: $LINT_LOG)"

# ------------------------------------------------------------ Phase P 预计算 --
if [ ! -f "$D/gkd_offpolicy.parquet" ]; then
    echo "== Phase P: teacher precompute (8-way shard)"
    # 预取是唯一允许打 hub 的时刻(setup.sh: "Unset it only when fetching");
    # HF_HOME 已指向 /mgfs 共享缓存,战役资产大概率在,此步多为免费校验。
    HF_HUB_OFFLINE=0 python - <<'PY'
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
    # 屏障以产物文件为准,不 wait 进程:实测(本任务首跑,2026-08-17)vLLM 引擎
    # 拆除会挂死已完成的主进程,8/8 parquet 落盘 + 完成行打印后 wait 十分钟不
    # 返回,8 卡空烧。文件齐 → 60s 体面退出期 → 按 GPU 占用收割残留进程(僵死
    # 引擎占显存,不清彩排必 OOM)→ 合并。90 分钟仍不齐才判真失败。
    n=0
    for t in $(seq 1 90); do
        n=$(ls "$D"/gkd_offpolicy.parquet.shard*of8 2>/dev/null | wc -l)
        [ "$n" -ge 8 ] && break
        sleep 60
    done
    if [ "$n" -lt 8 ]; then
        echo "PRECOMPUTE STALLED at $n/8 after 90min"; tail -3 "$LOGD"/shard*.log; exit 1
    fi
    echo "== all shards on disk; grace 60s, then reaping stragglers"
    sleep 60
    pkill -f "[g]en_offpolicy.py --shard" 2>/dev/null || true
    sleep 10
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -n "$pids" ] && { echo "killing GPU holdouts: $pids"; echo "$pids" | xargs -r kill -9; sleep 5; }
    python scripts/gen_offpolicy.py --merge 8 \
        --train-parquet "$D/simopd_math/train.parquet" \
        --out "$D/gkd_offpolicy.parquet" || { echo "MERGE FAILED"; tail -3 "$LOGD"/shard*.log; exit 1; }
fi
[ -f "$D/gkd_offpolicy.parquet.dry" ] || python scripts/gen_offpolicy.py --dry \
    --train-parquet "$D/simopd_math/train.parquet" --out "$D/gkd_offpolicy.parquet"
echo "== Phase P done: $(ls -la "$D/gkd_offpolicy.parquet" | awk '{print $5}') bytes"

# -------------------------------------------------------------- Phase R 彩排 --
echo "== Phase R: 3-step rehearsals (4 arms parallel, one 2-GPU pair each)"
if [ "$SEED" != 0 ]; then
    # seed!=0 只等标记:并发彩排会互撞日志与 rehearsal ckpt 命名空间。标记由
    # seed-0 作业或手工彩排(gpu193 实践)落盘,seed 无关可复用。
    until [ -f "$LOGD/rehearsal_${ARMS[0]}.OK" ] && [ -f "$LOGD/rehearsal_${ARMS[1]}.OK" ] \
       && [ -f "$LOGD/rehearsal_${ARMS[2]}.OK" ] && [ -f "$LOGD/rehearsal_${ARMS[3]}.OK" ]; do
        echo "seed $SEED: waiting for 4/4 rehearsal .OK markers in $LOGD ($(date))"
        sleep 600
    done
    echo "== Phase R done (markers found; rehearsed elsewhere)"
else
_rpids=()
for i in 0 1 2 3; do
    ARM=${ARMS[$i]}
    [ -f "$LOGD/rehearsal_${ARM}.OK" ] && { echo "rehearsal $ARM already OK, skip"; continue; }
    ( bash deploy/dsw/rehearse_a_axis.sh "$ARM" "${PAIRS[$i]}" \
        && touch "$LOGD/rehearsal_${ARM}.OK" ) &
    _rpids+=($!)
done
# 裸 wait 是禁忌(2026-08-18 实测:四臂全跳过后任务在此无限挂死):此 bash 的
# 无参 wait 连 exec 的 tee 进程替换一起等,而 tee 到脚本终结才退——08-17 的
# "Phase P wait 挂死"同根因,当时的文件屏障恰好绕开。显式 pid 数组只等真子
# 任务;空数组必须整个跳过(wait 空参数又退化成裸 wait)。
[ ${#_rpids[@]} -gt 0 ] && wait "${_rpids[@]}"
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
fi

# -------------------------------------------------------------- Phase L 发射 --
echo "== Phase L: 4 lanes x 250 steps, seed ${SEED} (TEST/SAVE freq 25, auto-resume from ckpt)"
# 无 wandb 凭证时降级 offline:DLC pod 的 /root 是易失的,凭证只可能来自任务
# 环境面板或镜像;缺了它 wandb.init 会在 step 0 前杀死 lane。offline 下步级
# 指标仍在控制台日志里(gkd_* 走同一通道),post-eval 流水线不受影响。
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE=offline

_lpids=()
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
                export EXPERIMENT_NAME="${ARM}_s${SEED}_16k"
                export CUDA_VISIBLE_DEVICES=${PAIRS[$i]}
                export TOTAL_TRAINING_STEPS=250
                export WANDB_RUN_GROUP="Qwen3-1.7B-Base__from__Qwen3-4B-Instruct-2507__s${SEED}"
                export WANDB_TAGS="${ARM},A,seed${SEED},dlc_a_axis"
                bash scripts/run_opd_baseline.sh \
                    data.seed="$SEED" \
                    actor_rollout_ref.rollout.seed="$SEED"
            ) && { echo "lane ${ARM}: attempt $attempt completed"; break; }
            echo "lane ${ARM}: attempt $attempt failed ($(date)); resume-retry"
            sleep 30
        done
    ) > "$LOGD/lane_${ARM}_s${SEED}.log" 2>&1 &
    _lpids+=($!)
    echo "lane ${ARM} (seed ${SEED}) -> GPUs ${PAIRS[$i]}, log $LOGD/lane_${ARM}_s${SEED}.log"
done
# 同 Phase R:显式 pid,绝不裸 wait(tee 进程替换死锁)。
wait "${_lpids[@]}"
echo "== Phase L done"
ok=0
for ARM in "${ARMS[@]}"; do
    ck=$(ls -d "$D/ckpt/simopd/${ARM}_s${SEED}_16k/global_step_"* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    echo "  ${ARM}_s${SEED}_16k: last checkpoint step ${ck:-NONE}"
    [ -n "$ck" ] && ok=$((ok+1))
done
# 全灭防假成功(bug 11 连带发现):四 lane 重试烧光后脚本曾照常打 DONE 退出 0,
# DLC 记"成功"且不再重启,8 卡任务零产出静默终结。零存档一律挂起待查(重启
# 修不了环境错;挂起让失败可见、不烧重启配额)。
if [ "$ok" -eq 0 ]; then
    while true; do
        echo "ALL LANES DEAD, zero checkpoints banked -- NOT declaring success; inspect $LOGD/lane_*_s${SEED}.log ($(date))"
        sleep 600
    done
fi
echo "A_AXIS_FLEET_DONE"
