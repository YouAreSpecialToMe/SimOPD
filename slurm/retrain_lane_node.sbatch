#!/usr/bin/env bash
#SBATCH --job-name=simopd-lane
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# 分区、排除名单、日志目录都不在这里写死 —— 见 simopd_env.example.sh。
#   * 分区走 Slurm 原生的输入环境变量 SBATCH_PARTITION(man sbatch 的 INPUT ENVIRONMENT
#     VARIABLES 一节),export 了就生效,不需要改脚本。
#   * 排除名单没有对应的 SBATCH_* 变量,所以由提交方(campaign_scheduler.sh /
#     night_window.sh)读 SLURM_EXCLUDE_NODES 后用 --exclude 传进来。手工提交时自己带。
#     这个名单有用:本轮就踩到过两类节点级故障 —— 一台被 Slurm 账外的进程占掉约 65 G
#     显存(Slurm 仍把它报成 idle,teacher vLLM 起不来、整个作业当场报废),另一台本地
#     SSD 被占满(Ray 的 spill 落在计算节点的 /tmp,七条臂同时 ENOSPC)。两者都只能靠
#     排除躲开,清干净了就从名单里删掉。
#   * 日志目录走 SBATCH_OUTPUT/SBATCH_ERROR,同样是 Slurm 原生变量。
# --time 在这里只有两个作用,都不是"上限"本身:
#   (a) 给 Slurm 的回填调度一个界,满载集群里有界的作业更容易插进空档;
#   (b) 兜底,防止无人看管时跑飞。
# 真正的停机是全局的 —— 用户 2026-09-05 定"从某一刻起算总共 8 小时,不是每个作业各算
# 各的",由 slurm/hard_stop.sh 按一个绝对截止时间统一 scancel。注意 Slurm 只允许用户
# **降低**自己作业的时限,不允许提高:这里写小了就再也放不回去(5167 就是这么被锁在
# 8 小时上、比全局截止线早一小时被砍的)。所以宁可写宽。
#SBATCH --time=12:00:00
#SBATCH --requeue
# 日志路径走 Slurm 原生的 SBATCH_OUTPUT/SBATCH_ERROR 环境变量(见 simopd_env.example.sh)。
# 不在这里写 #SBATCH --output=$VAR —— sbatch 不展开指令行里的 shell 变量,那样只会
# 在当前目录建出一个名叫 '$VAR' 的目录。
#
# One node = 8 H100 = four 2-GPU lanes. Two is the floor per run: verl registers the
# teacher pool as its own Ray resource pool and will not share the actor's GPU.
#
#   RUNS="vanilla_corr:0" LANES=1 STEPS=250 sbatch slurm/retrain_lane_node.sbatch
#   RUNS="b1_skew_kl_corr:0 b2_forward_kl_corr:0 c3_intersection_corr:0 d2_selectkd_corr:0" \
#       STEPS=200 sbatch slurm/retrain_lane_node.sbatch
#
set -uo pipefail

source $SIMOPD_ROOT/simopd_env.sh
cd "$SIMOPD_ROOT"

RUNS=${RUNS:-}
if [ -z "$RUNS" ]; then
    echo "FATAL: RUNS is empty. Refusing to fall back to run_parallel.sh's default work" >&2
    echo "       list (every stock arm), which is 99 arms and not the rerun roster." >&2
    echo "       RUNS=\"vanilla_corr:0\" LANES=1 sbatch $0" >&2
    exit 1
fi
export LANES=${LANES:-4}
export STEPS=${STEPS:-250}
export LOG_DIR=${LOG_DIR:-$OPD_LOGS/lanes}
mkdir -p "$LOG_DIR"

echo "=== $(hostname)  job $SLURM_JOB_ID  restarts=${SLURM_RESTART_COUNT:-0}  $(date -Is)"
echo "=== RUNS: $RUNS"
echo "=== LANES=$LANES STEPS=$STEPS SUITE_K=${SIMOPD_SUITE_K:-unset} ARCHIVE=${SIMOPD_ARCHIVE:-1}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo

# ---- ckpt to the hub (AGENT.md 2.3) -------------------------------------------
# The data loss this campaign is recovering from happened because the trainer wrote
# ONE copy to a shared disk: the repo has no upload path at all (verl's FSDP manager
# marks hdfs_path "Unused"), the cluster went away, and 308 unevaluated checkpoints
# went with it. So the syncer starts WITH the lanes and does not depend on anyone
# remembering. Opt-in: no CKPT_SYNC_REPO is a no-op. CKPT_SYNC_REPO set with no
# HF_TOKEN is LOUD -- silence is exactly how the data was lost.
SYNC_LOG="$LOG_DIR/ckpt_sync_job${SLURM_JOB_ID}.log"
if [ -n "${CKPT_SYNC_REPO:-}" ]; then
    if [ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
        echo "!! CKPT_SYNC_REPO is set but there is no HF_TOKEN in the environment --"
        echo "!! the syncer did NOT start and these checkpoints exist in one place only."
    else
        ( HF_HUB_OFFLINE= nohup python scripts/ckpt_sync.py \
            --repo "$CKPT_SYNC_REPO" --watch "${CKPT_SYNC_EVERY:-600}" \
            --with-optimizer "${CKPT_SYNC_FULL:-}" >> "$SYNC_LOG" 2>&1 & )
        echo "== ckpt syncer up -> $CKPT_SYNC_REPO   (log $SYNC_LOG)"
        echo "   'automatic upload' means an OK line in that log and SYNCED.json in the"
        echo "   repo -- not the fact that this message printed. ckpt_sync.py had a"
        echo "   SyntaxError until 2026-09-02 and had never run anywhere."
    fi
else
    echo "== CKPT_SYNC_REPO unset: no upload. These checkpoints will exist in ONE place."
fi

# ---- 本地盘:开工先扫死 Ray 目录 --------------------------------------------------
# 2026-09-08 09:26,某个节点上四条 lane 同时死于
#   OSError: [Errno 28] No space left on device: <本地盘>/<user>/ray_lane3/ray/...
# 七条臂全 FAIL、整节点报废。RAY_TMPDIR=/tmp/$USER,而计算节点的 /tmp 就是 5.9T 的
# 本地 SSD。run_parallel.sh 只查"有没有活的 Ray 进程",**不清盘上的死目录** —— 同一
# 个节点被反复分配几十次之后,历次崩掉的 session 目录(尤其是 spill 出来的对象)就把
# 盘吃光了。节点是 EXCLUSIVE 独占,这时候盘上不可能有本作业以外我们的活儿,直接扫。
_RT="${RAY_TMPDIR:-/tmp}"
if [ -d "$_RT" ]; then
    echo "== 本地盘开工前: $(df -h "$_RT" | tail -1)"
    _n=0
    for _d in "$_RT"/ray_lane*; do
        [ -d "$_d" ] || continue
        pgrep -f "${_d}(/|$)" >/dev/null 2>&1 && { echo "   保留 $_d(还有活进程)"; continue; }
        rm -rf -- "$_d" && _n=$((_n + 1))
    done
    [ "$_n" -gt 0 ] && echo "   扫掉 $_n 个死 ray_lane 目录"
    echo "== 本地盘清完:  $(df -h "$_RT" | tail -1)"
    # 剩余空间兜底:低于 200G 就别开工了,否则跑几个小时再 ENOSPC 更亏。
    _avail=$(df -BG --output=avail "$_RT" | tail -1 | tr -dc '0-9')
    if [ "${_avail:-999}" -lt 200 ]; then
        echo "FATAL: $_RT 只剩 ${_avail}G,清完仍不够(阈值 200G)。这台机器的本地盘被别的东西占着。" >&2
        echo "占盘大头(看得见的部分):" >&2
        du -xh -d2 "$(dirname "$_RT")" 2>/dev/null | sort -rh | head -12 >&2
        exit 1
    fi
fi

# ---- lanes ---------------------------------------------------------------------
# 200 步是 2026-09 名册的协议步长,不是泄漏的短跑。run_opd_baseline.sh 有个守卫:
# STEPS<250 且没有 TAG/REHEARSAL/SHORT_RUN_OK 就拒跑 —— 立意是对的("短跑会写 -> OK、
# 被记成 DONE,是没人会重试的假绿"),但它给的三个逃生口里只有第三个能用:TAG 会把 run
# 改名成 <arm>_s0_<tag>(_lane.sh:93),名册、ckpt 路径、verdict 匹配、评测队列全部对不上;
# REHEARSAL=1 会把步数压成 3。所以这里用 SHORT_RUN_OK,并且只在真的 <250 时给,
# 250 步的臂(vanilla_corr 等)env 不变、指纹不动、续跑不受影响。
if [ "${STEPS:-250}" -lt 250 ]; then
    export SIMOPD_SHORT_RUN_OK=1
    echo "== STEPS=$STEPS < 250:按 AGENT.md 2.2 的名册步长,SIMOPD_SHORT_RUN_OK=1 已置"
fi

# 一个节点上可以同时跑两个步数组。为什么要这样:run_parallel.sh 一次调用只有一个 STEPS,
# 而名册里 250 步只有 5 条、200 步有 24 条 —— 只按步数组打包的话,250 组派完之后剩下的
# 那一两条会独占一个节点,8 张卡里 6 张空着。GPU_LIST 把卡分开、RAY_TMPDIR_TAG 把 Ray 的
# 每-lane 目录分开(否则两组的 lane0 都会去抢 $RAY_TMPDIR/ray_lane0),两次调用就能共存。
# B 组可选:不给 RUNS_B 就完全是原来的行为。
_gpu_group() {   # $1=起始卡号 $2=lane 数 -> "0,1 2,3 ..."
    local first=$1 n=$2 out="" i
    for i in $(seq 0 $((n - 1))); do
        out="$out $((first + i*2)),$((first + i*2 + 1))"
    done
    echo "${out# }"
}

GPU_LIST="$(_gpu_group 0 "$LANES")" \
    # 错峰拉大到 45 秒(默认 15)。verl 的 single_controller/base/worker.py:59 用的是
# 不安全的取端口法:bind(("",0)) 拿到号之后**把 socket 关掉**,再在稍后真正绑 ——
# 中间这段空窗里别的 lane 可能抢走同一个号。2026-09-06 实测撞过一次
# (DistNetworkError: failed to listen on port 35277),一条 lane 当场死掉、
# 两张卡空转一整夜。verl 自己在 net_utils.py:73 有 with_alive_sock 的安全版本,
# 但 worker.py 没用它 —— 那才是根治,这里先把撞车概率压下去。
export LANE_STAGGER_SEC=${LANE_STAGGER_SEC:-45}

bash deploy/dsw/run_parallel.sh "$RUNS"

if [ -n "${RUNS_B:-}" ]; then
    LANES_B=${LANES_B:-$(set -- $RUNS_B; echo $#)}
    STEPS_B=${STEPS_B:-200}
    if [ "$STEPS_B" -lt 250 ]; then export SIMOPD_SHORT_RUN_OK=1; fi
    echo "== B 组:$LANES_B lane,STEPS=$STEPS_B,卡 $(_gpu_group $((LANES * 2)) "$LANES_B")"
    sleep 20                     # 别和 A 组同时冷启动:模型/Ray/vLLM 一起拉会互相拖慢
    # LANES 必须显式传给 B 组:run_parallel.sh 读的是自己的 LANES,不给的话它沿用
    # A 组那次的值。2026-09-06 实测:A 组只有 1 条臂(LANES=1),B 组 3 条臂就被塞进
    # 了一条 lane、只用 GPU 2-3,4-7 整整四张卡空转一整夜。
    GPU_LIST="$(_gpu_group $((LANES * 2)) "$LANES_B")" \
    LANES="$LANES_B" RAY_TMPDIR_TAG=b STEPS="$STEPS_B" \
        bash deploy/dsw/run_parallel.sh "$RUNS_B"
fi
rc=$?
# Slurm 批处理环境里没有 USER,而脚本开了 set -u —— 2026-09-13 起每晚的 lane 作业
# 都死在 "line 163: USER: unbound variable",窗口开着却一步没跑。为什么之前没事:
# 09-08/09 的守卫是从交互 shell 手工拉起的(有 USER);后来守卫被 crontab 的看门狗
# 重起过,cron 的最小环境没有 USER,于是一路传到 sbatch。和调度器那个 ME=$(id -un)
# 是同一个坑,这里用同样的办法。
ME=$(id -un)
[ "$rc" = 0 ] || { echo "FATAL: run_parallel.sh exited $rc before the lanes were up" >&2; exit "$rc"; }

# run_parallel.sh nohups each lane and returns immediately -- correct on DSW, fatal
# here. Hold the allocation until the last _lane.sh is gone.
sleep 90
if ! pgrep -u "$ME" -f "deploy/dsw/_lane.sh" >/dev/null 2>&1; then
    echo "FATAL: no lane process 90s after launch -- see $LOG_DIR/lane*.log" >&2
    exit 1
fi
echo "=== lanes up; holding the allocation. tail -f $LOG_DIR/lane0_*.log"
while pgrep -u "$ME" -f "deploy/dsw/_lane.sh" >/dev/null 2>&1; do sleep 120; done

# One last sync pass before the node goes away, so the final checkpoint is not the
# one that misses the boat.
if [ -n "${CKPT_SYNC_REPO:-}" ] && [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "=== lanes done; final ckpt_sync pass"
    HF_HUB_OFFLINE= timeout 7200 python scripts/ckpt_sync.py --repo "$CKPT_SYNC_REPO" \
        --with-optimizer "${CKPT_SYNC_FULL:-}" >> "$SYNC_LOG" 2>&1 \
        || echo "WARN: final sync pass failed -- see $SYNC_LOG" >&2
    pkill -u "$ME" -f "ckpt_sync.py --repo ${CKPT_SYNC_REPO}" 2>/dev/null
fi
echo "LANE_NODE_DONE $(date -Is)"
