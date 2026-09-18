#!/usr/bin/env bash
#SBATCH --job-name=simopd-evalfarm
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
#SBATCH --requeue
# 日志路径走 Slurm 原生的 SBATCH_OUTPUT/SBATCH_ERROR 环境变量(见 simopd_env.example.sh)。
# 不在这里写 #SBATCH --output=$VAR —— sbatch 不展开指令行里的 shell 变量,那样只会
# 在当前目录建出一个名叫 '$VAR' 的目录。
#
# 评测专列:一个节点 8 张卡 = 8 个 worker,外加一个 refill 常驻补队列。
#
#   sbatch slurm/retrain_eval_farm.sbatch
#   EVAL_GRID=light sbatch slurm/retrain_eval_farm.sbatch    # 只在锚点跑 AIME
#   EMPTY_EXIT=0    sbatch slurm/retrain_eval_farm.sbatch    # 队列空了也不放节点
#
# 这是 deploy/dlc/eval_farm.sh 的 Slurm 版。差别只在载体:DLC 那版靠平台自动重启,
# 这里靠 --requeue + 前台守夜循环;worker 与 refill 是同两个脚本,协议一个字没改。
# 时限:分区 MaxTime=UNLIMITED,所以不写 --time —— 一格 AIME avg@8 要跑上小时,
# 掐表只会把跑到一半的格子变成僵尸 claim。
set -u
# 不要依赖环境里的 USER:Slurm 的批处理环境没有它,而这里开了 set -u。从登录 shell
# 提交时 sbatch 默认 --export=ALL 会把 USER 带进来,所以手工提交看不出问题;一旦改由
# cron -> window_watchdog.sh -> night_window.sh 这条链提交,整条链都不带 USER,作业
# 一起来就 'USER: unbound variable' 当场死。2026-09-13/15/16 三个晚上的 lane 作业就是
# 这么零产出的。id -un 不依赖环境,永远对。
ME=$(id -un)
cd $SIMOPD_ROOT
. ./simopd_env.sh
# 双保险:2026-08-23 就是容器注入了 VLLM_USE_MODELSCOPE=True、venv 没装 modelscope,
# 32 个 worker 一起 import 就死,把整条队列烧成 FAILED。env 文件缺行也不许回退。
export VLLM_USE_MODELSCOPE=False VERL_USE_MODELSCOPE=False

Q=$SIMOPD_STORE/evalq_exp
# 不是随手挑的目录:scripts/eval_refill_exp.py 的 workers() 按 $SIMOPD_STORE/corr_wave/
# evalw_slot*.log 这个 glob 统计产能。写到别处,产能表就永远显示 0 个 worker —— 队列还在
# 跑,页面却安静地说没人干活(AGENT.md section 7 那类"过滤器写死"的坑,同一个形状)。
LOGD=${EVAL_LOG_DIR:-$SIMOPD_STORE/corr_wave}
GRID=${EVAL_GRID:-full}
EMPTY_EXIT=${EMPTY_EXIT:-7200}     # 队列连续空这么多秒就放节点回去给训练;0=永不退
NGPU=${NGPU:-8}
mkdir -p "$LOGD" "$Q/claims" "$Q/failed" "$SIMOPD_EVALS"

echo "=== $(hostname)  job $SLURM_JOB_ID  restarts=${SLURM_RESTART_COUNT:-0}  $(date -Is)"
echo "=== queue $Q   evals -> $SIMOPD_EVALS   grid=$GRID   SUITE_K=${SIMOPD_SUITE_K:-32}"
echo "=== logs $LOGD"

_start_worker() {
    ( nohup bash deploy/dsw/eval_worker_exp.sh "$1" "$Q" \
        >> "$LOGD/evalw_slot${SLURM_JOB_ID}_gpu$1.log" 2>&1 & )
}
# 日志名保持 evalw_slot*_gpu*.log:scripts/eval_refill_exp.py 的 workers() 按这个 glob
# 统计产能,换个名字那张产能表就永远是空的。
_start_refill() {
    ( nohup python scripts/eval_refill_exp.py --write --watch 1200 --grid "$GRID" \
        >> "$LOGD/eval_refill_${SLURM_JOB_ID}.log" 2>&1 & )
}

# refill 先跑一次(同步),否则 8 个 worker 会对着一个空队列干等 20 分钟。
python scripts/eval_refill_exp.py --write --grid "$GRID" | tail -20
_start_refill
for g in $(seq 0 $((NGPU - 1))); do _start_worker "$g"; sleep 2; done

_reap() { echo "=== $(date -Is) 收尾:停 worker 与 refill"
          pkill -u "$ME" -f "eval_worker_exp.sh .* $Q" 2>/dev/null
          pkill -u "$ME" -f "eval_refill_exp.py" 2>/dev/null
          sleep 5; exit 0; }
trap _reap TERM INT

empty_for=0
while true; do
    sleep 300
    for g in $(seq 0 $((NGPU - 1))); do
        pgrep -u "$ME" -f "eval_worker_exp.sh $g $Q" >/dev/null 2>&1 || {
            echo "=== $(date -Is) worker gpu$g 不在了,拉起来"; _start_worker "$g"; }
    done
    pgrep -u "$ME" -f "eval_refill_exp.py" >/dev/null 2>&1 || {
        echo "=== $(date -Is) refill 不在了,拉起来"; _start_refill; }

    n=$(wc -l < "$Q/pending.txt" 2>/dev/null || echo 0)
    busy=$(ls -A "$Q/claims" 2>/dev/null | wc -l)
    echo "[eval_farm] 值守 $(date '+%T')  队列剩 $n 条,在跑 $busy 格"

    # 空队列 + 没有格子在跑,才算真没活;只看队列会在最后 8 格跑着的时候提前放节点。
    if [ "$n" -eq 0 ] && [ "$busy" -eq 0 ]; then
        empty_for=$((empty_for + 300))
        if [ "$EMPTY_EXIT" -gt 0 ] && [ "$empty_for" -ge "$EMPTY_EXIT" ]; then
            echo "=== 队列空了 ${empty_for}s,把节点还回去(EMPTY_EXIT=$EMPTY_EXIT)"; _reap
        fi
    else
        empty_for=0
    fi
done
