#!/usr/bin/env bash
#SBATCH --job-name=simopd-agate
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
#SBATCH --time=02:00:00
# 日志路径走 Slurm 原生的 SBATCH_OUTPUT/SBATCH_ERROR 环境变量(见 simopd_env.example.sh)。
# 不在这里写 #SBATCH --output=$VAR —— sbatch 不展开指令行里的 shell 变量,那样只会
# 在当前目录建出一个名叫 '$VAR' 的目录。
#
# A 轴四条臂的预演门:一个节点四对卡并行跑 3 步机判彩排,全过才写放行标记。
#
# 为什么要有这道门(AGENT.md 2.2 + arms.yaml 的臂注):A 轴靠 gkd 缓存/教师注册表喂数据,
# 这些东西坏掉时训练**不会报错,只会静默退化成香草**。rehearse_a_axis.sh 的注释里记着
# 实测过的那次:注册表被 Ray GC 之后 331/331 条全降级,3 步彩排照样绿灯退出 —— 所以判据
# 里有"教师参与度全零即 FAIL"这种硬闸。不过这道门再铺 4 条 200 步的臂,便宜得多。
#
# 彩排用 EXPERIMENT_NAME=rehearsal_<arm> 的独立命名空间、REHEARSAL=1 走短跑守卫的
# 合法通行证、3 步 < SAVE_FREQ 不落盘,不会碰真正的 <arm>_s0 目录。
set -u
cd $SIMOPD_ROOT
. ./simopd_env.sh >/dev/null 2>&1

ARMS=${ARMS:-"a1_gkd_mix0.5_n0 a3_offpolicy_n0 a4_dagger_anneal_n0 a5_aggrevate_n0"}
GATE=$SIMOPD_STORE/gates/a_axis_ok
LOGD=$SIMOPD_STORE/a_axis
mkdir -p "$(dirname "$GATE")" "$LOGD"

echo "=== $(hostname) job $SLURM_JOB_ID $(date -Is)"
echo "=== 彩排: $ARMS"
C=${DATA_DIR:?}/gkd_offpolicy.parquet
[ -f "$C" ] || { echo "FATAL: gkd 缓存不在 ($C) —— a1/a3/a4 没法彩排"; exit 1; }
[ -f "$C.dry" ] || { echo "FATAL: a5 的 keys 文件不在 ($C.dry)"; exit 1; }

i=0; pids=(); names=()
for a in $ARMS; do
    g="$((i*2)),$((i*2+1))"
    echo "== $a -> gpu $g"
    bash deploy/dsw/rehearse_a_axis.sh "$a" "$g" > "$LOGD/gate_${a}.log" 2>&1 &
    pids+=($!); names+=("$a"); i=$((i+1))
    sleep 20          # 错峰:四个 vLLM 同时冷启动会互相拖慢
done

rc=0
for k in "${!pids[@]}"; do
    if wait "${pids[$k]}"; then echo "  PASS ${names[$k]}"
    else echo "  FAIL ${names[$k]} —— 见 $LOGD/gate_${names[$k]}.log"; rc=1; fi
done

if [ "$rc" -eq 0 ]; then
    date -Is > "$GATE"
    echo "=== 四条全过,放行标记已写: $GATE"
else
    echo "=== 有臂没过,**不写**放行标记 —— A 轴不会被铺上去,先看日志"
    for n in "${names[@]}"; do
        echo "--- $n 尾部 ---"; tail -12 "$LOGD/gate_${n}.log" 2>/dev/null | sed 's/^/    /'
    done
fi
exit $rc
