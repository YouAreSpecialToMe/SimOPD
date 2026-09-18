#!/usr/bin/env bash
# simopd_env.example.sh —— 复制成 simopd_env.sh 后按本机情况填。
#
#   cp simopd_env.example.sh simopd_env.sh && $EDITOR simopd_env.sh
#
# simopd_env.sh 在 .gitignore 里(第 20 行),不会被提交 —— 密钥只放在那份里,别放这份。
# 仓库里几乎所有脚本(slurm/*、deploy/dsw/*、scripts/*)开头都 `. ./simopd_env.sh`,
# 所以这一个文件就是全部的机器相关配置。source 之后再提交作业,Slurm 的 SBATCH_* 变量
# 才会对 sbatch 生效。

# ── 1. 四个路径根 ──────────────────────────────────────────────────────────
# 代码里出现的 $SIMOPD_ROOT / $SIMOPD_STORE / $OPD_LOGS / $OPD 全部来自这里。
# 四个都必须设,脚本不再带任何硬编码默认值。
export OPD=$HOME/opd                     # 工作区顶层,下面放代码、数据、日志
export SIMOPD_ROOT=$OPD/SimOPD           # 本仓库的 checkout(含 verl/ 子克隆)
export SIMOPD_STORE=$OPD/simopd_data     # 数据根:ckpt / traj / evals / 队列 / 闸门
export OPD_LOGS=$OPD/logs                # Slurm 作业日志(%x-%j.out 落在这里)
mkdir -p "$SIMOPD_STORE" "$OPD_LOGS"

# 从 $SIMOPD_STORE 派生的几个子根。改布局的话改这里,不要改脚本。
export CKPT_ROOT=$SIMOPD_STORE/ckpt
export DATA_DIR=$SIMOPD_STORE/simopd_math
export SIMOPD_EVAL_ROOT=$SIMOPD_STORE/evals

# ── 2. Slurm ───────────────────────────────────────────────────────────────
# 分区、日志路径走 Slurm 原生的输入环境变量(man sbatch 的 INPUT ENVIRONMENT VARIABLES),
# 不写死在 #SBATCH 指令里 —— 指令行不展开 shell 变量,写 $VAR 只会建出一个叫 '$VAR'
# 的目录。改集群只改这里。
export SBATCH_PARTITION=CHANGE_ME        # sinfo -s 看有哪些分区
export SBATCH_OUTPUT=$OPD_LOGS/%x-%j.out
export SBATCH_ERROR=$OPD_LOGS/%x-%j.out

# 排除名单没有对应的 SBATCH_* 变量,所以由提交方(slurm/campaign_scheduler.sh 与
# slurm/night_window.sh)读这个变量后用 --exclude 传。留空就是不排除。
#
# 这个名单有用。2026-09 这一轮踩到过两类节点级故障,都不是代码问题、也都不会让作业
# 提前退出,只会安静地烧机时:
#   (a) 一台节点被 Slurm 账外的进程占掉约 65 G 显存(Slurm 仍把它报成 idle)。作业拿到
#       它之后四条 lane 的 teacher vLLM 全部 OOM 起不来,整个作业当场报废。
#   (b) 一台节点的本地 SSD 100% 满。Ray 的 spill 落在计算节点的 /tmp,七条臂同时
#       ENOSPC 而死;清掉我们自己的残留目录只回收 5 G,占盘的不是我们。
# slurm/retrain_lane_node.sbatch 里有开工前的本地盘预检(可用 < 200 G 直接 FATAL 退出),
# 能把 (b) 变成"快速失败"而不是"跑一半炸",但躲开还是得靠这个名单。
export SLURM_EXCLUDE_NODES=""            # 例:node-a,node-b

# ── 3. Hugging Face ────────────────────────────────────────────────────────
export HF_HOME=$SIMOPD_STORE/hf
export HF_HUB_OFFLINE=1                  # 计算节点通常不出网;要下载时临时置 0
# export HF_ENDPOINT=https://hf-mirror.com      # 需要镜像时打开
# export HF_HUB_DISABLE_XET=1                   # xet 传输在某些网络下不稳时打开
# export HF_TOKEN=...                           # 只放在 simopd_env.sh 里,别提交

# ── 4. 实验跟踪 ────────────────────────────────────────────────────────────
# 2026-09 这一轮全程 offline —— 计算节点不出网,在线模式会让每一步都卡在重试上。
# 代价是没有在线曲线,全部指标只在 $SIMOPD_STORE/<run>/metrics/launch_*.jsonl 里。
export WANDB_MODE=offline
export WANDB_DIR=$SIMOPD_STORE/wandb
# export WANDB_API_KEY=...                      # 只放在 simopd_env.sh 里,别提交

# ── 5. 评测采样数 ──────────────────────────────────────────────────────────
# 协议原文是 AIME/AMC avg@32(eval_suite.py 的 docstring 也这么写),但 2026-09 这一轮
# 全部 155 份产物实测是 **avg@8**(aime24/aime25 每题 8 次、amc23 每题 8 次;minerva 与
# math500 是 avg@3,与协议一致)。当时这个值是从 shell 环境给的,没有进代码,脚本默认
# 仍是 32。
#
# 要**接着补**那 114 格,必须设成 8。eval_suite.py 的 newest() 按每题采样数过滤产物,
# k=8 与 k=32 的 parquet 不会互相顶替、也不会混进同一个 composite —— 用默认值跑出来的
# 是另一套读数,和已有的 36 格齐全单元不可比。
# 要**全部重评**成协议口径,就把这里设成 32 并把已有 155 份挪走,别混放。
export SIMOPD_SUITE_K=8

# ── 6. 可选:国内镜像 ──────────────────────────────────────────────────────
# export VERL_USE_MODELSCOPE=1
# export VLLM_USE_MODELSCOPE=1

# ── 7. 虚拟环境 ────────────────────────────────────────────────────────────
# deploy/dsw/setup.sh 默认把 venv 装在 $SIMOPD_ROOT/simopd。装在别处就设这个。
# export SIMOPD_VENV=$SIMOPD_ROOT/simopd
[ -f "$SIMOPD_ROOT/simopd/bin/activate" ] && . "$SIMOPD_ROOT/simopd/bin/activate"
