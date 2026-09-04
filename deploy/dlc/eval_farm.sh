#!/usr/bin/env bash
# 评测专列 payload(slot7-10 通用,task.sh set/swap 装上)。
#
# v2(2026-08-22 终检):v1 两个坑 ——
#   a) worker 裸继承 pod 环境:这批 4 节点单的容器注入了 VLLM_USE_MODELSCOPE=True,
#      venv 没装 modelscope,vLLM 一 import 就死,32 个 worker 把整条队列烧成
#      FAILED(还顺带留下一地删不掉的 claim)。训练侧没事是因为舰队脚本 source 了
#      simopd_env.sh —— 现在 payload 顶上也 source,worker/refill 全体继承。
#   b) 四个 pod 各起一个 refill 竞写 pending.txt:原子替换不损文件,但视图打架
#      (谁后写谁说了算)。refill 只在 slot7 起;slot7 整锅端了由平台重投,照旧回来。
set -u
D=${D:-/mgfs/shared/Group_GY/changhao/simopd_data}
ROOT=${ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
LOGD=${LOGD:-$D/corr_wave}; Q=$D/evalq_exp
# 换集群必改(AGENT.md §6 1c):原先这一行是硬写死、没有覆盖口的
V=${SIMOPD_PY:-/mgfs/shared/Group_GY/changhao/SimOPD/simopd/bin/python}
[ -x "$V" ] || V=$(command -v python3)
cd "$ROOT" && . ./simopd_env.sh
export VLLM_USE_MODELSCOPE=False VERL_USE_MODELSCOPE=False   # 双保险:env 文件缺行也不回退
# eval worker 的位置(2026-09-02):**仓库里那份是权威**。上一份只躺在 $D 上、从未进过
# git,随旧集群一起没了,而这里三处调用都指着它 —— 缺了就是「卡先闲着」,静默不评。
# 盘上副本只作老 pod 的兜底。
EVALW=${EVALW:-$ROOT/deploy/dlc/eval_worker_exp.sh}
[ -f "$EVALW" ] || EVALW=$D/eval_worker_exp.sh
[ -f "$EVALW" ] || { echo "[eval_farm] 找不到 eval_worker_exp.sh(仓库与盘上都没有),没有 worker 可起"; exit 1; }
RFLOG=$LOGD/eval_refill_slot${SLOT:-7}.log
_start_worker() { ( cd "$ROOT" && nohup bash "$EVALW" "$1" "$Q" >> "$LOGD/evalw_slot${SLOT:-7}_gpu$1.log" 2>&1 & ); }
_start_refill() { [ "${SLOT:-7}" = 7 ] || return 0
    ( cd "$ROOT" && nohup "$V" scripts/eval_refill_exp.py --write --watch 1200 >> "$RFLOG" 2>&1 & ); }
for g in 0 1 2 3 4 5 6 7; do _start_worker "$g"; done
_start_refill
while true; do sleep 300
    for g in 0 1 2 3 4 5 6 7; do pgrep -f "eval_worker_exp.sh $g $Q" >/dev/null 2>&1 || _start_worker "$g"; done
    if [ "${SLOT:-7}" = 7 ]; then pgrep -f "eval_refill_exp.py" >/dev/null 2>&1 || _start_refill; fi
    echo "[eval_farm] 值守 $(date "+%T") 队列剩 $(wc -l < "$Q/pending.txt" 2>/dev/null || echo ?) 条"
done
