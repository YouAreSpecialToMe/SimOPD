#!/usr/bin/env bash
# 单卡评测 worker:从共享队列认领一格(run × step),跑 eval_suite,放开。
#
#   bash deploy/dlc/eval_worker_exp.sh <gpu-index> [队列目录]
#
# 2026-09-02 重写。上一份从未提交进 git、只躺在 $D 上,随旧集群一起没了(AGENT.md §2.6);
# 而 eval_farm.sh / worker.sh / corr_wave_fleet.sh 三处都指着它,缺了就是「卡先闲着」——
# 静默不评。所以这一份放在仓库里,盘上那份(如果哪天又有)只作后备。
#
# 契约不是猜的,全部从现存代码反推:
#   队列  $Q/pending.txt   每行 "run step [benches] [maxtok]",由 scripts/eval_refill_exp.py
#                          原子替换写出(os.replace),整行读不会读到半截;**行序即优先级**,
#                          从上往下取(refill 的 prio():vanilla_corr → 本轮新格 → 其余 → 补漏)。
#   认领  $Q/claims/<run>__<step>/   mkdir 原子抢单,先到先得;里面写 owner。
#   心跳  每 300 秒 touch **claim 目录本身**。收尸方读的是目录 mtime
#         (refill: os.path.getmtime;worker.sh: stat -c %Y),touch 里面的文件不更新目录
#         mtime,那样心跳等于没打,2 小时后自己被别人收尸。
#   收尸  超过 REAP_SILENT_SEC(7200)没心跳 = 主人死了,任何一方都可回收。
#   产物  $EVALS/<run>__<bench>__step<n>__seed<s>__<stamp>.parquet
#   干活  scripts/eval_suite.py run —— 它自己按 SUITE 决定每基准的 k(SIMOPD_SUITE_K,
#         本轮 8)、温度 0.7 / top_p 0.95 / 32768 token,并且**按基准续跑**:一格中途死了,
#         已出的基准不会重跑。所以 worker 不需要自己拆基准,重试也很便宜。
#
# 三个会咬人的地方,写死在代码里别再犯:
#   1) 删 claim 只能 rm -rf,**不能 rmdir** —— 目录里有 owner 文件,rmdir 删不动却不报错,
#      2026-08-23 就是这么攒出 722 个僵尸 claim,把 651 行队列「合法地」清成 0 行。
#   2) 判「我这张卡空不空」必须**按 index 精确取行**。nvidia-smi 不认数字形式的
#      CUDA_VISIBLE_DEVICES,它照样报全部卡的全局 index;上一份用 head -1 读到的永远是
#      GPU0,于是同 pod 八个 worker 一起误判卡忙、集体坐等。
#   3) 日志里的状态词是 refill 的 workers() 在 grep 的:等卡要出现 "sitting out"/"busy",
#      退出要出现 "exiting"。改词等于让产能面板瞎掉。
#
# 测试口(deploy/dlc/test_eval_worker.sh 用):
#   EVALW_DRY=1     不碰 GPU、不真跑评测(假装成功并落一个空 parquet)
#   EVALW_PASSES=N  跑 N 轮队列后退出(0 = 永远)
#   EVALW_IDLE_SEC  空队列/等卡时的睡眠(默认 120)
set -uo pipefail

GPU=${1:?用法: eval_worker_exp.sh <gpu-index> [队列目录]}
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
D=${SIMOPD_STORE:-${D:-/mgfs/shared/Group_GY/changhao/simopd_data}}
Q=${2:-$D/evalq_exp}
EVALS=${SIMOPD_EVALS:-$D/evals}
REAP_SILENT_SEC=${REAP_SILENT_SEC:-7200}
HEARTBEAT_SEC=${HEARTBEAT_SEC:-300}
IDLE=${EVALW_IDLE_SEC:-120}
PASSES=${EVALW_PASSES:-0}
DRY=${EVALW_DRY:-0}
BENCHES_ALL="aime24 aime25 amc23 minerva math500"

cd "$ROOT" || exit 1
# 顶上 source:2026-08-23 的教训是容器注入了 VLLM_USE_MODELSCOPE=True 而 venv 没装
# modelscope,vLLM 一 import 就死,32 个 worker 把整条队列烧成 FAILED。eval_farm.sh 已经
# 做过一次,这里再做一次 —— worker 也可能被 worker.sh / corr_wave_fleet.sh 直接拉起。
# shellcheck disable=SC1091
[ -f ./simopd_env.sh ] && . ./simopd_env.sh
export VLLM_USE_MODELSCOPE=False VERL_USE_MODELSCOPE=False
PY=${SIMOPD_PY:-$ROOT/simopd/bin/python}
[ -x "$PY" ] || PY=$(command -v python3)

log() { echo "[evalw gpu$GPU $(date +%T)] $*"; }

# GNU 与 BSD 的 stat 不同型。worker.sh 里写死了 `stat -c %Y`(集群是 Linux,没问题),
# 但那也意味着这份脚本在开发机上根本跑不起来、于是从来没人在提交前测过它 —— 上一份
# 就是这么进的坑。探测一次,让它在 mac 上也能跑测试。
if stat -c %Y . >/dev/null 2>&1; then _mtime() { stat -c %Y "$1"; }
else                                  _mtime() { stat -f %m "$1"; }; fi

# ---- 状态 --------------------------------------------------------------------
CLAIM=""          # 当前持有的 claim 目录(空 = 没持有)
HB_PID=""
# 本进程内失败过的格,避免热重试同一格。用字符串集合而不是关联数组:后者要 bash 4,
# 而开发机(macOS)是 3.2 —— 一个只能在集群上运行的脚本就是一个不会被测试的脚本。
# 同 worker.sh 里 `case "$launched_gpus" in *" $g "*)` 的写法。
FAIL1=" "; DEAD=" "
_fail_n()    { case "$DEAD" in *" $1 "*) echo 2; return;; esac
               case "$FAIL1" in *" $1 "*) echo 1; return;; esac
               echo 0; }
_fail_bump() { if [ "$(_fail_n "$1")" = 0 ]; then FAIL1="$FAIL1$1 "; else DEAD="$DEAD$1 "; fi; }
_fail_kill() { DEAD="$DEAD$1 "; }

release() {       # 放开当前 claim。rm -rf,不是 rmdir(见抬头 1)
    [ -n "$HB_PID" ] && { kill "$HB_PID" 2>/dev/null; wait "$HB_PID" 2>/dev/null; HB_PID=""; }
    [ -n "$CLAIM" ] && { rm -rf "$CLAIM"; CLAIM=""; }
}
# 被 pkill(worker.sh 的 eviction 会 pkill 本脚本)或正常退出时,别把格锁在手里带走
trap 'log "exiting (signal)"; release; exit 0' INT TERM HUP
trap 'release' EXIT

# ---- GPU:按 index 精确取(见抬头 2)-------------------------------------------
gpu_used_mib() {
    # env -u:万一 CUDA_VISIBLE_DEVICES 是 UUID 形式,nvidia-smi 真会过滤,index 就对不上了
    env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader 2>/dev/null \
    | awk -F, -v g="$GPU" '{gsub(/[^0-9]/,"",$1); gsub(/[^0-9]/,"",$2);
                            if ($1+0 == g+0) {print $2+0; found=1}}
                           END {if (!found) print -1}'
}
gpu_free() {
    [ "$DRY" = 1 ] && return 0
    local used; used=$(gpu_used_mib)
    [ "$used" -ge 0 ] 2>/dev/null || { log "nvidia-smi 没有 index $GPU 这张卡 -> exiting"; exit 1; }
    [ "$used" -lt "${GPU_FREE_MIB:-500}" ]
}

# ---- 队列 --------------------------------------------------------------------
cell_complete() {  # RUN STEP BENCH...  -> 0 当这些基准的产物都在
    local run=$1 step=$2 b; shift 2
    for b in "$@"; do
        compgen -G "$EVALS/${run}__${b}__step${step}__seed*.parquet" > /dev/null || return 1
    done
    return 0
}

try_claim() {      # RUN STEP -> 0 抢到(并已起心跳)
    local run=$1 step=$2 dir="$Q/claims/${run}__${step}" age
    mkdir -p "$Q/claims" 2>/dev/null
    if ! mkdir "$dir" 2>/dev/null; then
        # 已被认领。只有「陈掉且没跑完」才回收 —— 契约就是 2 小时没心跳算主人死了
        # (refill 的注释:「claim 过期由 worker 侧处理」)。活的一律尊重。
        age=$(( $(date +%s) - $(_mtime "$dir" 2>/dev/null || echo 0) ))
        if [ "$age" -gt "$REAP_SILENT_SEC" ]; then
            log "claim ${run}@${step} 已静默 ${age}s -> 收尸重抢"
            rm -rf "$dir"
            mkdir "$dir" 2>/dev/null || return 1
        else
            return 1
        fi
    fi
    printf 'host=%s pid=%s gpu=%s started=%s\n' "$(hostname)" "$$" "$GPU" "$(date +%FT%T)" > "$dir/owner"
    CLAIM=$dir
    # 心跳:touch 目录本身(见抬头,touch 里面的文件不更新目录 mtime)。
    # `trap - EXIT ...` 是必须的:子壳会继承父壳的 EXIT trap,而那个 trap 是 release ——
    # 心跳一退出就会把父壳正在跑的那一格 rm -rf 掉,并且 kill 掉「自己」。
    ( trap - EXIT INT TERM HUP
      while :; do sleep "$HEARTBEAT_SEC"; touch "$dir" 2>/dev/null || exit 0; done ) &
    HB_PID=$!
    return 0
}

run_cell() {       # RUN STEP BENCH...  -> 0 成功
    local run=$1 step=$2; shift 2
    local benches="$*" model="$D/ckpt/simopd/${run}/global_step_${step}/actor/huggingface"
    local bench_arg=()
    # 整套五基准时不传 --bench,让 eval_suite 走它自己的默认(与已入库产物同口径)
    [ "$benches" = "$BENCHES_ALL" ] || bench_arg=(--bench "$(echo "$benches" | tr ' ' ',')")
    if [ ! -d "$model" ]; then
        log "跳过 ${run}@${step}:$model 不存在(ckpt 没落全或已被清理)"
        return 2                       # 2 = 别重试,不是我们的错
    fi
    log "开跑 ${run}@${step} [${benches}] k=${SIMOPD_SUITE_K:-32} -> $EVALS"
    if [ "$DRY" = 1 ]; then
        local b; for b in $benches; do
            : > "$EVALS/${run}__${b}__step${step}__seed0__dry.parquet"
        done
        return 0
    fi
    # < /dev/null:主循环的 stdin 是 pending.txt(while read 在读它)。子进程若读一口
    # stdin 就会吃掉队列剩下的行,这一轮之后的格全部凭空消失 —— 而且不会报错。
    CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/eval_suite.py run \
        --run-id "$run" --step "$step" --model "$model" \
        --out-dir "$EVALS" "${bench_arg[@]+"${bench_arg[@]}"}" < /dev/null
}

# ---- 主循环 ------------------------------------------------------------------
mkdir -p "$EVALS" "$Q/claims" 2>/dev/null
log "启动:队列 $Q,产物 $EVALS,python $PY,k=${SIMOPD_SUITE_K:-32}${DRY:+ (DRY=$DRY)}"
pass=0
while :; do
    pass=$((pass + 1))
    if ! gpu_free; then
        log "gpu $GPU busy($(gpu_used_mib) MiB) -- sitting out ${IDLE}s"
        sleep "$IDLE"; [ "$PASSES" -gt 0 ] && [ "$pass" -ge "$PASSES" ] && break; continue
    fi
    if [ ! -s "$Q/pending.txt" ]; then
        log "队列空 -- sitting out ${IDLE}s"
        sleep "$IDLE"; [ "$PASSES" -gt 0 ] && [ "$pass" -ge "$PASSES" ] && break; continue
    fi

    took=0
    # 整份读进来再遍历:refill 用 os.replace 原子换文件,遍历中途被换掉也不会读到半截,
    # 我们这一轮按旧快照走完,下一轮自然看到新的。
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        # set -f:拆字段要词分割,但不要通配符展开 —— 队列行里一个 * 会被展成目录内容
        set -f; set -- $line; set +f
        run=${1:-}; step=${2:-}; benches=${3:-}; maxtok=${4:-}
        [ -n "$run" ] && [ -n "$step" ] || continue
        if [ -n "$maxtok" ]; then
            # 队列格式留了这一列,但 eval_suite 把 32768 钉死(AGENT.md §2.2b:截断率是
            # 头号仪表,降帽等于改测量对象)。refill 从不写它;真写了也只警告不照做。
            log "WARN ${run}@${step}: 队列行带 maxtok=$maxtok,已忽略(suite 钉死 32768)"
        fi
        if [ -n "$benches" ]; then benches=${benches//,/ }; else benches=$BENCHES_ALL; fi

        key="${run}__${step}"
        [ "$(_fail_n "$key")" -ge 2 ] && continue
        cell_complete "$run" "$step" $benches && continue
        try_claim "$run" "$step" || continue

        # 抢到之后再查一次:抢单与上一个 worker 写完产物之间有窗口
        if cell_complete "$run" "$step" $benches; then
            log "${run}@${step} 在抢单窗口内已被别人评完,放开"
            release; continue
        fi
        took=1
        if run_cell "$run" "$step" $benches; then
            if cell_complete "$run" "$step" $benches; then
                log "完成 ${run}@${step}"
            else
                # eval_suite 退 0 却没落齐产物:当失败处理,否则格会被当成做完而永远漏掉
                log "ERROR ${run}@${step}: eval_suite 退出 0 但产物不齐,记为失败"
                _fail_bump "$key"
            fi
        else
            rc=$?
            if [ "$rc" = 2 ]; then _fail_kill "$key"; else _fail_bump "$key"; fi
            log "失败 ${run}@${step} rc=$rc(本进程累计 $(_fail_n "$key") 次,满 2 次本进程不再试)"
        fi
        release          # 无论成败都放开:成了产物在盘上,败了让别人/下轮再试
        break            # 回到外层重新读队列(优先级可能已变)
    done < "$Q/pending.txt"

    if [ "$took" = 0 ]; then
        log "队列里没有可认领的格(都在跑或已完成) -- sitting out ${IDLE}s"
        sleep "$IDLE"
    fi
    [ "$PASSES" -gt 0 ] && [ "$pass" -ge "$PASSES" ] && break
done
log "exiting (passes=$pass)"
