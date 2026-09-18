#!/usr/bin/env bash
# 评测 worker:一张卡,一个进程,从 evalq_exp 队列里抢格子跑 eval_suite。
#
#   bash deploy/dsw/eval_worker_exp.sh <gpu> <队列目录>      # cwd = $SIMOPD_ROOT
#
# 重写于 2026-09-04。上一版住在 $D/eval_worker_exp.sh —— 数据盘上,**不在 git 里** ——
# 所以旧集群一没,它就没了。这一版进仓库,这才是那次教训的实际内容。
#
# 契约(由 scripts/eval_refill_exp.py 与 deploy/dlc/eval_farm.sh 反推,逐条对齐):
#   * 队列行 = "run step [benches] [maxtok]",整行读;pending.txt 由 refill 原子替换。
#   * 抢单 = mkdir $Q/claims/<run>__<step>,靠 mkdir 的原子性,失败即别人在跑。
#   * **心跳 = 每 5 分钟 touch 一次 claim 目录**;refill 按 "2 小时没心跳算死" 收尸。
#     这个数字不是随便定的:refill 的 scan() 里写死 7200s,两边必须对得上。
#   * 日志里 "exiting" / "sitting out" / "busy" 是 refill 的 workers() 用来判状态的
#     关键词,别改措辞。
#   * 产物由 eval_suite.py 落到 $SIMOPD_EVALS,文件名 <run>__<bench>__step<n>__seed<s>__*.parquet。
set -u

GPU=${1:?usage: eval_worker_exp.sh <gpu> <queue-dir>}
Q=${2:?usage: eval_worker_exp.sh <gpu> <queue-dir>}
CLAIMS=$Q/claims
FAILED=$Q/failed
CKPT=${CKPT_ROOT:-$SIMOPD_STORE/ckpt}/simopd
IDLE_SEC=${EVALW_IDLE_SEC:-120}          # 队列空了歇多久
BEAT_SEC=${EVALW_BEAT_SEC:-300}          # 心跳间隔,必须 << refill 的 7200
MAX_FAILS=${EVALW_MAX_FAILS:-3}          # 一个格子连挂几次就不再碰
mkdir -p "$CLAIMS" "$FAILED"

_say() { echo "[$(date '+%m-%d %H:%M:%S')] gpu$GPU $*"; }

# nvidia-smi 不认数字 CUDA_VISIBLE_DEVICES(AGENT.md section 7):设了它之后 `-i 3` 指的是
# 重映射后的第 3 张,不是物理第 3 张。整表拉出来按 index 精确匹配是唯一不会错的读法。
_gpu_used_mib() {
    env -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F'[, ]+' -v g="$GPU" '$1==g {print $2; exit}'
}

CUR=""                                   # 手上的 claim,给 trap 用
_release() { [ -n "$CUR" ] && rm -rf "$CLAIMS/$CUR"; CUR=""; }
# 失败路径必须 rm -rf,不能 rmdir:claim 目录里有 owner 文件,rmdir 删不动 —— 2026-08-22
# 终检实锤,722 个僵尸 claim 就是这么来的,把 651 行队列"合法地"清成了 0。
trap '_say "exiting (signal)"; _release; exit 0' TERM INT

_fail_count() { local f=$FAILED/$1; [ -f "$f" ] && wc -l < "$f" | tr -dc 0-9 || echo 0; }

# 失败后必须退避,否则 release 之后下一圈立刻从队首重抢同一格:三次失败在一秒内耗尽,
# 一次瞬时故障(NFS 抖一下、邻居 OOM)就把一个好 ckpt 永久毒化。退避让 MAX_FAILS 变成
# "几分钟里挂了三次",这才是 poisoned 该有的含义。
# 两条 local 分开写:`local n=$1 s=$((n*n*B))` 里的算术在 local 赋值 **之前** 展开,拿到的
# 是外层作用域残留的 n(抢单循环里那个),第一次退避会算成 0 秒。
_backoff() { local n=$1; local s=$(( n * n * ${EVALW_BACKOFF_SEC:-30} ))
             _say "backing off ${s}s before looking at the queue again"; sleep "$s"; }

_say "worker up, queue $Q, ckpt root $CKPT, evals -> ${SIMOPD_EVALS:-<unset!>}"

while true; do
    # ---- 卡还给我用吗 --------------------------------------------------------
    used=$(_gpu_used_mib)
    if [ -n "$used" ] && [ "$used" -gt "${EVALW_BUSY_MIB:-2000}" ]; then
        _say "sitting out: gpu $GPU busy (${used} MiB in use)"
        sleep "$IDLE_SEC"; continue
    fi

    # ---- 抢一格 --------------------------------------------------------------
    # 从队首往下试,抢到第一个算数。多个 worker 因此自然错开(A 拿第 1 行,B 在第 1 行
    # mkdir 失败就顺到第 2 行),不需要任何协调。
    line=""; cell=""
    while read -r run step rest; do
        [ -z "${run:-}" ] && continue
        case "$run" in '#'*) continue ;; esac
        c="${run}__${step}"
        n=$(_fail_count "$c")
        if [ "$n" -ge "$MAX_FAILS" ]; then continue; fi
        if mkdir "$CLAIMS/$c" 2>/dev/null; then
            cell=$c; line="$run $step $rest"; break
        fi
    done < <(cat "$Q/pending.txt" 2>/dev/null)

    if [ -z "$cell" ]; then
        _say "queue empty or fully claimed; idling ${IDLE_SEC}s"
        sleep "$IDLE_SEC"; continue
    fi
    CUR=$cell
    echo "$(hostname) gpu$GPU pid=$$ $(date -Is)" > "$CLAIMS/$cell/owner"

    set -- $line
    run=$1; step=$2; benches=${3:-}

    # ---- 权重在哪 ------------------------------------------------------------
    # ckpt_sync 只传 actor/huggingface(bf16 权重 + config + tokenizer),refill 也按这个
    # 目录判"这个 ckpt 能不能评",所以它就是评测入口;actor/ 是 FSDP 分片,vLLM 读不了。
    model=$CKPT/$run/global_step_$step/actor/huggingface
    if [ ! -d "$model" ]; then
        _say "!! $cell: no $model -- ckpt vanished or never finished writing; releasing"
        echo "$(date -Is) missing $model" >> "$FAILED/$cell"
        _release; _backoff "$(_fail_count "$cell")"; continue
    fi

    # ---- 跑 ------------------------------------------------------------------
    # eval_suite.py 是套件的权威定义(权重、tau=0.7/top_p=0.95、32k 预算、SIMOPD_SUITE_K),
    # worker 不重复实现任何一条。它自己按 newest() 跳过已完成的 benchmark,所以中途挂掉
    # 重来只补缺的那几个。--bench 把一格拆成子集,对应队列行的第三段(refill --grid light)。
    args=(scripts/eval_suite.py run --model "$model" --run-id "$run" --step "$step")
    [ -n "${SIMOPD_EVALS:-}" ] && args+=(--out-dir "$SIMOPD_EVALS")
    [ -n "$benches" ] && args+=(--bench "$benches")
    _say "start $cell${benches:+ [$benches]}  k=${SIMOPD_SUITE_K:-32}"

    t0=$SECONDS
    CUDA_VISIBLE_DEVICES=$GPU python "${args[@]}" &
    job=$!
    # 心跳:eval 在后台跑,这里每 BEAT_SEC 摸一次 claim。摸的是目录本身的 mtime,
    # refill 读的也是它 —— 一格 AIME avg@8 要跑上小时,不摸就会被当成死人收尸。
    while kill -0 "$job" 2>/dev/null; do
        touch "$CLAIMS/$cell" 2>/dev/null
        sleep "$BEAT_SEC" &
        wait $! 2>/dev/null
    done
    wait "$job"; rc=$?
    dt=$((SECONDS - t0))

    if [ "$rc" -eq 0 ]; then
        _say "done  $cell in ${dt}s"
        rm -f "$FAILED/$cell"
    else
        # 失败也要放格子回队列(下一轮 refill 会重排),但记一笔:同一格连挂 MAX_FAILS 次
        # 之后所有 worker 都跳过它,否则一个真坏的 ckpt 会在整个 farm 里轮着烧卡。
        echo "$(date -Is) rc=$rc after ${dt}s on $(hostname) gpu$GPU" >> "$FAILED/$cell"
        n=$(_fail_count "$cell")
        _say "!! FAIL $cell rc=$rc after ${dt}s (fail $n/$MAX_FAILS)"
        [ "$n" -ge "$MAX_FAILS" ] && _say "!! $cell poisoned: $MAX_FAILS failures, no worker will retry it"
        _release; _backoff "$n"; continue
    fi
    _release
done
