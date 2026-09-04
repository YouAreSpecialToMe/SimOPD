#!/usr/bin/env bash
# eval_worker_exp.sh 的无 GPU 证明台。任何机器都能跑(mac / 登录节点 / pod):
#
#   bash deploy/dlc/test_eval_worker.sh
#
# 真的是:队列解析、抢单(mkdir 原子)、心跳打在目录上、陈 claim 收尸、带 owner 文件的
# claim 能被删掉(rmdir 删不动的那个坑)、按 index 判卡、--bench 拆分、失败退避、日志里
# refill 要 grep 的状态词。假的:GPU(PATH 上的假 nvidia-smi)、评测(假 python,只记参数
# 并落空 parquet)。证明不了:真 vLLM 起得来 —— 那要等第一条 lane。
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
W="$ROOT/deploy/dlc/eval_worker_exp.sh"
SB=$(mktemp -d /tmp/simopd_evalw.XXXX)
FAILS=0
say() { echo; echo "== $*"; }
chk() { if eval "$2"; then echo "  PASS  $1"; else echo "  FAIL  $1"; FAILS=$((FAILS+1)); fi; }

mkdir -p "$SB/bin" "$SB/evalq_exp/claims" "$SB/evals" "$SB/logs"
# 假卡:index 0 占满(70000 MiB),index 1..7 空闲(4 MiB)。判 index 0 空闲 = 踩了 head -1 的坑。
cat > "$SB/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
echo "0, 70000 MiB"
for i in 1 2 3 4 5 6 7; do echo "$i, 4 MiB"; done
EOF
# 假 python:记下收到的完整参数,按 --benchmarks/--bench 落空 parquet,可选 sleep
cat > "$SB/bin/fakepy" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$CALLS"
run=""; step=""; out=""; bench=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) run=$2; shift 2;; --step) step=$2; shift 2;;
    --out-dir) out=$2; shift 2;; --bench) bench=$2; shift 2;;
    *) shift;;
  esac
done
[ -n "${FAKE_SLEEP:-}" ] && sleep "$FAKE_SLEEP"
[ "${FAKE_RC:-0}" = 0 ] || exit "$FAKE_RC"
[ -n "$bench" ] || bench="aime24,aime25,amc23,minerva,math500"
IFS=,; for b in $bench; do : > "$out/${run}__${b}__step${step}__seed0__t.parquet"; done
EOF
chmod +x "$SB/bin/"*
export PATH="$SB/bin:$PATH"
CALLS="$SB/calls.txt"; : > "$CALLS"; export CALLS
# `wc -l` 在 BSD 上输出带前导空格("       1"),字符串比较全部对不上 —— 测试台自己的坑,
# 头一遍就踩了。awk 的 NR 是干净的。
ncalls() { awk 'END{print NR}' "$CALLS"; }

# ckpt 树:两条 run 有权重,一条没有
for r in runA_s0_16k runB_s0_16k; do mkdir -p "$SB/ckpt/simopd/$r/global_step_25/actor/huggingface"; done

wrun() {  # GPU 额外env... -> 跑一轮 worker,输出到 $SB/out.txt
    local gpu=$1; shift
    env "$@" SIMOPD_STORE="$SB" SIMOPD_EVALS="$SB/evals" SIMOPD_PY="$SB/bin/fakepy" \
        EVALW_PASSES=1 EVALW_IDLE_SEC=1 HEARTBEAT_SEC=1 \
        bash "$W" "$gpu" "$SB/evalq_exp" > "$SB/out.txt" 2>&1
}
q() { printf '%s\n' "$@" > "$SB/evalq_exp/pending.txt"; }

say "A 空队列:sitting out + exiting(refill 的产能面板 grep 这两个词)"
: > "$SB/evalq_exp/pending.txt"; wrun 3
chk "空队列说 sitting out"      'grep -q "sitting out" "$SB/out.txt"'
chk "退出说 exiting"            'grep -q "exiting" "$SB/out.txt"'

say "B 正常一格:抢单 -> 跑 -> 产物齐 -> claim 放开"
q "runA_s0_16k 25"; : > "$CALLS"; wrun 3
chk "调了一次评测"              '[ "$(ncalls)" = 1 ]'
chk "传了 run-id/step/model/out-dir" 'grep -q "eval_suite.py run --run-id runA_s0_16k --step 25 --model $SB/ckpt/simopd/runA_s0_16k/global_step_25/actor/huggingface --out-dir $SB/evals" "$CALLS"'
chk "整套五基准时不传 --bench"  '! grep -q -- "--bench" "$CALLS"'
chk "五个产物都落了"            '[ "$(ls "$SB/evals" | grep -c "^runA_s0_16k__.*__step25__")" = 5 ]'
chk "claim 已放开"              '[ ! -d "$SB/evalq_exp/claims/runA_s0_16k__25" ]'
chk "日志报完成"                'grep -q "完成 runA_s0_16k@25" "$SB/out.txt"'

say "C 已完成的格不再抢(cell_complete 短路)"
: > "$CALLS"; wrun 3
chk "没有再调评测"              '[ "$(ncalls)" = 0 ]'
chk "说了没有可认领的格"        'grep -q "没有可认领的格" "$SB/out.txt"'

say "D 轻档网格:队列行给了基准子集 -> 只传那几个"
rm -f "$SB/evals"/*.parquet; q "runA_s0_16k 25 amc23,minerva,math500"; : > "$CALLS"; wrun 3
chk "--bench 精确传三项"        'grep -q -- "--bench amc23,minerva,math500" "$CALLS"'
chk "只落三个产物"              '[ "$(ls "$SB/evals" | grep -c "step25__")" = 3 ]'
chk "子集齐了就算完成"          'grep -q "完成 runA_s0_16k@25" "$SB/out.txt"'

say "E 按 index 判卡(head -1 的坑):index 0 忙、index 3 空"
rm -f "$SB/evals"/*.parquet; q "runA_s0_16k 25"; : > "$CALLS"; wrun 0
chk "gpu0 忙 -> sitting out"    'grep -q "gpu 0 busy" "$SB/out.txt"'
chk "gpu0 没有开跑"             '[ "$(ncalls)" = 0 ]'
wrun 3
chk "gpu3 空 -> 照跑"           '[ "$(ncalls)" = 1 ]'

say "F 活 claim 要尊重,陈 claim(>2h 没心跳)要收尸"
rm -f "$SB/evals"/*.parquet; q "runA_s0_16k 25" "runB_s0_16k 25"
mkdir -p "$SB/evalq_exp/claims/runA_s0_16k__25"; echo "host=other" > "$SB/evalq_exp/claims/runA_s0_16k__25/owner"
: > "$CALLS"; wrun 3
chk "活 claim 被跳过,改跑 runB" 'grep -q "run-id runB_s0_16k" "$CALLS" && ! grep -q "run-id runA_s0_16k" "$CALLS"'
chk "别人的 claim 没被动"        '[ -d "$SB/evalq_exp/claims/runA_s0_16k__25" ]'
python3 - "$SB/evalq_exp/claims/runA_s0_16k__25" <<'PY'
import os, sys, time
os.utime(sys.argv[1], (time.time()-9000, time.time()-9000))   # 2.5 小时没心跳
PY
rm -f "$SB/evals"/*.parquet; q "runA_s0_16k 25"; : > "$CALLS"; wrun 3
chk "陈 claim 被收尸并重抢"      'grep -q "已静默.*收尸重抢" "$SB/out.txt"'
chk "带 owner 文件也删得掉(rmdir 删不动)" '[ ! -d "$SB/evalq_exp/claims/runA_s0_16k__25" ]'
chk "收尸后真的把这格跑了"        '[ "$(ncalls)" = 1 ]'

say "G 心跳打在 claim 目录本身(打在里面的文件上不更新目录 mtime)"
rm -f "$SB/evals"/*.parquet; q "runA_s0_16k 25"; : > "$CALLS"
( env SIMOPD_STORE="$SB" SIMOPD_EVALS="$SB/evals" SIMOPD_PY="$SB/bin/fakepy" FAKE_SLEEP=4 \
      EVALW_PASSES=1 EVALW_IDLE_SEC=1 HEARTBEAT_SEC=1 \
      bash "$W" 3 "$SB/evalq_exp" > "$SB/out_hb.txt" 2>&1 ) &
HBW=$!
CD="$SB/evalq_exp/claims/runA_s0_16k__25"
for _ in 1 2 3 4 5 6 7 8 9 10; do [ -d "$CD" ] && break; sleep 0.3; done
M1=$(python3 -c "import os,sys;print(int(os.path.getmtime(sys.argv[1])))" "$CD" 2>/dev/null || echo 0)
sleep 2.5
M2=$(python3 -c "import os,sys;print(int(os.path.getmtime(sys.argv[1])))" "$CD" 2>/dev/null || echo 0)
wait $HBW
chk "抢单期间 claim 目录被 touch 过"  '[ "$M1" != 0 ] && [ "$M2" -gt "$M1" ]'
chk "跑完后 claim 放开"               '[ ! -d "$CD" ]'
chk "心跳子壳没有误删 claim(跑成功了)" 'grep -q "完成 runA_s0_16k@25" "$SB/out_hb.txt"'

say "H 两个 worker 抢同一格:只有一个真跑"
rm -f "$SB/evals"/*.parquet; q "runA_s0_16k 25"; : > "$CALLS"
for g in 3 4; do
  ( env SIMOPD_STORE="$SB" SIMOPD_EVALS="$SB/evals" SIMOPD_PY="$SB/bin/fakepy" FAKE_SLEEP=2 \
        EVALW_PASSES=1 EVALW_IDLE_SEC=1 HEARTBEAT_SEC=1 \
        bash "$W" "$g" "$SB/evalq_exp" > "$SB/out_c$g.txt" 2>&1 ) &
done
wait
chk "mkdir 抢单:恰好跑了一次"   '[ "$(ncalls)" = 1 ]'
chk "另一个说没有可认领的格"     'grep -lq "没有可认领的格" "$SB"/out_c3.txt "$SB"/out_c4.txt'

say "I ckpt 不存在 -> 跳过且不反复重试"
rm -f "$SB/evals"/*.parquet; q "ghost_s0_16k 25"; : > "$CALLS"
env SIMOPD_STORE="$SB" SIMOPD_EVALS="$SB/evals" SIMOPD_PY="$SB/bin/fakepy" \
    EVALW_PASSES=3 EVALW_IDLE_SEC=1 HEARTBEAT_SEC=1 \
    bash "$W" 3 "$SB/evalq_exp" > "$SB/out_ghost.txt" 2>&1
chk "报了 ckpt 不存在"           'grep -q "不存在" "$SB/out_ghost.txt"'
chk "没有调评测"                 '[ "$(ncalls)" = 0 ]'
chk "三轮里只试了一次(不热重试)" '[ "$(grep -c "不存在" "$SB/out_ghost.txt")" = 1 ]'
chk "没把 ghost 的 claim 留下"    '[ ! -d "$SB/evalq_exp/claims/ghost_s0_16k__25" ]'

say "J 评测失败 -> claim 放开、计一次失败、满 2 次本进程不再试"
rm -f "$SB/evals"/*.parquet; q "runA_s0_16k 25"; : > "$CALLS"
env SIMOPD_STORE="$SB" SIMOPD_EVALS="$SB/evals" SIMOPD_PY="$SB/bin/fakepy" FAKE_RC=1 \
    EVALW_PASSES=4 EVALW_IDLE_SEC=1 HEARTBEAT_SEC=1 \
    bash "$W" 3 "$SB/evalq_exp" > "$SB/out_fail.txt" 2>&1
chk "试了 2 次就不再试"          '[ "$(grep -c "失败 runA_s0_16k@25" "$SB/out_fail.txt")" = 2 ]'
chk "失败后 claim 放开了"        '[ ! -d "$SB/evalq_exp/claims/runA_s0_16k__25" ]'

say "K 队列行序即优先级 + maxtok 列只警告不照做"
rm -f "$SB/evals"/*.parquet; q "runB_s0_16k 25" "runA_s0_16k 25"; : > "$CALLS"; wrun 3
chk "取的是第一行 runB"          'grep -q "run-id runB_s0_16k" "$CALLS"'
rm -f "$SB/evals"/*.parquet; q "runA_s0_16k 25 aime24 65536"; : > "$CALLS"; wrun 3
chk "maxtok 有警告"              'grep -q "maxtok=65536,已忽略" "$SB/out.txt"'
chk "仍按 --bench aime24 跑"     'grep -q -- "--bench aime24" "$CALLS"'

echo
echo "RESULT: $([ $FAILS -eq 0 ] && echo "ALL PASS" || echo "$FAILS FAILURE(S)")   (sandbox: $SB)"
[ $FAILS -eq 0 ] && rm -rf "$SB"
exit $([ $FAILS -eq 0 ] && echo 0 || echo 1)
