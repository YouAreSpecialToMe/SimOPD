#!/usr/bin/env bash
#SBATCH --job-name=simopd-gkdgen
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
#SBATCH --time=12:00:00
# 日志路径走 Slurm 原生的 SBATCH_OUTPUT/SBATCH_ERROR 环境变量(见 simopd_env.example.sh)。
# 不在这里写 #SBATCH --output=$VAR —— sbatch 不展开指令行里的 shell 变量,那样只会
# 在当前目录建出一个名叫 '$VAR' 的目录。
#
# A-axis unlock: the teacher's off-policy demonstrations, one per training prompt.
# a1 / a3 / a4 read it as SIMOPD_GKD_CACHE, a5 reads the keys-only .dry as
# SIMOPD_A5_KEYS. The previous cluster's copy was lost with that cluster's share.
#
# Scale (docs/v2-inloop-decline-probe.md 7.1): the teacher's median demo on THESE
# prompts is 6449 tokens and 8.6% hit the 16k cap -- ~100M tokens of generation, not
# the 722-token math500 teacher. Eight shards, one per GPU; SamplingParams carries a
# per-request seed so N shards are token-identical to one single-GPU pass.
set -uo pipefail

source $SIMOPD_ROOT/simopd_env.sh
cd "$SIMOPD_ROOT"

OUT=${OUT:-$DATA_DIR/gkd_offpolicy.parquet}
TRAIN=$DATA_DIR/train.parquet
N=${SHARDS:-8}
LIMIT=${LIMIT:-}          # LIMIT=64 for a smoke pass
[ -n "$LIMIT" ] && LIMIT="--limit $LIMIT"

echo "=== gkd cache -> $OUT   ($(hostname), $(date -Is))"

# Cache preflight: fail before burning the allocation, not 40 minutes in.
python - <<'PYEOF' || exit 1
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-1.7B-Base"):
    try:
        snapshot_download(m, local_files_only=True)
    except Exception:
        raise SystemExit(f"FATAL: {m} not in local HF cache -- run scripts/fetch_assets.py first")
print("preflight ok: teacher + student resolve from the local cache")
PYEOF

# ---- keys-only pass for a5 (CPU, minutes). The script appends .dry itself, so it
# ---- cannot clobber the real cache with [1,2,3] stub rows.
if [ -s "$OUT.dry" ]; then
    echo "== $OUT.dry exists, skipping the dry pass"
else
    python scripts/gen_offpolicy.py --dry --out "$OUT" --train-parquet "$TRAIN" $LIMIT \
        || { echo "FATAL: dry pass failed -- a5 has no keys file"; exit 1; }
fi

# ---- the real thing -----------------------------------------------------------
if [ -s "$OUT" ]; then
    echo "== $OUT exists, nothing to generate"
    exit 0
fi

pids=()
for i in $(seq 0 $((N - 1))); do
    if [ -s "$OUT.shard${i}of${N}" ]; then
        echo "== shard $i already on disk, skipping"; continue
    fi
    CUDA_VISIBLE_DEVICES=$i nohup python scripts/gen_offpolicy.py \
        --shard "$i/$N" --out "$OUT" --train-parquet "$TRAIN" $LIMIT \
        > "$OPD_LOGS/gkdgen_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "== shard $i -> gpu $i (pid ${pids[-1]})"
done

rc=0
# set -u makes "${pids[@]}" an unbound-variable error when every shard was already
# on disk, which is exactly the resubmit-after-partial-failure case.
for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || rc=1; done
if [ "$rc" != 0 ]; then
    echo "FATAL: a shard failed -- see logs/gkdgen_shard*.log. Finished shards are kept;"
    echo "       resubmitting skips them, so only the failures re-run."
    exit 1
fi

# --merge re-checks key uniqueness and that the shards agree on provenance.
python scripts/gen_offpolicy.py --merge "$N" --out "$OUT" --train-parquet "$TRAIN" || exit 1

python - "$OUT" "$TRAIN" <<'PYEOF'
import sys, pandas as pd
out, train = sys.argv[1], sys.argv[2]
c, t = pd.read_parquet(out), pd.read_parquet(train)
assert c["prefix_hash"].is_unique, "duplicate keys survived the merge"
print(f"rows {len(c)} (train {len(t)}; the gap is duplicate prompts sharing a row)")
print(f"resp len p50 {c.response_tokens.median():.0f}  p90 {c.response_tokens.quantile(.9):.0f}  "
      f"capped {(c.response_tokens >= c.gen_max_tokens.iloc[0]).mean():.1%}")
print(f"teacher {c.teacher.iloc[0]}  cap {c.gen_max_tokens.iloc[0]}  tau {c.temperature.iloc[0]}")
PYEOF

rm -f "$OUT".shard*of"$N"
echo "GKDGEN_DONE $(date -Is)"
