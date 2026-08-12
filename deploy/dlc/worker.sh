#!/usr/bin/env bash
# SimOPD DLC fleet worker -- one instance per 8-GPU worker pod of a single big
# pytorchjob. Written 2026-08-13 for the 512-card shape.
#
# The design inverts the cluster's usual DLC pattern (tools/dlc/exp*.sh: one big
# Ray job, rank 0 drives, workers `ray start --block`). Our workload is the
# opposite shape -- hundreds of INDEPENDENT 2-GPU lanes plus single-GPU eval
# workers -- so there is no master, no cross-node NCCL, no rendezvous. Every
# worker runs the same supervisor loop against the shared filesystem, and the
# campaign's existing governance (manifest pool + mkdir claims + fingerprints +
# banked checkpoints) already makes that multi-node-safe: DLC here is just a
# provisioner of 8-GPU boxes, exactly what DSW was.
#
# What each pass of the loop does, in priority order:
#   1. TRAINING FIRST -- deploy/campaign.sh claims manifest rows onto free GPU
#      pairs (4 lanes/worker). Preempted/restarted pods resume from the last
#      25-step bank; the fingerprint refuses config drift, as always.
#   2. EVAL BACKFILL -- every GPU that holds no lane gets a single-GPU
#      eval_worker on the shared claim queue. This is the utilization
#      guarantee: when training rows drain, all 8 cards flip to eval within one
#      pass; when new waves land in the manifest, step 3 flips them back.
#   3. PRIORITY EVICTION -- if startable training rows exist but fewer than 2
#      GPUs are free, evict eval workers (cheapest work first: their items are
#      resumable by construction -- artifacts are per-benchmark and complete()
#      skips finished ones).
#
# Identity: DLC pod hostnames are EPHEMERAL across restarts, so machines are
# keyed by worker rank (stable for the life of the job): d<rank>. The boot
# upserts this worker's row in MACHINE_MAP under a lock -- without the upsert,
# a restarted pod trips campaign.sh's identity guard ("d3 is already registered
# to <old hostname>"), which is the guard doing its job against a rule DSW
# never needed.
set -uo pipefail

EXP_ROOT=${EXP_ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
DATA=${DATA:-/mgfs/shared/Group_GY/changhao/simopd_data}
EVALQ=${EVALQ:-$DATA/evalq_exp}
LOOP_SEC=${LOOP_SEC:-900}
# Domains this fleet serves, in priority order. Each is a (manifest, claim
# namespace, env triple); math is the default namespace and needs no overrides.
# A domain whose train.parquet is missing is skipped loudly, not fatally -- the
# IF set is gated on a license token and may land later than code.
DOMAINS=${DOMAINS:-"math if code"}

domain_env() {  # print export statements for one domain; empty for math
    case "$1" in
        if)   echo "export MANIFEST=configs/campaign_if.tsv CLAIM_DIR=.campaign_if"
              echo "export DATA_DIR=$DATA/simopd_if VAL_FILE_BASENAME=ifeval.parquet"
              echo "export MAX_RESPONSE_LENGTH=4096"
              echo "export CUSTOM_REWARD_PATH=$EXP_ROOT/src/simopd/domain_reward.py" ;;
        code) echo "export MANIFEST=configs/campaign_code.tsv CLAIM_DIR=.campaign_code"
              echo "export DATA_DIR=$DATA/simopd_code VAL_FILE_BASENAME=val_holdout.parquet"
              echo "export MAX_RESPONSE_LENGTH=4096" ;;
        math) : ;;
    esac
}

domain_data_dir() { case "$1" in if) echo "$DATA/simopd_if";; code) echo "$DATA/simopd_code";; math) echo "$DATA/simopd_math";; esac; }
domain_claim_dir() { case "$1" in if) echo ".campaign_if";; code) echo ".campaign_code";; math) echo ".campaign";; esac; }

RANK=${MLP_ROLE_INDEX:-${MLP_WORKER_RACK_RANK_INDEX:-${RANK:-0}}}
MACHINE="d${RANK}"
LOG_DIR="$DATA/dlc_logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/${MACHINE}_$(date +%Y%m%d_%H%M%S).log") 2>&1
echo "== dlc worker $MACHINE on $(hostname) $(date -u +%FT%TZ) =="

cd "$EXP_ROOT"
. ./simopd_env.sh

# ---- per-domain namespaces + rank-keyed identity upsert ----------------------
# Each domain has its own claim dir (own BATCH_TAG -> run names <arm>_s<s>_<tag>,
# so checkpoints/fingerprints never collide across domains) and its own
# MACHINE_MAP. Seeding is idempotent; the upsert is locked because 64 workers
# boot at once.
upsert_map() {  # $1 = claim dir
    local MAP="$EXP_ROOT/$1/MACHINE_MAP"
    for i in $(seq 1 60); do
        if mkdir "$MAP.lock" 2>/dev/null; then
            grep -vP "\t${MACHINE}\t" "$MAP" 2>/dev/null > "$MAP.tmp" || true
            printf '%s\t%s\t%s\n' "$(hostname)" "$MACHINE" "$(date -u +%FT%TZ)" >> "$MAP.tmp"
            mv "$MAP.tmp" "$MAP"
            rmdir "$MAP.lock"
            return 0
        fi
        sleep 2
    done
}
for DOM in $DOMAINS; do
    CD="$EXP_ROOT/$(domain_claim_dir "$DOM")"
    mkdir -p "$CD"
    case "$DOM" in
        if)   [ -s "$CD/BATCH_TAG" ] || printf 'if4k' > "$CD/BATCH_TAG" ;;
        code) [ -s "$CD/BATCH_TAG" ] || printf 'code4k' > "$CD/BATCH_TAG" ;;
    esac
    upsert_map "$(domain_claim_dir "$DOM")"
done
export MACHINE

# ---- boot-time hygiene -------------------------------------------------------
# Reap vLLM engines orphaned by a previous incarnation of this pod (ppid==1;
# training engines have live Ray parents and are untouched), and release eval
# claims whose owner never wrote a complete artifact set and has been silent
# for 2h -- evalq_exp has no refill reaper, so the fleet reaps for itself.
bash "$DATA/reap_orphan_vllm.sh" 2>/dev/null || true
now=$(date +%s)
for c in "$EVALQ"/claims/*__*; do
    [ -d "$c" ] || continue
    age=$(( now - $(stat -c %Y "$c") ))
    [ "$age" -gt 7200 ] && rmdir --ignore-fail-on-non-empty "$c" 2>/dev/null \
        && rm -rf "$c" 2>/dev/null && echo "reaped stale eval claim $(basename "$c")"
done

free_gpus() {  # indices with <500MiB used
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        | awk -F, '{gsub(/ |MiB/,"",$2); if ($2+0 < 500) print $1}'
}

eval_workers_here() { pgrep -f "eval_worker_exp.sh" 2>/dev/null | wc -l; }

startable_rows() {  # aggregated across every served domain
    local n=0 d
    for DOM in $DOMAINS; do
        [ -f "$(domain_data_dir "$DOM")/train.parquet" ] || continue
        d=$( (eval "$(domain_env "$DOM")"; MACHINE=$MACHINE bash deploy/campaign.sh --dry 2>/dev/null) \
            | grep -cE '^\s+any\s+\S+\s+POOL|assigned\s+\S' ) || d=0
        n=$((n + d))
    done
    echo "$n"
}

while :; do
    # -- 3. priority eviction (checked first so pass N's training sees the GPUs)
    rows=$(startable_rows)
    nfree=$(free_gpus | wc -l)
    if [ "${rows:-0}" -gt 0 ] && [ "$nfree" -lt 2 ] && [ "$(eval_workers_here)" -gt 0 ]; then
        echo "eviction: $rows startable training rows, $nfree free GPUs -> stopping eval workers"
        pkill -f "eval_worker_exp.sh" 2>/dev/null || true
        pkill -f "eval_offline.py" 2>/dev/null || true
        sleep 3
        bash "$DATA/reap_orphan_vllm.sh" 2>/dev/null || true
    fi

    # -- 1. training first, domains in priority order; a domain missing its
    #       dataset is skipped loudly (the IF set waits on a license token)
    for DOM in $DOMAINS; do
        if [ ! -f "$(domain_data_dir "$DOM")/train.parquet" ]; then
            echo "domain $DOM: no train.parquet yet, skipped"
            continue
        fi
        ( eval "$(domain_env "$DOM")"
          MACHINE=$MACHINE bash deploy/campaign.sh 2>&1 | tail -3 ) || true
    done

    # -- 2. eval backfill on whatever is still idle
    sleep 20   # let freshly-launched lanes grab their memory before we measure
    for g in $(free_gpus); do
        pgrep -f "eval_worker_exp.sh $g\b" >/dev/null 2>&1 && continue
        nohup bash "$DATA/eval_worker_exp.sh" "$g" "$EVALQ" \
            > "$LOG_DIR/${MACHINE}_evalw_gpu${g}.log" 2>&1 &
        echo "eval backfill: worker on gpu $g"
    done

    sleep "$LOOP_SEC"
done
