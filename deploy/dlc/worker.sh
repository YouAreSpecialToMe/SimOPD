#!/usr/bin/env bash
# SimOPD DLC fleet worker -- one instance per 8-GPU worker pod of a single big
# pytorchjob. Written 2026-08-13 for the ~500-card shape; the worker count is
# elastic (rank-keyed identity), so the loop is size-agnostic by construction.
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
# IF-domain reward: two constraint families (number_sentences,
# capital_word_frequency -- 12.5% of IF rows) need NLTK punkt at score time;
# without this they raise, score 0 forever, and read as "hard prompts"
# (caught live-firing real rows, 2026-08-13). Data lives on the shared disk.
export NLTK_DATA=${NLTK_DATA:-$DATA/nltk_data}
LOOP_SEC=${LOOP_SEC:-900}
# Test seams, both default-off. WORKER_DRY=1 makes every pass observational:
# training uses campaign.sh --dry (claims nothing), eviction logs its decision
# instead of killing, backfill echoes instead of spawning. WORKER_PASSES=N exits
# after N passes. The boot-time reaper stays REAL in dry mode on purpose -- point
# EVALQ at a sandbox queue to test it (deploy/dlc/test_worker_dry.sh does).
WORKER_DRY=${WORKER_DRY:-0}
WORKER_PASSES=${WORKER_PASSES:-0}
# Domains this fleet serves, in priority order. Each is a (manifest, claim
# namespace, env triple); math is the default namespace and needs no overrides.
# A domain whose train.parquet is missing is skipped loudly, not fatally -- the
# IF set is gated on a license token and may land later than code.
# Pairs are namespaces too: same mechanism, bigger lanes. w8b rejoins the banked
# W cell (TAG=w, anchor 0.664 already minted); p4b is fully gated on its probe
# file. Order = priority: math resume, cheap domains, then whole-box pairs.
DOMAINS=${DOMAINS:-"math if code w8b p4b"}

domain_env() {  # print export statements for one domain; empty for math
    case "$1" in
        if)   echo "export MANIFEST=configs/campaign_if.tsv CLAIM_DIR=.campaign_if"
              echo "export DATA_DIR=$DATA/simopd_if VAL_FILE_BASENAME=ifeval.parquet"
              echo "export MAX_RESPONSE_LENGTH=4096"
              echo "export CUSTOM_REWARD_PATH=$EXP_ROOT/src/simopd/domain_reward.py" ;;
        code) echo "export MANIFEST=configs/campaign_code.tsv CLAIM_DIR=.campaign_code"
              echo "export DATA_DIR=$DATA/simopd_code VAL_FILE_BASENAME=val_holdout.parquet"
              echo "export MAX_RESPONSE_LENGTH=4096" ;;
        w8b)  # the measured W shape, verbatim from w_pair_launch.sh (81 s/step):
              # student FSDP-4 + teacher TP2 x2 replicas, whole box, mem 0.40,
              # cap 8192. TAG stays `w` so rows join the banked cell family.
              echo "export MANIFEST=configs/campaign_w8b.tsv CLAIM_DIR=.campaign_w8b"
              echo "export STUDENT_MODEL=Qwen/Qwen3-8B-Base TEACHER_MODEL=Qwen/Qwen3-32B"
              echo "export NGPUS_PER_NODE=4 TEACHER_WORLD_SIZE=4 GPUS_PER_RUN=8"
              echo "export EXTRA_HYDRA=distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2"
              echo "export ROLLOUT_GPU_MEM_UTIL=0.40 MAX_RESPONSE_LENGTH=8192 PPO_MAX_TOKEN_LEN_PER_GPU=12288" ;;
        p4b)  # arithmetic shape, probe-gated (every manifest row carries needs=):
              # student 4B FSDP-2 + teacher 14B TP1 x2 replicas, half box, mem 0.40.
              echo "export MANIFEST=configs/campaign_p4b.tsv CLAIM_DIR=.campaign_p4b"
              echo "export STUDENT_MODEL=Qwen/Qwen3-4B-Base TEACHER_MODEL=Qwen/Qwen3-14B"
              echo "export NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=2 GPUS_PER_RUN=4"
              echo "export ROLLOUT_GPU_MEM_UTIL=0.40 MAX_RESPONSE_LENGTH=8192 PPO_MAX_TOKEN_LEN_PER_GPU=12288" ;;
        math) : ;;
    esac
}

domain_data_dir() { case "$1" in if) echo "$DATA/simopd_if";; code) echo "$DATA/simopd_code";; *) echo "$DATA/simopd_math";; esac; }
domain_claim_dir() { case "$1" in if) echo ".campaign_if";; code) echo ".campaign_code";; w8b) echo ".campaign_w8b";; p4b) echo ".campaign_p4b";; math) echo ".campaign";; esac; }
domain_width() { case "$1" in w8b) echo 8;; p4b) echo 4;; *) echo 2;; esac; }

# rank resolution, VERBATIM order from the colleagues' proven payloads
# (tools/dlc/exp*.sh line 44): rack-rank first, then role-index, then RANK.
# No silent default: a rankless boot would make every worker claim d0 and the
# identity guard would kill all but one -- die here with the fix instead.
RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-}}}
if [ -z "$RANK" ]; then
    echo "FATAL: no rank env (MLP_WORKER_RACK_RANK_INDEX / MLP_ROLE_INDEX / RANK)." >&2
    echo "       A DLC pytorchjob always sets one; for a manual boot:" >&2
    echo "       MLP_ROLE_INDEX=<n> bash deploy/dlc/worker.sh" >&2
    exit 1
fi
MACHINE="d${RANK}"
LOG_DIR="$DATA/dlc_logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/${MACHINE}_$(date +%Y%m%d_%H%M%S).log") 2>&1
echo "== dlc worker $MACHINE on $(hostname) $(date -u +%FT%TZ) =="

cd "$EXP_ROOT"
# mounted-workspace git quirk, straight from the colleagues' proven DLC
# payloads (tools/dlc/exp*.sh): never let an ownership check across the fs
# boundary block the job -- campaign.sh runs rev-parse/status/diff constantly.
git config --global --add safe.directory "$EXP_ROOT" 2>/dev/null || true
git config --global --add safe.directory /mgfs/shared/Group_GY/changhao/SimOPD 2>/dev/null || true
. ./simopd_env.sh

# ---- per-domain namespaces + rank-keyed identity upsert ----------------------
# Each domain has its own claim dir (own BATCH_TAG -> run names <arm>_s<s>_<tag>,
# so checkpoints/fingerprints never collide across domains) and its own
# MACHINE_MAP. Seeding is idempotent; the upsert is locked because 64 workers
# boot at once.
upsert_map() {  # $1 = claim dir
    local MAP="$EXP_ROOT/$1/MACHINE_MAP"
    for i in $(seq 1 60); do
        # a pod killed mid-upsert leaves the lock forever; steal after 180s
        if [ -d "$MAP.lock" ] && [ "$(( $(date +%s) - $(stat -c %Y "$MAP.lock") ))" -gt 180 ]; then
            rmdir "$MAP.lock" 2>/dev/null || true
        fi
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
        if|code)
              case "$DOM" in if) t=if4k;; *) t=code4k;; esac
              [ -s "$CD/BATCH_TAG" ] || printf '%s' "$t" > "$CD/BATCH_TAG"
              # d2/d3/d4 are the 4-GPU boxes for the topology-carrying family
              # (b3/d1/d2/d3/g2/n8/j1 pin their own NGPUS=2+TWS=2 = 4 GPUs);
              # the manifests pin those rows there. Cost: pool rows those boxes
              # claim also run at width 4, idling 2 GPUs per such lane -- three
              # boxes of bounded waste against seven arms that otherwise fail
              # at boot on every 2-GPU lane.
              for m in d2 d3 d4; do
                  [ -s "$CD/GPUS_PER_RUN.$m" ] || printf '4' > "$CD/GPUS_PER_RUN.$m"
              done ;;
        w8b)  [ -s "$CD/BATCH_TAG" ] || printf 'w' > "$CD/BATCH_TAG" ;;
        p4b)  [ -s "$CD/BATCH_TAG" ] || printf 'p4b' > "$CD/BATCH_TAG" ;;
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
EVAL_OUT="$DATA/evals"
claim_complete() {  # RUN STEP -> 0 when all five benchmark artifacts exist
    local n=0 b
    for b in aime24 aime25 amc23 minerva math500; do
        compgen -G "$EVAL_OUT/${1}__${b}__step${2}__seed*" > /dev/null && n=$((n + 1))
    done
    [ "$n" -ge 5 ]
}
# Reap a claim only when it is FINISHED (artifacts complete -- leftover claim,
# safe to clear any time) or ANCIENT (>36h; a healthy checkpoint-eval is ~13h,
# so 36h means a wrecked owner). The first draft reaped anything >2h old, which
# would have released claims under LIVE long evals on other boxes on every pod
# restart -- constant duplicated work. Caught in review.
now=$(date +%s)
for c in "$EVALQ"/claims/*__*; do
    [ -d "$c" ] || continue
    name=$(basename "$c"); RUN=${name%__*}; STEP=${name##*__}
    age=$(( now - $(stat -c %Y "$c") ))
    if claim_complete "$RUN" "$STEP"; then
        rm -rf "$c" && echo "reaped finished-claim leftover $name"
    elif [ "$age" -gt 129600 ]; then
        rm -rf "$c" && echo "reaped ancient claim $name (${age}s)"
    fi
done

free_gpus() {  # indices with <500MiB used
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        | awk -F, '{gsub(/ |MiB/,"",$2); if ($2+0 < 500) print $1}'
}

eval_workers_here() { pgrep -f "eval_worker_exp.sh" 2>/dev/null | wc -l; }

startable_rows() {  # "count maxwidth" aggregated across every served namespace --
                    # the eviction check needs the WIDTH too: pair lanes need a
                    # whole/half box, and evicting down to 2 free GPUs would
                    # starve them forever behind eval backfill
    local n=0 w=2 d
    for DOM in $DOMAINS; do
        [ -f "$(domain_data_dir "$DOM")/train.parquet" ] || continue
        # count entries on the `assigned` / `pool free` lines, excluding the
        # literal "none" -- the first version grepped 'assigned\s+\S', which
        # matches "assigned      none" and would have evicted eval workers on
        # every pass forever (caught in review, never shipped)
        #
        # `|| true`, NOT `|| d=0`: on a fully busy box --dry prints the whole
        # plan and THEN exits FATAL (<2 free GPUs), so under pipefail the
        # assignment "fails" AFTER capturing a perfectly good count -- and
        # `|| d=0` overwrote it, which made the eviction gate permanently
        # closed on exactly the boxes it exists for (caught by the dry
        # harness, 2026-08-13). awk's END prints 0 on empty input, so a
        # campaign that dies before printing still yields d=0 via the guard.
        d=$( (eval "$(domain_env "$DOM")"; MACHINE=$MACHINE timeout 180 bash deploy/campaign.sh --dry 2>/dev/null) \
            | awk '$1=="assigned"{for(i=2;i<=NF;i++)if($i!="none")n++}
                   $1=="pool"&&$2=="free"{for(i=3;i<=NF;i++)if($i!="none")n++}
                   END{print n+0}' ) || true
        [ -n "${d:-}" ] || d=0
        if [ "${d:-0}" -gt 0 ]; then
            n=$((n + d))
            [ "$(domain_width "$DOM")" -gt "$w" ] && w=$(domain_width "$DOM")
        fi
    done
    echo "$n $w"
}

FEED_TAGS=${FEED_TAGS:-"16k w"}   # p4b joins via env once its gate opens
FEED_SEC=${FEED_SEC:-1800}
feed_evalq() {
    local stamp="$EVALQ/last_feed" lock="$EVALQ/feed.lock" now age ck run step tag added=0
    now=$(date +%s)
    if [ -f "$stamp" ]; then
        age=$((now - $(stat -c %Y "$stamp" 2>/dev/null || echo 0)))
        [ "$age" -lt "$FEED_SEC" ] && return 0
    fi
    if ! mkdir "$lock" 2>/dev/null; then
        # steal only a dead feeder's lock, same rule as the map lock
        age=$((now - $(stat -c %Y "$lock" 2>/dev/null || echo "$now")))
        { [ "$age" -gt 1800 ] && rmdir "$lock" 2>/dev/null && mkdir "$lock" 2>/dev/null; } || return 0
    fi
    touch "$stamp"
    mkdir -p "$EVALQ/claims"; touch "$EVALQ/pending.txt"
    for tag in $FEED_TAGS; do
        # BASELINE FIRST: every comparison is anchored to vanilla, yet the 16k
        # sweep left it at 9/30 suite cells (mid-curve s75-s225 all missing)
        # while b1 got 30/30 -- alphabetical glob order put vanilla last. The
        # queue is consumed top-down, so feed order IS priority; the
        # already-queued guard below makes the double glob safe.
        for ck in "$DATA/ckpt/simopd/"vanilla*"_${tag}"/global_step_*/actor \
                  "$DATA/ckpt/simopd/"*"_${tag}"/global_step_*/actor; do
            [ -d "$ck" ] || continue
            step=${ck%/actor}; step=${step##*global_step_}
            run=${ck%/global_step_*}; run=${run##*/}
            # settled = the step dir stopped moving >=10 min ago; a bank still
            # being written must not be evaluated mid-copy
            [ $((now - $(stat -c %Y "${ck%/actor}" 2>/dev/null || echo "$now"))) -lt 600 ] && continue
            local n=0 b
            for b in aime24 aime25 amc23 minerva math500; do
                compgen -G "$DATA/evals/${run}__${b}__step${step}__seed*" >/dev/null && n=$((n+1))
            done
            [ "$n" -ge 5 ] && continue                            # already measured
            [ -d "$EVALQ/claims/${run}__${step}" ] && continue    # in flight
            grep -qx "$run $step" "$EVALQ/pending.txt" && continue # already queued
            echo "$run $step" >> "$EVALQ/pending.txt"
            added=$((added+1))
        done
    done
    rmdir "$lock" 2>/dev/null
    [ "$added" -gt 0 ] && echo "eval feed: +$added cells (pending $(grep -c . "$EVALQ/pending.txt" 2>/dev/null || echo 0))"
    return 0
}

while :; do
    # -- 3. priority eviction (checked first so pass N's training sees the GPUs).
    #    startable_rows costs one campaign --dry PER DOMAIN -- a full lustre scan
    #    of checkpoints and logs each (~30-60s measured on the hop pod) -- and its
    #    only consumer is the eviction decision, which is moot when this box runs
    #    no eval workers. Gate on the cheap check first: a fresh fleet skips the
    #    whole scan every pass.
    evalw=$(eval_workers_here)
    if [ "$evalw" -gt 0 ]; then
        read -r rows need <<< "$(startable_rows)"
    else
        rows=0 need=2
    fi
    nfree=$(free_gpus | wc -l)
    # one line per pass: the gate's actual inputs, so silence is diagnosable
    echo "pass: evalw=$evalw free=$nfree startable=${rows:-0} need=${need:-2}"
    if [ "${rows:-0}" -gt 0 ] && [ "$nfree" -lt "${need:-2}" ] && [ "$evalw" -gt 0 ]; then
        echo "eviction: $rows startable rows need width $need, $nfree free -> stopping eval workers"
        if [ "$WORKER_DRY" = "1" ]; then
            echo "DRY: would pkill eval workers + reap orphans"
        else
            pkill -f "eval_worker_exp.sh" 2>/dev/null || true
            pkill -f "eval_offline.py" 2>/dev/null || true
            sleep 3
            bash "$DATA/reap_orphan_vllm.sh" 2>/dev/null || true
        fi
    fi

    # -- 1. training first, domains in priority order; a domain missing its
    #       dataset is skipped loudly (the IF set waits on a license token)
    launched_gpus=" "
    for DOM in $DOMAINS; do
        if [ ! -f "$(domain_data_dir "$DOM")/train.parquet" ]; then
            echo "domain $DOM: no train.parquet yet, skipped"
            continue
        fi
        if [ "$WORKER_DRY" = "1" ]; then
            out=$( (eval "$(domain_env "$DOM")"
                    MACHINE=$MACHINE bash deploy/campaign.sh --dry 2>&1) ) || true
        else
            out=$( (eval "$(domain_env "$DOM")"
                    MACHINE=$MACHINE bash deploy/campaign.sh 2>&1) ) || true
        fi
        echo "$out" | tail -3
        # GPUs this pass just handed to lanes: a lane spends minutes loading
        # weights before it holds memory, so the <500MiB check alone would let
        # backfill land an eval engine on a lane's card and OOM its boot
        launched_gpus+="$(echo "$out" | grep -oE 'GPU_LIST\s+[0-9, ]+' \
                          | sed 's/GPU_LIST//; s/,/ /g') "
    done

    # -- 1.5 feed the eval queue: the fleet feeds itself. The m-fleet era fed
    #    evalq from an hourly hop-pod cron (exp_patrol.sh) with a hand-picked
    #    13-arm roster -- scarce eval cards then, and hop pods die (dsw243
    #    did). Here any worker feeds, one at a time (lock), at most once per
    #    FEED_SEC fleet-wide (stamp), roster = every run under the tags this
    #    fleet serves. Domain tags (if4k/code4k) are deliberately NOT fed:
    #    the 5-bench suite is the MATH metric; domain runs carry their metric
    #    in-loop (val every 25 steps), transfer probes are D6/final-ckpt work.
    feed_evalq

    # -- 2. eval backfill on whatever is still idle
    [ "$WORKER_DRY" = "1" ] && sleep 1 || sleep 60   # settle; launched_gpus is the real guard
    for g in $(free_gpus); do
        case "$launched_gpus" in *" $g "*) continue ;; esac
        # trailing space, not \b: pgrep -f is POSIX ERE and \b silently matches a
        # literal b, so "gpu 1" would collide with "gpu 10" AND every duplicate
        # check would fail -- one extra eval worker per GPU per pass
        pgrep -f "eval_worker_exp.sh $g " >/dev/null 2>&1 && continue
        if [ "$WORKER_DRY" = "1" ]; then
            echo "DRY: would start eval worker on gpu $g"
            continue
        fi
        nohup bash "$DATA/eval_worker_exp.sh" "$g" "$EVALQ" \
            > "$LOG_DIR/${MACHINE}_evalw_gpu${g}.log" 2>&1 &
        echo "eval backfill: worker on gpu $g"
    done

    # test seam: WORKER_PASSES=N exits after N supervisor passes (0 = run forever)
    PASS_N=$(( ${PASS_N:-0} + 1 ))
    if [ "$WORKER_PASSES" -gt 0 ] && [ "$PASS_N" -ge "$WORKER_PASSES" ]; then
        echo "worker: WORKER_PASSES=$WORKER_PASSES reached, exiting"
        exit 0
    fi
    sleep "$LOOP_SEC"
done
