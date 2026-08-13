#!/usr/bin/env bash
# GPU-free proof harness for the DLC fleet worker. Run on any box that mounts
# /mgfs (a hop pod works): it simulates two worker ranks CONCURRENTLY with fake
# hostnames and a fake nvidia-smi, drives worker.sh in WORKER_DRY mode against
# the REAL trees and manifests, and asserts every decision the supervisor makes.
#
#   bash deploy/dlc/test_worker_dry.sh
#
# What is REAL here: campaign.sh row parsing, claim-namespace seeding, identity
# upsert under contention (incl. stale-lock steal), startable-row counting, the
# eviction decision, backfill targeting, and the eval-claim reaper (against a
# SANDBOX queue with fabricated claims). What is FAKE: GPUs (PATH shim),
# hostnames (PATH shim -- two ranks on one box would otherwise trip campaign.sh's
# identity guard, which is the guard being right), eval workers (a sleeping stub
# whose cmdline matches the pgrep pattern). What this CANNOT prove: anything
# involving a real lane or a real vLLM boot -- that stays on the smoke job.
set -uo pipefail

EXP_ROOT=${EXP_ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
DATA=${DATA:-/mgfs/shared/Group_GY/changhao/simopd_data}
SB=$(mktemp -d /tmp/simopd_workertest.XXXX)
FAILS=0
say()  { echo "== $*"; }
chk()  { if eval "$2"; then echo "  PASS  $1"; else echo "  FAIL  $1"; FAILS=$((FAILS+1)); fi; }

# ---- fakes -------------------------------------------------------------------
mkdir -p "$SB/bin" "$SB/evalq/claims" "$SB/evals" "$SB/logs"
cat > "$SB/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# fake fleet card: 8 GPUs; memory per GPU read from $GPU_STATE (one MiB value per line)
STATE=${GPU_STATE:?}
i=0
while read -r mem; do echo "$i, ${mem} MiB"; i=$((i+1)); done < "$STATE"
EOF
cat > "$SB/bin/hostname" <<'EOF'
#!/usr/bin/env bash
echo "${FAKE_HOSTNAME:-testhost}"
EOF
chmod +x "$SB/bin/"*
printf '70000\n70000\n70000\n70000\n70000\n70000\n70000\n70000\n' > "$SB/busy"
printf '4\n4\n4\n4\n4\n4\n4\n4\n' > "$SB/free"

# a stub whose cmdline matches worker.sh's pgrep pattern
cat > "$SB/eval_worker_exp.sh" <<'EOF'
#!/usr/bin/env bash
sleep 120
EOF
chmod +x "$SB/eval_worker_exp.sh"

# ---- sandbox eval queue: three fabricated claims for the reaper --------------
# (a) FINISHED leftover: all five artifacts exist -> must be reaped
mkdir -p "$SB/evalq/claims/fakerun_s0_16k__25"
for b in aime24 aime25 amc23 minerva math500; do
    touch "$SB/evals/fakerun_s0_16k__${b}__step25__seed0__t.parquet"
done
# (b) LIVE: fresh, incomplete -> must survive
mkdir -p "$SB/evalq/claims/liverun_s1_16k__50"
# (c) ANCIENT: incomplete but 40h old -> must be reaped
mkdir -p "$SB/evalq/claims/deadrun_s2_16k__75"
touch -d '40 hours ago' "$SB/evalq/claims/deadrun_s2_16k__75"
# the reaper globs $DATA/evals -- point it at the sandbox via a worker env seam?
# it hardcodes EVAL_OUT="$DATA/evals"; keep DATA real but plant the fake
# artifacts THERE would pollute -- instead run the reaper with DATA=$SB and a
# fake reap_orphan_vllm + eval trees:
mkdir -p "$SB/dlc_logs"
cat > "$SB/reap_orphan_vllm.sh" <<'EOF'
#!/usr/bin/env bash
echo "(fake orphan reaper)"
EOF
chmod +x "$SB/reap_orphan_vllm.sh"
cp "$DATA/eval_worker_exp.sh" "$SB/eval_worker_exp_real_unused.sh" 2>/dev/null || true
mkdir -p "$SB/simopd_math" && touch "$SB/simopd_math/train.parquet"   # math namespace: dataset "exists"
cp -r "$DATA/simopd_code" "$SB/simopd_code" 2>/dev/null || mkdir -p "$SB/simopd_code"
[ -f "$SB/simopd_code/train.parquet" ] || touch "$SB/simopd_code/train.parquet"
# NOTE: DATA=$SB redirects EVALQ/evals/logs/reapers; EXP_ROOT stays REAL, so
# campaign.sh parses the real manifests and real claim namespaces.

# ---- stale lock: pre-seed one to prove the steal ------------------------------
mkdir -p "$EXP_ROOT/.campaign_code"
mkdir -p "$EXP_ROOT/.campaign_code/MACHINE_MAP.lock" 2>/dev/null || true
touch -d '10 minutes ago' "$EXP_ROOT/.campaign_code/MACHINE_MAP.lock"

cleanup() {  # strip this harness's fake identities from every real map
    for cd_ in .campaign .campaign_if .campaign_code .campaign_w8b .campaign_p4b; do
        m="$EXP_ROOT/$cd_/MACHINE_MAP"
        [ -f "$m" ] && grep -v '^fake-dlc-' "$m" > "$m.clean" && mv "$m.clean" "$m"
    done
}
trap cleanup EXIT

run_worker() {  # rank, gpu-state-file
    local rank=$1 state=$2; shift 2
    echo "   [$(date +%H:%M:%S)] worker rank$rank starting (domains: math if code)"
    PATH="$SB/bin:$PATH" GPU_STATE="$state" FAKE_HOSTNAME="fake-dlc-w${rank}" \
    MLP_ROLE_INDEX=$rank EXP_ROOT="$EXP_ROOT" DATA="$SB" EVALQ="$SB/evalq" \
    DOMAINS="math if code" \
    WORKER_DRY=1 WORKER_PASSES=1 LOOP_SEC=3 \
    bash "$EXP_ROOT/deploy/dlc/worker.sh" > "$SB/w${rank}.out" 2>&1
    echo $? > "$SB/w${rank}.rc"
    echo "   [$(date +%H:%M:%S)] worker rank$rank done"
}

say "TEST A: two ranks boot concurrently, all GPUs busy, fake eval workers alive"
# args mimic a real eval worker's cmdline (gpu + queue), which is what pgrep sees
bash "$SB/eval_worker_exp.sh" 7 "$SB/evalq" & STUB=$!
run_worker 0 "$SB/busy" & P0=$!
run_worker 1 "$SB/busy" & P1=$!
wait $P0 $P1
kill $STUB 2>/dev/null || true

chk "rank0 exited cleanly"                '[ "$(cat "$SB/w0.rc")" = 0 ]'
chk "rank1 exited cleanly"                '[ "$(cat "$SB/w1.rc")" = 0 ]'
chk "d0 registered in math map"           'grep -qP "\td0\t" "$EXP_ROOT/.campaign/MACHINE_MAP"'
chk "d1 registered in math map"           'grep -qP "\td1\t" "$EXP_ROOT/.campaign/MACHINE_MAP"'
chk "d0+d1 in code map (stale lock stolen)" 'grep -qP "\td0\t" "$EXP_ROOT/.campaign_code/MACHINE_MAP" && grep -qP "\td1\t" "$EXP_ROOT/.campaign_code/MACHINE_MAP"'
chk "code BATCH_TAG seeded"               '[ "$(cat "$EXP_ROOT/.campaign_code/BATCH_TAG")" = code4k ]'
chk "if BATCH_TAG seeded"                 '[ "$(cat "$EXP_ROOT/.campaign_if/BATCH_TAG")" = if4k ]'
chk "wide-box width seeded (code)"        '[ "$(cat "$EXP_ROOT/.campaign_code/GPUS_PER_RUN.d2")" = 4 ]'
# w8b seeding checked via a minimal single-namespace pass (below), keeping the
# main concurrency test to three domains for speed
chk "reaper: finished leftover reaped"    '[ ! -d "$SB/evalq/claims/fakerun_s0_16k__25" ]'
chk "reaper: live claim survives"         '[ -d "$SB/evalq/claims/liverun_s1_16k__50" ]'
chk "reaper: ancient claim reaped"        '[ ! -d "$SB/evalq/claims/deadrun_s2_16k__75" ]'
chk "IF domain skipped (no dataset)"      'grep -q "domain if: no train.parquet yet" "$SB/w0.out"'
chk "code rows counted startable"         'grep -qE "eviction: [1-9][0-9]* startable" "$SB/w0.out"'
chk "eviction stayed DRY"                 'grep -q "DRY: would pkill" "$SB/w0.out"'
chk "no backfill while busy"              '! grep -q "would start eval worker" "$SB/w0.out"'
cp "$SB/w0.out" "$SB/wA0.out"; cp "$SB/w1.out" "$SB/wA1.out"   # TEST B reuses w0.out; keep A's for forensics

say "TEST A2: single pass on w8b only -- namespace seeding for the pair"
PATH="$SB/bin:$PATH" GPU_STATE="$SB/busy" FAKE_HOSTNAME="fake-dlc-w0" \
MLP_ROLE_INDEX=0 EXP_ROOT="$EXP_ROOT" DATA="$SB" EVALQ="$SB/evalq" DOMAINS="w8b" \
WORKER_DRY=1 WORKER_PASSES=1 LOOP_SEC=1 bash "$EXP_ROOT/deploy/dlc/worker.sh" > "$SB/wpair.out" 2>&1
chk "w8b tag joins banked cell"           '[ "$(cat "$EXP_ROOT/.campaign_w8b/BATCH_TAG")" = w ]'

say "TEST B: rank0 again, all GPUs free -- backfill should target all 8"
pkill -f "$SB/eval_worker_exp.sh" 2>/dev/null || true
run_worker 0 "$SB/free"
chk "backfill targeted 8 GPUs"            '[ "$(grep -c "DRY: would start eval worker" "$SB/w0.out")" = 8 ]'
chk "no eviction when idle"               '! grep -q "DRY: would pkill" "$SB/w0.out"'
chk "dry training claims nothing"         '! grep -rq "fake-dlc-w0" "$EXP_ROOT/.campaign_code/claims" 2>/dev/null'

echo
echo "RESULT: $([ $FAILS -eq 0 ] && echo ALL PASS || echo "$FAILS FAILURE(S)")   (sandbox: $SB)"
exit $([ $FAILS -eq 0 ] && echo 0 || echo 1)
