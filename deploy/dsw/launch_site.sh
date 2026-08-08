#!/usr/bin/env bash
# S-wave bring-up for one local DSW box (24-card plan, 2026-08-07). Idempotent --
# rerun freely; finished phases skip themselves via markers in .campaign/.
#
#   bash deploy/dsw/launch_m1.sh        # floor lanes (2-card), migration + repin
#   bash deploy/dsw/launch_m2.sh        # n8 cell control (4-card lanes)
#   bash deploy/dsw/launch_m3.sh        # n8 cell treatment (4-card lanes)
#
# Run m1 FIRST (it owns the one-time shared-fs steps), then m2/m3 in any order.
# Run inside tmux: the probe phase is a real 50-step run (~4-5h) in the foreground;
# the daemon it starts at the end survives logout on its own.
#
# Env: PROBE=0 skip the probe phase   APPLY=1 migration without the y/N prompt
#      FORCE=1 proceed over busy GPUs (only after you have verified what they are)
set -euo pipefail
MACHINE=${1:?usage: launch_site.sh <m1|m2|m3>}
case "$MACHINE" in
  m1) LANE_SHAPE=2; PROBE_RUN="vanilla:0" ;;
  m2) LANE_SHAPE=4; PROBE_RUN="vanilla_n8:0" ;;
  m3) LANE_SHAPE=4; PROBE_RUN="j1_kdrl:0" ;;
  *) echo "FATAL: unknown machine '$MACHINE'" >&2; exit 1 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CLAIM_DIR=${CLAIM_DIR:-.campaign}; mkdir -p "$CLAIM_DIR" logs
banner(){ printf '\n=== [%s] %s ===\n' "$MACHINE" "$*"; }

banner "phase 0: code + guards"
git pull --ff-only
# A pilot-era daemon must not feed the new manifest: on m2/m3 it would launch the
# n8 rows with 2-card lanes. Its heartbeat tells us whether one is alive.
if [ -f "$CLAIM_DIR/daemon.alive.$MACHINE" ]; then
    _hb=$(date -d "$(cat "$CLAIM_DIR/daemon.alive.$MACHINE")" +%s 2>/dev/null || echo 0)
    if [ $(( $(date +%s) - _hb )) -lt 1800 ]; then
        touch "$CLAIM_DIR/daemon.stop.$MACHINE"
        echo "old daemon for $MACHINE is alive -- stop file written; it exits within its"
        echo "15-min loop. Rerun this script afterwards."
        exit 0
    fi
fi
_busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)
if [ "$_busy" -gt 0 ] && [ "${FORCE:-0}" != 1 ]; then
    echo "FATAL: $_busy compute process(es) on the GPUs -- 8k drain not done, or something" >&2
    echo "       else is training. Wait it out, or FORCE=1 after verifying:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    exit 1
fi
source "${SIMOPD_VENV:-simopd}/bin/activate" 2>/dev/null || true
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# First-ever launch on a box must bind hostname->label in the shared map; a
# registered box refuses a contradicting label instead of obeying it.
if ! awk -F'\t' -v m="$MACHINE" '$2==m' "$CLAIM_DIR/MACHINE_MAP" 2>/dev/null | grep -q .; then
    MACHINE=$MACHINE bash deploy/campaign.sh --dry >/dev/null || true
fi

if [ "$MACHINE" = m1 ]; then
    banner "phase 1: pilot-8k roster migration (shared fs -- runs once, m1 owns it)"
    if [ -f "$CLAIM_DIR/migrated_pilot8k" ]; then
        echo "already done at $(cat "$CLAIM_DIR/migrated_pilot8k")"
    else
        PILOT8K="vanilla b1_skew_kl b2_forward_kl b3_eopd_gate c2_quantile_budget \
d1_tip d2_selectkd d3_teachability e1_pl_rank f1_soft_log f2_hard_clip f3_power \
g1_verified_only g2_fire_likelihood h1_first_segment vanilla_n8 j1_kdrl"
        _names=$(for a in $PILOT8K; do printf '%s_s0 ' "$a"; done)
        python scripts/migrate_stale.py --suffix __pilot8k --names $_names
        if [ "${APPLY:-0}" != 1 ]; then
            echo; read -r -p "dry run above (absent names skip themselves) -- apply? [y/N] " _yn
            [ "${_yn:-n}" = y ] || { echo "not applied; rerun when ready."; exit 1; }
        fi
        python scripts/migrate_stale.py --suffix __pilot8k --names $_names --apply
        date -u +%FT%TZ > "$CLAIM_DIR/migrated_pilot8k"
        echo "NOTE: resuming an old-fingerprint ckpt takes a one-time RESUME=force --"
        echo "      if you use it, record why in $CLAIM_DIR/PIN_HISTORY."
    fi
    banner "phase 2: repin (assignment changed + 16k defaults now run-defining)"
    REASON="S wave: 24-card local plan, 16k batch" MACHINE=m1 bash deploy/campaign.sh --repin || true
else
    banner "phase 1-2: owned by m1"
    if [ ! -f "$CLAIM_DIR/migrated_pilot8k" ]; then
        echo "FATAL: migration marker missing -- run launch_m1.sh first." >&2
        exit 1
    fi
fi

banner "phase 3: 16k probe -- $PROBE_RUN, 50 steps, ${LANE_SHAPE}-card lane"
if [ "${PROBE:-1}" = 1 ] && [ ! -f "$CLAIM_DIR/probe16k.$MACHINE.ok" ]; then
    _pgpus=$(seq -s, 0 $((LANE_SHAPE-1)))
    # VAL_BEFORE_TRAIN mints the 16k step-0 anchor on m1's floor probe, and the
    # cell's own step-0 point on m2/m3. SAVE_FREQ=-1: a probe leaves no ckpts.
    VAL_BEFORE_TRAIN=True STEPS=50 TEST_FREQ=25 SAVE_FREQ=-1 \
      TAG="${MACHINE}probe" GPU_LIST="$_pgpus" LANES=1 \
      RAY_TMPDIR_TAG="${MACHINE}probe$(date +%s)_" \
      bash deploy/dsw/run_parallel.sh "$PROBE_RUN"
    date -u +%FT%TZ > "$CLAIM_DIR/probe16k.$MACHINE.ok"
    echo
    echo "probe OK. The FIRST validation in its log (step 0) is the 16k anchor for this"
    echo "lane shape -- record the number in $CLAIM_DIR/PIN_HISTORY. Newest log:"
    ls -t logs/lane*.log 2>/dev/null | head -1 || true
else
    echo "skipped (PROBE=0 or $CLAIM_DIR/probe16k.$MACHINE.ok exists)"
fi

banner "phase 4: lane shape on record + daemon"
echo "$LANE_SHAPE" > "$CLAIM_DIR/GPUS_PER_RUN.$MACHINE"
echo "recorded: GPUS_PER_RUN.$MACHINE = $LANE_SHAPE (daemon invocations read this)"
nohup bash deploy/campaign_daemon.sh > "logs/$(hostname)_daemon.log" 2>&1 &
sleep 3
tail -n 5 "logs/$(hostname)_daemon.log" || true
echo
echo "done. watch:   python scripts/progress.py"
echo "      daemon:  logs/$(hostname)_daemon.log   stop: touch $CLAIM_DIR/daemon.stop.$MACHINE"
