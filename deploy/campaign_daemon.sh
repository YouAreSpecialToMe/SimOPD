#!/usr/bin/env bash
# Keep this machine's lanes fed: re-invoke campaign.sh whenever there might be work.
#
#   nohup bash deploy/campaign_daemon.sh > logs/$(hostname)_daemon.log 2>&1 &
#   touch .campaign/daemon.stop.<machine>     # stop it, from ANY machine (shared fs)
#
# Exists because nothing else refills a lane. _lane.sh drains the list it was handed
# and exits; campaign.sh claims work only at the moment it is invoked. m1 sat idle
# with two runs finished and three pool rows waiting -- by design, twice over, with
# no piece whose job it was to notice. This is that piece.
#
# It is deliberately nothing more than a loop around campaign.sh. Idempotence lives
# THERE (done runs skipped, in-flight respected, claims atomic, launches flock'd), so
# the daemon cannot double-launch even overlapping a manual invocation. It never
# passes INFLIGHT_HOURS=0 or ALLOW_UNKNOWN_GPU_USERS -- those overrides exist for a
# human who has just verified a corpse, and a daemon has verified nothing.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOOP_SEC=${LOOP_SEC:-900}
CLAIM_DIR=${CLAIM_DIR:-.campaign}

# Resolve identity the same way campaign.sh does; refuse to babysit an unregistered box.
MACHINE=$(awk -F'\t' -v h="$(hostname)" '$1==h {print $2; exit}' "$CLAIM_DIR/MACHINE_MAP" 2>/dev/null)
[ -n "$MACHINE" ] || { echo "FATAL: $(hostname) not in $CLAIM_DIR/MACHINE_MAP; register first:"; \
                       echo "       MACHINE=<name> bash deploy/campaign.sh --dry"; exit 1; }
STOP="$CLAIM_DIR/daemon.stop.$MACHINE"
rm -f "$STOP"
echo "[daemon] $MACHINE on $(hostname), every ${LOOP_SEC}s; stop with: touch $STOP"

while :; do
    if [ -e "$STOP" ]; then echo "[daemon] stop file seen, exiting"; rm -f "$STOP"; exit 0; fi
    echo "[daemon] $(date -u +%FT%TZ) invoking campaign.sh"
    bash deploy/campaign.sh
    rc=$?
    # 0 = launched or nothing to do; anything else is a refusal worth a human's eyes,
    # but not worth dying over -- the condition (drift, busy GPUs) may clear itself.
    [ $rc -ne 0 ] && echo "[daemon] campaign.sh exited $rc (see above); retrying in ${LOOP_SEC}s"
    sleep "$LOOP_SEC"
done
