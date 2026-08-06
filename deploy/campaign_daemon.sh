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

# The daemon runs CAMPAIGN defaults, period. It inherits the interactive shell it was
# nohup'd from, and that shell's leftovers become silent config: a stale STEPS=3 from
# a rehearsal turned two arms into 3-step runs that would have been recorded as DONE
# -- a false OK is worse than a failure, because nothing ever retries it. Same lesson
# as ModelScope: inheritance is an override channel, so close it explicitly.
unset STEPS TEST_FREQ SAVE_FREQ TAG REHEARSAL PROJECT_NAME EXPERIMENT_NAME \
      LANES GPU_LIST GPUS_PER_RUN RAY_TMPDIR_TAG RAY_TMPDIR LANE_RUNS LANE_TAG \
      INFLIGHT_HOURS ALLOW_UNKNOWN_GPU_USERS RESUME ROLLOUT_GPU_MEM_UTIL \
      TOTAL_TRAINING_STEPS MAX_RESPONSE_LENGTH DISTILLATION_LOSS_MODE 2>/dev/null

LOOP_SEC=${LOOP_SEC:-900}
CLAIM_DIR=${CLAIM_DIR:-.campaign}

# Resolve identity the same way campaign.sh does; refuse to babysit an unregistered box.
MACHINE=$(awk -F'\t' -v h="$(hostname)" '$1==h {print $2; exit}' "$CLAIM_DIR/MACHINE_MAP" 2>/dev/null)
[ -n "$MACHINE" ] || { echo "FATAL: $(hostname) not in $CLAIM_DIR/MACHINE_MAP; register first:"; \
                       echo "       MACHINE=<name> bash deploy/campaign.sh --dry"; exit 1; }
# Exactly one daemon per machine. Without this, a second `nohup` quietly makes two:
# the launch flock keeps them from double-launching, but the first to see the stop
# file deletes it and the OTHER survives the stop request -- an unstoppable daemon is
# worse than none. The lock is held for the daemon's lifetime; /tmp because the
# singleton scope is this machine, and flock on the shared NAS is the semantics not
# to rely on.
exec 8> "/tmp/simopd_daemon_${MACHINE}.lock"
flock -n 8 || {
    echo "FATAL: a daemon for $MACHINE is already running on this box:" >&2
    pgrep -af "campaign_daemon" | grep -v $$ | sed 's/^/       /' >&2
    exit 1
}
echo $$ >&8

STOP="$CLAIM_DIR/daemon.stop.$MACHINE"
HEART="$CLAIM_DIR/daemon.alive.$MACHINE"
rm -f "$STOP"
echo "[daemon] $MACHINE on $(hostname) pid $$, every ${LOOP_SEC}s; stop with: touch $STOP"

while :; do
    # Heartbeat on the shared mount: progress.py shows it from any machine, so "is
    # m3's daemon alive" is a glance, not an ssh. A stale heartbeat IS the alarm.
    date -u +%FT%TZ > "$HEART"
    if [ -e "$STOP" ]; then echo "[daemon] stop file seen, exiting"; rm -f "$STOP" "$HEART" "$CLAIM_DIR/daemon.status.$MACHINE"; exit 0; fi
    echo "[daemon] $(date -u +%FT%TZ) invoking campaign.sh (STEPS=${STEPS:-250-default})"
    _out=$(bash deploy/campaign.sh 2>&1)
    rc=$?
    printf '%s\n' "$_out"
    # A refusing daemon looks EXACTLY like a working one from the progress screen:
    # heartbeat fresh, rows QUEUED, nothing moving -- m2 and m3 sat in that state with
    # eight free GPUs while the guard (correctly) refused over leftover processes, and
    # nothing surfaced the contradiction. So the outcome of every invocation lands on
    # the shared mount, and progress.py turns a refusal into an ATTENTION line.
    if [ $rc -ne 0 ]; then
        { echo "REFUSING rc=$rc at $(date -u +%FT%TZ)"; printf '%s\n' "$_out" | grep -E "FATAL|held|QUAR" | head -4; } \
            > "$CLAIM_DIR/daemon.status.$MACHINE"
        echo "[daemon] campaign.sh exited $rc; status written; retrying in ${LOOP_SEC}s"
    else
        printf 'ok %s\n' "$(date -u +%FT%TZ)" > "$CLAIM_DIR/daemon.status.$MACHINE"
    fi
    sleep "$LOOP_SEC"
done
