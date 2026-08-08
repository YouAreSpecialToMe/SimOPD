#!/usr/bin/env bash
# Start this box's campaign daemon. Nothing else.
#
#   bash deploy/dsw/start_daemon.sh m1              # start if absent
#   RESTART=1 bash deploy/dsw/start_daemon.sh m1    # kill any running one first
#
# Deliberately minimal: no busy-GPU guard, no migration, no `set -e`. Those
# belong to launch_site.sh, and when one of them exits early the operator is
# left with a box that ran a script and got no daemon -- the 08-08 failure.
# This one registers the machine if needed, records the lane shape, clears a
# leftover stop file, starts the daemon, and then PROVES it is alive.
MACHINE=${1:?usage: start_daemon.sh <m1|m2|m3>}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CLAIM_DIR=${CLAIM_DIR:-.campaign}
mkdir -p "$CLAIM_DIR" logs
source "${SIMOPD_VENV:-simopd}/bin/activate" 2>/dev/null
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

if ! awk -F'\t' -v h="$(hostname)" '$1==h' "$CLAIM_DIR/MACHINE_MAP" 2>/dev/null | grep -q .; then
    echo "registering $(hostname) as $MACHINE"
    MACHINE=$MACHINE bash deploy/campaign.sh --dry >/dev/null 2>&1
fi
_mapped=$(awk -F'\t' -v h="$(hostname)" '$1==h {print $2; exit}' "$CLAIM_DIR/MACHINE_MAP" 2>/dev/null)
if [ "$_mapped" != "$MACHINE" ]; then
    echo "FATAL: this box is registered as '${_mapped:-nothing}', not '$MACHINE'." >&2
    echo "       Registration is hostname->label and refusing beats obeying here." >&2
    exit 1
fi
echo 2 > "$CLAIM_DIR/GPUS_PER_RUN.$MACHINE"
rm -f "$CLAIM_DIR/daemon.stop.$MACHINE" "$CLAIM_DIR/MAX_LANES.$MACHINE"

if pgrep -f campaign_daemon.sh >/dev/null 2>&1; then
    if [ "${RESTART:-0}" = 1 ]; then
        # A daemon that outlived a reset spent those passes refusing (its box was
        # briefly unregistered) -- replacing it is cheaper than reasoning about
        # which pass it is on.
        echo "RESTART=1: killing the running daemon (pid $(pgrep -f campaign_daemon.sh | head -1))"
        pkill -f campaign_daemon.sh; sleep 3
    else
        echo "daemon already running (pid $(pgrep -f campaign_daemon.sh | head -1)); nothing to do"
        echo "  (RESTART=1 to replace it)"
        exit 0
    fi
fi
# Break a stale singleton lock. The daemon holds /tmp/simopd_daemon_<m>.v2.lock
# through fd 8, and fd 8 is INHERITED: a leaked child (ray worker, vllm server,
# an orphaned shepherd) keeps the lock long after the daemon is gone, so every
# restart refuses with "a daemon is already running" while pgrep finds none --
# the 08-08 deadlock. No daemon process here means no daemon, whoever holds the
# fd; removing the path lets the new one lock a fresh inode while the stragglers
# keep the old one.
_lock="/tmp/simopd_daemon_${MACHINE}.v2.lock"
if [ -e "$_lock" ] && ! pgrep -f campaign_daemon.sh >/dev/null 2>&1; then
    _holder=$(fuser "$_lock" 2>/dev/null | tr -s ' ')
    [ -n "$_holder" ] && echo "stale lock held by pid(s)$_holder (no daemon process) -- breaking it"
    rm -f "$_lock"
fi
LOG="logs/$(hostname)_daemon.log"
nohup bash deploy/campaign_daemon.sh > "$LOG" 2>&1 &
for _ in $(seq 12); do
    sleep 1
    if pgrep -f campaign_daemon.sh >/dev/null 2>&1 && [ -f "$CLAIM_DIR/daemon.alive.$MACHINE" ]; then
        echo "daemon: ALIVE  pid $(pgrep -f campaign_daemon.sh | head -1)  heartbeat $(cat "$CLAIM_DIR/daemon.alive.$MACHINE")"
        echo "log: $LOG   (its first campaign.sh pass prints below within ~a minute)"
        sleep 25; tail -n 20 "$LOG"
        exit 0
    fi
done
echo >&2
echo "FATAL: daemon did not come up. Log:" >&2
tail -n 20 "$LOG" >&2
exit 1
