#!/usr/bin/env bash
# Start this box's campaign daemon. Nothing else.
#
#   bash deploy/dsw/start_daemon.sh m1
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
    echo "daemon already running (pid $(pgrep -f campaign_daemon.sh | head -1)); nothing to do"
    exit 0
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
