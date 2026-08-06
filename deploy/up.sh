#!/usr/bin/env bash
# Bring this machine into the campaign, or confirm it already is. One line, any box:
#
#   bash deploy/up.sh                 # registered machine
#   MACHINE=m2 bash deploy/up.sh      # first time on a box (registers it)
#
# Safe to run at any moment, repeatedly: everything underneath is idempotent --
# campaign.sh skips done runs, respects in-flight ones, claims atomically, and the
# daemon refuses to duplicate itself. This script adds nothing but the sequence.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CLAIM_DIR=${CLAIM_DIR:-.campaign}

echo "== up.sh on $(hostname) =="
git pull --ff-only -q && echo "repo: $(git rev-parse --short HEAD) (pulled)" \
    || echo "repo: $(git rev-parse --short HEAD) (pull failed -- proceeding with what is here)"

MACHINE=${MACHINE:-$(awk -F'\t' -v h="$(hostname)" '$1==h {print $2; exit}' "$CLAIM_DIR/MACHINE_MAP" 2>/dev/null)}
if [ -z "$MACHINE" ]; then
    echo "FATAL: this box is not registered and MACHINE was not given." >&2
    echo "       First time here:  MACHINE=<m1|m2|m3> bash deploy/up.sh" >&2
    exit 1
fi
# Registration (and every refusal campaign.sh can produce) surfaces here once,
# visibly, rather than inside a daemon log nobody reads.
MACHINE="$MACHINE" bash deploy/campaign.sh --dry || exit 1

if bash -c 'exec 9> "/tmp/simopd_daemon_'"$MACHINE"'.lock"; flock -n 9' 2>/dev/null; then
    nohup bash deploy/campaign_daemon.sh > "logs/$(hostname)_daemon.log" 2>&1 &
    sleep 2
    if kill -0 $! 2>/dev/null; then
        echo "daemon: started (pid $!, log logs/$(hostname)_daemon.log)"
    else
        echo "daemon: FAILED to start -- last lines of its log:"; tail -3 "logs/$(hostname)_daemon.log"
        exit 1
    fi
else
    echo "daemon: already running for $MACHINE (heartbeat in progress.py header)"
fi

command -v python >/dev/null && [ -f scripts/progress.py ] && {
    source "${SIMOPD_VENV:-simopd}/bin/activate" 2>/dev/null
    python scripts/progress.py 2>/dev/null | head -30
}
