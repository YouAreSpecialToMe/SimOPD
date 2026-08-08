#!/usr/bin/env bash
# Stop the pilot-era workload on THIS box -- daemon, lane shepherds, training,
# teacher vLLM -- so the S-wave bring-up starts clean. Run on each of m1/m2/m3
# (any order), verify the final "clean" line, then run deploy/dsw/launch_mX.sh.
#
# Why hard-stop is safe here: the 8k batch is pilot data by amendment (sec 3.8);
# ckpts up to each run's last SAVE boundary survive; the retired manifest waves
# cannot relaunch anything; launch_m1's migration renames the names regardless.
# Why daemon-stop alone is NOT enough: a lane drains the list it was handed at
# launch -- finish one run, start the next -- without ever re-reading the
# manifest. The shepherds have to go before the GPUs are swept, or a dying run
# is simply replaced by its lane's next one.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CLAIM_DIR=${CLAIM_DIR:-.campaign}
source "${SIMOPD_VENV:-simopd}/bin/activate" 2>/dev/null || true

_ME=$(awk -F'\t' -v h="$(hostname)" '$1==h {print $2; exit}' "$CLAIM_DIR/MACHINE_MAP" 2>/dev/null)
echo "=== daemon: stop THIS machine's (${_ME:-unregistered}); pass --all for every machine ==="
if [ "${1:-}" = "--all" ]; then
    for m in m1 m2 m3; do touch "$CLAIM_DIR/daemon.stop.$m" 2>/dev/null || true; done
    rm -f "$CLAIM_DIR"/daemon.alive.m* "$CLAIM_DIR"/daemon.status.m* 2>/dev/null || true
elif [ -n "$_ME" ]; then
    touch "$CLAIM_DIR/daemon.stop.$_ME" 2>/dev/null || true
    # Stale heartbeat/status would trip the launch scripts' is-a-daemon-alive guard.
    rm -f "$CLAIM_DIR/daemon.alive.$_ME" "$CLAIM_DIR/daemon.status.$_ME" 2>/dev/null || true
fi
pkill -f campaign_daemon.sh 2>/dev/null && echo "  daemon process on this box killed" \
                                        || echo "  no daemon process on this box"

echo "=== lane shepherds, before the GPUs ==="
pkill -f "dsw/_lane.sh" 2>/dev/null || true
pkill -f "dsw/run_parallel.sh" 2>/dev/null || true
sleep 3

echo "=== GPU processes (training + teacher vLLM): TERM, grace, then KILL ==="
_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | sort -u)
if [ -n "$_pids" ]; then
    echo "$_pids" | xargs -r kill 2>/dev/null || true
    sleep 20
    _left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | sort -u)
    [ -n "$_left" ] && { echo "$_left" | xargs -r kill -9 2>/dev/null || true; sleep 5; }
else
    echo "  nothing on the GPUs"
fi
command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true

echo "=== stamp FAIL markers into THIS machine's now-dead logs ==="
# Everything local is dead by construction at this point, so every unfinished
# RUN marker in this machine's logs is a kill we just performed (or an older
# one) -- write the end it will never write itself, or the corpse renders
# RUNNING forever, blocks migration, and counts as in-flight for 6 hours.
for _f in "logs/${_ME:-__none__}"/lane*.log $([ "$_ME" = m1 ] && echo logs/lane*.log); do
    [ -f "$_f" ] || continue
    grep -oE '^#+ RUN: [A-Za-z0-9_.]+' "$_f" 2>/dev/null | awk '{print $3}' | sort -u | \
    while read -r _n; do
        grep -qE "^#+ ${_n} -> (OK|FAIL)" "$_f" || {
            echo "## ${_n} -> FAIL (hard-stopped by stop_pilot $(date -u +%FT%TZ))" >> "$_f"
            echo "  stamped FAIL: ${_n}  ($_f)"
        }
    done
done

echo "=== after ==="
_n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)
if [ "$_n" = 0 ]; then
    _me=$(awk -F'\t' -v h="$(hostname)" '$1==h {print $2; exit}' "$CLAIM_DIR/MACHINE_MAP" 2>/dev/null)
    echo "clean: 0 compute processes on this box."
    echo "next:  bash deploy/dsw/launch_${_me:-mX}.sh"
else
    echo "still $_n GPU process(es) -- look before forcing further:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi
