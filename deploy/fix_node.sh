#!/usr/bin/env bash
# State-aware bring-up/repair for one box. Invoked by fix_m1.sh / fix_m2.sh / fix_m3.sh.
#
# Decides from the machine's ACTUAL state, because the operator runs this at an
# unknown moment: a box that has launched healthy lanes since the last look must not
# have them killed by instructions written for the state before.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CLAIM_DIR=${CLAIM_DIR:-.campaign}
EXPECT=${1:?wrapper passes the machine name}

MACHINE=$(awk -F'\t' -v h="$(hostname)" '$1==h {print $2; exit}' "$CLAIM_DIR/MACHINE_MAP" 2>/dev/null)
if [ -z "$MACHINE" ]; then
    echo "this box is unregistered; registering as $EXPECT (first launch here)"
    MACHINE=$EXPECT
elif [ "$MACHINE" != "$EXPECT" ]; then
    echo "FATAL: this box is registered as '$MACHINE' but you ran fix_${EXPECT}.sh." >&2
    echo "       Running another machine's script here is the wrong-box accident;" >&2
    echo "       use fix_${MACHINE}.sh on this box." >&2
    exit 1
fi
echo "== fix_node on $MACHINE ($(hostname)) @ $(date -u +%FT%TZ) =="

# --- 1. retire the old daemon (it may be mid-sleep for up to 15 min and would not
#        see a stop file; the lock file holds its pid, so retire it directly) -------
_pid=$(cat "/tmp/simopd_daemon_${MACHINE}.v2.lock" 2>/dev/null | head -1)
if [ -n "${_pid:-}" ] && grep -q campaign_daemon "/proc/$_pid/cmdline" 2>/dev/null; then
    kill "$_pid" 2>/dev/null && echo "daemon: retired old instance (pid $_pid)"
else
    for _p in $(pgrep -f '[c]ampaign_daemon'); do
        kill "$_p" 2>/dev/null && echo "daemon: retired old instance (pid $_p)"
    done
fi
rm -f "$CLAIM_DIR/daemon.stop.$MACHINE"

# --- 2. processes: three states, three answers ------------------------------------
if pgrep -f '[_]lane.sh' >/dev/null 2>&1; then
    echo "lanes: LIVE lane driver(s) present -- touching no process on this box"
    echo "       $(pgrep -fc '[_]lane.sh') driver(s): $(pgrep -f '[_]lane.sh' | tr '\n' ' ')"
else
    _left=""
    for _p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        case "$(tr '\0' ' ' < "/proc/$_p/cmdline" 2>/dev/null)" in
            *python*|*ray*|*vllm*|*VLLM*) _left="$_left $_p" ;;
        esac
    done
    if [ -n "$_left" ]; then
        echo "lanes: no drivers, but GPU leftovers:$_left -- reclaiming"
        for _s in TERM KILL; do
            for _p in $_left; do kill -"$_s" "$_p" 2>/dev/null; done
            [ "$_s" = TERM ] && sleep 5
        done
    else
        echo "lanes: none running, GPUs clean"
    fi
fi

# --- 3. failure digest, for pasting rather than describing ------------------------
_report="logs/${MACHINE}_failures_$(date -u +%H%M%S).txt"
_found=0
{
    echo "== failure digest for $MACHINE @ $(date -u +%FT%TZ) =="
    for _f in logs/lane*.log "logs/$MACHINE"/lane*.log; do
        [ -f "$_f" ] || continue
        grep -q -- "-> FAIL" "$_f" || continue
        _found=1
        echo; echo "──── $_f (tail of final attempt) ────"
        tail -60 "$_f"
    done
    echo; echo "──── deduplicated error spectrum (all logs) ────"
    grep -hE "Error|Exception|FATAL|Traceback|refuse" logs/lane*.log "logs/$MACHINE"/lane*.log 2>/dev/null \
        | sort | uniq -c | sort -rn | head -15
} > "$_report" 2>/dev/null
if [ "$_found" = 1 ] || [ -s "$_report" ]; then
    echo "failures: digest written -> $_report  (paste this file)"
    grep -m4 -A0 "────" "$_report" | sed 's/^/   /'
else
    rm -f "$_report"; echo "failures: none recorded on this box"
fi

# --- 4. fresh daemon (status-reporting build) + visible dry + progress ------------
[ "${SKIP_UP:-0}" = 1 ] && { echo "(SKIP_UP=1: stopping before up.sh -- test hook)"; exit 0; }
MACHINE="$MACHINE" bash deploy/up.sh
