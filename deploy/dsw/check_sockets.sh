#!/usr/bin/env bash
# Are the lanes' weight-transfer sockets actually distinct?
#
#   bash deploy/dsw/check_sockets.sh [seconds]
#
# This bug cost two rounds of lane time because nothing reported it: the losing lane
# blocks in send_pyobj with every engine resident, 0% GPU, and no error anywhere. The
# only outward sign is two PIDs bound to one path -- which is a one-line query nobody
# thought to run, so it is a script now.
#
# The sockets exist only during a weight transfer (verl unlinks them in _cleanup every
# step), so a single snapshot usually sees nothing. Sample for a while instead.
set -u
SECS="${1:-90}"
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT

echo "sampling ${SECS}s (sockets are only bound during a weight transfer)..."
END=$((SECONDS + SECS))
while [ $SECONDS -lt $END ]; do
    ss -xlp 2>/dev/null | grep -o '/tmp/rl-colocate-zmq-[^ ]*\.sock[^)]*)' >> "$TMP"
    sleep 0.2
done

echo
if [ ! -s "$TMP" ]; then
    echo "no weight-transfer socket seen in ${SECS}s."
    echo "  Either no lane is training, or every lane is already stuck (a blocked sender"
    echo "  stays bound, so it WOULD show). Check step counts before believing the former."
    exit 2
fi

# path -> how many distinct pids ever held it
awk '{ n=split($0,a,"pid="); split(a[2],b,","); print $1, b[1] }' "$TMP" \
    | sed 's/[0-9]*$//;s/ .*pid=/ /' >/dev/null 2>&1
grep -o '/tmp/rl-colocate-zmq-[^ ]*\.sock' "$TMP" | sort -u > "$TMP.paths"
echo "distinct socket paths seen: $(wc -l < "$TMP.paths")"
sed 's/^/  /' "$TMP.paths"

BAD=0
while read -r p; do
    pids=$(grep -F "$p" "$TMP" | grep -o 'pid=[0-9]*' | sort -u | tr '\n' ' ')
    n=$(printf '%s' "$pids" | wc -w)
    if [ "$n" -gt 1 ]; then
        echo
        echo "COLLISION: $p"
        echo "  held by $n processes: $pids"
        BAD=1
    fi
done < "$TMP.paths"

echo
if [ "$BAD" = 1 ]; then
    echo "-> Lanes share a socket path. _init_socket does an unconditional os.remove"
    echo "   before binding, so they delete each other's live socket once per step."
    echo "   The loser hangs with no error. Confirm the patch fired:"
    echo "     grep -h 'weight-transfer .* lane-scoped' <lane logs>"
    exit 1
fi
echo "-> No collision: every path is held by one process."
if ! grep -q -- '-replica-' "$TMP.paths" || ! grep -qE 'zmq-[^-]+-[0-9a-f]{8}-replica' "$TMP.paths"; then
    echo "   NOTE: paths carry no lane id. Distinct today by luck of timing, not by"
    echo "   construction -- expected after 'zmq_lane' is active is <jobid>-<8 hex>."
    exit 3
fi
echo "   Paths carry a lane id, so they are distinct by construction."
