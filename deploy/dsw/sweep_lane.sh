#!/usr/bin/env bash
# Reclaim ONE lane without touching the others.
#
#   bash deploy/dsw/sweep_lane.sh 1        # lane 1 -> GPUs 2,3 and /tmp/ray_lane1
#   bash deploy/dsw/sweep_lane.sh 1 3      # several
#
# For the case a campaign is half alive: two lanes hung and two are at step 126 with
# checkpoints on disk. `ray stop --force` would take the healthy ones too, and killing
# by hand is how the wrong process gets signalled. This is the same three nets _lane.sh
# uses, so the reasoning behind each is there; the short version is that Ray renames
# its workers with setproctitle, which erases both the command line AND the adjacent
# environ block, leaving only kernel bookkeeping to match on.

set -uo pipefail
SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GPUS_PER_RUN=${GPUS_PER_RUN:-2}
[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 2; }

for lane in "$@"; do
    tmp="${RAY_TMPDIR_BASE:-/tmp}/ray_lane${lane}"
    first=$((lane * GPUS_PER_RUN))
    gpus=$(seq -s, "$first" $((first + GPUS_PER_RUN - 1)))
    echo "=== lane $lane: GPUs [$gpus], $tmp"

    before=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpus" 2>/dev/null | wc -l)
    echo "  $before compute process(es) on those GPUs"

    for sig in TERM KILL; do
        # 2. raylet/gcs keep --temp_dir in argv. Anchored so lane1 cannot match lane10.
        pkill -"$sig" -f "${tmp}(/|$)" 2>/dev/null
        # 3. renamed python workers, identified by the GPU they hold. Each pid is
        #    confirmed to look like a training process first: inside a container
        #    nvidia-smi can report host pids that name something unrelated here.
        for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpus" 2>/dev/null); do
            [ -r "/proc/$pid/cmdline" ] || continue
            case "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" in
                *python*|*ray::*|*vllm*|*VLLM*|*raylet*) kill -"$sig" "$pid" 2>/dev/null ;;
            esac
        done
        [ "$sig" = TERM ] && sleep 5
    done

    # the lane worker itself, matched by the runs it was given
    pkill -f "LANE_RUNS.*" 2>/dev/null
    after=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpus" 2>/dev/null | wc -l)
    echo "  now $after"
    [ "$after" -gt 0 ] && { echo "  still occupied:" >&2
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv -i "$gpus" 2>/dev/null | sed 's/^/    /' >&2; }
    rm -rf "$tmp" 2>/dev/null
done

echo
echo "other lanes, untouched:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
