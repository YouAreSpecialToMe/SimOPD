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

# Both must mirror run_parallel.sh exactly, or this reclaims the wrong lane. When a
# relaunch used RAY_TMPDIR_TAG=r2_ and GPU_LIST="2,3 6,7", lane 0 means /tmp/ray_laner2_0
# on GPUs 2,3 -- while the defaults here would compute /tmp/ray_lane0 on GPUs 0,1,
# which is a DIFFERENT, healthy lane. Reclaiming a stuck lane by killing a running one
# is the exact accident this script exists to prevent.
#   RAY_TMPDIR_TAG=r2_ GPU_LIST="2,3 6,7" bash deploy/dsw/sweep_lane.sh 0 1
for lane in "$@"; do
    tmp="${RAY_TMPDIR_BASE:-/tmp}/ray_lane${RAY_TMPDIR_TAG:-}${lane}"
    if [ -n "${GPU_LIST:-}" ]; then
        gpus=$(set -- $GPU_LIST; eval echo "\${$((lane + 1))}")
        [ -n "$gpus" ] || { echo "GPU_LIST has no entry for lane $lane" >&2; continue; }
    else
        first=$((lane * GPUS_PER_RUN))
        gpus=$(seq -s, "$first" $((first + GPUS_PER_RUN - 1)))
    fi
    # Refuse to act on a lane whose temp dir does not exist: that means the tag or the
    # number is wrong, and the GPUs computed alongside it are then someone else's.
    if [ ! -d "$tmp" ]; then
        echo "=== lane $lane: no $tmp -- wrong RAY_TMPDIR_TAG or lane number?" >&2
        echo "    existing lane temp dirs:" >&2
        ls -d "${RAY_TMPDIR_BASE:-/tmp}"/ray_lane* 2>/dev/null | sed 's/^/      /' >&2
        echo "    refusing, because the GPUs computed for it belong to another lane" >&2
        continue
    fi
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

    # The lane worker itself. It cannot be matched by command line -- every lane runs
    # the identical `bash deploy/dsw/_lane.sh` -- and `pkill -f LANE_RUNS` (what this
    # first did) is worse than useless: env assignments are not part of argv so it
    # matches nothing, and were it ever to match it is not lane-scoped and would take
    # the healthy lanes too. The lane worker is plain bash, never renamed by
    # setproctitle, so unlike a Ray worker its environment IS readable -- and that is
    # what says which lane it belongs to.
    for pid in $(pgrep -u "$(id -u)" -f "[_]lane\.sh" 2>/dev/null); do
        if { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null |
                grep -qxF "RAY_TMPDIR=$tmp"; then
            echo "  lane worker pid $pid belongs to lane $lane -- stopping it"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    after=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpus" 2>/dev/null | wc -l)
    echo "  now $after"
    [ "$after" -gt 0 ] && { echo "  still occupied:" >&2
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv -i "$gpus" 2>/dev/null | sed 's/^/    /' >&2; }
    rm -rf "$tmp" 2>/dev/null
done

echo
echo "other lanes, untouched:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
