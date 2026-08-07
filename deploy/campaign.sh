#!/usr/bin/env bash
# Queue this machine's share of the campaign.
#
#   bash deploy/campaign.sh --fingerprint       # is this box the same as the others?
#   bash deploy/campaign.sh --fs-probe          # do the machines share a filesystem?
#   REASON="..." bash deploy/campaign.sh --repin   # move the pin, on the record
#   bash deploy/campaign.sh --plan              # whole manifest, audited, nothing run
#   MACHINE=m2 bash deploy/campaign.sh --dry    # what m2 would launch, and why
#   MACHINE=m2 bash deploy/campaign.sh          # launch it
#   MACHINE=m2 bash deploy/campaign.sh --machine-control
#
# configs/campaign.tsv says who runs what; this reads that, subtracts what this
# machine has already finished, takes only the GPUs nothing is using, and hands the
# remainder to run_parallel.sh -- which keeps its pre-flight, its snapshot, and its
# per-lane isolation. Nothing here re-implements those.
#
# Safe to re-run. A machine that was interrupted picks up where it stopped: finished
# runs are skipped outright and a half-finished one resumes from its checkpoint, which
# run_opd_baseline.sh now refuses to do if the config on disk disagrees.

set -uo pipefail

SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$SIMOPD_ROOT"
MANIFEST=${MANIFEST:-configs/campaign.tsv}
LOG_DIR=${LOG_DIR:-$SIMOPD_ROOT/logs}
# The protocol-batch tag. Resolved HERE, before any row->name matching: a row's
# log name is "${arm}_s${seed}_${TAG}" once a tag is set, and matching without it
# would let the 8k pilot's "vanilla_s0 -> OK" mark the 16k rerun of the same row
# as already done -- silently, fleet-wide, via the shared logs/ directory.
TAG="${TAG:-$(cat "${CLAIM_DIR:-.campaign}/BATCH_TAG" 2>/dev/null || true)}"
# A batch tag makes every OLD row look unfinished: wave-1's "vanilla_s0 -> OK" no
# longer matches the tagged name "vanilla_s0_16k", so without a floor the fleet
# would quietly re-run its entire pre-16k history -- with each arm's seeds packed
# on the machines that happened to run them last time. The floor says which waves
# BELONG to the current batch; rows below it are prior-protocol history, not work.
BATCH_MIN_WAVE="$(cat "${CLAIM_DIR:-.campaign}/BATCH_MIN_WAVE" 2>/dev/null || echo 0)"

# One value for the whole campaign. 0.55 is the script default and is faster, but a
# second value makes a second batch, and a second batch needs its own vanilla x3 noise
# floor -- ~168 GPU-hours against the ~73 that 0.55 saves across 13 runs. It is also
# not numerically neutral: rollout sampling uses one engine-level seed, so cache size
# changes batch composition and with it the RNG stream. See configs/campaign.tsv.
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}

MODE=run
for a in "$@"; do
    case "$a" in
        --plan) MODE=plan ;;
        --dry|--dry-run) MODE=dry ;;
        --machine-control) MODE=control ;;
        --fingerprint) MODE=fingerprint ;;
        --fs-probe) MODE=fsprobe ;;
        --repin) MODE=repin ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# --fingerprint: everything that could make one box compute a different number.
#
# All four run the same image, which settles the software half -- but the driver is
# host-side and outside the image, and the image cannot make an A100 out of whatever
# is actually in the box. Diff this across machines. Identical output means the
# 50-step machine control below is a formality; a difference means it is required,
# because every arm is measured against a vanilla x3 floor that lives on m1 and a
# machine effect would ride silently on top of every one of them.
# ---------------------------------------------------------------------------
if [ "$MODE" = fingerprint ]; then
    source "${SIMOPD_VENV:-simopd}/bin/activate" 2>/dev/null
    echo "host          $(hostname)"
    echo "gpu           $(nvidia-smi 2>/dev/null --query-gpu=name --format=csv,noheader | sort -u | tr '\n' '/')"
    echo "gpu count     $(nvidia-smi 2>/dev/null -L | wc -l)"
    echo "driver        $(nvidia-smi 2>/dev/null --query-gpu=driver_version --format=csv,noheader | sort -u)"
    echo "cuda(smi)     $(nvidia-smi 2>/dev/null | awk -F'CUDA Version: *' '/CUDA Version/{print $2}' | tr -d ' |')"
    python - <<'PYEOF' 2>/dev/null
import torch, importlib
print(f"torch         {torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")
for m in ("vllm", "transformers", "flash_attn", "ray"):
    try:
        print(f"{m:13s} {importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"{m:13s} <{type(e).__name__}>")
print(f"tf32 matmul   {torch.backends.cuda.matmul.allow_tf32}")
print(f"sdp kernels   flash={torch.backends.cuda.flash_sdp_enabled()} mem={torch.backends.cuda.mem_efficient_sdp_enabled()}")
PYEOF
    echo "repo          $(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
    exit 0
fi

# ---------------------------------------------------------------------------
# --fs-probe: is CLAIM_DIR really shared? One machine writing and reading its own
# marker proves nothing -- run this on every machine, then look for the others.
# ---------------------------------------------------------------------------
if [ "$MODE" = fsprobe ]; then
    : "${CLAIM_DIR:=${SIMOPD_SHARED:-$SIMOPD_ROOT/.campaign}}"
    mkdir -p "$CLAIM_DIR/_probe" 2>/dev/null || {
        echo "cannot write $CLAIM_DIR/_probe -- set CLAIM_DIR to a path you expect to be shared" >&2
        exit 1; }
    printf '%s\t%s\n' "$(hostname)" "$(date -u +%FT%TZ)" > "$CLAIM_DIR/_probe/$(hostname)"
    echo "wrote  $CLAIM_DIR/_probe/$(hostname)"
    echo
    echo "markers visible from here:"
    for f in "$CLAIM_DIR"/_probe/*; do
        [ -f "$f" ] && printf '  %-24s %s\n' "$(basename "$f")" "$(cat "$f" | cut -f2)"
    done
    n=$(ls -1 "$CLAIM_DIR/_probe" 2>/dev/null | wc -l)
    echo
    if [ "$n" -le 1 ]; then
        echo "only this machine so far. Run --fs-probe on the OTHERS, then run it here"
        echo "again: seeing one marker proves the directory exists, not that it is shared."
    else
        echo "$n machines visible -> CLAIM_DIR is shared. Rows with machine 'any' in the"
        echo "manifest can then be claimed at runtime instead of assigned by hand."
    fi
    exit 0
fi

rows() {   # wave, machine, arm, seed, note -- comments and blank lines gone
    awk -F'\t' '!/^[[:space:]]*#/ && NF>=4 { print $1, $2, $3, $4, $5 }' "$MANIFEST"
}

# ---------------------------------------------------------------------------
# --plan: audit the manifest against the registry.
# A missing arm is a hole in the paper that nothing else would report -- the lanes
# would all run green and one column would simply never exist.
# ---------------------------------------------------------------------------
if [ "$MODE" = plan ]; then
    source "${SIMOPD_VENV:-simopd}/bin/activate" 2>/dev/null
    export PYTHONPATH="$SIMOPD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    echo "=== manifest: $MANIFEST ==="
    printf '%-5s %-9s %-26s %s\n' wave machine arm seed
    rows | sort -k1,1n -k2,2 | while read -r w m arm s; do
        printf '%-5s %-9s %-26s %s\n' "$w" "$m" "$arm" "$s"
    done
    echo
    echo "=== per machine ==="
    rows | awk '{c[$2]++} END {for (m in c) printf "  %-9s %2d runs -> %d lanes x 2 GPUs\n", m, c[m], c[m]}' | sort
    echo
    echo "=== coverage against configs/arms.yaml ==="
    _stock=$(python scripts/arm.py list --status stock 2>/dev/null)
    [ -n "$_stock" ] || { echo "  cannot read the registry; is PYTHONPATH/venv right?" >&2; exit 1; }
    _miss=0
    for arm in $_stock; do
        n=$(rows | awk -v a="$arm" '$3==a' | wc -l)
        [ "$n" -ge 1 ] || { printf '  MISSING  %-26s runnable, assigned to nobody\n' "$arm"; _miss=1; }
    done
    rows | awk '{print $3}' | sort -u | while read -r arm; do
        echo "$_stock" | grep -qx "$arm" || printf '  UNKNOWN  %-26s in the manifest, not in the registry\n' "$arm"
    done
    for arm in $(python scripts/arm.py list --status needs 2>/dev/null); do
        printf '  blocked  %-26s registry status=needs, correctly absent\n' "$arm"
    done
    [ "$_miss" = 0 ] && echo "  every runnable arm is assigned exactly once or more"
    echo
    echo "=== duplicates (same arm+seed assigned twice = the same work run twice) ==="
    d=$(rows | awk '{print $3"_s"$4}' | sort | uniq -d)
    [ -z "$d" ] && echo "  none" || { echo "$d" | sed 's/^/  DUPLICATE /'; exit 1; }
    exit 0
fi

# ---------------------------------------------------------------------------
# The pool. A row whose machine is 'any' belongs to whichever box claims it first,
# which needs a shared filesystem -- verify with --fs-probe before using it, because
# an UNSHARED CLAIM_DIR makes every machine claim everything and silently run the
# whole campaign several times over. mkdir is the claim: it is atomic and it fails
# for the loser, which is the property a lock file written with > does not have.
# ---------------------------------------------------------------------------
: "${CLAIM_DIR:=${SIMOPD_SHARED:-$SIMOPD_ROOT/.campaign}}"

claim() {   # claim <name> -> 0 if this machine now owns it
    mkdir -p "$CLAIM_DIR/claims" 2>/dev/null || return 1
    mkdir "$CLAIM_DIR/claims/$1" 2>/dev/null || return 1
    printf 'machine=%s host=%s at=%s\n' "$MACHINE" "$(hostname)" "$(date -u +%FT%TZ)" \
        > "$CLAIM_DIR/claims/$1/owner"
    return 0
}
claim_owner() { cat "$CLAIM_DIR/claims/$1/owner" 2>/dev/null; }

# ---------------------------------------------------------------------------
# Which machine is this? m1/m2/m3 are manifest labels, and a box does not know its
# own label -- the operator did, per launch, from memory. Typing m2 on the box that
# already ran as m1 double-assigns m2's rows and orphans m1's, silently. So the first
# launch REGISTERS hostname->label in the shared map; after that MACHINE can be
# omitted, and a label that contradicts the map is refused instead of obeyed.
# ---------------------------------------------------------------------------
_map="$CLAIM_DIR/MACHINE_MAP"
_mapped=$(awk -F'\t' -v h="$(hostname)" '$1==h {print $2; exit}' "$_map" 2>/dev/null)
MACHINE=${MACHINE:-$_mapped}
[ -n "$MACHINE" ] || {
    echo "FATAL: this box ($(hostname)) is not yet registered, so it must be told its" >&2
    echo "       manifest name ONCE:   MACHINE=<name> bash deploy/campaign.sh --dry" >&2
    echo "       That records hostname->name in $_map; afterwards MACHINE can be omitted." >&2
    echo "       Names in $MANIFEST: $(rows | awk '{print $2}' | sort -u | tr '\n' ' ')" >&2
    [ -f "$_map" ] && { echo "       already registered:" >&2; sed 's/^/         /' "$_map" >&2; }
    exit 1
}
if [ -n "$_mapped" ] && [ "$_mapped" != "$MACHINE" ]; then
    echo "FATAL: $(hostname) is registered as '$_mapped' but MACHINE=$MACHINE was given." >&2
    echo "       Obeying it would run another machine's rows here. If the registration" >&2
    echo "       itself is wrong, edit $_map." >&2
    exit 1
fi
_other=$(awk -F'\t' -v m="$MACHINE" -v h="$(hostname)" '$2==m && $1!=h {print $1; exit}' "$_map" 2>/dev/null)
if [ -n "$_other" ]; then
    echo "FATAL: '$MACHINE' is already registered to $_other. Two boxes answering to one" >&2
    echo "       name is the duplicate-run accident by another door. Pick this box's own" >&2
    echo "       name, or fix $_map." >&2
    exit 1
fi
if [ -z "$_mapped" ]; then
    mkdir -p "$CLAIM_DIR"
    printf '%s\t%s\t%s\n' "$(hostname)" "$MACHINE" "$(date -u +%FT%TZ)" >> "$_map"
    echo "  registered: $(hostname) = $MACHINE ($_map)"
fi

# A unique tmpdir tag PER INVOCATION. Lane numbering restarts at 0 every launch, so a
# second invocation on a box with surviving lanes computed the same /tmp/ray_lane<m>_0
# as a cluster still alive under it -- and run_parallel's attach-guard then correctly
# refused, wedging the machine: m1 could not top up because its own healthy vanilla_s1
# owned lane 0's tmpdir. Timestamping the tag makes the collision impossible instead
# of guarded; GPU ownership (not tags) is what prevents double-launches, and fix_node
# reclaims by GPU truth, so nothing else keyed on tag stability.
LAUNCH_TAG="${MACHINE}_$(date +%s)_"
# One launch at a time per machine. A daemon firing while a human runs campaign.sh
# by hand would have both count the same free GPUs and both launch lanes onto them.
# The lock is machine-LOCAL (/tmp) on purpose: the race being closed is same-machine
# concurrency, and flock on the shared NAS mount is exactly the semantics not to
# depend on. plan/probe/fingerprint modes take no lock -- they launch nothing.
# .v2 in the lock path, and 9>&- on every launch below: nohup'd lanes inherit open
# file descriptors, so the flock taken here was held by the entire ray/vllm tree for
# the lifetime of the lanes -- every machine with running lanes refused all further
# campaign invocations, manual and daemon alike, as "mid-launch". Closing the fd in
# the children fixes the cause; versioning the path releases the fleet from locks the
# CURRENTLY-running lanes already hold, without touching them.
if [ "$MODE" = run ] || [ "$MODE" = control ]; then
    # Every launching invocation records its own transcript on the shared mount.
    # Manual runs used to exist only in someone's terminal scrollback: three times
    # today the words "还是有点问题" arrived without the output that would have
    # named the problem in one read, and diagnosis stalled on asking for a paste.
    # The daemon already records itself; now the human path does too.
    mkdir -p logs
    exec > >(tee "logs/campaign_last_${MACHINE}.txt") 2>&1
    echo "(transcript: logs/campaign_last_${MACHINE}.txt @ $(date -u +%FT%TZ), HEAD $(git rev-parse --short HEAD 2>/dev/null))"
    exec 9> "/tmp/simopd_campaign_${MACHINE}.v2.lock"
    flock -n 9 || {
        echo "FATAL: another campaign.sh invocation is mid-launch on this machine" >&2
        echo "       (daemon and manual run overlapping is the usual cause; wait a" >&2
        echo "       minute or stop the daemon: touch .campaign/daemon.stop.$MACHINE)" >&2
        exit 1
    }
fi

# A machine participates if the manifest names it OR the pool has rows -- a box with
# zero named rows and an open pool is exactly what m4 will be on arrival, and the old
# named-rows-only gate refused it outright. (The gate also produced a vacuously "green"
# audit test: nothing was claimed because the run was refused, and an assertion that
# checked for the claim's absence read that refusal as a successful release.)
rows | awk -v m="$MACHINE" -v A=any '$2==m || $2==A' | grep -q . || {
    echo "FATAL: '$MACHINE' has no rows in $MANIFEST and the pool is empty." >&2
    echo "       Known: $(rows | awk '{print $2}' | sort -u | tr '\n' ' ')" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# One commit for the whole campaign.
#
# The machines share a repo -- a git pull on one is a pull on all -- so code can change
# between one machine's runs and another's. That is invisible: the config fingerprint
# in run_opd_baseline.sh records config, not code, so editing a loss in src/ leaves it
# unchanged while the arms become incomparable in a dimension nothing recorded.
#
# The snapshot run_parallel.sh takes makes each RUN internally consistent; it cannot
# make two runs launched hours apart agree. Only a pinned ref can, and a shared
# filesystem is exactly what makes it enforceable across machines rather than a note.
#
# (Pulling while lanes run is safe on its own: measured, git checkout replaces the
# inode, so a running bash keeps reading the file it started with. An in-place rewrite
# -- an editor, sed -i -- does corrupt it, which is what campaign.sbatch warns about.)
# ---------------------------------------------------------------------------
_head=$(git rev-parse HEAD 2>/dev/null || echo nogit)
_ref_file="$CLAIM_DIR/CAMPAIGN_REF"
# The pin gate compares two COMMITS; uncommitted edits to run-defining files would
# otherwise be snapshotted into runs with no record (audit 2026-08-07 F7).
_dirty=$(git status --porcelain -- src/ configs/arms.yaml scripts/run_opd_baseline.sh scripts/arm.py 2>/dev/null)
if [ -n "$_dirty" ] && [ "$MODE" != plan ] && [ "$MODE" != repin ]; then
    echo "FATAL: uncommitted changes in run-defining files -- commit (and repin) first:" >&2
    echo "$_dirty" | sed 's/^/       /' >&2
    exit 1
fi

# Re-pinning is sometimes correct -- 62b33e6 could not run on any machine, because
# src/simopd/zmq_lane.py was missing from git. What must not happen is re-pinning
# leaving no trace. "Note it in the ledger" is advice nobody follows at 2am, so the
# move writes its own entry: old ref, new ref, who, when, and why.
if [ "$MODE" = repin ]; then
    mkdir -p "$CLAIM_DIR"
    _was=$(cut -d' ' -f1 < "$_ref_file" 2>/dev/null || echo "<unpinned>")
    [ -n "${REASON:-}" ] || {
        echo "FATAL: set REASON. A pin move that does not say why is the drift it exists" >&2
        echo "       to catch, one level up." >&2
        echo "       REASON=\"zmq_lane.py was missing from git; 62b33e6 cannot run\" \\" >&2
        echo "         MACHINE=$MACHINE bash deploy/campaign.sh --repin" >&2
        exit 1; }
    printf '%s pinned-by=%s at=%s\n' "$_head" "${MACHINE:-?}@$(hostname)" "$(date -u +%FT%TZ)" > "$_ref_file"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$_was" "$_head" \
        "${MACHINE:-?}@$(hostname)" "$REASON" >> "$CLAIM_DIR/PIN_HISTORY"
    echo "re-pinned $_was -> $_head"
    echo "  reason: $REASON"
    echo "  history: $CLAIM_DIR/PIN_HISTORY ($(wc -l < "$CLAIM_DIR/PIN_HISTORY") entries)"
    echo
    echo "Runs launched before this pin used the old code. Whether they stay comparable"
    echo "is a judgement about the diff, and it belongs in the paper's appendix:"
    git diff --stat "$_was" "$_head" -- scripts src configs 2>/dev/null | sed 's/^/    /'
    exit 0
fi
if [ -f "$_ref_file" ]; then
    _ref=$(cut -d' ' -f1 < "$_ref_file")
    if [ "$_ref" != "$_head" ] && [ "$_head" != nogit ]; then
        # Only files whose content actually flows into a training process. The first
        # version gated all of scripts/ and blocked a launch because progress.py -- a
        # read-only monitor -- had been added; refusing over a monitoring script is
        # this check crying wolf, the same failure the fingerprint avoids by exempting
        # SIMOPD_SHADOW. Monitoring, eval tooling and deploy glue move the pin with a
        # printed note instead. configs/campaign.tsv is deliberately out too: it
        # assigns work, it does not define what a run computes, and editing waves
        # mid-campaign is normal operation.
        _RUN_DEFINING="src configs/arms.yaml scripts/run_opd_baseline.sh scripts/arm.py"
        _drift=$(git diff --name-only "$_ref" "$_head" -- $_RUN_DEFINING 2>/dev/null)
        if [ -n "$_drift" ]; then
            echo "FATAL: the campaign is pinned to $_ref (see $_ref_file)" >&2
            echo "       and HEAD is $_head. Files that decide what a run computes changed:" >&2
            echo "$_drift" | sed 's/^/         /' >&2
            echo "       Arms launched on either side of this are not comparable, and nothing" >&2
            echo "       downstream would say so -- the config fingerprint records config," >&2
            echo "       not code." >&2
            echo "       git checkout $_ref                    # run the campaign's code" >&2
            echo "       REASON=\"why\" MACHINE=$MACHINE bash deploy/campaign.sh --repin" >&2
            echo "                                             # OR move the pin, on the record" >&2
            exit 1
        fi
        _moved=$(git diff --name-only "$_ref" "$_head" 2>/dev/null | head -6 | tr '\n' ' ')
        echo "  note: HEAD moved since the pin ($_moved...) but nothing run-defining" >&2
        echo "        (src/, arms.yaml, run_opd_baseline.sh, arm.py) -- runs stay comparable." >&2
    fi
else
    mkdir -p "$CLAIM_DIR" 2>/dev/null
    printf '%s pinned-by=%s at=%s\n' "$_head" "${MACHINE}@$(hostname)" "$(date -u +%FT%TZ)" > "$_ref_file"
    echo "  pinned campaign to $_head ($_ref_file)"
fi

# ---------------------------------------------------------------------------
# What this machine has already done, read from the lane logs _lane.sh writes.
# No new bookkeeping file: the summary line is already the record, and a marker
# written by something other than the run itself can outlive the thing it describes.
# ---------------------------------------------------------------------------
# OK and FAIL are NOT the same thing. An earlier version put both in `finished`, which
# meant a run that crashed was skipped on every later invocation -- the lanes stay
# green, the arm never gets written, and the only symptom is a column missing from the
# paper. FAIL is retried (it resumes from its checkpoint) and reported, because a run
# that keeps failing needs an operator, not a quiet retry.
done_ok=" "; failed=" "; recent=" "
declare -A _started _ended _failn
if [ -d "$LOG_DIR" ]; then
    # Both layouts: logs/lane*.log from before machines were separated, and
    # logs/<machine>/lane*.log from now on -- which matters on a shared filesystem,
    # where two boxes starting in the same second would otherwise write one filename.
    for f in "$LOG_DIR"/lane*.log "$LOG_DIR"/*/lane*.log; do
        [ -f "$f" ] || continue
        # done/FAIL evidence is fleet-wide (pool rows finish on other machines), but
        # IN-FLIGHT must be machine-local: another box's fresh log otherwise satisfies
        # the unknown-GPU-user cross-check and masks a local zombie -- the exact
        # double-launch the guard exists to prevent (audit 2026-08-07 F3).
        case "$f" in
            "$LOG_DIR"/lane*.log|"$LOG_DIR/${MACHINE:-__none__}"/lane*.log) _mine_log=1 ;;
            *) _mine_log=0 ;;
        esac
        while read -r n; do done_ok="$done_ok$n "; _ended[$n]=$(( ${_ended[$n]:-0} + 1 )); done < <(
            grep -oE '^#+ [A-Za-z0-9_.]+ -> OK' "$f" 2>/dev/null | awk '{print $2}')
        while read -r n; do failed="$failed$n "; _ended[$n]=$(( ${_ended[$n]:-0} + 1 ))
            _failn[$n]=$(( ${_failn[$n]:-0} + 1 )); done < <(
            grep -oE '^#+ [A-Za-z0-9_.]+ -> FAIL' "$f" 2>/dev/null | awk '{print $2}')
        while read -r n; do _started[$n]=$(( ${_started[$n]:-0} + 1 )); done < <(
            grep -oE '^#+ RUN: [A-Za-z0-9_.]+' "$f" 2>/dev/null | awk '{print $3}')
        # How long may a HEALTHY run be silent? Not 30 minutes, which is what this
        # said first and is the fourth time on this project that a threshold was
        # picked from what failure looks like without checking what health looks
        # like. A validation is ~75 minutes at this tier, and a run in one writes
        # nothing at all -- so 30 minutes calls a healthy run dead, and the machine
        # then offers to start a second copy of it. Six hours clears the longest
        # legitimate silence with room; the GPU cross-check below is what catches a
        # genuine crash, and being conservative here errs toward not duplicating.
        if [ "$_mine_log" = 1 ] && [ -n "$(find "$f" -mmin -$(( ${INFLIGHT_HOURS:-6} * 60 )) 2>/dev/null)" ]; then
            while read -r n; do recent="$recent$n "; done < <(
                grep -oE '^#+ RUN: [A-Za-z0-9_.]+' "$f" 2>/dev/null | awk '{print $3}')
        fi
        :
    done
fi

# Two lists, not one. Rows named for this machine are its own; 'any' rows are the pool
# and are only CLAIMED once the free-GPU count is known, because a machine that claims
# what it cannot start holds it away from the machine that could. A first version
# claimed during the manifest walk -- which also meant --dry took the whole pool and
# left nothing for anyone, a side effect a dry run must not have.
mine=""; pool=""; skipped=""; live=""; retry=""; quarantined=""
while read -r w m arm s note; do
    [ "$w" -lt "$BATCH_MIN_WAVE" ] && continue   # prior-batch history, see floor note
    name="${arm}_s${s}${TAG:+_$TAG}"
    case "$done_ok" in *" $name "*) skipped="$skipped $name(done)"; continue ;; esac
    # needs=<path>: a prerequisite this machine may not have. a2_coldstart pins an SFT
    # checkpoint under /scratch, which exists on no DSW box; unchecked it sorted first
    # in the pool, would have taken a lane, failed, and been retried on every pass.
    case "$note" in
        needs=*)
            _need=${note#needs=}
            [ -e "$_need" ] || { skipped="$skipped $name(needs $_need)"; continue; } ;;
    esac
    # Three strikes and the run is quarantined instead of retried. The daemon makes
    # retries unattended, and an arm with a real bug would otherwise crash-loop all
    # night, burning a lane per attempt with nobody watching. Three attempts is enough
    # to rule out transience; past that the failure wants eyes, not repetition.
    case "$failed" in *" $name "*)
        if [ "${_failn[$name]:-0}" -ge "${MAX_RUN_RETRIES:-3}" ]; then
            quarantined="$quarantined $name(${_failn[$name]}x)"
            continue
        fi
        retry="$retry $name" ;;
    esac
    # In flight = a recent log names it AND it has more starts than results. A run
    # whose FAIL is on disk is definitionally not in flight, but the lane log it sits
    # in stays fresh for hours while the lane works through its remaining runs -- the
    # old "log moved recently" test therefore held every FAILed run hostage until its
    # whole lane went quiet, which for a 4-run lane is more than a day.
    case "$recent" in
        *" $name "*)
            if [ "${_started[$name]:-0}" -gt "${_ended[$name]:-0}" ]; then
                live="$live $name"; continue
            fi ;;
    esac
    if [ "$m" = any ]; then
        if [ -d "$CLAIM_DIR/claims/$name" ]; then
            _owner=$(claim_owner "$name" | sed -n 's/.*machine=\([^ ]*\).*/\1/p')
            if [ "$_owner" = "$MACHINE" ]; then
                # Our own claim, and the run is not done (done_ok returned above). This
                # is the FAIL-retry path for pool rows: without it, a machine that
                # claimed a row and crashed skips it forever as "held by" itself, and
                # no other machine can take it either -- the row just dies.
                mine="$mine ${arm}:${s}"
            else
                skipped="$skipped $name(held by $_owner)"
            fi
        else
            pool="$pool ${arm}:${s}"
        fi
        continue
    fi
    mine="$mine ${arm}:${s}"
done < <(rows | awk -v m="$MACHINE" -v A=any '$2==m || $2==A' | sort -s -k1,1n)

echo "=== $MACHINE ==="
echo "  manifest      $(rows | awk -v m="$MACHINE" '$2==m' | wc -l) rows named for it, $(rows | awk '$2=="any"' | wc -l) in the shared pool"
[ -n "$skipped" ] && echo "  not mine     $skipped"
[ -n "$live" ]    && echo "  in flight    $live   (lane log moved within ${INFLIGHT_HOURS:-6}h)"
[ -n "$quarantined" ] && {
    echo "  QUARANTINED $quarantined"
    echo "              failed ${MAX_RUN_RETRIES:-3}+ times; not retried automatically. Read the"
    echo "              traceback first:  python scripts/triage.py logs/*/lane*.log"
    echo "              After a fix:      MAX_RUN_RETRIES=99 bash deploy/campaign.sh   # or clear its logs"
}
[ -n "$retry" ]   && {
    echo "  RETRYING    $retry"
    echo "              these reported FAIL before and are being run again -- they will"
    echo "              resume from their checkpoints. If one fails a second time, read it"
    echo "              rather than relaunching:  python scripts/triage.py \$LOG_DIR/lane*.log"
}
echo "  assigned     ${mine:- none}"
echo "  pool free    ${pool:- none}"
echo "  vllm mem     $ROLLOUT_GPU_MEM_UTIL  (pinned campaign-wide)"

if [ -z "${mine// /}${pool// /}" ] && [ "$MODE" != control ]; then
    echo
    echo "nothing left for $MACHINE. If that is a surprise, check --plan: a machine with"
    echo "no wave-2 rows finishes early by design, and its GPUs are then free for another"
    echo "machine's overflow (edit the manifest -- assignment is deliberate, not implicit)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Only GPUs nothing is using. m1 has lanes running right now, and a campaign that
# assumed the whole box would land on top of them.
# ---------------------------------------------------------------------------
busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)
free_idx=""
while read -r idx uuid; do
    case "$busy" in *"$uuid"*) continue ;; esac
    free_idx="$free_idx $idx"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ',')
set -- $free_idx
n_free=$#
n_mine=$(set -- $mine; echo $#)
n_pending=$(( n_mine + $(set -- $pool; echo $#) ))
GPUS_PER_RUN=${GPUS_PER_RUN:-2}
lanes=$(( n_free / GPUS_PER_RUN ))
[ "$lanes" -gt "$n_pending" ] && lanes=$n_pending
# control runs vanilla:0 regardless of pending, so an all-done manifest must not
# clamp it to zero lanes -- that clamp is what made the mode unreachable in test.
[ "$MODE" = control ] && [ "$lanes" -lt 1 ] && [ "$n_free" -ge "$GPUS_PER_RUN" ] && lanes=1

echo "  GPUs         $n_free free of $(nvidia-smi 2>/dev/null -L | wc -l) ->$([ "$lanes" -gt 0 ] && echo " $lanes lanes" || echo " 0 lanes")"
if [ "$lanes" -lt 1 ]; then
    echo >&2
    echo "FATAL: fewer than 2 free GPUs. A run needs an actor GPU and a teacher GPU," >&2
    echo "       and the teacher is a separate Ray resource pool that will not share." >&2
    echo "       Busy now:" >&2
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv >&2
    exit 1
fi

# Cross-check the log-derived in-flight set against the GPUs, which is orthogonal to
# it: busy GPUs come from the driver, not from anything this project writes. Every run
# holds GPUS_PER_RUN cards, so busy/2 is how many runs are alive on this box. If that
# exceeds what the logs accounted for, something is training that we could not name --
# and the next lane would happily start a second copy of it on the free cards. On m1
# that is exactly the shape of the accident: four lanes alive, four cards free.
_busy_n=$(( $(nvidia-smi -L 2>/dev/null | wc -l) - n_free ))
_expect=$(( _busy_n / GPUS_PER_RUN ))
_detected=$(set -- $live; echo $#)
if [ "$_expect" -gt "$_detected" ] && [ "${ALLOW_UNKNOWN_GPU_USERS:-0}" != 1 ]; then
    echo >&2
    echo "FATAL: $_busy_n GPUs are busy -- about $_expect run(s) -- but the lane logs" >&2
    echo "       account for only $_detected. Something is training that this cannot name," >&2
    echo "       so a lane started now could be a second copy of it." >&2
    echo >&2
    echo "       Unfinished runs it found, with how long since their log last moved --" >&2
    echo "       anything under ${INFLIGHT_HOURS:-6}h counts as alive, and a validation is ~75 min of silence:" >&2
    # Newest log per run only. Several rounds of failed launches leave a dead log
    # each, and printing them all made vanilla_s1 appear four times -- which buried
    # the fact that mattered: two runs went quiet 490 and 492 minutes ago, within two
    # minutes of each other, which is one event and not two slow runs.
    for _f in "$LOG_DIR"/lane*.log "$LOG_DIR/${MACHINE:-__none__}"/lane*.log; do
        [ -f "$_f" ] || continue
        _last=$(grep -oE '^#+ RUN: [A-Za-z0-9_.]+' "$_f" 2>/dev/null | tail -1 | awk '{print $3}')
        [ -n "$_last" ] || continue
        grep -qE "^#+ $_last -> (OK|FAIL)" "$_f" && continue
        printf '%s\t%s\t%s\n' "$(stat -c %Y "$_f")" "$_last" "$_f"
    done | sort -rn | awk -F'\t' -v now="$(date +%s)" '
        !seen[$2]++ { printf "         %-28s %4d min   %s\n", $2, (now-$1)/60, $3 }
        seen[$2]>1  { hidden++ }
        END { if (hidden) printf "         (%d older log(s) from earlier attempts hidden)\n", hidden }
    ' >&2
    echo >&2
    echo "       ps -o pid,etime,cmd -p \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\\n' ',' | sed 's/,\$//')" >&2
    echo "       INFLIGHT_HOURS=12 MACHINE=$MACHINE bash deploy/campaign.sh   # if a run is simply slow" >&2
    echo "       ALLOW_UNKNOWN_GPU_USERS=1 ...                                # if the busy cards are something else" >&2
    exit 1
fi

# The pool is open only where the claim directory is PROVEN shared: cornell has its
# own checkout on its own filesystem, so its .campaign is an island -- a claim there
# stops nobody, and the same row runs twice. Proof = --fs-probe has recorded another
# hostname in THIS CLAIM_DIR. Until two machines have probed, pool rows wait, loudly.
pool_open=""
[ -n "$(ls "$CLAIM_DIR/_probe" 2>/dev/null | grep -vx "$(hostname)")" ] && pool_open=1
if [ -z "$pool_open" ] && [ -n "${pool// /}" ]; then
    echo "  pool: CLOSED on this machine -- no other host has probed $CLAIM_DIR,"
    echo "        so a claim here would not be visible to the machines it must block."
    echo "        Open it:  bash deploy/campaign.sh --fs-probe   (here AND on one other box)"
    pool=""
fi

# Claim now that the lane count is known, and only as many as there are slots left
# after this machine's own rows. --dry deliberately claims nothing: it reports what it
# would take, so running it on four machines does not hand the pool to whoever ran it
# first.
claimed=""; would=""
slots=$(( lanes - n_mine ))
for entry in $pool; do
    [ "$slots" -gt 0 ] || break
    name="${entry%%:*}_s${entry##*:}${TAG:+_$TAG}"
    if [ "$MODE" = dry ]; then
        would="$would $name"
    else
        claim "$name" || { echo "  pool: $name went to $(claim_owner "$name" | sed -n 's/.*machine=\([^ ]*\).*/\1/p') first"; continue; }
        echo "  pool: claimed $name"
    fi
    claimed="$claimed $entry"; slots=$(( slots - 1 ))
done
[ -n "$would" ] && echo "  pool: would claim$would  (--dry claims nothing)"
pending="$mine$claimed"
echo "  pending     ${pending:- none}"
[ -n "${pending// /}" ] || [ "$MODE" = control ] || { echo; echo "nothing this machine can start right now."; exit 0; }

gpu_list=""
# Groups of GPUS_PER_RUN, not hardcoded pairs (audit 2026-08-07 latent item): the
# n8 cell runs 4-card lanes (NGPUS_PER_NODE=2 + TEACHER_WORLD_SIZE=2) on a
# dedicated GPUS_PER_RUN=4 daemon; run_parallel passes each comma-group verbatim
# to CUDA_VISIBLE_DEVICES, so any width flows through.
for i in $(seq 0 $((lanes - 1))); do
    grp=""
    for j in $(seq 1 "$GPUS_PER_RUN"); do
        idx=$(( i * GPUS_PER_RUN + j ))
        grp="$grp,${!idx}"
    done
    gpu_list="$gpu_list ${grp#,}"
done
gpu_list="${gpu_list# }"
_n_real=$(set -- $pending; echo $#)
if [ "$_n_real" -lt "$lanes" ] && [ "$MODE" != control ]; then
    lanes=$_n_real
    gpu_list=$(set -- $gpu_list; c=""; for i in $(seq 1 $lanes); do c="$c ${!i}"; done; echo "${c# }")
fi

echo "  GPU_LIST     $gpu_list"
echo "  steps        ${STEPS:-250}   save/${SAVE_FREQ:-25}  test/${TEST_FREQ:-25}"

# Object-store share by machine CAPACITY, not by this launch's lane count.
# run_parallel divides 0.30 by ITS lane count, which was right when one launch owned
# the box. campaign.sh makes staggered launches normal -- three lanes started today
# next to one still running from yesterday -- and then each launch divides by its own
# smaller number: 0.30/2 + 3 x 0.30/3 sums to 45% of available RAM, drifting back
# toward the /dev/shm exhaustion that hung lanes 1 and 3 at step 5. Dividing by the
# most lanes this box can hold keeps any mix of launches at or under 0.30 total.
if [ -z "${RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION:-}" ]; then
    _cap_lanes=$(( $(nvidia-smi -L 2>/dev/null | wc -l) / GPUS_PER_RUN ))
    export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=$(awk -v n="${_cap_lanes:-4}" \
        'BEGIN{printf "%.4f", 0.30/(n<1?1:n)}')
    echo "  object store $RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION of available RAM per lane (capacity $_cap_lanes lanes)"
fi

if [ "$MODE" = dry ]; then
    echo
    echo "would run:"
    echo "  GPU_LIST=\"$gpu_list\" LANES=$lanes RAY_TMPDIR_TAG=$LAUNCH_TAG \\"
    echo "  ROLLOUT_GPU_MEM_UTIL=$ROLLOUT_GPU_MEM_UTIL STEPS=${STEPS:-250} \\"
    echo "  bash deploy/dsw/run_parallel.sh \"${pending# }\""
    exit 0
fi

# ---------------------------------------------------------------------------
# --machine-control: is this box the same as m1?
#
# Only needed if --fingerprint differs between this box and m1. The machines are one
# image on identical cards with the same driver, so the expected answer is that it is
# unnecessary -- but "expected" is why --fingerprint exists, and this is what to run
# if it comes back different. Every arm is measured against the vanilla x3 floor on
# m1, so a machine effect would ride silently on top of every one of them.
#
# Same seed, same config, 50 steps. Re-running seed 0 rather than adding a fourth seed
# makes it a direct comparison instead of a "within the band" one, and m1's vanilla_s0
# already has validations at 25 and 50 to compare against. A fifth of a full run.
# ---------------------------------------------------------------------------
if [ "$MODE" = control ]; then
    echo
    echo "=== machine control: vanilla seed 0, 50 steps, tagged _${MACHINE} ==="
    echo "compare its val@25 and val@50 with m1's vanilla_s0 at the same steps."
    exec env GPU_LIST="$(echo "$gpu_list" | awk '{print $1}')" LANES=1 \
        RAY_TMPDIR_TAG="${MACHINE}ctl$(date +%s)_" TAG="$MACHINE" \
        STEPS=50 TEST_FREQ=25 SAVE_FREQ=-1 \
        bash deploy/dsw/run_parallel.sh "vanilla:0" 9>&- 8>&-
fi

echo
env GPU_LIST="$gpu_list" LANES="$lanes" RAY_TMPDIR_TAG="$LAUNCH_TAG" \
    LOG_DIR="$LOG_DIR/$MACHINE" TAG="$TAG" \
    STEPS="${STEPS:-250}" TEST_FREQ="${TEST_FREQ:-25}" SAVE_FREQ="${SAVE_FREQ:-25}" \
    bash deploy/dsw/run_parallel.sh "${pending# }" 9>&- 8>&-
rc=$?
if [ "$rc" != 0 ] && [ -n "${claimed// /}" ]; then
    # The launch never started, so the pool rows this invocation claimed are not
    # being worked on by anyone -- release them, or they stay "held by $MACHINE"
    # forever while the machine that could run them skips past. (exec was wrong here
    # for exactly this reason: nothing can run after an exec fails downstream.)
    for entry in $claimed; do
        name="${entry%%:*}_s${entry##*:}${TAG:+_$TAG}"
        rm -f "$CLAIM_DIR/claims/$name/owner"
        rmdir "$CLAIM_DIR/claims/$name" 2>/dev/null && echo "released pool claim: $name"
    done
fi
exit "$rc"
