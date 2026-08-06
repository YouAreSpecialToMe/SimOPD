#!/usr/bin/env bash
# Queue this machine's share of the campaign.
#
#   bash deploy/campaign.sh --fingerprint       # is this box the same as the others?
#   bash deploy/campaign.sh --fs-probe          # do the machines share a filesystem?
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

rows() {   # wave, machine, arm, seed -- comments and blank lines gone
    awk -F'\t' '!/^[[:space:]]*#/ && NF>=4 { print $1, $2, $3, $4 }' "$MANIFEST"
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

MACHINE=${MACHINE:-}
[ -n "$MACHINE" ] || {
    echo "FATAL: set MACHINE. The manifest assigns work by machine name, and there is no" >&2
    echo "       shared filesystem for a box to claim work at runtime -- it can only be told." >&2
    echo "       Names in $MANIFEST: $(rows | awk '{print $2}' | sort -u | tr '\n' ' ')" >&2
    exit 1
}
rows | awk -v m="$MACHINE" '$2==m' | grep -q . || {
    echo "FATAL: '$MACHINE' has no rows in $MANIFEST." >&2
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
if [ -f "$_ref_file" ]; then
    _ref=$(cut -d' ' -f1 < "$_ref_file")
    if [ "$_ref" != "$_head" ] && [ "$_head" != nogit ]; then
        _drift=$(git diff --name-only "$_ref" "$_head" -- scripts src configs 2>/dev/null)
        if [ -n "$_drift" ]; then
            echo "FATAL: the campaign is pinned to $_ref (see $_ref_file)" >&2
            echo "       and HEAD is $_head. Files that decide what a run computes changed:" >&2
            echo "$_drift" | sed 's/^/         /' >&2
            echo "       Arms launched on either side of this are not comparable, and nothing" >&2
            echo "       downstream would say so -- the config fingerprint records config," >&2
            echo "       not code." >&2
            echo "       git checkout $_ref                    # run the campaign's code" >&2
            echo "       echo $_head > $_ref_file              # OR re-pin, deliberately," >&2
            echo "                                              and note it in the ledger" >&2
            exit 1
        fi
        echo "  note: HEAD moved since the campaign was pinned, but nothing under" >&2
        echo "        scripts/ src/ configs/ changed -- runs stay comparable." >&2
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
done_ok=" "; failed=" "; attempted=" "; recent=" "
if [ -d "$LOG_DIR" ]; then
    # Both layouts: logs/lane*.log from before machines were separated, and
    # logs/<machine>/lane*.log from now on -- which matters on a shared filesystem,
    # where two boxes starting in the same second would otherwise write one filename.
    for f in "$LOG_DIR"/lane*.log "$LOG_DIR"/*/lane*.log; do
        [ -f "$f" ] || continue
        while read -r n; do done_ok="$done_ok$n "; done < <(
            grep -oE '^#+ [A-Za-z0-9_.]+ -> OK' "$f" 2>/dev/null | awk '{print $2}')
        while read -r n; do failed="$failed$n "; done < <(
            grep -oE '^#+ [A-Za-z0-9_.]+ -> FAIL' "$f" 2>/dev/null | awk '{print $2}')
        while read -r n; do attempted="$attempted$n "; done < <(
            grep -oE '^#+ RUN: [A-Za-z0-9_.]+' "$f" 2>/dev/null | awk '{print $3}')
        # How long may a HEALTHY run be silent? Not 30 minutes, which is what this
        # said first and is the fourth time on this project that a threshold was
        # picked from what failure looks like without checking what health looks
        # like. A validation is ~75 minutes at this tier, and a run in one writes
        # nothing at all -- so 30 minutes calls a healthy run dead, and the machine
        # then offers to start a second copy of it. Six hours clears the longest
        # legitimate silence with room; the GPU cross-check below is what catches a
        # genuine crash, and being conservative here errs toward not duplicating.
        if [ -n "$(find "$f" -mmin -$(( ${INFLIGHT_HOURS:-6} * 60 )) 2>/dev/null)" ]; then
            while read -r n; do recent="$recent$n "; done < <(
                grep -oE '^#+ RUN: [A-Za-z0-9_.]+' "$f" 2>/dev/null | awk '{print $3}')
        fi
    done
fi

# Two lists, not one. Rows named for this machine are its own; 'any' rows are the pool
# and are only CLAIMED once the free-GPU count is known, because a machine that claims
# what it cannot start holds it away from the machine that could. A first version
# claimed during the manifest walk -- which also meant --dry took the whole pool and
# left nothing for anyone, a side effect a dry run must not have.
mine=""; pool=""; skipped=""; live=""; retry=""
while read -r w m arm s; do
    name="${arm}_s${s}"
    case "$done_ok" in *" $name "*) skipped="$skipped $name(done)"; continue ;; esac
    case "$failed"  in *" $name "*) retry="$retry $name" ;; esac
    case "$recent" in
        *" $name "*)
            case "$attempted" in *" $name "*) live="$live $name" ; continue ;; esac ;;
    esac
    if [ "$m" = any ]; then
        if [ -d "$CLAIM_DIR/claims/$name" ]; then
            skipped="$skipped $name(held by $(claim_owner "$name" | sed -n 's/.*machine=\([^ ]*\).*/\1/p'))"
        else
            pool="$pool ${arm}:${s}"
        fi
        continue
    fi
    mine="$mine ${arm}:${s}"
done < <(rows | awk -v m="$MACHINE" -v A=any '$2==m || $2==A' | sort -k1,1n)

echo "=== $MACHINE ==="
echo "  manifest      $(rows | awk -v m="$MACHINE" '$2==m' | wc -l) rows named for it, $(rows | awk '$2=="any"' | wc -l) in the shared pool"
[ -n "$skipped" ] && echo "  not mine     $skipped"
[ -n "$live" ]    && echo "  in flight    $live   (lane log moved within ${INFLIGHT_HOURS:-6}h)"
[ -n "$retry" ]   && {
    echo "  RETRYING    $retry"
    echo "              these reported FAIL before and are being run again -- they will"
    echo "              resume from their checkpoints. If one fails a second time, read it"
    echo "              rather than relaunching:  python scripts/triage.py \$LOG_DIR/lane*.log"
}
echo "  assigned     ${mine:- none}"
echo "  pool free    ${pool:- none}"
echo "  vllm mem     $ROLLOUT_GPU_MEM_UTIL  (pinned campaign-wide)"

if [ -z "${mine// /}${pool// /}" ]; then
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
    for _f in "$LOG_DIR"/lane*.log "$LOG_DIR"/*/lane*.log; do
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

# Claim now that the lane count is known, and only as many as there are slots left
# after this machine's own rows. --dry deliberately claims nothing: it reports what it
# would take, so running it on four machines does not hand the pool to whoever ran it
# first.
claimed=""; would=""
slots=$(( lanes - n_mine ))
for entry in $pool; do
    [ "$slots" -gt 0 ] || break
    name="${entry%%:*}_s${entry##*:}"
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
[ -n "${pending// /}" ] || { echo; echo "nothing this machine can start right now."; exit 0; }

gpu_list=""
for i in $(seq 0 $((lanes - 1))); do
    a=$(( i * 2 + 1 )); b=$(( i * 2 + 2 ))
    gpu_list="$gpu_list ${!a},${!b}"
done
gpu_list="${gpu_list# }"
_n_real=$(set -- $pending; echo $#)
if [ "$_n_real" -lt "$lanes" ]; then
    lanes=$_n_real
    gpu_list=$(set -- $gpu_list; c=""; for i in $(seq 1 $lanes); do c="$c ${!i}"; done; echo "${c# }")
fi

echo "  GPU_LIST     $gpu_list"
echo "  steps        ${STEPS:-250}   save/${SAVE_FREQ:-50}  test/${TEST_FREQ:-25}"

if [ "$MODE" = dry ]; then
    echo
    echo "would run:"
    echo "  GPU_LIST=\"$gpu_list\" LANES=$lanes RAY_TMPDIR_TAG=${MACHINE}_ \\"
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
        RAY_TMPDIR_TAG="${MACHINE}ctl_" TAG="$MACHINE" \
        STEPS=50 TEST_FREQ=25 SAVE_FREQ=-1 \
        bash deploy/dsw/run_parallel.sh "vanilla:0"
fi

echo
exec env GPU_LIST="$gpu_list" LANES="$lanes" RAY_TMPDIR_TAG="${MACHINE}_" \
    LOG_DIR="$LOG_DIR/$MACHINE" \
    STEPS="${STEPS:-250}" TEST_FREQ="${TEST_FREQ:-25}" SAVE_FREQ="${SAVE_FREQ:-50}" \
    bash deploy/dsw/run_parallel.sh "${pending# }"
