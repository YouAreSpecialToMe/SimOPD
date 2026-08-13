#!/usr/bin/env bash
# T-0 preflight for the DLC fleet: one read-only sweep of everything the
# submission needs, run on the box that will submit (a hop pod with /mgfs).
#
#   bash deploy/dlc/preflight_fleet.sh
#
# Reports OK/MISS per item and ends GO or NO-GO. It changes nothing: the
# campaign's own guards (pin, identity, needs= fences) stay authoritative --
# this exists so the operator sees every gap in one screen instead of
# discovering them one FATAL at a time after submission.
set -uo pipefail

EXP_ROOT=${EXP_ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}
DATA=${DATA:-/mgfs/shared/Group_GY/changhao/simopd_data}
HF_HUB=$DATA/hf_cache/hub
DLC=${DLC:-/mgfs/shared/Group_GY/changhao/tools/pai/bin/dlc}
MISS=0
ok()   { echo "  OK    $*"; }
miss() { echo "  MISS  $*"; MISS=$((MISS+1)); }
note() { echo "  --    $*"; }

echo "== tree =="
if [ -d "$EXP_ROOT/.git" ]; then
    _head=$(git -C "$EXP_ROOT" rev-parse --short HEAD 2>/dev/null)
    _dirty=$(git -C "$EXP_ROOT" status --porcelain 2>/dev/null | head -3)
    [ -z "$_dirty" ] && ok "exp tree clean at $_head" \
                     || miss "exp tree DIRTY at $_head (uncommitted-edit fuse): $(echo "$_dirty" | tr '\n' ' ')"
else
    miss "exp tree not found at $EXP_ROOT (is /mgfs mounted?)"
fi

echo "== submission =="
[ -x "$DLC" ] && ok "dlc CLI at $DLC" || miss "dlc CLI not executable at $DLC"
[ -e "$HOME/.dlc/config" ] && ok "dlc CLI configured (~/.dlc/config)" \
    || miss "dlc CLI unconfigured -- run 'dlc config' once with your AK (user step; keys never in chat)"
for v in WORKSPACE_ID RESOURCE_ID IMAGE; do
    [ -n "$(printenv $v)" ] && ok "$v set" || miss "$v unset (PAI console)"
done
[ -n "${DATA_SOURCES:-}" ] && ok "DATA_SOURCES set" \
    || note "DATA_SOURCES unset -- fine IF the workspace mounts /mgfs implicitly; verify on the smoke job"

echo "== datasets (per domain) =="
[ -f "$DATA/simopd_math/train.parquet" ] && ok "math train.parquet" || miss "math train.parquet"
[ -f "$DATA/simopd_code/train.parquet" ] && ok "code  train.parquet + $(ls "$DATA/simopd_code" | wc -l) files" || miss "code train.parquet"
[ -f "$DATA/simopd_if/train.parquet" ]   && ok "IF    train.parquet" \
    || note "IF train.parquet absent -- workers skip the domain loudly; unblocks when the HF token lands at $DATA/.hf_token"

echo "== models (per pair, shared hf_cache) =="
pair() {  # label repo1 repo2 -- dir alone is not enough (an in-flight
          # snapshot_download creates it immediately); demand config + weights.
          # The true completeness gate is the fetch script's offline-load verify.
    local lbl=$1; shift
    local m=""
    for r in "$@"; do
        local d="$HF_HUB/models--${r//\//--}/snapshots"
        { [ -n "$(find "$d" -name config.json 2>/dev/null | head -1)" ] && \
          [ -n "$(find "$d" -name '*.safetensors' 2>/dev/null | head -1)" ]; } || m="$m $r"
    done
    [ -z "$m" ] && ok "$lbl" || miss "$lbl missing/incomplete:$m"
}
pair "math  (1.7B-Base <- 4B-2507)" "Qwen/Qwen3-1.7B-Base" "Qwen/Qwen3-4B-Instruct-2507"
pair "w8b   (8B-Base   <- 32B)"     "Qwen/Qwen3-8B-Base"   "Qwen/Qwen3-32B"
pair "p4b   (4B-Base   <- 14B)"     "Qwen/Qwen3-4B-Base"   "Qwen/Qwen3-14B"

echo "== pins (campaign guard is authoritative; this just reports) =="
for cd_ in .campaign .campaign_if .campaign_code .campaign_w8b .campaign_p4b; do
    f="$EXP_ROOT/$cd_/CAMPAIGN_REF"
    if [ -f "$f" ]; then
        _ref=$(cut -d' ' -f1 < "$f"); _h=$(git -C "$EXP_ROOT" rev-parse HEAD 2>/dev/null)
        if [ "$_ref" = "$_h" ]; then ok "$cd_ pinned at HEAD"
        else
            _rd=$(git -C "$EXP_ROOT" diff --name-only "$_ref" "$_h" -- src/ configs/arms.yaml scripts/run_opd_baseline.sh scripts/arm.py 2>/dev/null | head -1)
            [ -z "$_rd" ] && ok "$cd_ pin behind HEAD, nothing run-defining moved (guard will note+proceed)" \
                          || miss "$cd_ pin behind HEAD with run-defining diff ($_rd ...) -- REASON=... --repin before launch"
        fi
    else
        note "$cd_ unpinned -- worker seeds it on first boot"
    fi
done

echo "== gates + secrets =="
[ -f "$EXP_ROOT/gates/p4b_ok" ] && ok "gates/p4b_ok" \
    || note "gates/p4b_ok absent -- p4b rows stay fenced (by design until probe+rehearsal)"
[ -f "$DATA/.wandb_key" ] && ok ".wandb_key present ($(stat -c %a "$DATA/.wandb_key" 2>/dev/null))" || miss ".wandb_key"
[ -f "$DATA/.hf_token" ] && ok ".hf_token present" || note ".hf_token absent (only blocks IF prep, not launch)"

echo "== proof harness =="
note "run it fresh if worker.sh moved since the last ALL PASS:"
note "  bash $EXP_ROOT/deploy/dlc/test_worker_dry.sh   (~20s, no GPUs)"

echo
if [ "$MISS" -eq 0 ]; then
    echo "GO: nothing missing. Smoke first:  WORKERS=8 bash deploy/dlc/submit_fleet.sh"
    echo "    then full:                     bash deploy/dlc/submit_fleet.sh <card-count>"
else
    echo "NO-GO: $MISS item(s) missing above."
fi
exit "$([ $MISS -eq 0 ] && echo 0 || echo 1)"
