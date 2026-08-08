#!/usr/bin/env bash
# Full S-wave reset -- run on m1 ONLY (shared fs), AFTER stop_pilot.sh has shown
# "clean: 0" on ALL THREE boxes. Archives every piece of run state -- logs/,
# $CKPT_ROOT, .campaign -- into one timestamped place so the namespace restarts
# empty and the fleet relaunches from zero. Nothing is deleted: pilot evidence
# and half-run 16k debris move together (same volume, instant mv), and
# PIN_HISTORY carries forward for audit continuity. After this:
#   launch_m1.sh, then launch_m2.sh / launch_m3.sh -- a completely fresh start.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TS=$(date -u +%Y%m%d_%H%MZ)
ARCH="archive/pre_reset_$TS"

_busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)
[ "$_busy" = 0 ] || { echo "FATAL: $_busy GPU proc(s) on THIS box -- stop_pilot.sh first (m2/m3 too)." >&2; exit 1; }
echo "This box is clean. m2/m3 must ALSO have shown 'clean: 0' -- a run still writing"
echo "from another box would land its logs/ckpts in the fresh namespace mid-archive."
if [ "${APPLY:-0}" != 1 ]; then
    read -r -p "archive logs/ + \${CKPT_ROOT} + .campaign and start over? [y/N] " _yn
    [ "${_yn:-n}" = y ] || { echo "aborted"; exit 1; }
fi

mkdir -p "$ARCH"
if [ -d logs ]; then mv logs "$ARCH/logs"; fi
mkdir -p logs
echo "logs    -> $ARCH/logs"

if [ -n "${CKPT_ROOT:-}" ] && [ -d "$CKPT_ROOT" ]; then
    mv "$CKPT_ROOT" "${CKPT_ROOT%/}__pre_reset_$TS"
    mkdir -p "$CKPT_ROOT"
    echo "ckpts   -> ${CKPT_ROOT%/}__pre_reset_$TS"
else
    echo "WARN: CKPT_ROOT unset or absent -- checkpoints NOT archived. If any run ever" >&2
    echo "      wrote ckpts on this volume, export CKPT_ROOT and rerun: a stale" >&2
    echo "      fingerprint under a reused name refuses its relaunch (vanilla_s2 class)." >&2
fi

if [ -d .campaign ]; then
    cp .campaign/PIN_HISTORY "$ARCH/PIN_HISTORY.copy" 2>/dev/null || true
    mv .campaign "$ARCH/campaign_state"
    echo "state   -> $ARCH/campaign_state"
fi
mkdir -p .campaign
cp "$ARCH/PIN_HISTORY.copy" .campaign/PIN_HISTORY 2>/dev/null || true
# Nothing left to migrate, by construction -- launch_m1's phase 1 self-skips.
date -u +%FT%TZ > .campaign/migrated_pilot8k

echo
echo "reset complete -> $ARCH"
echo "next: bash deploy/dsw/launch_m1.sh   (re-registers, re-pins, re-mints the anchor)"
echo "      bash deploy/dsw/launch_m2.sh ; bash deploy/dsw/launch_m3.sh"
