#!/usr/bin/env bash
# One paste, whole machine. Written 2026-08-09 after several rounds of diagnosing
# a stall from status lines alone: what a hung lane looks like and what a slow one
# looks like differ only in numbers nobody had printed.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "########## HOST ##########"
uptime; free -g | head -3
echo "-- top RSS consumers"
ps -eo rss,pid,comm --sort=-rss | head -8 | awk '{printf "  %6.1fG  pid=%-8s %s\n", $1/1048576, $2, $3}'
echo "-- process counts: ray=$(pgrep -cf ray:: 2>/dev/null) vllm=$(pgrep -cf vllm 2>/dev/null) lane=$(pgrep -cf dsw/_lane.sh 2>/dev/null) train=$(pgrep -cf run_opd_baseline 2>/dev/null)"
echo "########## GPU ##########"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "-- compute apps"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | head -12
echo "########## SNAPSHOT ON THE NEWEST LANE ##########"
d=$(ls -dt /mnt/workspace/simopd_data/snapshots/* 2>/dev/null | head -1)
printf "%s\n  device=%s pop=%s chunk=%s shadow-stat=%s %s %s\n" "$(basename "$d")" \
  "$(grep -c 'to(teacher_topk_log_probs.device)' "$d"/src/simopd/topk_losses.py 2>/dev/null)" \
  "$(grep -c 'def _pop' "$d"/src/simopd/topk_losses.py 2>/dev/null)" \
  "$(grep -c 'SIMOPD_ENTROPY_CHUNK' "$d"/src/simopd/topk_losses.py 2>/dev/null)" \
  "$(grep -c '_stat_mask(teacher_topk_log_probs, data' "$d"/src/simopd/topk_losses.py 2>/dev/null)" \
  "$(grep -o 'FSDP_PARAM_OFFLOAD:-[A-Za-z]*' "$d"/scripts/run_opd_baseline.sh 2>/dev/null | head -1)" \
  "$(grep -o 'FSDP_OPTIMIZER_OFFLOAD:-[A-Za-z]*' "$d"/scripts/run_opd_baseline.sh 2>/dev/null | head -1)"
echo "########## NEWEST LANE LOG: is it moving? ##########"
_me=$(awk -F'\t' -v h="$(hostname)" '$1==h {print $2; exit}' .campaign/MACHINE_MAP 2>/dev/null)
f=$(ls -t logs/${_me:-.}/lane*.log 2>/dev/null | head -1)
echo "log: $f"
echo "-- run in it: $(grep -oE '^#+ RUN: [A-Za-z0-9_.]+' "$f" 2>/dev/null | tail -1)"
a=$(grep -oE "finished: [0-9]+" "$f" 2>/dev/null | tail -1); s1=$(stat -c %s "$f" 2>/dev/null)
tail -6 "$f" 2>/dev/null | cut -c1-160
sleep 90
b=$(grep -oE "finished: [0-9]+" "$f" 2>/dev/null | tail -1); s2=$(stat -c %s "$f" 2>/dev/null)
echo "-- after 90s:  finished '$a' -> '$b'   logsize $s1 -> $s2"
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
echo "########## ERRORS ANYWHERE IN IT ##########"
grep -inE "out of memory|oom|killed|Traceback|RuntimeError|Error:" "$f" 2>/dev/null | tail -8 | cut -c1-200
