#!/usr/bin/env bash
# 一条命令回答:代码到底有没有上车,以及最近一次失败说了什么
echo "==== 本机 git ===="; git log --oneline -1
echo "==== 最新 3 个快照携带的修复 ===="
for d in $(ls -dt /mnt/workspace/simopd_data/snapshots/* 2>/dev/null | head -3); do
  printf "%-30s device=%s pop-guard=%s chunk1024=%s shadow-stat=%s optim-offload-default=%s\n" \
    "$(basename $d)" \
    "$(grep -c 'to(teacher_topk_log_probs.device)' $d/src/simopd/topk_losses.py 2>/dev/null)" \
    "$(grep -c 'def _pop' $d/src/simopd/topk_losses.py 2>/dev/null)" \
    "$(grep -c 'SIMOPD_ENTROPY_CHUNK' $d/src/simopd/topk_losses.py 2>/dev/null)" \
    "$(grep -c '_stat_mask(teacher_topk_log_probs, data' $d/src/simopd/topk_losses.py 2>/dev/null)" \
    "$(grep -o 'FSDP_OPTIMIZER_OFFLOAD:-[A-Za-z]*' $d/scripts/run_opd_baseline.sh 2>/dev/null | head -1)"
done
echo "==== 最近的失败快照 ===="
ls -t logs/failures/*.log 2>/dev/null | head -3
for f in $(ls -t logs/failures/*.log 2>/dev/null | head -2); do
  echo "---- $f"
  grep -nE "RuntimeError|Error:|out of memory|non-empty|same device|FATAL|Killed" "$f" | tail -6
done
