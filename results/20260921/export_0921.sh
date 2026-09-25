#!/usr/bin/env bash
# export_0921.sh          (run on w0; read-only on the nodes; writes /home/tiger/opd/results_export/20260921/)
# Periodic snapshot of the 0921 scaling campaign (haotian 2026-09-22: "when partial results are done, periodically sync to wandb and this
# machine, as the server may be ephemeral"). Gathers the small archive files of all four nodes on w0 (node-to-node rsync only), then
# tools/export_0921_tables.py builds the tables + per-sample eval files, and the logs / trajectory samples are copied in gzipped/as-is.
# The laptop pulls the folder with results/fetch_export.sh 20260921 all (plain ssh-cat + md5; never scp/rsync/push from the laptop).
set -uo pipefail
O=/home/tiger/opd; E=$O/results_export; TAG=20260921; OUT=$E/$TAG; G=$E/_gather_0921; PY=$O/SimOPD/simopd/bin/python
mkdir -p $OUT $G/train $G/logs $O/logs
say() { echo "[$(date +%T)] $*"; }
say "0. eval artifacts of all nodes -> wandb_inbox/evals_0921 (same filter as the W&B daemon)"
mkdir -p $O/wandb_inbox/evals_0921
rsync -a --include='*.parquet' --exclude='*' $O/simopd_data/store_0921/evals/ $O/wandb_inbox/evals_0921/ 2>/dev/null
for n in w1 w2 w3; do rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=10" --include='*.parquet' --exclude='*' $n:$O/simopd_data/store_0921/evals/ $O/wandb_inbox/evals_0921/ 2>/dev/null; done
say "1. training archive files (metrics jsonl, manifests, contracts) + campaign logs of all nodes -> $G"
INC=(--include='*_seed0/' --include='*_seed0/metrics/' --include='*_seed0/metrics/launch_*.jsonl' --include='*_seed0/manifest/' --include='*_seed0/manifest/*' --include='*_seed0/run_manifest.json' --include='*_seed0/simopd_stop_contract.txt' --include='*_seed0/simopd_fingerprint.txt' --include='*_seed0/latest_checkpointed_iteration.txt' --exclude='*')
for n in w0 w1 w2 w3; do
  mkdir -p $G/train/$n $G/logs/$n
  if [ $n = w0 ]; then src=""; rsh=(); else src="$n:"; rsh=(-e "ssh -o BatchMode=yes -o ConnectTimeout=10"); fi
  rsync -a "${rsh[@]}" "${INC[@]}" --prune-empty-dirs "${src}$O/simopd_data/ckpt/simopd_scaling/" $G/train/$n/ 2>/dev/null
  rsync -a "${rsh[@]}" --include='scaling/' --include='scaling/*.log' --include='scaling/*.exit' --include='scaling/queue_*.txt' --include='scaling/wandb_ids.tsv' --include='scaling/scaling_runs.log' --include='evalw0921/***' --include='eval0921.log' --include='gpu_guard.log' --include='disk_guard.log' --include='holder_swap.log' --exclude='*' "${src}$O/logs/" $G/logs/$n/ 2>/dev/null
  rsync -a "${rsh[@]}" "${src}$O/simopd_data/store_0921/evalq_exp/refill.log" $G/logs/$n/refill_0921.log 2>/dev/null
  for f in $G/logs/$n/scaling/*.log; do [ -f "$f" ] && ln -sf "$f" "$G/logs/$n/$(basename "$f")"; done   # export_0921_tables.py reads <node>/scaling_runs.log, <node>/wandb_ids.tsv
  ln -sf $G/logs/$n/scaling/scaling_runs.log $G/logs/$n/scaling_runs.log 2>/dev/null; ln -sf $G/logs/$n/scaling/wandb_ids.tsv $G/logs/$n/wandb_ids.tsv 2>/dev/null
done
say "2. tables, per-sample eval files, train archive"
PYTHONPATH= nice -n 10 $PY $O/tools/export_0921_tables.py $OUT $O/wandb_inbox/evals_0921 $G/train $G/logs $O/wandb_inbox/traj_samples_0921 2>&1 | tail -n 3
say "3. trajectory samples + gzipped logs"
mkdir -p $OUT/traj_samples $OUT/logs
rsync -a $O/wandb_inbox/traj_samples_0921/ $OUT/traj_samples/ 2>/dev/null
for n in w0 w1 w2 w3; do
  mkdir -p $OUT/logs/$n
  for f in $G/logs/$n/scaling/*.log $G/logs/$n/scaling/scaling_runs.log $G/logs/$n/eval0921.log $G/logs/$n/gpu_guard.log $G/logs/$n/disk_guard.log $G/logs/$n/refill_0921.log $G/logs/$n/evalw0921/*.log; do
    [ -f "$f" ] || continue; dst=$OUT/logs/$n/$(basename "$f").gz
    if [ ! -f "$dst" ] || [ "$f" -nt "$dst" ]; then gzip -6 -c "$f" > "$dst.tmp" && mv "$dst.tmp" "$dst"; fi
  done
  for f in $G/logs/$n/scaling/*.exit $G/logs/$n/scaling/queue_*.txt $G/logs/$n/scaling/wandb_ids.tsv; do [ -f "$f" ] && cp -p "$f" $OUT/logs/$n/; done
done
# the manifest must cover the copied material too -> rebuild it after step 3
PYTHONPATH= $PY - "$OUT" <<'EOF'
import hashlib, os, sys
OUT = sys.argv[1]; man = []
for root, _, fs in os.walk(OUT):
    for f in sorted(fs):
        p = os.path.join(root, f); rel = os.path.relpath(p, OUT)
        if rel == "MANIFEST.tsv" or f.startswith("."): continue
        man.append((rel, os.path.getsize(p), hashlib.md5(open(p, "rb").read()).hexdigest()))
open(f"{OUT}/MANIFEST.tsv", "w").write("path\tbytes\tmd5\n" + "".join(f"{r}\t{b}\t{m}\n" for r, b, m in sorted(man)))
big = [(r, b) for r, b, _ in man if b > 50 * 1024 * 1024]
print(f"manifest: {len(man)} files, {sum(b for _, b, _ in man) / 1e6:.1f} MB, over 50 MB: {len(big)} {big[:3]}")
EOF
cp -p $O/tools/export_0921.sh $O/tools/export_0921_tables.py $OUT/ 2>/dev/null
say "done -> $OUT ($(du -sh $OUT | cut -f1))"
