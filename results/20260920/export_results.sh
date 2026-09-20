#!/usr/bin/env bash
# export_results.sh <tag e.g. 20260920> [final]   (run on w0; reads the run directories only, writes to /home/tiger/opd/results_export/<tag>/)
# Rebuilds the results data set in the format of the repo's results/20260909/ + docs/data/post_eval_*.csv:
#   0. refresh the union of the four nodes' evals/*.parquet on w0 (same rsync filter as the W&B daemon)
#   1. per-node training tables = the repo's results/20260909/extract.py, unchanged (tools/export_train_tables.py), in parallel
#   2. offline tables = the repo's scripts/extract_post_eval.py --roster ALL, unchanged
#   3. merge + derived offline tables (tools/export_merge.py) + README with consistency checks (tools/export_readme.py)
#   4. (final only) tools/export_extras.sh: logs/, traj_samples/, non_roster_dirs.tsv, extras_manifest.tsv + README section
# Nothing leaves the cluster. haotian commits/pushes himself (hard rule 5 in the laptop CLAUDE.md).
set -uo pipefail
TAG=${1:?tag}; KIND=${2:-snapshot}; O=/home/tiger/opd; E=$O/results_export; OUT=$E/$TAG; PY=$O/SimOPD/simopd/bin/python
on() { local n=$1; shift; if [ "$n" = w0 ]; then bash -c "$*"; else ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$n" "$*"; fi; }
mkdir -p $OUT $E/store_view $O/logs; ln -sfn $O/wandb_inbox/evals $E/store_view/evals
echo "[$(date +%T)] 0. refreshing the union of eval artifacts"
rsync -a --include='*.parquet' --exclude='*' $O/simopd_data/evals/ $O/wandb_inbox/evals/
for n in w1 w2 w3; do rsync -a --include='*.parquet' --exclude='*' $n:$O/simopd_data/evals/ $O/wandb_inbox/evals/; done
echo "[$(date +%T)] 1. per-node training tables"
for n in w0 w1 w2 w3; do ( on $n "mkdir -p $E/nodes/$n $O/logs; cd $O && PYTHONPATH= nice -n 10 $PY tools/export_train_tables.py $n $E/nodes/$n > logs/export_train_$n.log 2>&1" ) & done; wait
for n in w1 w2 w3; do mkdir -p $E/nodes/$n; rsync -a --delete $n:$E/nodes/$n/ $E/nodes/$n/; done
for n in w0 w1 w2 w3; do tail -n 1 $( [ $n = w0 ] && echo $O/logs/export_train_w0.log || echo /dev/null ) 2>/dev/null; done
echo "[$(date +%T)] 2. offline tables"
cd $O && PYTHONPATH= SIMOPD_STORE=$E/store_view nice -n 10 $PY SimOPD/scripts/extract_post_eval.py --roster ALL --out-dir $OUT | tail -n 4
echo "[$(date +%T)] 3. merge, aligned offline table, README"
PYTHONPATH= $PY tools/export_merge.py $OUT | tail -n 1
PYTHONPATH= $PY tools/offline_results.py $OUT | tail -n 1
PYTHONPATH= $PY tools/paper_results.py > /dev/null 2>&1
( cd $O/SimOPD && : > $OUT/verdict_ledger.md && for st in 250 200 100; do for b in amc23 aime24 aime25; do PYTHONPATH= SIMOPD_EVAL_ROOT=$O/wandb_inbox/evals $PY scripts/verdict.py --base vanilla_corr --bench $b --step $st 2>/dev/null | grep -v -E "PENDING|MISSING" >> $OUT/verdict_ledger.md; echo >> $OUT/verdict_ledger.md; done; done )
EXPORT_KIND=$KIND PYTHONPATH= $PY tools/export_readme.py $OUT "$(ls -t $O/logs/paper_results/cells_*.csv | head -1)" | tail -n 1
rm -rf $OUT/aligned $OUT/aligned_tables.py $OUT/offline_best.tsv
cp tools/export_train_tables.py tools/export_merge.py tools/offline_results.py tools/export_readme.py tools/export_results.sh $OUT/; rm -f $OUT/.counts
# final exports also carry the additional material (logs, sampled trajectories, non-roster dirs) and its README section
[ "$KIND" = final ] && bash $O/tools/export_extras.sh $TAG | tail -n 1
echo "[$(date +%T)] done -> $OUT"; du -sh $OUT | cut -f1
awk -F'\t' 'NR>1{t++; c+=$6} END{printf "grid cells: %d, fully evaluated: %d\n", t, c}' $OUT/offline_grid_status.tsv
