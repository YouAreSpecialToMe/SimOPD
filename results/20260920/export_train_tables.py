#!/usr/bin/env python3
"""export_train_tables.py <node> <out_dir>   (run ON a node with SimOPD/simopd/bin/python; read-only on the run directories)
Runs the repo's OWN exporter results/20260909/extract.py -- same columns, same formulas, same "later launch file overrides an earlier one"
rule -- but (a) restricted to the roster arms whose run directory lives on THIS node (the 2026-09-19 campaign is sharded over four nodes) and
(b) writing to <out_dir> instead of into the repository. tools/export_results.sh merges the four per-node outputs."""
import importlib.util, os, sys
node, out = sys.argv[1], sys.argv[2]
src = "/home/tiger/opd/SimOPD/results/20260909/extract.py"
spec = importlib.util.spec_from_file_location("extract_20260909", src); ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
os.makedirs(os.path.join(out, "metrics_full"), exist_ok=True)
ex.OUT = out
_roster = ex.roster
ex.roster = lambda: [r for r in _roster() if os.path.isdir(f"{ex.CK}/{r[0]}_s0")]
print(f"[{node}] store={ex.STORE} arms here: {[r[0] for r in ex.roster()]}", flush=True)
ex.main()
print(f"[{node}] done -> {out}", flush=True)
