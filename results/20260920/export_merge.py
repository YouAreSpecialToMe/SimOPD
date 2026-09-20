#!/usr/bin/env python3
"""export_merge.py <out_dir>   (run on w0)  -- merges the four per-node outputs of tools/export_train_tables.py (= the repo's
results/20260909/extract.py, run per node) into one directory with the SAME files and columns as results/20260909/, and adds the offline-suite
tables. Data movement only: every column is a number found on disk or computed by the formula stated in README.md."""
import csv, glob, os, shutil, subprocess, sys, time
import pandas as pd
OUT = sys.argv[1]; E = os.path.dirname(OUT.rstrip("/")); NODES = ["w0", "w1", "w2", "w3"]; REPO = "/home/tiger/opd/SimOPD"
REF = 32.8173; ANCHOR, CHEAP = {100, 200, 250}, {25, 50, 150}; ALL5 = ["aime24", "aime25", "amc23", "minerva", "math500"]; LIGHT = ["amc23", "minerva", "math500"]
roster = [r.split("\t") for r in open(f"{REPO}/configs/retrain_roster.tsv").read().splitlines() if r and not r.startswith("#")]
order = {r[0]: i for i, r in enumerate(roster)}; target = {r[0]: int(r[1]) for r in roster}
def rd(n, name):
    p = f"{E}/nodes/{n}/{name}"
    if not os.path.exists(p): return None, []
    rows = list(csv.reader(open(p), delimiter="\t")); return rows[0], rows[1:]
def wr(name, header, rows):
    with open(f"{OUT}/{name}", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n"); w.writerow(header); w.writerows(rows)
    counts[name] = len(rows)
counts = {}; node_of = {}
os.makedirs(f"{OUT}/metrics_full", exist_ok=True)
for name in ["arms.tsv", "progress.tsv", "checkpoints.tsv", "metrics_summary.tsv", "val_gen_summary.tsv", "archive_inventory.tsv", "teacher_columns.tsv"]:
    header, rows = None, []
    for n in NODES:
        h, r = rd(n, name)
        if h is None: continue
        header = header or h; assert h == header, (name, n)
        rows += r
        if name == "progress.tsv":
            for x in r: node_of[x[0]] = n
    rows.sort(key=lambda x: (order.get(x[0], 999), int(x[1]) if len(x) > 1 and x[1].lstrip("-").isdigit() and name != "arms.tsv" and name != "progress.tsv" and name != "archive_inventory.tsv" else 0))
    wr(name, header, rows)
# val_accuracy_wide: the step columns differ per node (250-step arms live on w0 only) -> union of columns
wide = {}; steps = set()
for n in NODES:
    h, r = rd(n, "val_accuracy_wide.tsv")
    if h is None: continue
    for x in r:
        wide[x[0]] = dict(zip(h[1:], x[1:])); steps |= set(h[1:])
cols = sorted(steps, key=int)
wr("val_accuracy_wide.tsv", ["arm"] + cols, [[a] + [wide[a].get(c, "") for c in cols] for a in sorted(wide, key=lambda a: order.get(a, 999))])
nfull = 0
for n in NODES:
    for p in glob.glob(f"{E}/nodes/{n}/metrics_full/*.tsv"): shutil.copy2(p, f"{OUT}/metrics_full/"); nfull += 1
counts["metrics_full/"] = nfull
wr("arm_nodes.tsv", ["arm", "node"], [[a, node_of[a]] for a in sorted(node_of, key=lambda a: order.get(a, 999))])
# ---------------- offline suite: tables derived from post_eval_cells.csv / post_eval_bystep.csv (written by scripts/extract_post_eval.py)
cells = pd.read_csv(f"{OUT}/post_eval_cells.csv"); by = pd.read_csv(f"{OUT}/post_eval_bystep.csv")
counts["post_eval_cells.csv"] = len(cells); counts["post_eval_bystep.csv"] = len(by)
# grid status: which (arm, step) cells of the light grid exist as checkpoints, and how many of their required benches are finished
ck = pd.read_csv(f"{OUT}/checkpoints.tsv", sep="\t"); have = cells.groupby(["arm", "step"])["bench"].apply(set).to_dict()
rows = []
for r in ck.itertuples():
    s = int(r.global_step)
    if s not in ANCHOR and s not in CHEAP: continue
    need = ALL5 if s in ANCHOR else LIGHT; done = [b for b in need if b in have.get((r.arm, s), set())]
    rows.append([r.arm, s, "anchor" if s in ANCHOR else "light", len(need), len(done), int(len(done) == len(need)), ",".join(b for b in need if b not in done)])
wr("offline_grid_status.tsv", ["arm", "step", "grid_kind", "n_bench_required", "n_bench_done", "complete", "missing"], rows)
json_counts = ";".join(f"{k}={v}" for k, v in counts.items()); open(f"{OUT}/.counts", "w").write(json_counts + "\n")
print(json_counts)
