#!/usr/bin/env python3
"""export_0921_tables.py <out_dir> <evals_dir> <train_gather_dir> <logs_gather_dir> <traj_dir>      (run on w0 by tools/export_0921.sh)

Builds the 0921 (scaling campaign, docs/SCALING-RECIPE.md) results snapshot from what is on disk. Data movement only: every number is read
from an artifact or computed by the formula named in README.md (the composite is the one of scripts/eval_suite.py: equal macro average of
AIME24+25 pooled, AMC23, Minerva, MATH500). Idempotent: re-running refreshes the tables and adds only the per-sample files that are new.
  eval_cells.csv          one row per (run, step, bench) artifact of the LOCKED suite (T 0.7, k 32/32/32/3/3): avg@k, pass@k, lengths, truncation
  eval_composites.tsv     one row per (run, step): the four components and the composite when all five benchmarks exist
  eval_grid_status.tsv    every (run, step) cell the campaign needs (+ the three untrained-student cells) with what is done
  per_sample/<artifact>.parquet   the artifact without its `response` and `token_ids` columns (per-problem, per-sample binaries: what the
                          recipe's paired tests need) -- a few hundred KB per artifact instead of tens of MB
  train/runs.tsv, train/inloop_val.tsv, train/metrics/<run>/launch_*.jsonl.gz, train/manifests/<run>/...
  traj_samples/, logs/ are copied by the shell wrapper; MANIFEST.tsv + README.md are written here.
"""
import csv, glob, gzip, hashlib, json, os, re, shutil, sys, time
import pandas as pd
import pyarrow.parquet as pq

OUT, EV, TR, LG, TJ = sys.argv[1:6]
BENCH = ["aime24", "aime25", "amc23", "minerva", "math500"]
K = {"aime24": 32, "aime25": 32, "amc23": 32, "minerva": 3, "math500": 3}
COMP = [("aime", ["aime24", "aime25"]), ("amc23", ["amc23"]), ("minerva", ["minerva"]), ("math500", ["math500"])]
STEPS = [50, 100, 150, 200, 250]
BASES = {"student_base_1p7b": "Qwen/Qwen3-1.7B-Base", "student_base_4b": "Qwen/Qwen3-4B-Base", "student_base_8b": "Qwen/Qwen3-8B-Base"}
PAT = re.compile(r"(.+?)__([a-z0-9]+)__step(\d+)__seed(\d+)__(\d{8}T\d{6}Z)\.parquet$")
DROP = {"response", "token_ids"}
os.makedirs(f"{OUT}/per_sample", exist_ok=True); os.makedirs(f"{OUT}/train/metrics", exist_ok=True); os.makedirs(f"{OUT}/train/manifests", exist_ok=True)

# ---------------- eval artifacts: newest per (run, step, bench) of the locked suite --------------------------------------------------------
newest = {}
for p in glob.glob(f"{EV}/*.parquet"):
    m = PAT.match(os.path.basename(p))
    if not m: continue
    run, bench, step, seed, ts = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5)
    key = (run, step, bench)
    if key not in newest or ts > newest[key][0]: newest[key] = (ts, p)
cells = []
for (run, step, bench), (ts, p) in sorted(newest.items()):
    pf = pq.ParquetFile(p); cols = [c for c in pf.schema_arrow.names if c not in DROP]
    df = pf.read(columns=cols).to_pandas()
    t = float(df["temperature"].iloc[0]) if "temperature" in df else float("nan")
    k = int(df.groupby("problem_id").size().iloc[0])
    if abs(t - 0.7) > 1e-6 or k != K.get(bench, k):       # not an artifact of the locked suite (era gate, same rule as eval_suite.newest)
        continue
    per = df.groupby("problem_id")["correct"]
    cells.append(dict(run=run, step=step, bench=bench, k=k, n_problems=int(per.ngroups), n_samples=int(len(df)),
                      avg_at_k=round(float(per.mean().mean()), 6), pass_at_k=round(float(per.max().mean()), 6),
                      len_mean=round(float(df["resp_len"].mean()), 1), len_p50=float(df["resp_len"].median()), len_p90=float(df["resp_len"].quantile(0.9)),
                      len_max=int(df["resp_len"].max()), trunc_rate=round(float(df["truncated"].mean()), 6),
                      fr_stop=round(float((df["finish_reason"] == "stop").mean()), 6) if "finish_reason" in df else "",
                      git_sha=str(df["git_sha"].iloc[0]) if "git_sha" in df else "", temperature=t,
                      top_p=float(df["top_p"].iloc[0]) if "top_p" in df else "", max_tokens=int(df["max_tokens"].iloc[0]) if "max_tokens" in df else "",
                      stop_set=str(df["stop_token_ids"].iloc[0]) if "stop_token_ids" in df else "", artifact_ts=ts, artifact_file=os.path.basename(p)))
    dst = f"{OUT}/per_sample/{os.path.basename(p)}"
    if not os.path.exists(dst):
        df.to_parquet(dst + ".tmp", index=False); os.replace(dst + ".tmp", dst)
    cells[-1]["_per_problem"] = per.mean()
pd.DataFrame([{k: v for k, v in c.items() if not k.startswith("_")} for c in cells]).to_csv(f"{OUT}/eval_cells.csv", index=False)

# composites (eval_suite.py DEFAULT_WEIGHTS: 0.25 each; aime24+25 pooled by problem)
by = {}
for c in cells: by.setdefault((c["run"], c["step"]), {})[c["bench"]] = c
rows = []
for (run, step), d in sorted(by.items()):
    r = dict(run=run, step=step, n_bench=len(d), complete=int(all(b in d for b in BENCH)))
    for name, bs in COMP:
        if all(b in d for b in bs):
            r[name] = round(float(pd.concat([d[b]["_per_problem"] for b in bs]).mean()), 6)
        else: r[name] = ""
    r["composite"] = round(sum(0.25 * r[name] for name, _ in COMP), 6) if r["complete"] else ""
    rows.append(r)
with open(f"{OUT}/eval_composites.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["run", "step", "n_bench", "complete", "aime", "amc23", "minerva", "math500", "composite"], delimiter="\t", lineterminator="\n")
    w.writeheader(); w.writerows(rows)

# ---------------- training runs: gathered archive files -----------------------------------------------------------------------------------
runs = {}
for d in sorted(glob.glob(f"{TR}/w[0-3]/*_seed0")):
    run = os.path.basename(d); node = d.split("/")[-2]; runs[run] = dict(run=run, node=node, dir=d)
def scaling_log_times(run):
    st = en = rc = ""
    for n in ["w0", "w1", "w2", "w3"]:
        p = f"{LG}/{n}/scaling_runs.log"
        if not os.path.exists(p): continue
        for l in open(p, errors="replace"):
            if f" {run}: START " in l: st = l[1:20]
            if f" {run}: END " in l: en = l[1:20]; m = re.search(r"rc=(\d+)", l); rc = m.group(1) if m else ""
    return st, en, rc
def wall_h(st, en):
    try:
        f = "%Y-%m-%d_%H:%M:%S"; return round((time.mktime(time.strptime(en, f)) - time.mktime(time.strptime(st, f))) / 3600, 2)
    except Exception: return ""
wandb_ids = {}
for p in glob.glob(f"{LG}/w[0-3]/wandb_ids.tsv"):
    for l in open(p):
        x = l.rstrip("\n").split("\t")
        if len(x) >= 4: wandb_ids[x[2]] = x[0]
run_rows, val_rows = [], []
for run, r in sorted(runs.items()):
    d = r["dir"]; m = re.match(r"t([0-9.]+)_s([0-9.]+)_(vanilla|ours)_seed(\d+)", run)
    files = sorted(glob.glob(f"{d}/metrics/launch_*.jsonl")); steps = {}; vals = {}
    for f in files:                                          # a later launch overrides an earlier one at the same step (resume semantics)
        for line in open(f, errors="replace"):
            try: o = json.loads(line)
            except Exception: continue
            steps[o["step"]] = o["data"]
            for k, v in o["data"].items():
                if k.startswith("val-core/") and k.endswith("acc/mean@1"): vals[o["step"]] = v
        os.makedirs(f"{OUT}/train/metrics/{run}", exist_ok=True); dst = f"{OUT}/train/metrics/{run}/{os.path.basename(f)}.gz"
        if not os.path.exists(dst) or os.path.getmtime(f) > os.path.getmtime(dst):
            with open(f, "rb") as fi, gzip.open(dst + ".tmp", "wb") as fo: shutil.copyfileobj(fi, fo)
            os.replace(dst + ".tmp", dst)
    md = f"{OUT}/train/manifests/{run}"; os.makedirs(md, exist_ok=True)
    for name in ["run_manifest.json", "simopd_stop_contract.txt", "simopd_fingerprint.txt", "latest_checkpointed_iteration.txt"]:
        if os.path.exists(f"{d}/{name}"): shutil.copy2(f"{d}/{name}", md)
    for p in glob.glob(f"{d}/manifest/*"): shutil.copy2(p, md)
    st, en, rc = scaling_log_times(run)
    times = [v.get("timing_s/step") for v in steps.values() if isinstance(v.get("timing_s/step"), (int, float))]
    run_rows.append(dict(run=run, node=r["node"], teacher=m.group(1) if m else "", student=m.group(2) if m else "", method=m.group(3) if m else "",
                         start_cst=st, end_cst=en, wall_h=wall_h(st, en), rc=rc, steps_logged=len(steps), last_step=max(steps) if steps else "",
                         mean_s_per_step=round(sum(times) / len(times), 1) if times else "",
                         latest_ckpt=open(f"{d}/latest_checkpointed_iteration.txt").read().strip() if os.path.exists(f"{d}/latest_checkpointed_iteration.txt") else "",
                         stop_contract=open(f"{d}/simopd_stop_contract.txt").read().strip() if os.path.exists(f"{d}/simopd_stop_contract.txt") else "",
                         wandb_id=wandb_ids.get(run, ""), **{f"val{s}": (round(vals[s], 4) if s in vals else "") for s in STEPS}))
    for s in sorted(vals): val_rows.append(dict(run=run, step=s, val_math500_greedy=round(vals[s], 4)))
pd.DataFrame(run_rows).to_csv(f"{OUT}/train/runs.tsv", sep="\t", index=False)
pd.DataFrame(val_rows).to_csv(f"{OUT}/train/inloop_val.tsv", sep="\t", index=False)

# ---------------- grid status: what the campaign needs vs what exists --------------------------------------------------------------------
have = {(c["run"], c["step"]): set() for c in cells}
for c in cells: have[(c["run"], c["step"])].add(c["bench"])
grid = []
for run in sorted(runs):
    for s in STEPS:
        done = have.get((run, s), set()); grid.append(dict(run=run, step=s, kind="checkpoint", n_bench_done=len(done), complete=int(len(done) == 5), missing=",".join(b for b in BENCH if b not in done)))
for b, model in BASES.items():
    done = have.get((b, 0), set()); grid.append(dict(run=b, step=0, kind=f"untrained student {model}", n_bench_done=len(done), complete=int(len(done) == 5), missing=",".join(x for x in BENCH if x not in done)))
pd.DataFrame(grid).to_csv(f"{OUT}/eval_grid_status.tsv", sep="\t", index=False)

# ---------------- README + MANIFEST ---------------------------------------------------------------------------------------------------------
n_complete = sum(1 for g in grid if g["complete"]); n_cells = len(grid)
readme = f"""# SimOPD scaling campaign ("0921") -- results snapshot

Built {time.strftime('%Y-%m-%d %H:%M:%S')} server time (CST = PT + 15 h) by `tools/export_0921.sh` on w0 from the four Merlin nodes; refreshed periodically while
the campaign runs, so a snapshot may be partial. Recipe: `docs/SCALING-RECIPE.md` of branch `codex/scaling-selectkd-skl` @ 9e9076d (teachers
Qwen3-4B/8B/32B x students Qwen3-1.7B/4B/8B-Base x {{vanilla, ours}}, seed 0, 250 iterations, checkpoints every 50). Offline suite = the
branch's locked one: AIME24/25 + AMC23 avg@32, Minerva + MATH500 avg@3, T 0.7, top-p 0.95, 32,768 tokens; the composite is the equal macro
average of scripts/eval_suite.py (AIME24+25 pooled by problem, AMC23, Minerva, MATH500 -- 0.25 each). The three untrained students are
evaluated once each as `student_base_<size>` at step 0.

State of this snapshot: {len(runs)} training runs archived, {len(cells)} suite artifacts, {n_complete}/{n_cells} fully evaluated (run, step) cells.

Files
- `eval_cells.csv` -- one row per suite artifact (newest per run/step/benchmark): avg@k = mean over problems of the per-problem mean of `correct`;
  pass@k = share of problems with at least one correct sample; lengths in tokens; trunc_rate = share of samples that hit the token budget.
- `eval_composites.tsv` -- per (run, step): the four components and the composite (blank until all five benchmarks exist).
- `eval_grid_status.tsv` -- the 93 cells the campaign needs (18 runs x 5 steps + 3 untrained students) and what is done.
- `per_sample/` -- every artifact without its `response` / `token_ids` columns: per problem and sample `correct`, `resp_len`, `truncated`,
  `finish_reason`, `last_token_id`, plus the artifact's provenance columns. The full responses stay on the nodes (`simopd_data/store_0921/evals/`).
- `train/runs.tsv` -- per run: node, wall time, exit code, steps, mean s/step, stop contract, W&B id, in-loop MATH500 greedy at 50..250.
- `train/inloop_val.tsv` -- the in-loop validation curve (verl's MATH500 greedy mean@1; health telemetry, not the offline suite).
- `train/metrics/<run>/launch_*.jsonl.gz` -- the verl file logger: every scalar per step, one file per launch (a resume = a later launch).
- `train/manifests/<run>/` -- run_manifest.json, the launch manifests, stop contract, fingerprint, latest checkpoint step.
- `traj_samples/<run>/stage_<step>.parquet|json` -- 100 sequences drawn uniformly from every 25th training step's batch (tools/traj_sample.py).
- `logs/` -- gzipped training / eval-farm / guard logs, queue files, exit records.
- `MANIFEST.tsv` -- path, bytes, md5 of every file in this snapshot (written last).
"""
open(f"{OUT}/README.md", "w").write(readme)
man = []
for root, _, fs in os.walk(OUT):
    for f in sorted(fs):
        p = os.path.join(root, f); rel = os.path.relpath(p, OUT)
        if rel == "MANIFEST.tsv" or f.startswith("."): continue
        man.append((rel, os.path.getsize(p), hashlib.md5(open(p, "rb").read()).hexdigest()))
with open(f"{OUT}/MANIFEST.tsv", "w") as fh:
    fh.write("path\tbytes\tmd5\n"); fh.writelines(f"{r}\t{b}\t{m}\n" for r, b, m in sorted(man))
print(f"runs={len(runs)} artifacts={len(cells)} cells_complete={n_complete}/{n_cells} per_sample_files={len(os.listdir(f'{OUT}/per_sample'))} files={len(man)} bytes={sum(b for _, b, _ in man)}")
