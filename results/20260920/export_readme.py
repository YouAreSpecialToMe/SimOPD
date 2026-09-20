#!/usr/bin/env python3
"""export_readme.py <out_dir> [<paper_results cells csv>]  -- writes README.md for a results export directory from the tables in it (no hand-typed numbers)."""
import csv, glob, os, sys, time
import pandas as pd

def _pt_now():
    """haotian's clock: Pacific Time first, UTC second (the servers run CST)."""
    import datetime
    try:
        from zoneinfo import ZoneInfo; pt = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    except Exception:
        pt = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-7)))
    return pt.strftime("%Y-%m-%d %H:%M") + " PT (" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M") + " UTC)"
OUT = sys.argv[1]; chk = sys.argv[2] if len(sys.argv) > 2 else None
t = lambda n: pd.read_csv(f"{OUT}/{n}", sep="\t", dtype=str, keep_default_na=False)
prog, ck, wide, vg, ms = t("progress.tsv"), t("checkpoints.tsv"), t("val_accuracy_wide.tsv"), t("val_gen_summary.tsv"), t("metrics_summary.tsv")
res, grid = t("offline_results.tsv"), t("offline_grid_status.tsv"); cells = pd.read_csv(f"{OUT}/post_eval_cells.csv"); by = pd.read_csv(f"{OUT}/post_eval_bystep.csv")
nodes = t("arm_nodes.tsv"); n_of = dict(zip(nodes.arm, nodes.node))
roster = [r.split("\t")[0] for r in open("/home/tiger/opd/SimOPD/configs/retrain_roster.tsv").read().splitlines() if r and not r.startswith("#")]
absent = [a for a in roster if a not in set(prog.arm)]
reached = prog[prog.reached_target == "1"]; ck_sum = prog.ckpt_max_step.astype(int).sum(); tgt_sum = prog.target_steps.astype(int).sum()
L = []; A = L.append
KIND = os.environ.get("EXPORT_KIND", "snapshot")
A(f"# 2026-09 roster retraining -- results data set (exported {_pt_now()}, " + ("FINAL: training and offline evaluation complete" if KIND == "final" else "mid-campaign snapshot") + ")\n")
A("Same kind of directory as `results/20260909/`. The training tables and `post_eval_*.csv` are **data movement**: every column is a number found on disk or computed from")
A("such numbers by a formula stated below. `offline_results.tsv` additionally applies the comparison rule the branch registers (section Alignment: matched vanilla `vanilla_corr`, fixed")
A("step) and `verdict_ledger.md` is the output of the branch\x27s own `scripts/verdict.py`; nothing here is a method of my own, no measured number is altered, no arm is labelled.\n")
A("What is new relative to `results/20260909/`: (1) the training tables are re-extracted after the training window of 2026-09-18 16:30 PT to 2026-09-20 (PT) on the 4-node")
A("cluster; (2) the **offline evaluation suite now exists** (gap 7 of the 0909 README: \"no offline evaluation yet\") -- `post_eval_cells.csv`,")
A("`post_eval_bystep.csv` in the schema of `docs/data/post_eval_*.csv`, and `offline_results.tsv`, which lines the results up with the paper protocol (section Alignment).\n")
A("- Training tables: the repo's own `results/20260909/extract.py`, unchanged, run once per node (the run directories are sharded over four nodes,")
A("  `arm_nodes.tsv`) and concatenated in roster order (`export_train_tables.py`, `export_merge.py`).")
A("- Offline tables: the repo's own `scripts/extract_post_eval.py --roster ALL`, unchanged, over the union of the four nodes' `evals/*.parquet`.")
A("- Sources: `<arm>_s0/metrics/launch_*.jsonl`, `val_gen/*.jsonl`, `global_step_*/`, `traj/`, `evals/*.parquet`, `configs/retrain_roster.tsv`, `configs/arms.yaml`.\n")
A("---\n\n## Files\n\n| file | rows | content |\n|---|---|---|")
desc = [("arms.tsv", "axis, target steps, roster priority, env delta vs the carrier, description, roster note (arms with a run directory on this cluster)"),
        ("arm_nodes.tsv", "which node holds each arm's run directory"),
        ("progress.tsv", "max checkpoint step, number of checkpoints, max logged step, logged steps, reached target, number of launch files"),
        ("checkpoints.tsv", "one row per checkpoint (arm, global_step)"), ("val_accuracy_wide.tsv", "arm x step in-loop validation accuracy (steps that are multiples of 25)"),
        ("metrics_summary.tsv", "arm x step long table of the 24 common metrics"), ("metrics_full/<arm>.tsv", "every logged metric of the arm per step"),
        ("val_gen_summary.tsv", "one row per in-loop validation (items, acc mean, reward mean, output chars)"), ("archive_inventory.tsv", "file counts of the analysis archive layer"),
        ("teacher_columns.tsv", "row count and teacher-column means of each `summary_<step>.parquet`"),
        ("post_eval_cells.csv", "OFFLINE: one row per (arm, seed, step, benchmark) -- newest artifact; acc, pass@k, lengths, truncation, finish reasons, stop set (+ two added columns: accuracy over the non-truncated samples, share of wrong samples that are truncated)"),
        ("post_eval_bystep.csv", "OFFLINE: one row per (arm, step) with a COMPLETE five-benchmark suite: composite and its four components"),
        ("offline_results.tsv", "OFFLINE, THE RESULT TABLE: one row per (arm, step) with a complete suite -- five scores and composite (x100, `scripts/eval_suite.py` definition), truncation, the composite of the campaign vanilla at the SAME step and the arithmetic difference, the registered base of `scripts/verdict.py`, the gap vs the fixed 32.8173 of the paper draft, `is_best_step`, and the paper row this arm corresponds to (table, label, composite, selected step, seeds)"),
        ("verdict_ledger.md", "OFFLINE: output of the branch's own `scripts/verdict.py --base vanilla_corr` (exact McNemar on per-problem paired results, noise floor from vanilla seeds) for the T=0.7 benchmarks at steps 100/200/250; rows without artifacts (PENDING/MISSING) are filtered out, nothing else is touched"),
        ("offline_grid_status.tsv", "OFFLINE: every checkpoint on the light grid, benches required / finished / missing at export time")]
cnt = dict(x.split("=") for x in open(f"{OUT}/.counts").read().strip().split(";"))
for f, d in desc: A(f"| `{f}` | {cnt.get(f if not f.startswith('metrics_full') else 'metrics_full/', '')}{' files' if f.startswith('metrics_full') else ''} | {d} |")
A("\nPer-item generations (in-loop `val_gen/<step>.jsonl`, offline `evals/*.parquet` with full response text, `traj/`) are NOT copied here; they stay in the run directories.\n")
A("---\n\n## Definitions\n")
A("Training-side columns, the three step notions, `env_delta_vs_carrier`, the shared training configuration and the \"later launch file overrides an")
A("earlier one for the same step\" rule are exactly those of `results/20260909/README.md` (same script).\n")
A("### Offline suite (handoff manual 8.1, `scripts/eval_suite.py`)\n")
A("```\naime24 / aime25   avg@8   30 problems each     temperature 0.7, top_p 0.95, 32768-token budget\namc23             avg@8   40 problems\nminerva           avg@3  272 problems\nmath500           avg@3  500 problems")
A("avg@k     = mean over problems of the per-problem mean of `correct` over its k samples\ncomposite = 0.25 * (AIME24+AIME25 pooled by problem) + 0.25 * AMC23 + 0.25 * Minerva + 0.25 * MATH500, complete five-benchmark suites only")
A("grid      = `eval_refill_exp.py --grid light`: steps 100/200/250 all five benchmarks; steps 25/50/150 amc23+minerva+math500 (no composite); other steps not evaluated\n```")
A("`k_small_benches` in `offline_results.tsv` is 8 (handoff manual 8.1: every artifact of this round is avg@8) or 32 (paper protocol); a cell uses avg@32 only when all three")
A("small benchmarks have it, sample counts are never mixed inside one estimate. All runs are seed 0.\n")
A("### Alignment with the paper (rules applied in `offline_results.tsv`; none changes a measured number)\n")
vc = res[res.arm == "vanilla_corr"]
A("1. **Matched control, fixed step.** Every `_corr` / `_n0` arm is built on the carrier `vanilla_corr`, which scores " + ", ".join(f"{float(x.composite):.2f}@{x.step}" for x in vc.itertuples()) + " here against the fixed")
A("   reference 32.82 of the paper draft (original run family). A gap against 32.82 therefore mixes the method with the base. `diff_vs_vanilla` compares each arm with the campaign vanilla")
A("   at the SAME step (`vanilla_corr`; `vanilla_te`, registered to reproduce it, where `vanilla_corr` has no complete suite). This is the rule the branch already registers")
A("   (handoff manual 8.4, `scripts/verdict.py`: baseline `vanilla_corr`, one fixed step, paired per-problem comparison, tie below the noise floor). `registered_base` lists the")
A("   `BASE_OVERRIDES` special cases; `gap_vs_paper_ref_32.8173` is kept only as the historical convention.")
A("2. **Test and noise floor are the branch's.** `verdict_ledger.md` is the output of `scripts/verdict.py --base vanilla_corr` at a fixed step: per-benchmark accuracy difference, exact McNemar on the")
A("   per-problem paired results, and the noise floor = range over vanilla seeds 0/1/2. Only seed 0 exists in this roster, so every verdict reads AWAIT-FLOOR; the ledger registers MATH500 and")
A("   Minerva as greedy (T=0) artifacts, which this campaign does not produce, so it covers amc23 / aime24 / aime25. `diff_vs_vanilla` in `offline_results.tsv` is plain arithmetic on composites.")
A("3. **Best checkpoint vs fixed step.** `is_best_step` marks the paper-style selection (highest composite among the complete suites of the arm); the fixed-step reading is the row at step 200.")
A("4. **Truncation is part of the result.** `trunc_pct` and `amc23_trunc_pct_by_step` stand next to every composite; a checkpoint whose AMC23 truncation is >= 90% has stopped terminating.\n")
A("---\n\n## Summary numbers\n\n| item | value |\n|---|---|")
A(f"| roster arms | {len(roster)} |\n| arms with a run directory on this cluster (training tables) | {len(prog)} |")
A(f"| reached target (`global_step_<target>` exists) | **{len(reached)}** |\n| checkpoint steps / target steps (these {len(prog)} arms) | **{ck_sum} / {tgt_sum}** |")
A(f"| checkpoints | {len(ck)} |\n| in-loop validations (`val_gen_summary.tsv` rows) | {len(vg)} |")
A(f"| offline benchmark artifacts (`post_eval_cells.csv` rows) | {len(cells)} |\n| complete five-benchmark suites (`post_eval_bystep.csv` rows) | **{len(by)}** over {by.arm.nunique()} arms |")
g = grid.copy(); g["complete"] = g.complete.astype(int)
A(f"| light-grid cells that exist as checkpoints here / fully evaluated | {len(g)} / **{g.complete.sum()}** (anchor {len(g[g.grid_kind=='anchor'])} / {g[g.grid_kind=='anchor'].complete.sum()}, light {len(g[g.grid_kind=='light'])} / {g[g.grid_kind=='light'].complete.sum()}) |")
A(f"\nReached target: {', '.join('`'+a+'`' for a in reached.arm)}.\n")
A("### In-loop validation accuracy (`val_accuracy_wide.tsv`; blank = no record at that step)\n\n```")
cols = list(wide.columns[1:]); A("arm".ljust(26) + "".join(c.rjust(7) for c in cols))
for r in wide.itertuples(index=False): A(r[0].ljust(26) + "".join((f"{float(v):.3f}" if v else "").rjust(7) for v in r[1:]))
A("```\n\nn = 500, so the binomial standard error of one point is about 0.021-0.022.\n")
A("### Offline results (`offline_results.tsv`; x100; diff = composite minus the campaign vanilla composite at the same step)\n\n```")
A("arm".ljust(26) + "step".rjust(5) + "k".rjust(4) + "composite".rjust(11) + "diff".rjust(8) + "trunc%".rjust(8) + "  paper row (composite@step, seeds)")
num = lambda v, f: (f % float(v)) if v not in ("", None) else ""
for st_ in sorted(set(res.step.astype(int)), reverse=True):
    for r in res[res.step.astype(int) == st_].assign(c=lambda d: d.composite.astype(float)).sort_values("c", ascending=False).itertuples():
        A(r.arm.ljust(26) + str(r.step).rjust(5) + str(r.k_small_benches).rjust(4) + num(r.composite, "%.2f").rjust(11) + num(r.diff_vs_vanilla, "%+.2f").rjust(8) + num(r.trunc_pct, "%.1f").rjust(8) + ("  " + r.paper_row + " (" + str(r.paper_composite) + "@" + str(r.paper_step) + ", " + str(r.paper_seeds) + " seeds)" if r.paper_row else ""))
    A("")
A("```\n")
A("Sampling noise of one composite at avg@8, from the calibration note in `scripts/eval_suite.py`: SE about 0.75 (x100); seed-to-seed SD of the legacy vanilla about 0.42.\n")
A("---\n\n## Consistency checks (recorded as found)\n")
m = ms[ms.val_acc != ""][["arm", "step", "val_acc"]].merge(vg[["arm", "step", "acc_mean"]], on=["arm", "step"])
d = (m.val_acc.astype(float) - m.acc_mean.astype(float)).abs()
A(f"1. **In-loop accuracy from two independent sources.** `metrics_summary.tsv:val_acc` vs `val_gen_summary.tsv:acc_mean` on the {len(m)} (arm, step) pairs present in both: max |difference| = {d.max():.4f}, pairs differing by more than 0.0005: {(d > 0.0005).sum()}.")
A(f"2. **Checkpoint counts.** sum of `progress.tsv:n_checkpoints` = {prog.n_checkpoints.astype(int).sum()}; `checkpoints.tsv` rows = {len(ck)}.")
if chk and os.path.exists(chk):
    c = pd.read_csv(chk); c = c[c.composite.notna()]; c["arm"] = c.arm.astype(str)
    j = by.assign(c100=100 * by.composite).merge(c[["arm", "step", "composite"]], on=["arm", "step"], suffixes=("", "_suite"))
    A(f"3. **Composite from two code paths.** `extract_post_eval.py` (newest artifact by file-name time stamp) vs an `eval_suite.py acc`-style recomputation (newest artifact with temperature 0.7 and the suite's samples/problem): {len(j)} of {len(by)} complete suites matched, max |difference| = {(j.c100 - j.composite_suite).abs().max():.6f} (x100).")
A(f"4. **Stop sets.** distinct `stop_set` values in `post_eval_cells.csv`: {', '.join(sorted(map(str, cells.stop_set.unique())))}; (run, step) groups with more than one stop set: {(cells.groupby(['run','step']).stop_set.nunique() > 1).sum()}.\n")
A("---\n\n## Known gaps and facts at export time (observations, no inference)\n")
A(f"1. **Arms below target.** Arms below target in `progress.tsv`: {', '.join('`'+r.arm+'` ('+r.ckpt_max_step+'/'+r.target_steps+')' for r in prog[prog.reached_target != '1'].itertuples())}. " + ("`a5_aggrevate_n0` was still training when this was exported; " if (prog[prog.arm == "a5_aggrevate_n0"].reached_target != "1").any() else "") + "`c5_union_fkl`, `c5_union_rkl`, `e2_set_coverage_a0_corr`, `g6_seqmean_corr`, `h2_last_segment_corr` were stopped on 2026-09-18 17:00 PT (2026-09-19 00:00 UTC) by decision and are evaluated at the checkpoints they have.")
A(f"2. **Offline evaluation coverage.** `offline_grid_status.tsv` lists {len(g) - g.complete.sum()} grid cells that are not fully evaluated yet (column `missing`). A benchmark appears in `post_eval_cells.csv` only when its artifact is final; a (arm, step) appears in `post_eval_bystep.csv` only with all five.")
A(f"3. **Roster arms without a run directory on this cluster** (they reached their target before the handoff; only their offline artifacts were shipped): {', '.join('`'+a+'`' for a in absent)}. They are absent from the training tables and present in the offline tables.")
tr = cells[cells.trunc_rate >= 0.5]
per = tr.groupby("arm").size().sort_values(ascending=False)
A(f"4. **Offline truncation** (32768-token budget): {len(tr)} of {len(cells)} benchmark artifacts have `trunc_rate` >= 0.5 (filter `post_eval_cells.csv`); artifacts per arm: " + ", ".join(f"`{a}` {n}" for a, n in per.items()) + ". Highest: " + "; ".join(f"`{r.arm}`@{int(r.step)} {r.bench} {r.trunc_rate:.2f}" for r in cells.sort_values("trunc_rate", ascending=False).head(6).itertuples()) + ".")
A("5. Steps re-run after a resume are logged twice; as in 0909 the later launch file wins. `n_launch_files` in `progress.tsv` gives the number of launches.")
A("6. W&B was offline in the 0909 export; in this window it is online (training curves, the offline table, 100 sampled trajectories per 25-step stage per arm). This directory remains the file record.")
import re as _re, collections as _co
_ev = os.path.join(os.path.dirname(OUT.rstrip("/")), "store_view", "evals"); _grp = _co.defaultdict(list)
for _f in sorted(glob.glob(_ev + "/*.parquet")):
    _m = _re.match(r"(.+)__([a-z0-9]+)__step(\d+)__seed(\d+)__(\w+)\.parquet$", os.path.basename(_f))
    if _m: _grp[(_m.group(1), _m.group(2), int(_m.group(3)))].append(_f)
_dup = {k: v for k, v in _grp.items() if len(v) > 1}
def _acc(_f):
    _d = pd.read_parquet(_f, columns=["problem_id", "correct"]); return _d.groupby("problem_id").correct.mean().mean()
_parts = []
for _k, _v in sorted(_dup.items()):
    _parts.append("`%s`@%d %s: %s" % (_k[0], _k[2], _k[1], ", ".join("artifact %s acc %.4f" % (os.path.basename(_x).split("__")[-1][:-8], _acc(_x)) for _x in _v)))
A("7. **Benchmarks evaluated more than once.** Artifacts in the union of the four nodes: %d; distinct (run, benchmark, step): %d; with more than one artifact: %d%s. The tables use the newest artifact (rule of `scripts/eval_suite.py` and `scripts/extract_post_eval.py`); a second artifact comes from a scheduling overlap between two nodes -- same checkpoint, same settings.\n" % (sum(len(v) for v in _grp.values()), len(_grp), len(_dup), ("; " + "; ".join(_parts)) if _parts else ""))
A("---\n\n## Reproduce\n\n```bash\n# on every node that holds run directories\npython export_train_tables.py <node> <scratch>/nodes/<node>\n# on the node that holds the union of evals/*.parquet\nSIMOPD_STORE=<dir with evals/> python scripts/extract_post_eval.py --roster ALL --out-dir <out>\npython export_merge.py <out> && python export_readme.py <out>\n```\nAll scripts only read the run directories.")
open(f"{OUT}/README.md", "w").write("\n".join(L) + "\n"); print("README.md written,", len(L), "lines")
