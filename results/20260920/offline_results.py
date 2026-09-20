#!/usr/bin/env python3
"""offline_results.py <export_dir>   (run on w0 with the repo venv python; reads evals/*.parquet + the export's post_eval_cells.csv)
Writes THE offline table of the export, already lined up with the paper's protocol, and nothing else:
  offline_results.tsv   one row per (arm, step) that has a COMPLETE five-benchmark suite
  post_eval_cells.csv   gets two extra columns (acc_untruncated_samples, wrong_truncated_share); the repo's columns are untouched
Rules (README.md, section "Alignment"): composite and artifact choice exactly as scripts/eval_suite.py; when a cell has avg@32 artifacts for all of
AIME24/AIME25/AMC23 they are used (paper protocol), otherwise avg@8 (handoff manual 8.1) -- never mixed inside a cell; every arm is compared with the
campaign's own vanilla at the SAME step (vanilla_corr; vanilla_te -- registered to reproduce it -- where vanilla_corr has no complete suite) by a
paired per-problem difference; scripts/verdict.py BASE_OVERRIDES gives the registered base of the special cases."""
import math, os, re, sys
import pandas as pd
OUT = sys.argv[1]; EV = "/home/tiger/opd/wandb_inbox/evals"; REF = 32.8173
BENCH = ["aime24", "aime25", "amc23", "minerva", "math500"]; SMALL = ["aime24", "aime25", "amc23"]
COMP = [("aime", ["aime24", "aime25"]), ("amc23", ["amc23"]), ("minerva", ["minerva"]), ("math500", ["math500"])]
REG_BASE = {"c2_qb_fixed8_corr": "c2_quantile_budget_corr", "c2_qb_perseq_corr": "c2_quantile_budget_corr", "c4_hq": "c4_carrier", "c4_state": "c4_carrier",
            "c5_union_rkl": "c1_lsm_topk32_renorm", "c5_union_fkl": "b2_forward_kl", "c4_pi_tail_budget_corr": "c4_pi_tail_budget",
            "c2_quantile_budget_corr": "c2_quantile_budget", "vanilla_te": "vanilla_corr"}
PAPER = {  # arm -> (table, row label, composite, selected step, seeds)   [Revisit_OPD.pdf Tables 2-7 and 9]
 "a1_gkd_mix0.5_n0": ("T2", "Fixed mix", 24.70, 50, 1), "a3_offpolicy_n0": ("T2", "Teacher", 20.59, 50, 1), "a4_dagger_anneal_n0": ("T2", "Annealed mix", 21.74, 50, 1),
 "g1_verified_only_corr": ("T3", "Correct", 28.23, 125, 3), "d2_selectkd_corr": ("T4", "SelecTKD", 33.09, 150, 3), "d3_teachability_corr": ("T4", "TA-OPD", 29.88, 100, 3),
 "h1_first_segment_corr": ("T4", "Prefix", 30.32, 250, 3), "h2_last_segment_corr": ("T4", "Suffix", 17.17, 250, 3), "h3_random_segment_corr": ("T4", "Window", 32.71, 250, 3),
 "h4_random_scatter_corr": ("T4", "Scatter", 30.09, 50, 3), "c2_quantile_budget_corr": ("T5", "Quantile: mean budget 8", 33.56, 75, 1),
 "c4_pi_tail_budget_corr": ("T5", "Student-tail: 95% mass", 35.04, 175, 1), "c3_intersection_corr": ("T5", "Intersection", 34.52, 250, 3),
 "c2_qb_fixed8_corr": ("T5", "Fixed |S|=8", 23.83, 25, 3), "c2_qb_perseq_corr": ("T5", "Quantile: sequence scope", 27.99, 25, 3), "c4_hq": ("T5", "Student-tail + HQ gate", 33.56, 75, 1),
 "c4_state": ("T5", "Student-tail + state gate", 33.42, 100, 1), "c5_union_fkl": ("T5", "FKL union", 33.01, 75, 1), "c5_union_rkl": ("T5", "RKL union", 29.18, 25, 1),
 "e2_set_coverage_a0_corr": ("T6", "Membership only", 24.46, 25, 3), "b1_skew_kl_corr": ("T7", "Skew", 33.81, 200, 3), "b2_forward_kl_corr": ("T7", "Forward", 32.77, 125, 1),
 "f2_hard_clip_corr": ("T7", "Hard clip", 34.34, 175, 3), "f3_power_corr": ("T7", "Power", 33.79, 125, 3), "vanilla_corr": ("ref", "Vanilla (original run family)", 32.82, 75, 3)}
# ---- artifacts: newest per (arm, step, bench, k) with temperature 0.7
art = {}
for f in os.listdir(EV):
    m = re.match(r"(.+?)_s0__([a-z0-9]+)__step(\d+)__seed\d+__(.+)\.parquet$", f)
    if not m or m.group(2) not in BENCH: continue
    p = os.path.join(EV, f)
    try: meta = pd.read_parquet(p, columns=["temperature", "problem_id"])
    except Exception: continue
    if abs(float(meta["temperature"].iloc[0]) - 0.7) > 1e-6: continue
    key = (m.group(1), int(m.group(3)), m.group(2), int(meta.groupby("problem_id").size().iloc[0]))
    if key not in art or os.path.getmtime(p) > os.path.getmtime(art[key]): art[key] = p
def cell_k(arm, step):
    """32 when all three small benches have an avg@32 artifact, else 8 when all have avg@8, else None."""
    for k in (32, 8):
        if all((arm, step, b, k) in art for b in SMALL): return k
    return None
_df = {}
def load(arm, step, b):
    k = 3 if b in ("minerva", "math500") else cell_k(arm, step)
    if k is None or (arm, step, b, k) not in art: return None
    p = art[(arm, step, b, k)]
    if p not in _df: _df[p] = pd.read_parquet(p, columns=["problem_id", "correct", "truncated", "resp_len"])
    return _df[p]
def pp(arm, step, b):
    d = load(arm, step, b); return None if d is None else d.groupby("problem_id")["correct"].mean()
def complete(arm, step): return all(load(arm, step, b) is not None for b in BENCH)
def composite(arm, step):
    return 100 * sum(0.25 * float(pd.concat([pp(arm, step, b) for b in bs]).mean()) for _, bs in COMP) if complete(arm, step) else None
def paired(arm, base, step):
    d, var = 0.0, 0.0
    for _, bs in COMP:
        a = pd.concat([pp(arm, step, b) for b in bs]); c = pd.concat([pp(base, step, b) for b in bs]); a, c = a.align(c, join="inner"); dif = a - c
        d += 0.25 * dif.mean(); var += (0.25 ** 2) * dif.var(ddof=1) / len(dif)
    se = math.sqrt(var); p = math.erfc(abs(d / se) / math.sqrt(2)) if se > 0 else 1.0
    return 100 * d, 100 * se, p
cells = pd.read_csv(f"{OUT}/post_eval_cells.csv"); stop = {(r.arm, int(r.step)): str(r.stop_set) for r in cells.itertuples()}
arms = sorted({a for a, _, _, _ in art}); steps = sorted({s for _, s, _, _ in art}); rows = []
for arm in arms:
    done = [s for s in steps if complete(arm, s)]
    if not done: continue
    comp = {s: composite(arm, s) for s in done}; best = max(comp, key=comp.get)
    amc = {}                                   # AMC23 is evaluated at EVERY grid step (also the 3-benchmark light cells) -> truncation trajectory
    for s in steps:
        pth = art.get((arm, s, "amc23", 32)) or art.get((arm, s, "amc23", 8))
        if pth: amc[s] = float(pd.read_parquet(pth, columns=["truncated"])["truncated"].mean())
    for s in done:
        dfs = [load(arm, s, b) for b in BENCH]; n = sum(len(x) for x in dfs)
        aime = float(pd.concat([pp(arm, s, "aime24"), pp(arm, s, "aime25")]).mean())
        r = dict(arm=arm, step=s, k_small_benches=cell_k(arm, s), seed=0, **{b: round(100 * float(pp(arm, s, b).mean()), 2) for b in BENCH}, aime_pooled=round(100 * aime, 2),
                 composite=round(comp[s], 2), trunc_pct=round(100 * sum(x["truncated"].sum() for x in dfs) / n, 1), len_mean=int(sum(x["resp_len"].sum() for x in dfs) / n))
        base = "vanilla_corr" if complete("vanilla_corr", s) else ("vanilla_te" if complete("vanilla_te", s) else None)
        r["vanilla_base_used"] = base or ""; r["vanilla_base_composite"] = round(composite(base, s), 2) if base else ""
        if base and base != arm:
            d, se, p = paired(arm, base, s); r.update(diff_vs_vanilla=round(d, 2), paired_se=round(se, 2), p_value=round(p, 4), ci95_lo=round(d - 1.96 * se, 2), ci95_hi=round(d + 1.96 * se, 2))
        rb = REG_BASE.get(arm, "vanilla_corr"); r["registered_base"] = rb
        if rb not in (arm, base) and complete(rb, s):
            d, se, p = paired(arm, rb, s); r.update(diff_vs_registered_base=round(d, 2), p_registered_base=round(p, 4))
        r["gap_vs_paper_ref_32.8173"] = round(comp[s] - REF, 2); r["is_best_step"] = int(s == best); r["n_complete_suites"] = len(done); r["stop_set"] = stop.get((arm, s), "")
        t = PAPER.get(arm); r.update(paper_table=t[0] if t else "", paper_row=t[1] if t else "", paper_composite=t[2] if t else "", paper_step=t[3] if t else "", paper_seeds=t[4] if t else "")
        r["amc23_trunc_pct_by_step"] = ";".join(f"{x}:{100 * v:.0f}" for x, v in amc.items()); rows.append(r)
cols = ["arm", "step", "k_small_benches", "seed"] + BENCH + ["aime_pooled", "composite", "trunc_pct", "len_mean", "vanilla_base_used", "vanilla_base_composite", "diff_vs_vanilla", "paired_se", "p_value",
        "ci95_lo", "ci95_hi", "registered_base", "diff_vs_registered_base", "p_registered_base", "gap_vs_paper_ref_32.8173", "is_best_step", "n_complete_suites", "stop_set",
        "paper_table", "paper_row", "paper_composite", "paper_step", "paper_seeds", "amc23_trunc_pct_by_step"]
if os.environ.get("EXPORT_STATS", "0") != "1":      # haotian 2026-09-19 PT: follow the 0917 branch strictly, invent nothing -> my paired SE / p / CI stay out
    cols = [c for c in cols if c not in ("paired_se", "p_value", "ci95_lo", "ci95_hi", "p_registered_base")]
res = pd.DataFrame(rows).reindex(columns=cols); res.to_csv(f"{OUT}/offline_results.tsv", sep="\t", index=False)
# ---- two extra columns on the repo-schema per-benchmark table (same artifact, identified by its time stamp)
ex = {}
for r in cells.itertuples():
    p = os.path.join(EV, f"{r.run}__{r.bench}__step{int(r.step)}__seed{int(r.seed)}__{r.ts}.parquet")
    try: d = pd.read_parquet(p, columns=["correct", "truncated"])
    except Exception: continue
    nt = d[d["truncated"] == 0]; wr = d[d["correct"] != 1]
    ex[r.Index] = (float(nt["correct"].mean()) if len(nt) else None, float(wr["truncated"].mean()) if len(wr) else None)
cells["acc_untruncated_samples"] = [ex.get(i, (None, None))[0] for i in cells.index]; cells["wrong_truncated_share"] = [ex.get(i, (None, None))[1] for i in cells.index]
cells.to_csv(f"{OUT}/post_eval_cells.csv", index=False)
for stale in ("offline_best.tsv",):
    if os.path.exists(f"{OUT}/{stale}"): os.remove(f"{OUT}/{stale}")
print(f"offline_results.tsv rows={len(res)} arms={res.arm.nunique()} k32_cells={(res.k_small_benches == 32).sum()}; post_eval_cells.csv +2 columns")
