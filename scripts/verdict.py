"""The verdict ledger: every arm against vanilla, by the pre-registered rules.

    python scripts/verdict.py                       # ledger from whatever artifacts exist
    python scripts/verdict.py --step 250 --write docs/verdict-ledger.md

Implements plan §4/§5 exactly:

  noise floor   per-domain range across vanilla seeds 0/1/2 at the same step. The
                floor is what gives "tie" an operational meaning; until all three
                seeds' artifacts exist the ledger says AWAIT-FLOOR, never guesses.
  tie           |Δ| < floor  ->  TIE, and tie takes precedence over significance:
                a p<0.05 win smaller than seed noise is a win over nothing.
  promote       exact McNemar on per-problem paired binaries, p < 0.05. Exact
                binomial on the discordant pairs, not the chi-square approximation --
                discordant counts here are routinely small, which is precisely where
                the approximation lies.
  fixed step    comparisons at one step for every arm (fixed-horizon amendment,
                2026-08-06). vanilla_s0 at 250 is 0.506, not its 0.63 peak; that
                decline is Mode A data, and the ledger compares survivors' endpoints.

Missing artifacts are rows too: each prints the eval_offline.py command that would
fill it, so this file doubles as the evaluation work-list. Runs on pandas + stdlib.
"""

import argparse
import glob
import math
import os
import sys

BASE = "vanilla"
SEEDS = (0, 1, 2)
ALPHA = 0.05

# Pre-registered deviations that gate specific rows (plan §4.6). Shown on the ledger
# so a reader cannot take the number without the asterisk.
CAVEATS = {
    "c1_lsm_topk32_renorm": "ran on cornell; vanilla floor is DSW -- same-cluster control (TAG=xc) pending",
    "f1_soft_log": "completed at 150 steps under the old default; +100-step resume pending before fixed-step entry",
    # 2607.23731 (Outcome-Confounded Local Supervision), read for this wording:
    # outcome-level filtering does not localize token-level signal -- ~68% of
    # response-token mass is agreement-on-failure even under such filters. The
    # verdict may say "trajectory selection helps/hurts", never "purified signal".
    "g1_verified_only": "outcome-level gate only; no token-level localization claim (2607.23731)",
    "g4_failure_only": "outcome-level gate only; no token-level localization claim (2607.23731); g1's mirror, same red line",
    "g5_rgopd_gate": "gate audited on protocol k1 base, NOT their top-50+tail base (registered); outcome-level caveat as g1",
    "vanilla_n8": "n=8 cell control; its row vs vanilla measures the group-sampling knob alone",
    "j1_kdrl": "judged vs vanilla_n8 (same cell); cross-cell comparison to the main table crosses the n boundary",
    # N2's verdict is about the length fixed point first and capability second: the
    # preregistered reads (P-EOS / P-support / P-teacher, arms.yaml) are decided on
    # the eos_* panels + the truncation trajectory, not on composite alone.
    "n0_termfix": "single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1",
    "n02_termfix_cal": "corrected-rerun wave 1: judged against n0_termfix (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right",
    "n0_f1_softlog": "corrected-rerun wave 1: judged against n0_termfix and banked f1_soft_log",
    "n0_f2_clip10": "corrected-rerun wave 1: judged against n0_termfix and banked f2_hard_clip",
    "n0_f3_power": "corrected-rerun wave 1: judged against n0_termfix and banked f3_power",
    "n0_h2_lastseg": "corrected-rerun wave 1: judged against n0_termfix and banked h2_last_segment",
    "n0_b1_skew": "corrected-rerun wave 1: judged against n0_termfix and banked b1_skew_kl",
    "n0_g1_verified": "corrected-rerun wave 1: judged against n0_termfix and banked g1_verified_only",
    "n0_g4_failure": "corrected-rerun wave 1: judged against n0_termfix and banked g4_failure_only",
    "n0_g6_seqmean": "corrected-rerun wave 1: judged against n0_termfix and banked g6_seqmean",
    "n0_f2_clip2.3": "corrected-rerun wave 1: judged against n0_termfix and the banked f2_clip2.3 row, seed-paired",
    "n0_f4_posclip": "corrected-rerun wave 1: judged against n0_termfix and the banked f4_posclip row, seed-paired",
    "n0_f5_tanh": "corrected-rerun wave 1: judged against n0_termfix and the banked f5_tanh row, seed-paired",
    "n0_b5_k2": "corrected-rerun wave 1: judged against n0_termfix and the banked b5_k2 row, seed-paired",
    "n0_h1_firstseg": "corrected-rerun wave 1: judged against n0_termfix and the banked h1_first_segment row, seed-paired",
    "n0_h3_randseg": "corrected-rerun wave 1: judged against n0_termfix and the banked h3_random_segment row, seed-paired",
    "n0_h4_randscatter": "corrected-rerun wave 1: judged against n0_termfix and the banked h4_random_scatter row, seed-paired",
    "n0_g5_rgopd": "corrected-rerun wave 1: judged against n0_termfix and the banked g5_rgopd_gate row, seed-paired",
    "n0_g2_fire": "corrected-rerun wave 1: judged against n0_termfix and the banked g2_fire_likelihood row, seed-paired",
    "n0_d1_tip": "corrected-rerun wave 2 (D, both paths): judged against n0_termfix and the banked d1_tip row, seed-paired",
    "n0_d2_selectkd": "corrected-rerun wave 2 (D, both paths): judged against n0_termfix and the banked d2_selectkd row, seed-paired",
    "n0_d3_teachability": "corrected-rerun wave 2 (D, both paths): judged against n0_termfix and the banked d3_teachability row, seed-paired",
    "n2_termcal": "single-knob vs vanilla; verdict reads the termination fixed point (trunc/len) and the eos_* panels before composite; E_S={151643} student side, E_T={151643,151645} teacher side (stop-token audit 2026-08-19: same event, disjoint tokens); eos_dl_at_stop is the live receipt of the -25-nat terminal punishment in the PG base",
}

ARMS = [  # ledger order: axis order from the plan
    "a1_gkd_mix0.5", "a3_offpolicy", "a4_dagger_anneal", "a5_aggrevate",
    "a2_coldstart", "b1_skew_kl", "b2_forward_kl", "b3_eopd_gate",
    "b4_jsd", "b4_jsd_b0.1", "b4_jsd_b0.9", "b5_k2",
    "c1_lsm_topk32_renorm", "c1_direct", "c1_tailbucket",
    "c3_intersection", "c4_pi_tail_budget",
    "c2_quantile_budget", "c2_qb_fixed8", "c2_qb_perseq",
    "d1_tip", "d2_selectkd", "d3_teachability",
    "e1_pl_rank", "e1_pl_rank_a0", "e2_set_coverage", "e2_set_coverage_a0", "e3_zvalue",
    "f1_soft_log", "f2_hard_clip", "f3_power",
    "f2_clip2.3", "f4_posclip", "f5_tanh", "g1_verified_only",
    "g1_quota", "g2_fire_likelihood", "g4_failure_only", "g4_quota", "g5_rgopd_gate",
    "g6_seqmean",
    "h1_first_segment", "h2_last_segment", "h3_random_segment",
    "h4_random_scatter", "h5_gen100", "h6_gen_sched", "h7_gen512",
    "h8_gen2048", "h9_prune_adapt", "h10_task_subset",
    "vanilla_n8", "j1_kdrl",
    "n0_termfix", "n2_termcal",
    "n02_termfix_cal", "n0_f1_softlog", "n0_f2_clip10", "n0_f3_power", "n0_h2_lastseg",
    "n0_b1_skew", "n0_g1_verified", "n0_g4_failure", "n0_g6_seqmean",
    "n0_f2_clip2.3", "n0_f4_posclip", "n0_f5_tanh", "n0_b5_k2", "n0_h1_firstseg", "n0_h3_randseg",
    "n0_h4_randscatter", "n0_g5_rgopd", "n0_g2_fire",
    "n0_d1_tip", "n0_d2_selectkd", "n0_d3_teachability",
]
# Arms judged against a non-vanilla base (self-contained mini-cells). The base row
# itself still appears vs vanilla, which reads out the cell's boundary knob.
BASE_OVERRIDES = {"j1_kdrl": "vanilla_n8"}
TRANSFER = ("humanevalplus", "mbppplus", "ifeval")  # amc23 is in-domain (suite); audit S3


# The ledger's sampling era per benchmark: MATH500's registered metric is greedy
# pass@1; the small benchmarks are the avg@k triple. Filenames carry no sampling
# params, and four eras have already written the same names into one directory --
# without this predicate, mtime silently decides which metric enters the paper
# (audit 2026-08-07 F1: one touch of a stale file moved suite_acc by 0.17).
LEDGER_TEMPERATURE = {"math500": 0.0, "minerva": 0.0}   # default for others: 0.7


def load_correct(evals, run_id, bench, step):
    """problem_id -> pass rate, from the newest artifact of the RIGHT sampling era."""
    import pandas as pd

    want_t = LEDGER_TEMPERATURE.get(bench.split("_")[0], 0.7)
    pats = [os.path.join(evals, f"{run_id}__{bench}__step{step}__*.parquet"),
            os.path.join(evals, f"{run_id}__{bench}__step{step}.parquet")]
    hits = [h for p in pats for h in glob.glob(p)]
    for h in sorted(hits, key=os.path.getmtime, reverse=True):
        df = pd.read_parquet(h)
        t = float(df["temperature"].iloc[0]) if "temperature" in df else want_t
        if abs(t - want_t) < 1e-6:
            return df.groupby("problem_id")["correct"].mean()
    return None


def mcnemar_exact(b, c):
    """Two-sided exact binomial on the discordant pairs: X ~ Bin(b+c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def eval_cmd(run_id, bench, step):
    return (f"python scripts/eval_offline.py --model <ckpt_of_{run_id}>/global_step_{step} "
            f"--benchmarks {bench} --run-id {run_id} --step {step}")


def main():
    p = argparse.ArgumentParser()
    # Same chain as eval_offline.py's --out-dir, in the same order -- reader and
    # writer resolving differently is how artifacts get stranded.
    p.add_argument("--evals", default=os.environ.get("SIMOPD_EVAL_ROOT",
                   os.environ.get("SIMOPD_EVALS",
                   os.path.expanduser("~/data/simopd_evals"))))
    p.add_argument("--bench", default="math500")
    p.add_argument("--step", type=int, default=250)
    p.add_argument("--seed", type=int, default=0, help="arm seed compared against vanilla_s<seed>")
    p.add_argument("--write", help="also write the ledger as markdown here")
    a = p.parse_args()
    out = []

    def emit(line=""):
        print(line)
        out.append(line)

    emit(f"# Verdict ledger -- {a.bench} @ step {a.step} (evals: {a.evals})")

    # ---- noise floor ---------------------------------------------------------
    vans = {s: load_correct(a.evals, f"{BASE}_s{s}", a.bench, a.step) for s in SEEDS}
    have = [s for s in SEEDS if vans[s] is not None]
    floor = None
    emit("\n## Noise floor (vanilla seeds, range of accuracy)")
    for s in SEEDS:
        if vans[s] is None:
            emit(f"- seed {s}: MISSING   ->  {eval_cmd(f'{BASE}_s{s}', a.bench, a.step)}")
        else:
            emit(f"- seed {s}: acc {vans[s].mean():.4f}  ({len(vans[s])} problems)")
    if len(have) == len(SEEDS):
        accs = [vans[s].mean() for s in SEEDS]
        floor = max(accs) - min(accs)
        emit(f"- **floor = {floor:.4f}** (range across {len(SEEDS)} seeds; |Δ| below this is a TIE)")
    else:
        emit(f"- floor: AWAIT-FLOOR ({len(have)}/{len(SEEDS)} seeds evaluated)")

    base = vans.get(a.seed)

    # ---- per-arm rows --------------------------------------------------------
    emit(f"\n## Arms vs {BASE}_s{a.seed}")
    emit("| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |")
    emit("|---|---|---|---|---|---|")
    for arm in ARMS:
        run_id = f"{arm}_s{a.seed}"
        cav = f" ⚠{CAVEATS[arm]}" if arm in CAVEATS else ""
        cur = load_correct(a.evals, run_id, a.bench, a.step)
        if cur is None:
            emit(f"| {arm} | — | — | — | — | PENDING: `{eval_cmd(run_id, a.bench, a.step)}`{cav} |")
            continue
        eff_base = base
        if arm in BASE_OVERRIDES:
            eff_base = load_correct(a.evals, f"{BASE_OVERRIDES[arm]}_s{a.seed}", a.bench, a.step)
        if eff_base is None:
            emit(f"| {arm} | — | — | — | — | PENDING baseline artifact{cav} |")
            continue
        common = eff_base.index.intersection(cur.index)
        vb, vc = eff_base.loc[common], cur.loc[common]
        delta = float(vc.mean() - vb.mean())
        # Paired binaries: avg@k rates collapse to right/wrong at 0.5 for pairing, the
        # same convention d6_matrix uses; pass@1 artifacts are already 0/1.
        b_ = int(((vb > 0.5) & (vc <= 0.5)).sum())
        c_ = int(((vb <= 0.5) & (vc > 0.5)).sum())
        pval = mcnemar_exact(b_, c_)
        if len(common) == 0:
            verdict = "NO-OVERLAP (no shared problem ids -- check id conventions)"
        elif floor is None:
            # Floor first: promoting before the noise floor exists publishes rows
            # that flip to TIE when the third seed lands (audit 2026-08-07 F2).
            verdict = f"AWAIT-FLOOR (p={'<' if pval < ALPHA else '≥'}{ALPHA})"
        elif abs(delta) < floor:
            verdict = "TIE (|Δ| < floor)"
        elif pval < ALPHA:
            verdict = "**PROMOTE**" if delta > 0 else "**WORSE**"
        else:
            verdict = "inconclusive"
        emit(f"| {arm} | {delta:+.4f} | {b_} | {c_} | {pval:.4f} | {verdict}{cav} |")

    # ---- side-effect panel: transfer deltas where artifacts exist ------------
    emit(f"\n## Side-effect panel (transfer deltas vs {BASE}_s{a.seed}; informational until per-bench floors exist)")
    any_transfer = False
    for bench in TRANSFER:
        vb = load_correct(a.evals, f"{BASE}_s{a.seed}", bench, a.step)
        if vb is None:
            continue
        for arm in ARMS:
            vc = load_correct(a.evals, f"{arm}_s{a.seed}", bench, a.step)
            if vc is None:
                continue
            any_transfer = True
            emit(f"- {arm} on {bench}: {vc.mean():.4f} vs {vb.mean():.4f} (Δ {vc.mean()-vb.mean():+.4f})")
    if not any_transfer:
        emit("- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)")

    if a.write:
        with open(a.write, "w") as f:
            f.write("\n".join(out) + "\n")
        print(f"\nwrote {a.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
