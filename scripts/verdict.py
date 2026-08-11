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
}

ARMS = [  # ledger order: axis order from the plan
    "a1_gkd_mix0.5", "a3_offpolicy", "a2_coldstart", "b1_skew_kl", "b2_forward_kl", "b3_eopd_gate",
    "b4_jsd", "b4_jsd_b0.1", "b4_jsd_b0.9", "b5_k2",
    "c1_lsm_topk32_renorm", "c3_intersection", "c4_pi_tail_budget",
    "c2_quantile_budget", "c2_qb_fixed8", "c2_qb_perseq",
    "d1_tip", "d2_selectkd", "d3_teachability",
    "e1_pl_rank", "e1_pl_rank_a0", "e2_set_coverage", "e2_set_coverage_a0", "e3_zvalue",
    "f1_soft_log", "f2_hard_clip", "f3_power", "g1_verified_only",
    "g1_quota", "g2_fire_likelihood", "g4_failure_only", "g4_quota", "g5_rgopd_gate",
    "h1_first_segment", "h2_last_segment", "h3_random_segment",
    "vanilla_n8", "j1_kdrl",
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
