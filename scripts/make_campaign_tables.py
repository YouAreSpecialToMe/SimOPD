#!/usr/bin/env python3
"""Regenerate the report's §3 suite grid and §3.1 step-250 breakdown from
post_eval_cells.csv, for EVERY arm that has cells.

    python scripts/make_campaign_tables.py            # rewrite docs/campaign_16k_report.md in place
    python scripts/make_campaign_tables.py --check    # print the tables, touch nothing

Both tables used to be hand-maintained ("Regenerated 2026-08-13 from the 489
complete checkpoint-evaluations"), so they went stale on every drain tick and
carried only the 16k cohort -- supplement arms such as h5_gen100 were absent
from the headline scoreboard even after their 149 cells landed. The cells table
is the canonical source (scripts/extract_post_eval.py writes it); everything
here is derived, so the numbers cannot drift from it again.

Composite口径 is eval_suite.py's: equal-macro mean of AIME(24+25 pooled), AMC23,
Minerva, MATH500. A (step, seed) counts as complete only when all five
benchmarks are present -- partial cells are excluded, never averaged over a
smaller macro set, because a missing AIME row would silently inflate the mean.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE, TEACHER = 0.1453, 0.6482          # §3 header anchors; GRR = gap recovered
STEPS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
MACRO = [("aime", ["aime24", "aime25"]), ("amc23", ["amc23"]),
         ("minerva", ["minerva"]), ("math500", ["math500"])]
BENCHES = ["aime24", "aime25", "amc23", "minerva", "math500"]


def composites(cells):
    """{(arm, step, seed): {macro: v, 'composite': v}} over complete cells only."""
    out = {}
    for (arm, step, seed), d in cells.groupby(["arm", "step", "seed"]):
        have = dict(zip(d.bench, d.avg_at_k))
        if any(b not in have for b in BENCHES):
            continue
        macro = {name: float(np.mean([have[b] for b in bs])) for name, bs in MACRO}
        macro["composite"] = float(np.mean(list(macro.values())))
        out[(arm, int(step), int(seed))] = macro
    return out


def cell(vals, n_seeds=3):
    if not vals:
        return "–"
    m = np.mean(vals)
    if len(vals) == 1:
        return f"{m:.3f}·1"
    s = np.std(vals, ddof=1)
    return f"{m:.3f}±{s:.3f}" + (f"·{len(vals)}" if len(vals) < n_seeds else "")


def grid_table(comp, arms):
    rows, order = {}, []
    for arm in arms:
        per_step = {st: [v["composite"] for (a, s, _), v in comp.items() if a == arm and s == st]
                    for st in STEPS}
        done = sum(len(v) for v in per_step.values())
        full = [np.std(v, ddof=1) for v in per_step.values() if len(v) == 3]
        final = per_step[250]
        grr = ((np.mean(final) - BASE) / (TEACHER - BASE) * 100) if final else None
        rows[arm] = dict(per_step=per_step, done=done,
                         sbar=float(np.mean(full)) if full else None, grr=grr,
                         key=np.mean(final) if final else -1)
        order.append(arm)
    order.sort(key=lambda a: rows[a]["key"], reverse=True)
    out = ["| method | " + " | ".join(f"s{s}" for s in STEPS) + " | σ̄ | GRR | ckpts done |",
           "|---" * 14 + "|"]
    for arm in order:
        r = rows[arm]
        cells_ = " | ".join(cell(r["per_step"][s]) for s in STEPS)
        sbar = f"{r['sbar']:.3f}" if r["sbar"] is not None else "–"
        grr = f"{r['grr']:.1f}%" if r["grr"] is not None else "–"
        out.append(f"| {arm} | {cells_} | {sbar} | {grr} | {r['done']}/30 |")
    return "\n".join(out)


def step250_table(comp, arms):
    out = ["| method | seeds | composite | Δ vs base | AIME24+25 | AMC23 | Minerva | MATH500 |",
           "|---|---|---|---|---|---|---|---|"]
    ranked = []
    for arm in arms:
        seeds = sorted(s for (a, st, s) in comp if a == arm and st == 250)
        if seeds:
            ranked.append((np.mean([comp[(arm, 250, s)]["composite"] for s in seeds]), arm, seeds))
    for mean_c, arm, seeds in sorted(ranked, reverse=True):
        v = {k: [comp[(arm, 250, s)][k] for s in seeds]
             for k in ("composite", "aime", "amc23", "minerva", "math500")}
        bold = "**" if mean_c >= 0.34 else ""
        out.append(f"| {bold}{arm}{bold} | {len(seeds)} | **{cell(v['composite'])}** | "
                   f"{mean_c - BASE:+.3f} | {cell(v['aime'])} | {cell(v['amc23'])} | "
                   f"{cell(v['minerva'])} | {cell(v['math500'])} |")
        for i, s in enumerate(seeds):
            c = comp[(arm, 250, s)]
            out.append(f"| ↳ s{s} | | {c['composite']:.4f} | | {c['aime']:.4f} | "
                       f"{c['amc23']:.4f} | {c['minerva']:.4f} | {c['math500']:.4f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(ROOT, "docs/data/post_eval_cells.csv"))
    ap.add_argument("--report", default=os.path.join(ROOT, "docs/campaign_16k_report.md"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    cells = pd.read_csv(a.cells)
    cells = cells[cells.run.str.contains("16k", na=False)]
    comp = composites(cells)
    arms = sorted({k[0] for k in comp})
    n_done = len(comp)

    grid = grid_table(comp, arms)
    s250 = step250_table(comp, arms)
    note = (f"_(Auto-generated by `scripts/make_campaign_tables.py` from the {n_done} complete "
            f"checkpoint-evaluations in `docs/data/post_eval_cells.csv`; every arm with cells is "
            f"listed, supplement arms included. Regenerate after each drain tick.)_")

    if a.check:
        print(note + "\n\n" + grid + "\n\n### 3.1\n\n" + s250)
        return

    md = open(a.report).read()
    start = md.index("| method | s25 |")
    end = md.index("### 3.1")
    tail = md.index("## 4.", end)
    head3_1 = md[end:md.index("|", end)]          # keep the 3.1 heading line verbatim
    md = md[:start] + grid + "\n\n" + head3_1 + s250 + "\n\n" + md[tail:]
    md = re.sub(r"_\(Regenerated 2026-[^_]*_", note, md, count=1)
    open(a.report, "w").write(md)
    print(f"rewrote §3 grid ({len(arms)} arms) and §3.1 from {n_done} complete cells")


if __name__ == "__main__":
    main()
