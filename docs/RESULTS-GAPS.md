# What the 16k campaign still owes — missing measurements and cells

_Status 2026-08-11, against `docs/campaign_16k_report.md` (04:21 regeneration:
72/87 rows at 250, suite 195/750). This doc is the gap list: what must still be
recorded, harvested, or run before the analysis phase can write its verdicts.
It changes no run-defining file; everything here is either log harvesting or
new manifest rows that go through the normal claim/repin path._

## 1. The compute ledger (the one real hole in our records)

Every curve in the report has **steps** on the x-axis. But arms differ wildly in
response length — c1 finishes a seed in 7.6 wall-h where vanilla takes 59.3
(8×) — so at equal steps the arms have generated, scored and trained on very
different token counts. Equal steps ≠ equal compute ≠ equal data. And because
OPD is on-policy, **data and compute are the same axis** (every training token
must be generated and teacher-scored at cost), so acc-vs-tokens is
simultaneously the data-efficiency and the compute-efficiency curve.

This matters most on the C axis: is c2's table-topping 0.684 partly *bought*
with longer generations, and is c1's 0.524 partly *excused* by an 8× cheaper
bill? Whether "adaptive budget allocation wins" survives compute normalization
decides how that headline sentence is written.

### 1.1 Why "FLOPs" reduces to token counters

Per response token the model FLOPs are one constant across all arms (loss-kernel
differences — top-8 vs top-32 gathers, quantile thresholds — are <0.1% noise):

```
FLOPs/token ≈ 2·N_T (teacher scoring) + 2·N_S (student decode) + 6·N_S (student update)
            ≈ 8.0 + 3.4 + 10.2  ≈ 21.6 GFLOPs/token      (N_S=1.7B, N_T=4B)
```

(prompt-prefill terms are common across arms and drop out of comparisons).
So analytic FLOPs is a linear map of the counters — **the counters are the
measurement**. Nothing needs re-running: the lane logs already carry
`response_length/mean`, `response_length/clip_ratio`, s/it, entropy and the verl
timing keys per step (`scripts/watch.py:200` parses them today). What is missing
is the harvest, not the instrumentation.

### 1.2 Ledger columns to harvest per run

| column | source | serves |
|---|---|---|
| cumulative generated tokens (per-step prefix sum of mean length × batch) | lane logs | acc-vs-token / acc-vs-FLOPs curves |
| cumulative **supervised** tokens | retention-scaled for D/H arms (D: `SIMOPD_D_RETENTION=0.5`; h1/h2/h3: K=100 window; SelecTKD: realized accept rate), full response otherwise | the H-axis budget claim is only readable on this axis — per supervised token h1 may top the table |
| teacher-scored tokens | = response tokens | teacher-side share of the bill |
| phase timing split (rollout / scoring / update) | verl timing keys in the same logs | quantifies the ~50% duty cycle; locates each arm's bottleneck |
| peak VRAM per arm | scattered OOM-war notes → one systematic column | the 16k-feasibility story; why the 4-card lanes exist |
| teacher payload width k (32 vs 64) | protocol constant | c2's bandwidth/memory cost note |
| analytic FLOPs | constant × counters | the paper's reproducible compute axis |

### 1.3 Deliverables built from the ledger

1. **Compute-normalized twin of report §1** — same in-loop table/curves with
   x = cumulative generated tokens (log axis). This is the correct coordinate
   system for every C-axis comparison.
2. **Compute-to-target table** — tokens and FLOPs to reach a threshold
   (e.g. 0.55 in-loop): the reviewer-friendly efficiency summary.
3. **Matched-budget readings** — interpolate every arm at a common token budget
   (e.g. vanilla's total) as the replacement for equal-step comparison.

### 1.4 Accounting discipline (proposed for the protocol)

Report **analytic FLOPs** (reproducible, derived from counters) and **measured
GPU·h** (hardware-bound, restart-inflated) side by side and never conflate
them. The report already proves the need: restart overhead, not the objective,
dominates measured cost variance (b3's 1,220 GPU·h is mostly OOM war, not
method).

Implementation: one fleet-side script (`scripts/compute_ledger.py`, to be
written) walking the same log globs as `campaign_table.py`. No new
instrumentation, no re-runs, no run-defining changes.

## 2. Two smaller harvests while we're in the logs

- **Grad-norm curves** for the collapse analysis — is the 150–200-step collapse
  preceded by a gradient spike, or silent? Same logs, same harvest pass.
- **Peak VRAM per arm** as a first-class column (see ledger above) instead of
  incident-note archaeology.

## 3. Missing experiment cells (backlog by axis)

Everything below goes through the normal manifest/claim path; sizes are
extrapolated from the arm's own measured wall-h.

| # | cell | why it exists | size |
|---|---|---|---|
| A1 | **a2 validity recipe cell**: rebuild stage 1 with `gen_coldstart_data.py --filter validity` + the pinned SFT hyperparameters (1 epoch, lr 1e-5, cosine, warmup 0.05), then 3 OPD seeds | the three flying a2 rows predate audit r6 and are the `--filter verifier` **ablation** cell (rename to `*_ablation` after they bank 250); the recipe cell is what the Rethinking citation may claim | ~400 GPU·h + 30 suite cells |
| A2 | a2 suite sweep (currently **0/30**) + entropy/length panel comparison against the collapse-point state of the sampled family | decides "collapse-immune" vs "collapse-delayed" for the only sampled arm still climbing | eval only |
| A3 | a1/a3 unlock: `gen_offpolicy` precompute (all train prompts, full-prefix keys) + 3-step rehearsal + **register opposing predictions in the ledger before launch** (direction-follows-sampler ⇒ a3 should show SFT-like dynamics) | the λ dose line {a3, a1, vanilla} under the protocol estimator — a literature-absent object | precompute + 6 runs × 2 cards |
| B1 | β ladder `b4_jsd_b0.1` / `b4_jsd_b0.9` — **already in the manifest as wave 9** (queues behind all existing waves; 2-card rows, keep off the quad boxes) | H1 (form) vs H2 (direction) opposing predictions, registered in `arm-provenance-r4.md` | ~358 GPU·h + 60 suite cells |
| C1 | **c1-direct paper-form ablation** (`USE_POLICY_GRADIENT=False`, 3 seeds) | c1 is the only top-k arm on the PG branch; its last-place 0.524 conflates "fixed-k is bad" with "PG drags divergence-valued losses". This cell decides whether "the literature default is the worst C cell" is writable. Highest analysis priority per GPU·h; opposing predictions P-PG / P-K registered in the ledger 2026-08-11 (stable entropy vs reproduced late drift) | ~50 GPU·h + 30 suite cells |
| C2 | c1 renorm-vs-tailbucket internal ablation (pre-registered with the arm) | second constructive test of the headline theorem: does explicitly bucketing the tail mass close the gap? | ~50 GPU·h |
| C3 | **full-vocab upper bound, one run** — feasible now that the streaming-lse rewrite removed the [T,V] materialization | the width axis needs its upper endpoint; how far c2's 0.684 sits from it completes the "average budget 8 suffices" story | 1 run, 4-card lane |
| C4 | **c2 pinning-granularity ladder** `c2_qb_fixed8` / `c2_qb_perseq` — **already in the manifest as wave 10** (`SIMOPD_QB_SCOPE`, same kernel/payload/branch, keep rule the only moving part; CPU battery 6/6) | c2's "batch-level" tau lives on the packed micro-batch, which collapses to a single sequence once 16k lengths saturate the 17408 packing cap — the shipped c2 rows drift from cross-sequence to de-facto per-sequence pinning (recorded deviation, 2026-08-11). perseq-vs-c2 decides whether that drift was harmless (P1/P2 registered in the ledger); fixed8 doubles as the missing matched-budget fixed control (c2-vs-c1 conflates budget size with allocation policy) | ~600 GPU·h + 60 suite cells |
| E1 | **E-ladder purity cells** `e1_pl_rank_a0` / `e2_set_coverage_a0` — **already in the manifest as wave 11** (anchor coef 0; the coefficient was registered as the internal ablation from day one) | e1/e2 as shipped are 'X + 0.1·values' mixtures, so no positive ladder claim is legal; a0 restores the pure order rung (P-order vs P-anchor registered) and e2@0 doubles as the roster's near-placebo control (P-frozen vs P-drift) | ~140 + ≤280 GPU·h |
| DH1 | **the D axis's missing random control**: `random-scatter @50%` and `@5%` — same mask-and-rescale kernel, criterion = random, so no KEEP_SAMPLED payload and **2-card lanes** (the d-family pays 4-card for its criteria) | (a) criterion vs random at matched budget — the selector literature's actual claim; (b) scatter-vs-window, deconfounding the position dose line's middle step; (c) if random matches the criteria, the d-family's entire 4-card premium bought nothing (`MECHANISMS.md` M-III) | 2 cells × 3 seeds, ~500 GPU·h |
| N1 | **M-I causal keystone (amendment candidate)**: `vanilla + truncation-zeroing reward term`, nothing else — j1 fixes termination but moves three knobs; this cell moves one (`MECHANISMS.md` M-I) | closes the EOS-starvation loop causally | ~350 GPU·h |
| S | suite completion: 555 checkpoint-evals remaining + the teacher suite row (eval queued) | every in-loop reading in this doc is a curve reading, not a verdict, until per-bench paired tests run on the full sweep | eval only |

### Priority order

C1 first (cheapest decision-per-GPU·h on the roster), then A1 (unblocks the A
axis entirely), B1/C4 run themselves via waves 9/10, C2/C3 opportunistic on
freed lanes, A3 after its precompute lands. The ledger harvest (§1) is
independent of all of these and can run today on any machine that sees the
lane logs.

## 4. Analysis tasks (log harvest only, no new runs)

| # | task | what it settles |
|---|---|---|
| M1 | **Compression dose-response formalization**: measured effective \|signal\| (p99 from the `_signal_quantiles` panels) vs lock step across {vanilla, f2, f1, b1, f3} — five arms, same support/branch/budget, magnitude transform the only knob; nominal bounds order ∞/10/log/2.3/1, locks order 122/198/208/247/never | whether unbounded sampled-token magnitude *is* the length-runaway driver (falsifiable: measured magnitudes must reproduce the nominal ordering); registered in the ledger 2026-08-11; **extended to the top-k family**: the only two capped top-k arms (e2, b2) are exactly the unbounded losses — prediction: their measured p99 is heavy-tailed vs the renorm-KL family |
| M2 | b3_gate open-rate inside the 50–100 transition window | the last nail in b3's degenerate-fixed-point account |
| M3 | first-boxed / any-boxed rescore of the greedy diagnostics (textdump infra exists) — the *capability curve* next to the *delivery curve* (protocol scoring unchanged) | how much of every reported fall is the grader, not the model (N2 in `MECHANISMS.md`) |
| M4 | deep-U mechanism: `gate_keep_frac` (g1/g4) + h2 panels — is the crater a gated-data famine that heals as accuracy rises? | the one entropy-dynamics regime still unexplained (N3) |
| M5 | the never-taken panel readings: overlap **mass** vs the "97–99%" claim, `pi_tail` K-scan, D-axis shadow Jaccard (redundancy prediction #4) | M-III's instruments are installed; these are the readings (N4) |
