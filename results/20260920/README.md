# 2026-09 roster retraining -- results data set (exported 2026-09-20 11:49 PT (2026-09-20 18:49 UTC), FINAL: training and offline evaluation complete)

Same kind of directory as `results/20260909/`. The training tables and `post_eval_*.csv` are **data movement**: every column is a number found on disk or computed from
such numbers by a formula stated below. `offline_results.tsv` additionally applies the comparison rule the branch registers (section Alignment: matched vanilla `vanilla_corr`, fixed
step) and `verdict_ledger.md` is the output of the branch's own `scripts/verdict.py`; nothing here is a method of my own, no measured number is altered, no arm is labelled.

What is new relative to `results/20260909/`: (1) the training tables are re-extracted after the training window of 2026-09-18 16:30 PT to 2026-09-20 (PT) on the 4-node
cluster; (2) the **offline evaluation suite now exists** (gap 7 of the 0909 README: "no offline evaluation yet") -- `post_eval_cells.csv`,
`post_eval_bystep.csv` in the schema of `docs/data/post_eval_*.csv`, and `offline_results.tsv`, which lines the results up with the paper protocol (section Alignment).

- Training tables: the repo's own `results/20260909/extract.py`, unchanged, run once per node (the run directories are sharded over four nodes,
  `arm_nodes.tsv`) and concatenated in roster order (`export_train_tables.py`, `export_merge.py`).
- Offline tables: the repo's own `scripts/extract_post_eval.py --roster ALL`, unchanged, over the union of the four nodes' `evals/*.parquet`.
- Sources: `<arm>_s0/metrics/launch_*.jsonl`, `val_gen/*.jsonl`, `global_step_*/`, `traj/`, `evals/*.parquet`, `configs/retrain_roster.tsv`, `configs/arms.yaml`.

---

## Files

| file | rows | content |
|---|---|---|
| `arms.tsv` | 24 | axis, target steps, roster priority, env delta vs the carrier, description, roster note (arms with a run directory on this cluster) |
| `arm_nodes.tsv` | 24 | which node holds each arm's run directory |
| `progress.tsv` | 24 | max checkpoint step, number of checkpoints, max logged step, logged steps, reached target, number of launch files |
| `checkpoints.tsv` | 113 | one row per checkpoint (arm, global_step) |
| `val_accuracy_wide.tsv` | 24 | arm x step in-loop validation accuracy (steps that are multiples of 25) |
| `metrics_summary.tsv` | 4652 | arm x step long table of the 24 common metrics |
| `metrics_full/<arm>.tsv` | 24 files | every logged metric of the arm per step |
| `val_gen_summary.tsv` | 171 | one row per in-loop validation (items, acc mean, reward mean, output chars) |
| `archive_inventory.tsv` | 24 | file counts of the analysis archive layer |
| `teacher_columns.tsv` | 184 | row count and teacher-column means of each `summary_<step>.parquet` |
| `post_eval_cells.csv` | 537 | OFFLINE: one row per (arm, seed, step, benchmark) -- newest artifact; acc, pass@k, lengths, truncation, finish reasons, stop set (+ two added columns: accuracy over the non-truncated samples, share of wrong samples that are truncated) |
| `post_eval_bystep.csv` | 57 | OFFLINE: one row per (arm, step) with a COMPLETE five-benchmark suite: composite and its four components |
| `offline_results.tsv` |  | OFFLINE, THE RESULT TABLE: one row per (arm, step) with a complete suite -- five scores and composite (x100, `scripts/eval_suite.py` definition), truncation, the composite of the campaign vanilla at the SAME step and the arithmetic difference, the registered base of `scripts/verdict.py`, the gap vs the fixed 32.8173 of the paper draft, `is_best_step`, and the paper row this arm corresponds to (table, label, composite, selected step, seeds) |
| `verdict_ledger.md` |  | OFFLINE: output of the branch's own `scripts/verdict.py --base vanilla_corr` (exact McNemar on per-problem paired results, noise floor from vanilla seeds) for the T=0.7 benchmarks at steps 100/200/250; rows without artifacts (PENDING/MISSING) are filtered out, nothing else is touched |
| `offline_grid_status.tsv` | 105 | OFFLINE: every checkpoint on the light grid, benches required / finished / missing at export time |

Per-item generations (in-loop `val_gen/<step>.jsonl`, offline `evals/*.parquet` with full response text, `traj/`) are NOT copied here; they stay in the run directories.

---

## Definitions

Training-side columns, the three step notions, `env_delta_vs_carrier`, the shared training configuration and the "later launch file overrides an
earlier one for the same step" rule are exactly those of `results/20260909/README.md` (same script).

### Offline suite (handoff manual 8.1, `scripts/eval_suite.py`)

```
aime24 / aime25   avg@8   30 problems each     temperature 0.7, top_p 0.95, 32768-token budget
amc23             avg@8   40 problems
minerva           avg@3  272 problems
math500           avg@3  500 problems
avg@k     = mean over problems of the per-problem mean of `correct` over its k samples
composite = 0.25 * (AIME24+AIME25 pooled by problem) + 0.25 * AMC23 + 0.25 * Minerva + 0.25 * MATH500, complete five-benchmark suites only
grid      = `eval_refill_exp.py --grid light`: steps 100/200/250 all five benchmarks; steps 25/50/150 amc23+minerva+math500 (no composite); other steps not evaluated
```
`k_small_benches` in `offline_results.tsv` is 8 (handoff manual 8.1: every artifact of this round is avg@8) or 32 (paper protocol); a cell uses avg@32 only when all three
small benchmarks have it, sample counts are never mixed inside one estimate. All runs are seed 0.

### Alignment with the paper (rules applied in `offline_results.tsv`; none changes a measured number)

1. **Matched control, fixed step.** Every `_corr` / `_n0` arm is built on the carrier `vanilla_corr`, which scores 32.24@100, 34.25@200, 34.82@250 here against the fixed
   reference 32.82 of the paper draft (original run family). A gap against 32.82 therefore mixes the method with the base. `diff_vs_vanilla` compares each arm with the campaign vanilla
   at the SAME step (`vanilla_corr`; `vanilla_te`, registered to reproduce it, where `vanilla_corr` has no complete suite). This is the rule the branch already registers
   (handoff manual 8.4, `scripts/verdict.py`: baseline `vanilla_corr`, one fixed step, paired per-problem comparison, tie below the noise floor). `registered_base` lists the
   `BASE_OVERRIDES` special cases; `gap_vs_paper_ref_32.8173` is kept only as the historical convention.
2. **Test and noise floor are the branch's.** `verdict_ledger.md` is the output of `scripts/verdict.py --base vanilla_corr` at a fixed step: per-benchmark accuracy difference, exact McNemar on the
   per-problem paired results, and the noise floor = range over vanilla seeds 0/1/2. Only seed 0 exists in this roster, so every verdict reads AWAIT-FLOOR; the ledger registers MATH500 and
   Minerva as greedy (T=0) artifacts, which this campaign does not produce, so it covers amc23 / aime24 / aime25. `diff_vs_vanilla` in `offline_results.tsv` is plain arithmetic on composites.
3. **Best checkpoint vs fixed step.** `is_best_step` marks the paper-style selection (highest composite among the complete suites of the arm); the fixed-step reading is the row at step 200.
4. **Truncation is part of the result.** `trunc_pct` and `amc23_trunc_pct_by_step` stand next to every composite; a checkpoint whose AMC23 truncation is >= 90% has stopped terminating.

---

## Summary numbers

| item | value |
|---|---|
| roster arms | 29 |
| arms with a run directory on this cluster (training tables) | 24 |
| reached target (`global_step_<target>` exists) | **19** |
| checkpoint steps / target steps (these 24 arms) | **4600 / 5000** |
| checkpoints | 113 |
| in-loop validations (`val_gen_summary.tsv` rows) | 171 |
| offline benchmark artifacts (`post_eval_cells.csv` rows) | 537 |
| complete five-benchmark suites (`post_eval_bystep.csv` rows) | **57** over 29 arms |
| light-grid cells that exist as checkpoints here / fully evaluated | 105 / **105** (anchor 45 / 45, light 60 / 60) |

Reached target: `vanilla_te`, `c2_qb_fixed8_corr`, `c2_qb_perseq_corr`, `n2_corr`, `b1_skew_kl_corr`, `b2_forward_kl_corr`, `c3_intersection_corr`, `d2_selectkd_corr`, `d3_teachability_corr`, `f2_hard_clip_corr`, `f3_power_corr`, `g1_verified_only_corr`, `h1_first_segment_corr`, `h3_random_segment_corr`, `h4_random_scatter_corr`, `a1_gkd_mix0.5_n0`, `a3_offpolicy_n0`, `a4_dagger_anneal_n0`, `a5_aggrevate_n0`.

### In-loop validation accuracy (`val_accuracy_wide.tsv`; blank = no record at that step)

```
arm                            25     50     75    100    125    150    175    200    225    250
vanilla_te                  0.630  0.610  0.610  0.628  0.604  0.630  0.654  0.666  0.592  0.654
c2_qb_fixed8_corr           0.600  0.630  0.570  0.510  0.472  0.496  0.522                     
c2_qb_perseq_corr           0.612  0.600  0.618  0.602  0.642  0.612  0.636  0.662              
c5_union_rkl                0.638  0.620  0.598  0.452  0.474  0.450                            
c5_union_fkl                0.550  0.570  0.514  0.480  0.462  0.472  0.394  0.440              
n2_corr                     0.538  0.618  0.592         0.592  0.544         0.654  0.656  0.642
b1_skew_kl_corr             0.608  0.622         0.608  0.606  0.618  0.650  0.630              
b2_forward_kl_corr          0.548  0.570  0.506  0.454  0.500  0.452  0.504  0.556              
c3_intersection_corr        0.528  0.568  0.598  0.578  0.562  0.554         0.570              
d2_selectkd_corr            0.614  0.634  0.598  0.626                0.666  0.658              
d3_teachability_corr        0.588  0.634  0.564  0.548  0.540         0.484                     
e2_set_coverage_a0_corr     0.596  0.584  0.572  0.568  0.556                                   
f2_hard_clip_corr           0.610  0.610  0.630  0.632  0.642         0.648  0.674              
f3_power_corr               0.548  0.564  0.554  0.616         0.544  0.558                     
g6_seqmean_corr             0.596  0.636  0.548  0.598  0.618                                   
g1_verified_only_corr       0.350  0.566  0.660  0.632  0.612  0.572  0.544  0.528              
h1_first_segment_corr       0.592  0.648  0.626  0.592  0.630  0.638  0.604  0.616              
h2_last_segment_corr        0.512  0.454  0.484  0.468                                          
h3_random_segment_corr      0.568  0.636  0.620  0.606  0.602  0.630  0.588  0.620              
h4_random_scatter_corr      0.604  0.640  0.618  0.618  0.626  0.576  0.590  0.574              
a1_gkd_mix0.5_n0            0.514  0.584  0.566  0.468  0.432  0.522  0.532  0.536              
a3_offpolicy_n0             0.506  0.572  0.540         0.486  0.452  0.428  0.438              
a4_dagger_anneal_n0         0.518  0.610  0.516         0.462  0.450  0.514  0.558              
a5_aggrevate_n0             0.520  0.586  0.578  0.488  0.488  0.484  0.494  0.518              
```

n = 500, so the binomial standard error of one point is about 0.021-0.022.

### Offline results (`offline_results.tsv`; x100; diff = composite minus the campaign vanilla composite at the same step)

```
arm                        step   k  composite    diff  trunc%  paper row (composite@step, seeds)
vanilla_te                  250   8      34.92   +0.10    13.3
vanilla_corr                250   8      34.82            11.3  Vanilla (original run family) (32.82@75, 3 seeds)
n2_corr                     250   8      34.15   -0.67    17.7

c2_quantile_budget_corr     200   8      35.49   +1.24    14.6  Quantile: mean budget 8 (33.56@75, 1 seeds)
c4_state                    200   8      34.86   +0.61     6.6  Student-tail + state gate (33.42@100, 1 seeds)
b1_skew_kl_corr             200   8      34.54   +0.29     5.1  Skew (33.81@200, 3 seeds)
n2_corr                     200   8      34.45   +0.20    22.7
vanilla_corr                200   8      34.25             6.0  Vanilla (original run family) (32.82@75, 3 seeds)
vanilla_te                  200   8      34.23   -0.02     6.7
d2_selectkd_corr            200   8      34.19   -0.06     7.9  SelecTKD (33.09@150, 3 seeds)
c2_qb_perseq_corr           200   8      34.16   -0.09    15.8  Quantile: sequence scope (27.99@25, 3 seeds)
f2_hard_clip_corr           200   8      33.93   -0.31     6.3  Hard clip (34.34@175, 3 seeds)
c4_pi_tail_budget_corr      200   8      33.93   -0.32    20.4  Student-tail: 95% mass (35.04@175, 1 seeds)
f3_power_corr               200   8      33.83   -0.42     6.4  Power (33.79@125, 3 seeds)
c4_hq                       200   8      33.77   -0.47     5.6  Student-tail + HQ gate (33.56@75, 1 seeds)
b2_forward_kl_corr          200   8      33.73   -0.51    15.2  Forward (32.77@125, 1 seeds)
h3_random_segment_corr      200   8      33.62   -0.62    24.4  Window (32.71@250, 3 seeds)
c3_intersection_corr        200   8      33.46   -0.79    17.3  Intersection (34.52@250, 3 seeds)
c2_qb_fixed8_corr           200   8      33.20   -1.05    12.2  Fixed |S|=8 (23.83@25, 3 seeds)
h4_random_scatter_corr      200   8      32.02   -2.23    34.6  Scatter (30.09@50, 3 seeds)
d3_teachability_corr        200   8      31.41   -2.84    18.8  TA-OPD (29.88@100, 3 seeds)
a5_aggrevate_n0             200   8      31.20   -3.05    35.4
a4_dagger_anneal_n0         200   8      31.19   -3.06    28.4  Annealed mix (21.74@50, 1 seeds)
h1_first_segment_corr       200   8      30.87   -3.38    10.6  Prefix (30.32@250, 3 seeds)
a1_gkd_mix0.5_n0            200   8      29.55   -4.69    32.3  Fixed mix (24.7@50, 1 seeds)
c5_union_fkl                200   8      27.29   -6.96    98.4  FKL union (33.01@75, 1 seeds)
g1_verified_only_corr       200   8      24.90   -9.35    46.3  Correct (28.23@125, 3 seeds)
a3_offpolicy_n0             200   8      20.31  -13.94    59.5  Teacher (20.59@50, 1 seeds)

vanilla_te                  100   8      34.12   +1.88    14.6
f2_hard_clip_corr           100   8      33.82   +1.58    14.6  Hard clip (34.34@175, 3 seeds)
d2_selectkd_corr            100   8      33.80   +1.56    18.5  SelecTKD (33.09@150, 3 seeds)
c4_pi_tail_budget_corr      100   8      33.79   +1.55    10.9  Student-tail: 95% mass (35.04@175, 1 seeds)
c3_intersection_corr        100   8      33.44   +1.20    24.4  Intersection (34.52@250, 3 seeds)
c2_quantile_budget_corr     100   8      33.41   +1.17    14.0  Quantile: mean budget 8 (33.56@75, 1 seeds)
c4_state                    100   8      33.24   +1.00    10.6  Student-tail + state gate (33.42@100, 1 seeds)
c5_union_fkl                100   8      32.85   +0.61     7.7  FKL union (33.01@75, 1 seeds)
n2_corr                     100   8      32.78   +0.54    33.2
f3_power_corr               100   8      32.76   +0.52    20.1  Power (33.79@125, 3 seeds)
c2_qb_perseq_corr           100   8      32.71   +0.47    13.0  Quantile: sequence scope (27.99@25, 3 seeds)
b1_skew_kl_corr             100   8      32.69   +0.45     8.2  Skew (33.81@200, 3 seeds)
c4_hq                       100   8      32.62   +0.38     9.1  Student-tail + HQ gate (33.56@75, 1 seeds)
b2_forward_kl_corr          100   8      32.43   +0.19     7.3  Forward (32.77@125, 1 seeds)
vanilla_corr                100   8      32.24            15.6  Vanilla (original run family) (32.82@75, 3 seeds)
c2_qb_fixed8_corr           100   8      32.00   -0.24    15.8  Fixed |S|=8 (23.83@25, 3 seeds)
g6_seqmean_corr             100   8      31.95   -0.29    28.1
g1_verified_only_corr       100   8      30.51   -1.73    18.5  Correct (28.23@125, 3 seeds)
d3_teachability_corr        100   8      30.04   -2.20    28.7  TA-OPD (29.88@100, 3 seeds)
h1_first_segment_corr       100   8      28.88   -3.36     8.1  Prefix (30.32@250, 3 seeds)
h3_random_segment_corr      100   8      28.86   -3.38    35.0  Window (32.71@250, 3 seeds)
h4_random_scatter_corr      100   8      28.71   -3.53    37.7  Scatter (30.09@50, 3 seeds)
a5_aggrevate_n0             100   8      27.53   -4.72    48.1
a4_dagger_anneal_n0         100   8      26.95   -5.29    41.1  Annealed mix (21.74@50, 1 seeds)
a1_gkd_mix0.5_n0            100   8      26.33   -5.91    52.4  Fixed mix (24.7@50, 1 seeds)
e2_set_coverage_a0_corr     100   8      25.57   -6.67    44.9  Membership only (24.46@25, 3 seeds)
c5_union_rkl                100   8      23.33   -8.91    93.5  RKL union (29.18@25, 1 seeds)
a3_offpolicy_n0             100   8      21.80  -10.44    30.6  Teacher (20.59@50, 1 seeds)
h2_last_segment_corr        100   8      21.44  -10.80    63.9  Suffix (17.17@250, 3 seeds)

```

Sampling noise of one composite at avg@8, from the calibration note in `scripts/eval_suite.py`: SE about 0.75 (x100); seed-to-seed SD of the legacy vanilla about 0.42.

---

## Consistency checks (recorded as found)

1. **In-loop accuracy from two independent sources.** `metrics_summary.tsv:val_acc` vs `val_gen_summary.tsv:acc_mean` on the 170 (arm, step) pairs present in both: max |difference| = 0.0000, pairs differing by more than 0.0005: 0.
2. **Checkpoint counts.** sum of `progress.tsv:n_checkpoints` = 113; `checkpoints.tsv` rows = 113.
3. **Composite from two code paths.** `extract_post_eval.py` (newest artifact by file-name time stamp) vs an `eval_suite.py acc`-style recomputation (newest artifact with temperature 0.7 and the suite's samples/problem): 57 of 57 complete suites matched, max |difference| = 0.000000 (x100).
4. **Stop sets.** distinct `stop_set` values in `post_eval_cells.csv`: 151645, off; (run, step) groups with more than one stop set: 0.

---

## Known gaps and facts at export time (observations, no inference)

1. **Arms below target.** Arms below target in `progress.tsv`: `c5_union_rkl` (150/250), `c5_union_fkl` (200/250), `e2_set_coverage_a0_corr` (125/200), `g6_seqmean_corr` (125/200), `h2_last_segment_corr` (100/200). `c5_union_fkl`, `c5_union_rkl`, `e2_set_coverage_a0_corr`, `g6_seqmean_corr`, `h2_last_segment_corr` were stopped on 2026-09-18 17:00 PT (2026-09-19 00:00 UTC) by decision and are evaluated at the checkpoints they have.
2. **Offline evaluation coverage.** `offline_grid_status.tsv` lists 0 grid cells that are not fully evaluated yet (column `missing`). A benchmark appears in `post_eval_cells.csv` only when its artifact is final; a (arm, step) appears in `post_eval_bystep.csv` only with all five.
3. **Roster arms without a run directory on this cluster** (they reached their target before the handoff; only their offline artifacts were shipped): `vanilla_corr`, `c2_quantile_budget_corr`, `c4_pi_tail_budget_corr`, `c4_hq`, `c4_state`. They are absent from the training tables and present in the offline tables.
4. **Offline truncation** (32768-token budget): 87 of 537 benchmark artifacts have `trunc_rate` >= 0.5 (filter `post_eval_cells.csv`); artifacts per arm: `a3_offpolicy_n0` 9, `c5_union_rkl` 9, `c5_union_fkl` 8, `a5_aggrevate_n0` 6, `h2_last_segment_corr` 5, `a4_dagger_anneal_n0` 5, `a1_gkd_mix0.5_n0` 5, `h4_random_scatter_corr` 5, `g1_verified_only_corr` 5, `c3_intersection_corr` 4, `h3_random_segment_corr` 4, `e2_set_coverage_a0_corr` 3, `n2_corr` 3, `c4_pi_tail_budget_corr` 2, `c2_quantile_budget_corr` 2, `g6_seqmean_corr` 2, `f2_hard_clip_corr` 2, `d3_teachability_corr` 2, `d2_selectkd_corr` 2, `c2_qb_perseq_corr` 1, `f3_power_corr` 1, `vanilla_corr` 1, `vanilla_te` 1. Highest: `c5_union_rkl`@100 aime24 1.00; `c5_union_rkl`@100 aime25 1.00; `c5_union_rkl`@150 amc23 1.00; `c5_union_rkl`@150 math500 1.00; `c5_union_fkl`@200 aime25 1.00; `c5_union_rkl`@150 minerva 1.00.
5. Steps re-run after a resume are logged twice; as in 0909 the later launch file wins. `n_launch_files` in `progress.tsv` gives the number of launches.
6. W&B was offline in the 0909 export; in this window it is online (training curves, the offline table, 100 sampled trajectories per 25-step stage per arm). This directory remains the file record.
7. **Benchmarks evaluated more than once.** Artifacts in the union of the four nodes: 538; distinct (run, benchmark, step): 537; with more than one artifact: 1; `h3_random_segment_corr_s0`@150 math500: artifact 20260919T202018Z acc 0.6440, artifact 20260920T030749Z acc 0.6380. The tables use the newest artifact (rule of `scripts/eval_suite.py` and `scripts/extract_post_eval.py`); a second artifact comes from a scheduling overlap between two nodes -- same checkpoint, same settings.


<!-- EXTRAS:START -->
---

## Additional material (not part of the 0909 layout)

| path | content |
|---|---|
| `logs/<node>/lanes/*.log.gz` | training logs of the lanes that ran on the node (console output of `slurm/retrain_lane_node.sbatch` / `deploy/dsw/_lane.sh`: every step line, validations, the `<arm> -> OK` result line) |
| `logs/<node>/evalw/*.log.gz` | eval worker logs, one per GPU (`evalw_<node>_gpu<k>`: the node's own queue; `evalw_<node>_foreign_gpu<k>`: cells handed over from another node) |
| `logs/<node>/*.gz` | launch / hand-over / rehearsal-gate logs, GPU guard and disk guard, W&B uploader (`wandb_sync`, `wandb_fast`, `wandb_live`), trajectory sampler, cell hand-over (`offload.log`, `offload_state.tsv`, `split.log`) |
| `logs/<node>/evalq/*` | eval-queue refill log and the failure records of the queues (`*_failed/<run>__<step>`: one line per failed attempt; 3 lines = the cell is skipped -- none reached 3) |
| `traj_samples/<run>/stage_<step>.parquet` | 100 training trajectories per 25-step stage, sampled uniformly without replacement from the step's batch (the tables shown on W&B under `traj/<run>`): prompt, response, token counts, score, advantage, truncation / stop flags, repetition, student log-prob, entropies, per-sequence FKL / RKL / JSD / TV / top-1 agreement with the teacher |
| `traj_samples/<run>/stage_<step>.json` | sampling record of that table: population size, seed, source file, sample vs population mean score |
| `non_roster_dirs.tsv` | directories in the checkpoint trees that are not roster runs (same columns as in 0909) |
| `extras_manifest.tsv` | path, bytes and md5 of every file listed in this section |

Logs: w0 41 files / 0.7 MB; w1 34 files / 0.7 MB; w2 32 files / 0.9 MB; w3 31 files / 1.0 MB (gzip, one file each; read with `zless` / `gzcat`, search with `zgrep`). Left on the servers: download and environment-setup logs, hidden cache files, the W&B live-streamer debug directory, per-item generations (`val_gen/`, `evals/*.parquet`, `traj/`) and checkpoints. Credential scan before packing (files=138 gz_bytes=3402702 redactions=key=value: 0; hf token: 0; signed url: 0; url credentials: 0; 40-hex near wandb: 0 hit_files=-).

Sampled trajectories: 184 stage tables over 24 runs, 18400 trajectories, 188 MB. Non-roster directories: 5.

<!-- EXTRAS:END -->

---

## Reproduce

```bash
# on every node that holds run directories
python export_train_tables.py <node> <scratch>/nodes/<node>
# on the node that holds the union of evals/*.parquet
SIMOPD_STORE=<dir with evals/> python scripts/extract_post_eval.py --roster ALL --out-dir <out>
python export_merge.py <out> && python export_readme.py <out>
```
All scripts only read the run directories.
