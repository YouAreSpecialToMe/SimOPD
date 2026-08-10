# SimOPD 16k Campaign — Live Report

_Student **Qwen3-1.7B-Base** ← Teacher **Qwen3-4B-Instruct-2507**, response cap 16,384 tokens, 250 steps,
3 seeds/arm, checkpoint every 25 steps (all 10 kept). 29 arms × 3 seeds = **87 rows**, all run on this
24-node / 192-GPU fleet — including the four arms (c4/e2/e3/g5) originally ceded to the collaborating
site and relaunched here on 2026-08-09._

_Regenerated 2026-08-11 02:25 from live logs, checkpoint trees and eval artifacts._

**Anchors.** In-loop greedy MATH500 of the base student: **0.468** (every curve's step-0).
Offline pipeline on the same model: 0.4740 (pipeline-consistency check).
Offline suite composite of the base student: **0.1453** (step −1 convention).

**Fleet state.** 24 of 29 arms have all three seeds at step 250; 4 arms (a2/e2/g5/h2) are in
their last steps; b2 is parked at its per-seed memory ceiling (§4.5). Post-hoc suite: **184 of 750
checkpoint-evaluations** complete, the rest grinding.

## 1. In-loop eval (greedy MATH500, every 25 steps)

`mean·N` over seeds with data at that step. Wall-family rows (b3/d1/d2/d3/g2) count 4-GPU-era logs only —
their 2-card attempts died at the length wall and are excluded. Sorted by step-250 value.

| arm | signal | s25 | s50 | s75 | s100 | s125 | s150 | s175 | s200 | s225 | s250 | now |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c2_quantile_budget | topk | 0.635·3 | 0.610·3 | 0.617·3 | 0.611·3 | 0.637·3 | 0.639·3 | 0.646·3 | 0.657·3 | 0.668·3 | 0.684·3 | 250/250/250 |
| j1_kdrl | sampled | 0.551·3 | 0.573·3 | 0.584·3 | 0.613·3 | 0.631·3 | 0.637·3 | 0.639·3 | 0.646·3 | 0.641·3 | 0.639·3 | 250/250/250 |
| c4_pi_tail_budget | topk | 0.627·3 | 0.617·3 | 0.599·3 | 0.589·3 | 0.593·3 | 0.593·3 | 0.619·3 | 0.631·3 | 0.638·3 | 0.639·3 | 250/250/250 |
| h1_first_segment | sampled | 0.587·3 | 0.628·3 | 0.627·3 | 0.629·3 | 0.615·3 | 0.630·3 | 0.617·3 | 0.632·3 | 0.626·3 | 0.625·3 | 250/250/250 |
| c3_intersection | topk | 0.566·3 | 0.617·3 | 0.599·3 | 0.519·3 | 0.555·3 | 0.578·3 | 0.589·3 | 0.587·3 | 0.589·3 | 0.599·3 | 250/250/250 |
| e3_zvalue | topk | 0.552·2 | 0.618·3 | 0.637·3 | 0.621·3 | 0.561·3 | 0.522·3 | 0.514·3 | 0.545·3 | 0.580·3 | 0.589·3 | 250/250/250 |
| h3_random_segment | sampled | 0.555·3 | 0.649·3 | 0.613·3 | 0.612·3 | 0.606·3 | 0.597·3 | 0.595·3 | 0.599·3 | 0.596·3 | 0.583·3 | 250/250/250 |
| e1_pl_rank | topk | 0.521·3 | 0.499·3 | 0.496·3 | 0.527·3 | 0.545·3 | 0.570·3 | 0.555·3 | 0.571·3 | 0.569·3 | 0.572·3 | 250/250/250 |
| g1_verified_only | sampled | 0.223·3 | 0.169·3 | 0.121·3 | 0.581·3 | 0.627·3 | 0.571·3 | 0.540·3 | 0.529·3 | 0.511·3 | 0.529·3 | 250/250/250 |
| b4_jsd | topk | 0.581·3 | 0.593·3 | 0.533·3 | 0.526·3 | 0.478·3 | 0.488·3 | 0.506·3 | 0.512·3 | 0.519·3 | 0.524·3 | 250/250/250 |
| c1_lsm_topk32_renorm | topk | 0.417·3 | 0.487·3 | 0.504·3 | 0.523·3 | 0.533·3 | 0.531·3 | 0.521·3 | 0.519·3 | 0.529·3 | 0.524·3 | 250/250/250 |
| g2_fire_likelihood | samp+sel | 0.619·3 | 0.612·3 | 0.615·3 | 0.624·3 | 0.554·3 | 0.501·3 | 0.481·3 | 0.461·3 | 0.495·3 | 0.511·3 | 250/250/250 |
| d2_selectkd | samp+sel | 0.622·3 | 0.620·3 | 0.595·3 | 0.612·3 | 0.641·3 | 0.641·3 | 0.559·3 | 0.473·3 | 0.472·3 | 0.480·3 | 250/250/250 |
| b1_skew_kl | sampled | 0.594·3 | 0.627·3 | 0.629·3 | 0.617·3 | 0.613·3 | 0.611·3 | 0.645·3 | 0.629·3 | 0.455·3 | 0.475·3 | 250/250/250 |
| vanilla | sampled | 0.604·3 | 0.627·3 | 0.625·3 | 0.575·3 | 0.517·3 | 0.475·3 | 0.438·3 | 0.441·3 | 0.474·3 | 0.473·3 | 250/250/250 |
| d1_tip | samp+sel | 0.580·3 | 0.620·3 | 0.617·3 | 0.577·3 | 0.529·3 | 0.459·3 | 0.466·3 | 0.459·3 | 0.489·3 | 0.473·3 | 250/250/250 |
| f3_power | sampled | 0.523·3 | 0.563·3 | 0.551·3 | 0.581·3 | 0.561·3 | 0.555·3 | 0.553·3 | 0.503·3 | 0.490·3 | 0.473·3 | 250/250/250 |
| vanilla_n8 | sampled | 0.623·3 | 0.629·3 | 0.614·3 | 0.555·3 | 0.528·3 | 0.480·3 | 0.433·3 | 0.471·3 | 0.487·3 | 0.466·3 | 250/250/250 |
| b5_k2 | sampled | 0.617·3 | 0.625·3 | 0.619·3 | 0.599·3 | 0.519·3 | 0.472·3 | 0.431·3 | 0.474·3 | 0.477·3 | 0.460·3 | 250/250/250 |
| f2_hard_clip | sampled | 0.603·3 | 0.631·3 | 0.618·3 | 0.633·3 | 0.630·3 | 0.631·3 | 0.648·3 | 0.464·3 | 0.467·3 | 0.457·3 | 250/250/250 |
| g4_failure_only | sampled | 0.615·3 | 0.615·3 | 0.629·3 | 0.582·3 | 0.497·3 | 0.465·3 | 0.425·3 | 0.435·3 | 0.466·3 | 0.450·3 | 250/250/250 |
| f1_soft_log | sampled | 0.588·3 | 0.609·3 | 0.624·3 | 0.631·3 | 0.620·3 | 0.632·3 | 0.623·3 | 0.457·3 | 0.463·3 | 0.447·3 | 250/250/250 |
| d3_teachability | samp+sel | 0.581·3 | 0.622·3 | 0.565·3 | 0.542·3 | 0.428·3 | 0.435·3 | 0.416·3 | 0.437·3 | 0.431·3 | 0.415·3 | 250/250/250 |
| b3_eopd_gate | samp+sel | 0.582·3 | 0.612·3 | 0.555·3 | 0.354·3 | 0.113·3 | 0.025·3 | 0.018·3 | 0.003·3 | 0.001·3 | 0.003·3 | 250/250/250 |
| g5_rgopd_gate | sampled | 0.618·3 | 0.612·3 | 0.627·3 | 0.581·3 | 0.496·3 | 0.464·3 | 0.425·3 | – | – | – | 186/188/187 |
| e2_set_coverage | topk | 0.596·3 | 0.597·3 | 0.588·3 | 0.581·3 | 0.498·3 | 0.469·3 | 0.460·3 | – | – | – | 180/180/182 |
| b2_forward_kl | sampled | 0.541·3 | 0.572·3 | 0.519·3 | 0.458·3 | 0.487·3 | 0.459·3 | 0.408·2 | 0.460·1 | – | – | 196/172/208 |
| a2_coldstart | sampled | 0.430·3 | 0.437·3 | 0.439·3 | 0.468·3 | 0.465·3 | 0.467·3 | 0.489·3 | 0.478·3 | – | – | 200/215/208 |
| h2_last_segment | sampled | 0.475·3 | 0.260·3 | 0.096·3 | 0.097·3 | 0.164·3 | 0.228·3 | 0.385·3 | 0.453·3 | – | – | 223/222/223 |

**Reading.** Three regimes are visible. (i) **Monotone climbers** — c2 (0.635→**0.684**), c4 (→0.639),
e1, c1, c3: every one is a top-k *value/order* objective, and none of them collapses. (ii) **Late
collapse** — most sampled-token arms (vanilla, n8, b5, g4, f1, f2, b1, d2) hold ~0.62-0.65 until step
150-200, then drop 0.15-0.19 within one or two evals; f2 is the sharpest (0.648@175 → 0.464@200).
(iii) **Deep-U recovery** — g1 (0.121@75 → 0.529), h2 (0.096@75 → 0.453@200 and still climbing), e3
(0.514@175 → 0.589): all three crater early and climb back, so a mid-run reading of those arms would
have inverted their ranking. b3 is the one arm that collapses and never returns (0.003), which is the
verdict on additive gated FKL at this scale, not an artifact — it ran to 250 on healthy hardware.

## 2. Efficiency

Wall-clock is summed over **every attempt** of each row (restarts, relays and revivals included) and
averaged over the arm's three seeds; GPU·h = wall-h × GPUs × 3 seeds. Widths are protocol: the
KEEP_SAMPLED family (b3/d1/d2/d3/g2) and the n-8 / KD-RL arms need a teacher pool on separate cards
(`NGPUS_PER_NODE=2 + TEACHER_WORLD_SIZE=2`), everything else runs 1+1.

| method | GPUs/run | seeds at 250 | ckpts saved | wall-h/seed | GPU·h (arm, 3 seeds) |
|---|---|---|---|---|---|
| a2_coldstart | 2 | 0/3 | 8/8/8 | 65.0 | 390 |
| b1_skew_kl | 2 | 3/3 | 10/10/10 | 34.6 | 208 |
| b2_forward_kl | 2 | 0/3 | 7/6/8 | 34.1 | 205 |
| b3_eopd_gate | 4 | 3/3 | 10/10/10 | 100.8 | 1210 |
| b4_jsd | 2 | 3/3 | 10/10/10 | 29.9 | 179 |
| b5_k2 | 2 | 3/3 | 10/10/10 | 58.6 | 352 |
| c1_lsm_topk32_renorm | 2 | 3/3 | 10/10/10 | 7.6 | 45 |
| c2_quantile_budget | 2 | 3/3 | 10/10/10 | 50.5 | 303 |
| c3_intersection | 2 | 3/3 | 10/10/10 | 25.0 | 150 |
| c4_pi_tail_budget | 2 | 3/3 | 10/10/10 | 32.9 | 198 |
| d1_tip | 4 | 3/3 | 10/10/10 | 43.4 | 521 |
| d2_selectkd | 4 | 3/3 | 10/10/10 | 32.4 | 389 |
| d3_teachability | 4 | 3/3 | 10/10/10 | 38.5 | 461 |
| e1_pl_rank | 2 | 3/3 | 10/10/10 | 23.1 | 139 |
| e2_set_coverage | 2 | 0/3 | 7/7/7 | 45.1 | 271 |
| e3_zvalue | 2 | 3/3 | 10/10/10 | 43.2 | 259 |
| f1_soft_log | 2 | 3/3 | 10/10/10 | 40.1 | 241 |
| f2_hard_clip | 2 | 3/3 | 10/10/10 | 42.5 | 255 |
| f3_power | 2 | 3/3 | 10/10/10 | 28.7 | 172 |
| g1_verified_only | 2 | 3/3 | 10/10/10 | 40.0 | 240 |
| g2_fire_likelihood | 4 | 3/3 | 10/10/10 | 47.8 | 573 |
| g4_failure_only | 2 | 3/3 | 10/10/10 | 59.5 | 357 |
| g5_rgopd_gate | 2 | 0/3 | 7/7/7 | 38.8 | 233 |
| h1_first_segment | 2 | 3/3 | 10/10/10 | 13.7 | 82 |
| h2_last_segment | 2 | 0/3 | 8/8/8 | 62.9 | 378 |
| h3_random_segment | 2 | 3/3 | 10/10/10 | 35.0 | 210 |
| j1_kdrl | 4 | 3/3 | 10/10/10 | 8.4 | 101 |
| vanilla | 2 | 3/3 | 10/10/10 | 59.3 | 356 |
| vanilla_n8 | 4 | 3/3 | 10/10/10 | 57.8 | 694 |
| **fleet total** | | | | | **9172** |

**Where the GPU·h went.** b3 alone burned 1209 GPU·h — ~13% of the campaign — for a scientifically dead
arm, almost all of it in the two-day OOM war (§4.2). The next three biggest (vanilla_n8 694, g2 573,
d1 521) are also 4-GPU rows. The cheapest completed arms (c1 45, h1 82, j1 101 GPU·h) are the ones that
never crashed: **restart overhead, not the objective, dominates cost variance**.

**Utilization.** Sampled fleet-wide while all arms were training (2026-08-09 02:55): rollout phases read
90-100% GPU, update phases 15-50%, VRAM 33-70 GiB/GPU depending on rollout-pool fraction and whether a
teacher engine shares the card. Instantaneous fleet average was ~50%, which is the expected duty cycle
for on-policy distillation at this response length (generation dominates wall-clock; the update phase is
memory-bound, not compute-bound).

**Storage.** 831 checkpoints × 27 GiB = **26 TiB** on the shared filesystem (per checkpoint: 7.6 GiB FSDP
model shards + 13 GiB optimizer state + 6.5 GiB HF export). Filesystem is at 94% (28 TiB free). The
optimizer shards of the 72 finished rows (~9 TiB) are dead weight once a row is done — reclaimable
without touching either the HF exports the eval suite reads or the model shards a re-run would need.

**Eval-line efficiency.** `eval_suite.py` runs vLLM with `tensor_parallel_size=1`, so a sweep dispatched
onto a GPU *pair* leaves the second card idle. Measured mid-campaign: eval workers pinned at 100% util /
73 GiB on the even card of each pair, partner at 4 MiB. Sweeps are now dispatched per-GPU, and long runs
are split across cards by `--step`, which roughly doubles post-eval throughput on the same hardware.

## 3. Post-hoc suite over saved checkpoints

Composite = equal-macro mean of AIME24+25 (avg@32), AMC23 (avg@32), Minerva (avg@3), MATH500 (avg@3),
τ=0.7 / top-p 0.95 / 32,768-token budget — the same protocol as the base-student anchor **0.1453**.
`mean·N` over seeds whose artifacts exist at that step; the last column counts completed
checkpoint-evaluations out of 30 (10 checkpoints × 3 seeds).

| method | s25 | s50 | s75 | s100 | s125 | s150 | s175 | s200 | s225 | s250 | ckpts done |
|---|---|---|---|---|---|---|---|---|---|---|---|
| j1_kdrl | 0.228·3 | 0.254·3 | 0.270·3 | 0.285·3 | 0.294·3 | 0.303·3 | 0.307·3 | 0.308·3 | 0.316·3 | 0.315·3 | 30/30 |
| c1_lsm_topk32_renorm | 0.137·3 | 0.161·3 | 0.171·3 | 0.186·3 | 0.191·3 | 0.200·3 | 0.211·3 | 0.232·2 | 0.233·2 | 0.234·2 | 27/30 |
| h1_first_segment | 0.275·3 | 0.292·3 | 0.287·3 | 0.292·3 | 0.294·3 | 0.295·3 | 0.297·3 | 0.301·3 | 0.299·3 | – | 27/30 |
| e1_pl_rank | 0.126·3 | 0.160·3 | 0.211·3 | 0.237·1 | – | – | – | – | – | – | 10/30 |
| f3_power | 0.188·3 | 0.241·3 | 0.295·3 | – | – | – | – | – | – | – | 9/30 |
| b1_skew_kl | 0.255·3 | 0.293·3 | 0.335·2 | – | – | – | – | – | – | – | 8/30 |
| b2_forward_kl | 0.224·3 | 0.257·3 | 0.330·1 | – | – | – | – | – | – | – | 7/30 |
| b4_jsd | 0.204·3 | 0.265·3 | – | – | – | – | – | – | – | – | 6/30 |
| c3_intersection | 0.134·3 | 0.263·3 | – | – | – | – | – | – | – | – | 6/30 |
| d2_selectkd | 0.279·3 | 0.295·3 | – | – | – | – | – | – | – | – | 6/30 |
| d3_teachability | 0.246·3 | 0.280·3 | – | – | – | – | – | – | – | – | 6/30 |
| f1_soft_log | 0.267·3 | 0.293·3 | – | – | – | – | – | – | – | – | 6/30 |
| h3_random_segment | 0.251·3 | 0.298·3 | – | – | – | – | – | – | – | – | 6/30 |
| f2_hard_clip | 0.277·3 | 0.293·2 | – | – | – | – | – | – | – | – | 5/30 |
| b3_eopd_gate | 0.253·3 | – | – | – | – | – | – | – | – | – | 3/30 |
| c2_quantile_budget | 0.280·3 | – | – | – | – | – | – | – | – | – | 3/30 |
| c4_pi_tail_budget | 0.266·3 | – | – | – | – | – | – | – | – | – | 3/30 |
| d1_tip | 0.260·3 | – | – | – | – | – | – | – | – | – | 3/30 |
| g1_verified_only | 0.106·3 | – | – | – | – | – | – | – | – | – | 3/30 |
| g2_fire_likelihood | 0.274·3 | – | – | – | – | – | – | – | – | – | 3/30 |
| vanilla | 0.280·3 | – | – | – | – | – | – | – | – | – | 3/30 |
| vanilla_n8 | 0.285·3 | – | – | – | – | – | – | – | – | – | 3/30 |
| b5_k2 | 0.276·1 | – | – | – | – | – | – | – | – | – | 1/30 |
| a2_coldstart | – | – | – | – | – | – | – | – | – | – | 0/30 |
| e2_set_coverage | – | – | – | – | – | – | – | – | – | – | 0/30 |
| e3_zvalue | – | – | – | – | – | – | – | – | – | – | 0/30 |
| g4_failure_only | – | – | – | – | – | – | – | – | – | – | 0/30 |
| g5_rgopd_gate | – | – | – | – | – | – | – | – | – | – | 0/30 |
| h2_last_segment | – | – | – | – | – | – | – | – | – | – | 0/30 |

### 3.1 Completed step-250 suites, per benchmark

| method | seed | step | composite | AIME24+25 | AMC23 | Minerva | MATH500 |
|---|---|---|---|---|---|---|---|
| c1_lsm_topk32_renorm | s0 | 250 | **0.2345** | 0.0156 | 0.2805 | 0.1311 | 0.5107 |
| c1_lsm_topk32_renorm | s1 | 250 | **0.2335** | 0.0172 | 0.2875 | 0.1287 | 0.5007 |
| j1_kdrl | s0 | 250 | **0.3167** | 0.0401 | 0.4281 | 0.1507 | 0.6480 |
| j1_kdrl | s1 | 250 | **0.3160** | 0.0547 | 0.4062 | 0.1642 | 0.6387 |
| j1_kdrl | s2 | 250 | **0.3136** | 0.0474 | 0.4078 | 0.1605 | 0.6387 |

**Reading.** The composite tracks the in-loop curve where both exist (j1: 0.315 composite / 0.639 in-loop
MATH500; the suite's own MATH500 column reads 0.639-0.648, i.e. the offline pipeline reproduces the
in-loop number to the third decimal), but it is **not** a monotone function of it: at step 75, b1 (0.335)
and b2 (0.330) are well above j1 (0.270) even though their in-loop MATH500 is comparable, because AMC23
and Minerva move differently from MATH500. Final rankings therefore wait for the full sweep — the arms
with complete curves so far are j1 (0.315), c1 (0.234) and h1 (0.299 @225).

## 4. Engineering changes that shipped mid-campaign

All changes below are **value-identical** (verified by CPU equivalence batteries) — they change peak
memory or launch safety, never a computed loss. Runs on different snapshots stay comparable.

### 4.1 `aggregate_dp` metric-count fix (`src/simopd/losses.py`)

DP-1 lanes never compare ranks, so `_signal_quantiles` skipping keys on empty masks stayed latent until
the 4-GPU (DP-2) lanes: a sparse selector leaves a positional bin empty on one rank but not the other,
per-rank append counts diverge, and verl raises `must have the same number of values: [5, 4]`. Empty sets
now emit zeros — metric emission only:

```python
if x.numel() == 0:
    # every rank must append every key on every micro-batch (Metric.aggregate_dp)
    x = losses.new_zeros(1).float()
```

### 4.2 The length wall, and what actually fixed it

**Symptom.** Once response lengths saturate the 16k cap (~step 76-110), the KEEP_SAMPLED family died in
`update_actor` asking for a **17-19 GiB single allocation**. b3 died at 76-84 on every attempt, d1 at
107-119, g2 at 127-143.

**Root cause.** `F.log_softmax(student_logits, -1)` in `_prepare` materializes an fp32 `[T, V]` tensor —
at ~33k tokens × 152k vocab that *is* 18.6 GiB, and `torch.topk` over the same tensor allocates a sort
workspace of comparable size.

**What did not work** (each was tried, measured, and falsified — recorded here because the negative
results are the expensive part):

| lever | expectation | measured outcome |
|---|---|---|
| `rollout.gpu_memory_utilization` 0.45→0.32→0.25→0.18 | free engine memory for the update | byte-identical death step and ask; `free_cache_engine=True` already sleeps vLLM during the update, so the pool never governed update-phase headroom |
| chunked `torch.topk` (ship 2026-08-08) | remove the sort workspace | b3 still died at 83-84 with the same 18.6 GiB ask — the workspace was never the dominant term |
| `ppo_max_token_len_per_gpu` 17408→8704 | halve the packed micro-batch | `AssertionError: max_token_len must be greater than the sequence length (16754)` — the packing cap cannot go below one sequence, and ours are 16k+ |
| `fsdp_config.optimizer_offload=True` | free ~19 GiB of AdamW state | knob reached hydra, resident memory unchanged, same death step |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | defragment | b2_s2 died **earlier** (202 vs 208) with a larger ask |

**What worked: never materialize `[T, V]`.** `log p(j) = logits[j] − lse`, so every consumer of the
student distribution (top-k gathers, sampled-token gathers, entropy, diagnostics) can be served from raw
logits plus a per-token `logsumexp`. Both halves of the flash-softmax trick are required — token-dim
chunking *and* recompute-in-backward; a naive chunk loop retains every chunk's fp32 intermediates and is
strictly worse than the monolith (measured: 64 MiB free at the same step 83):

```python
def _lse_chunked(student_logits, chunk=256):
    from torch.utils.checkpoint import checkpoint

    def _f(sl):
        return torch.logsumexp(sl.float(), dim=-1)

    return torch.cat([checkpoint(_f, sl, use_reentrant=False)
                      for sl in student_logits.split(chunk, dim=-2)], dim=-1)
```

Backward now touches one `[chunk, V]` softmax at a time (~150 MB at chunk=256). Equivalence battery on
CPU: fp32 identity ≤4e-6 on losses, auxiliary panels and input gradients; diagnostics key-and-value
identical between the materialized and streaming call paths; in bf16 the streaming path is **8× closer**
to the fp32 reference than the old fused-cast path.

**Result.** All nine wall-family rows (b3×3, d1×3, g2×3) reached step 250 on this code with zero rows
lost. The arms that had been parked for a day and a half resumed from their last checkpoints and ran to
completion without a single further OOM.

### 4.3 Porting the fix to the e-axis (`compute_set_coverage_topk`, `compute_zvalue_topk`)

e2 hit the same wall at steps 50-53 on all three seeds — earlier than the wall family, because Cornell-arm
generations are long from step 1. Its traceback pinned the ask to the *same* `_prepare` line, so the
kernel was ported to `_prepare_streaming` (and e3 pre-emptively, having identical consumption shape):

```python
lse, t_lp, t_id, _, _ = _prepare_streaming(...)
s = torch.gather(student_logits, -1, t_id) - lse.unsqueeze(-1)   # log p at teacher's top-k
stu_topk_ids = _student_topk_ids(student_logits, k=t_lp.shape[-1])  # rank is shift-invariant
out = _overlap_diagnostics(None, t_lp, t_id, stu_topk_ids, stu_at_teacher=s,
                           s_ent=_entropy_from_logits(student_logits, lse))
```

All three e2 seeds cleared the kill zone within the hour and banked step 75; e3 ran to 250 untouched by
the wall.

### 4.4 Materialized paths deliberately left alone

`compute_reverse_kl_topk` (c1) and `_prepare` itself still materialize for the arms that never saturate
the cap — 2-card rows that were stable all campaign. Changing them would have re-pinned 40+ flying rows
for no measured benefit.

### 4.5 b2 parked at its per-seed ceiling

b2's spike lives in the **vendored verl tree** (`verl/trainer/distillation/fsdp/losses.py:122`), which is
a live import — not snapshot-isolated like `src/simopd`. Editing it would have changed the code under
every future restart on the fleet, including wall-family relays that had just been stabilized. Each b2
seed's death step also sits *below* its next checkpoint (s0 196<200, s1 172<175, s2 208<225), so a relay
can never bank progress — "zombie march". Decision: **park all three at their last checkpoint**
(s0@175, s1@150, s2@200), fence the rows in `configs/campaign.tsv`, and run partial post-eval sweeps over
the checkpoints that exist. b2 is a collapsing baseline whose verdict was already written by step 175.

### 4.6 Launch-time twin defense

Two mechanisms, both needed:

1. **Poison pill** — manual relaunches pass `EXTRA_HYDRA="trainer.balance_batch=True …"` (a restatement of
   the hardcoded protocol value, so training is byte-identical) purely so it lands in the resume
   fingerprint. A stale baked lane firing the same row later computes a different fingerprint and
   self-refuses.
2. **Fingerprint sentinel** — the pill does *not* protect rows that were never relayed: their stored
   fingerprint is the plain one, so a baked twin matches it and would resume the original's checkpoints.
   This fired for real on 2026-08-10: b1_s2's lane finished and advanced to its baked `a2_coldstart:1`
   copy while the original was live at step 100. It was killed inside the boot window (zero
   contamination), then the five remaining baked-pending rows were pre-emptively blocked by overwriting
   their stored fingerprint with a sentinel value:

```bash
printf 'manual-twin-block-20260809' > $CKPT/<run>/simopd_fingerprint.txt   # original backed up alongside
```

All four twins that subsequently fired refused at launch with
`FATAL: … holds a checkpoint at step N from a DIFFERENT config` — one cosmetic FAIL each, zero GPU time,
zero contamination.

### 4.7 Incident ledger

Every crash recovered from the latest 25-step checkpoint (≤24 steps lost per incident). Across the whole
campaign: 0 rows lost to corruption, 0 checkpoint contaminations, 1 real duplicate caught in its boot
window, 4 twins refused by fingerprint sentinels, 6 more refused by poison pills. 84 of 87 rows will
reach step 250; the 3 exceptions are b2's seeds, stopped by a triple-confirmed hardware ceiling, with
their curves complete up to their stopping points. Full per-row history lives in `configs/campaign.tsv`.

## 5. What is still running

- **Training**: a2×3 (~200-215), h2×3 (~222), e2×3 (~180), g5×3 (~186) — all expected to reach 250 within
  the day. b2×3 stay parked.
- **Post-hoc suite**: 184/750 checkpoint-evaluations done, ~75 sweeps in flight. a2's partial sweeps
  will be re-dispatched after it completes so its last checkpoints are covered (`eval_suite` is
  idempotent — completed artifacts are skipped).
- **Deliverables pending**: full 29×10 composite table, per-benchmark breakdown at 250, and the
  seed-variance band per arm.
