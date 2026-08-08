# SimOPD 16k Campaign — Live Report

_Student **Qwen3-1.7B-Base** ← Teacher **Qwen3-4B-Instruct-2507**, response cap 16,384, 250 steps, 3 seeds/arm, checkpoints every 25 steps (all 10 kept). 25 arms run on this fleet; c4/e2/e3/g5 run at the collaborating site (rows kept empty here until their numbers land)._

_Generated 2026-08-09 ~03:00 from live logs; refreshed periodically while the campaign runs._

**Anchors.** In-loop greedy MATH500 of the base student: **0.468** (every curve's step-0). Offline pipeline on the same model: 0.4740 (pipeline-consistency check). Offline suite composite of the base student (step −1 convention): **0.1453**.

## 1. In-loop eval (greedy MATH500, every 25 steps)

`mean·N` over available seeds; wall-family (b3/d1/d2/d3/g2) readings are 4-GPU-era only (their 2-card attempts are excluded). ⚠ceded rows stop at the hand-off point.

| arm | signal | s25 | s50 | s75 | s100 | s125 | s150 | s175 | s200 | s225 | s250 | now |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h3_random_segment | sampled | 0.555·3 | 0.649·3 | 0.613·3 | 0.612·3 | – | – | – | – | – | – | 119/122/122 |
| j1_kdrl | sampled | 0.551·3 | 0.573·3 | 0.584·3 | 0.613·3 | 0.631·3 | 0.637·3 | 0.639·3 | 0.646·3 | 0.641·3 | 0.639·3 | 250/250/250 |
| b1_skew_kl | sampled | 0.594·3 | 0.627·3 | 0.629·3 | 0.617·3 | 0.613·3 | 0.611·3 | 0.645·3 | – | – | – | 178/177/177 |
| d2_selectkd | samp+sel | 0.622·3 | 0.620·3 | 0.595·3 | 0.612·3 | 0.641·3 | 0.641·3 | – | – | – | – | 168/166/164 |
| c2_quantile_budget | topk | 0.635·3 | 0.610·3 | 0.617·3 | 0.606·2 | – | – | – | – | – | – | 107/97/110 |
| f2_hard_clip | sampled | 0.603·3 | 0.631·3 | 0.618·3 | 0.633·3 | 0.630·3 | 0.631·3 | – | – | – | – | 165/165/163 |
| h1_first_segment | sampled | 0.587·3 | 0.628·3 | 0.627·3 | 0.629·3 | 0.615·3 | 0.630·3 | 0.617·3 | 0.632·3 | 0.626·3 | 0.625·3 | 250/250/250 |
| f1_soft_log | sampled | 0.588·3 | 0.609·3 | 0.624·3 | 0.631·3 | 0.620·3 | 0.632·3 | – | – | – | – | 172/172/172 |
| c4_pi_tail_budget ⚠ceded | topk | 0.630·2 | – | – | – | – | – | – | – | – | – | 12/28/28 |
| vanilla_n8 | sampled | 0.623·3 | 0.629·3 | 0.614·3 | 0.555·3 | – | – | – | – | – | – | 123/124/124 |
| g4_failure_only | sampled | 0.615·3 | 0.615·3 | 0.629·3 | 0.582·3 | – | – | – | – | – | – | 112/113/112 |
| vanilla | sampled | 0.604·3 | 0.627·3 | 0.625·3 | 0.575·3 | 0.517·3 | – | – | – | – | – | 130/126/128 |
| g1_verified_only | sampled | 0.223·3 | 0.169·3 | 0.121·3 | 0.581·3 | 0.627·3 | 0.571·3 | – | – | – | – | 172/174/172 |
| b5_k2 | sampled | 0.617·3 | 0.625·3 | 0.619·3 | 0.599·3 | – | – | – | – | – | – | 112/112/114 |
| g2_fire_likelihood | samp+sel | 0.619·3 | 0.612·3 | 0.615·3 | 0.624·3 | 0.542·1 | – | – | – | – | – | 122/123/135 |
| d3_teachability | samp+sel | 0.581·3 | 0.622·3 | 0.565·3 | 0.542·3 | 0.428·3 | – | – | – | – | – | 144/146/142 |
| d1_tip | samp+sel | 0.580·3 | 0.620·3 | 0.617·3 | 0.577·3 | – | – | – | – | – | – | 110/115/111 |
| g5_rgopd_gate ⚠ceded | sampled | 0.618·3 | – | – | – | – | – | – | – | – | – | 30/29/28 |
| c3_intersection | topk | 0.566·3 | 0.617·3 | 0.599·3 | 0.519·3 | 0.555·3 | 0.578·3 | – | – | – | – | 165/167/164 |
| b3_eopd_gate | samp+sel | 0.582·3 | 0.612·3 | 0.555·3 | – | – | – | – | – | – | – | 84/84/87 |
| e2_set_coverage ⚠ceded | topk | 0.596·3 | – | – | – | – | – | – | – | – | – | 30/31/29 |
| b4_jsd | topk | 0.581·3 | 0.593·3 | 0.533·3 | 0.526·3 | 0.478·3 | 0.488·3 | – | – | – | – | 154/152/155 |
| f3_power | sampled | 0.523·3 | 0.563·3 | 0.551·3 | 0.581·3 | 0.561·3 | 0.555·3 | 0.532·1 | – | – | – | 161/178/172 |
| e3_zvalue ⚠ceded | topk | 0.572·1 | – | – | – | – | – | – | – | – | – | 24/25/24 |
| b2_forward_kl | sampled | 0.541·3 | 0.572·3 | 0.519·3 | 0.458·3 | 0.484·2 | – | – | – | – | – | 124/149/149 |
| e1_pl_rank | topk | 0.521·3 | 0.499·3 | 0.496·3 | 0.527·3 | 0.545·3 | 0.570·3 | 0.555·3 | 0.565·2 | – | – | 191/214/224 |
| c1_lsm_topk32_renorm | topk | 0.417·3 | 0.487·3 | 0.504·3 | 0.523·3 | 0.533·3 | 0.531·3 | 0.521·3 | 0.519·3 | 0.529·3 | 0.524·3 | 250/250/250 |
| h2_last_segment | sampled | 0.475·3 | 0.260·3 | – | – | – | – | – | – | – | – | 74/74/74 |
| a2_coldstart | sampled | 0.430·3 | 0.437·3 | – | – | – | – | – | – | – | – | 52/68/61 |

## 2. Efficiency (live snapshot + coarse wall-clock)

GPU util/mem sampled fleet-wide at 02:55 (one instant — rollout phases read ~90-100%, update phases lower; treat as indicative). Wall-clock is summed over every attempt of the arm's 16k lanes (restarts included), averaged over seeds with data; GPU·h = GPUs × wall-clock. 4-GPU rows: memory spans student FSDP / student rollout engine / teacher engines.

| method | GPUs/run | util % (snap) | VRAM/GPU GiB (snap) | wall-h so far | GPU·h so far |
|---|---|---|---|---|---|
| vanilla | 2 | 49 | 33–70 | 20.4 | 41 |
| vanilla_n8 | 4 | 50 | 49–70 | 19.8 | 79 |
| a2_coldstart | 2 | 66 | 51–69 | 17.2 | 34 |
| b1_skew_kl | 2 | 49 | 40–69 | 19.7 | 39 |
| b2_forward_kl | 2 | 50 | 58–70 | 19.8 | 40 |
| b3_eopd_gate | 4 | 52 | 38–69 | 37.4 | 150 |
| b4_jsd | 2 | 50 | 58–70 | 15.5 | 31 |
| b5_k2 | 2 | 50 | 58–69 | 15.3 | 31 |
| c1_lsm_topk32_renorm | 2 | done | – | 7.6 | 15 |
| c2_quantile_budget | 2 | 16 | 58–70 | 19.8 | 40 |
| c3_intersection | 2 | 33 | 58–70 | 15.5 | 31 |
| c4_pi_tail_budget ⚠ceded | – | – | – | – | – |
| d1_tip | 4 | 41 | 38–70 | 18.0 | 72 |
| d2_selectkd | 4 | 50 | 49–70 | 30.1 | 120 |
| d3_teachability | 4 | 50 | 48–70 | 24.0 | 96 |
| e1_pl_rank | 2 | 33 | 58–70 | 19.8 | 40 |
| e2_set_coverage ⚠ceded | – | – | – | – | – |
| e3_zvalue ⚠ceded | – | – | – | – | – |
| f1_soft_log | 2 | 49 | 40–70 | 19.8 | 40 |
| f2_hard_clip | 2 | 49 | 40–70 | 19.8 | 40 |
| f3_power | 2 | 49 | 41–70 | 15.4 | 31 |
| g1_verified_only | 2 | 65 | 58–69 | 19.8 | 40 |
| g2_fire_likelihood | 4 | 50 | 38–69 | 22.4 | 90 |
| g4_failure_only | 2 | 50 | 58–69 | 15.5 | 31 |
| g5_rgopd_gate ⚠ceded | – | – | – | – | – |
| h1_first_segment | 2 | done | – | 14.3 | 29 |
| h2_last_segment | 2 | 34 | 26–69 | 15.5 | 31 |
| h3_random_segment | 2 | 52 | 35–70 | 15.1 | 30 |
| j1_kdrl | 4 | done | – | 8.4 | 34 |

## 3. Post-hoc suite over saved checkpoints (fills as sweeps complete)

Composite = equal-macro mean of AIME24+25 (avg@32), AMC23 (avg@32), Minerva (avg@3), MATH500 (avg@3) at τ=0.7 / top-p 0.95 / 32,768-token budget. Base student anchor: **0.1453**. One row per method, `mean·N` over seeds whose artifacts are complete at that step; every method completes all 10 checkpoints eventually — empty cells fill as the sweeps grind (~2.5-3 h per checkpoint per GPU pair).

| method | s25 | s50 | s75 | s100 | s125 | s150 | s175 | s200 | s225 | s250 |
|---|---|---|---|---|---|---|---|---|---|---|
| _base student_ | **0.145** (step −1) |  |  |  |  |  |  |  |  |  |
| vanilla | – | – | – | – | – | – | – | – | – | – |
| vanilla_n8 | – | – | – | – | – | – | – | – | – | – |
| a2_coldstart | – | – | – | – | – | – | – | – | – | – |
| b1_skew_kl | – | – | – | – | – | – | – | – | – | – |
| b2_forward_kl | – | – | – | – | – | – | – | – | – | – |
| b3_eopd_gate | – | – | – | – | – | – | – | – | – | – |
| b4_jsd | – | – | – | – | – | – | – | – | – | – |
| b5_k2 | – | – | – | – | – | – | – | – | – | – |
| c1_lsm_topk32_renorm | 0.147·2 | – | – | – | – | – | – | – | – | – |
| c2_quantile_budget | – | – | – | – | – | – | – | – | – | – |
| c3_intersection | – | – | – | – | – | – | – | – | – | – |
| c4_pi_tail_budget ⚠ceded | – | – | – | – | – | – | – | – | – | – |
| d1_tip | – | – | – | – | – | – | – | – | – | – |
| d2_selectkd | – | – | – | – | – | – | – | – | – | – |
| d3_teachability | – | – | – | – | – | – | – | – | – | – |
| e1_pl_rank | – | – | – | – | – | – | – | – | – | – |
| e2_set_coverage ⚠ceded | – | – | – | – | – | – | – | – | – | – |
| e3_zvalue ⚠ceded | – | – | – | – | – | – | – | – | – | – |
| f1_soft_log | – | – | – | – | – | – | – | – | – | – |
| f2_hard_clip | – | – | – | – | – | – | – | – | – | – |
| f3_power | – | – | – | – | – | – | – | – | – | – |
| g1_verified_only | – | – | – | – | – | – | – | – | – | – |
| g2_fire_likelihood | – | – | – | – | – | – | – | – | – | – |
| g4_failure_only | – | – | – | – | – | – | – | – | – | – |
| g5_rgopd_gate ⚠ceded | – | – | – | – | – | – | – | – | – | – |
| h1_first_segment | 0.275·3 | – | – | – | – | – | – | – | – | – |
| h2_last_segment | – | – | – | – | – | – | – | – | – | – |
| h3_random_segment | – | – | – | – | – | – | – | – | – | – |
| j1_kdrl | 0.228·3 | 0.254·3 | 0.270·3 | – | – | – | – | – | – | – |

## 4. Engineering changes that shipped mid-campaign (all value-identical)

### 4.1 `aggregate_dp` metric-count fix (`src/simopd/losses.py`)

DP-1 lanes never compare ranks, so `_signal_quantiles` skipping keys on empty masks was latent until the 4-GPU (DP-2) lanes: a sparse selector leaves a positional bin empty on one rank but not the other, per-rank append counts diverge, and verl raises `must have the same number of values: [5, 4]`. Empty sets now emit zeros — metric emission only, losses untouched:

```python
if x.numel() == 0:
    # every rank must append every key on every micro-batch (Metric.aggregate_dp)
    x = losses.new_zeros(1).float()
...
v = losses[b].detach().float().abs().mean() if b.any() else losses.new_zeros(()).float()
metrics[f"distillation/{name}_absmean_pos{tag}"] = Metric(aggregation=AggregationType.MEAN, value=v)
```

### 4.2 Chunked student top-k (`src/simopd/topk_losses.py`) — the length wall

`torch.topk` over the full `[*, T, 152k]` student log-probs allocates a sort workspace on the scale of its input. Once response lengths saturate the 16k cap (~step 76-110), that workspace is a **17-19 GiB single allocation** stacked on the log-softmax output and update-phase transients — it killed every b3 quad attempt at steps 76-84 through four rollout-pool settings (0.45→0.18) and capped d1/g2 relay legs at ~10 steps. Indices are per-token order statistics with no autograd surface, so slicing the token dimension is value-identical:

```python
def _student_topk_ids(student_log_probs, k, chunk=256):
    outs = [torch.topk(sl, k=k, dim=-1).indices
            for sl in student_log_probs.split(chunk, dim=-2)]
    return torch.cat(outs, dim=-2)
```

CPU equivalence test: index equality across chunk-boundary shapes (T=100/256/777/1000), tie-robust value sets in bf16. All 11 call sites migrated. Runs relaunched on the fixed snapshot: b3×3 (from step 75), d1_s0/s2 (from 100), g2_s2 (from 125); every other lane keeps its original snapshot — the change alters peak memory, not one computed value, so mixed versions stay comparable.

### 4.3 Engine-side memory measures (env-only, fingerprint-recorded)

```bash
# creep/length-wall relaunch recipe (start_wall4.sh callers)
RESUME=force \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EXTRA_HYDRA="trainer.balance_batch=True actor_rollout_ref.rollout.gpu_memory_utilization=0.32" \
bash start_wall4.sh <arm:seed> <gpu_quad>
```

### 4.4 Incident ledger (checkpoint-level losses only)

Every crash recovered from the latest 25-step checkpoint (≤24 steps lost per incident); zero checkpoint contamination across 7 duplicate-run kills (poison-pill fingerprints refused 6 more stale-queue twins at launch). Full per-row history lives in `configs/campaign.tsv` notes.

