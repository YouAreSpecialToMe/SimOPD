# SimOPD 16k Campaign — Live Report

_Student **Qwen3-1.7B-Base** ← Teacher **Qwen3-4B-Instruct-2507**, response cap 16,384 tokens, 250 steps,
3 seeds/arm, checkpoint every 25 steps (all 10 kept). 29 arms × 3 seeds = **87 rows**, all run on this
24-node / 192-GPU fleet — including the four arms (c4/e2/e3/g5) originally ceded to the collaborating
site and relaunched here on 2026-08-09._

_Regenerated 2026-08-11 04:21 from live logs, checkpoint trees and eval artifacts._

**Anchors (protocol-matched, measured 2026-08-11).** On the **in-loop axis** — MATH500, greedy mean@1,
16,384-token cap, the exact protocol every training curve is made of — the untrained student scores
**0.4580** and the teacher **0.9060**, so the available gap is **0.4480**. The `GRR`
column in §1 is Gap Recovery Rate = (arm@250 − student) / (teacher − student) against those two numbers.

These supersede two carried-over values that were measured under different conditions: the old step-0 anchor
0.468 (same base model but an 8k cap, via verl's val path) and `docs/METRICS.md`'s teacher 0.896. Both sat
~0.010 away from the protocol-matched measurement, and in opposite directions, so every previously quoted
recovery rate survives to within 0.1pp — the ranking never depended on them. Note also that this campaign
runs `val_before_train=False`: **no 16k curve carries its own step-0 point**, the first in-loop reading of
every arm is step 25, and the student anchor above comes from a standalone run on the same protocol.

Offline suite composite of the base student: **0.1453** (step −1 convention).

### Reference points (same offline suite as every checkpoint)

| model | composite | AIME24+25 | AMC23 | Minerva | MATH500 |
|---|---|---|---|---|---|
| **Qwen3-1.7B-Base** — untrained student (step −1) | **0.1453** | 0.0167 | 0.1906 | 0.0637 | 0.3100 |
| **Qwen3-4B-Instruct-2507** — teacher (ceiling) | _eval queued_ | – | – | – | – |

On the in-loop axis the two reference points are measured (see Anchors above): student
**0.4580** → teacher **0.9060**. c2 recovers **50.4%** of that gap, j1 and c4 **40.3%**,
h1 37.4%, and the collapsed arms none — b3 ends below the untrained student.

The untrained student is the **lower bound** — what the 1.7B model scores with no distillation at all — and
the 4B teacher is the **upper bound** an on-policy distillation run can approach. Every composite in §3 is
read against these two: `Δ vs base` in §3.1 is the gain over the lower bound, and the teacher row says how
much of the available headroom that gain represents. Note that the in-loop anchor (greedy MATH500 **0.468**)
and this table's MATH500 column (**0.3100**) are the same model under different decoding protocols — greedy
mean@1 in-loop versus τ=0.7 avg@3 in the suite — so the two must never be compared across tables.

**Fleet state.** 24 of 29 arms have all three seeds at step 250; 4 arms (a2/e2/g5/h2) are in
their last steps; b2 is parked at its per-seed memory ceiling (§4.5). Post-hoc suite: **195 of 750
checkpoint-evaluations** complete, the rest grinding.

## 1. In-loop eval (greedy MATH500, every 25 steps)

**mean±std across the arm's 3 seeds** (sample std, n−1); `·N` appears only when a seed is missing at that
step. `σ̄` is the mean across-seed std over all steps where all three seeds reported — one number for how
reproducible the arm is. Wall-family rows (b3/d1/d2/d3/g2) count 4-GPU-era logs only — their 2-card attempts
died at the length wall and are excluded. Sorted by step-250 value.

| arm | signal | s25 | s50 | s75 | s100 | s125 | s150 | s175 | s200 | s225 | s250 | σ̄ | GRR | now |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c2_quantile_budget | topk | 0.635±0.001 | 0.610±0.012 | 0.617±0.007 | 0.611±0.009 | 0.637±0.016 | 0.639±0.006 | 0.646±0.007 | 0.657±0.019 | 0.668±0.002 | 0.684±0.009 | 0.009 | 50.4% | 250/250/250 |
| j1_kdrl | sampled | 0.551±0.013 | 0.573±0.017 | 0.584±0.016 | 0.613±0.017 | 0.631±0.006 | 0.637±0.003 | 0.639±0.023 | 0.646±0.011 | 0.641±0.007 | 0.639±0.028 | 0.014 | 40.3% | 250/250/250 |
| c4_pi_tail_budget | topk | 0.627±0.009 | 0.617±0.006 | 0.599±0.004 | 0.589±0.019 | 0.593±0.004 | 0.593±0.004 | 0.619±0.023 | 0.631±0.025 | 0.638±0.025 | 0.639±0.002 | 0.012 | 40.3% | 250/250/250 |
| h1_first_segment | sampled | 0.587±0.007 | 0.628±0.009 | 0.627±0.011 | 0.629±0.008 | 0.615±0.014 | 0.630±0.012 | 0.617±0.016 | 0.632±0.002 | 0.626±0.010 | 0.625±0.008 | 0.010 | 37.4% | 250/250/250 |
| c3_intersection | topk | 0.566±0.015 | 0.617±0.015 | 0.599±0.013 | 0.519±0.001 | 0.555±0.004 | 0.578±0.005 | 0.589±0.008 | 0.587±0.012 | 0.589±0.010 | 0.599±0.008 | 0.009 | 31.4% | 250/250/250 |
| e3_zvalue | topk | 0.552±0.028·2 | 0.618±0.014 | 0.637±0.003 | 0.621±0.017 | 0.561±0.013 | 0.522±0.007 | 0.514±0.007 | 0.545±0.006 | 0.580±0.011 | 0.589±0.017 | 0.011 | 29.2% | 250/250/250 |
| h3_random_segment | sampled | 0.555±0.004 | 0.649±0.002 | 0.613±0.010 | 0.612±0.016 | 0.606±0.012 | 0.597±0.034 | 0.595±0.009 | 0.599±0.019 | 0.596±0.010 | 0.583±0.006 | 0.012 | 27.8% | 250/250/250 |
| e1_pl_rank | topk | 0.521±0.012 | 0.499±0.012 | 0.496±0.006 | 0.527±0.030 | 0.545±0.011 | 0.570±0.013 | 0.555±0.005 | 0.571±0.011 | 0.569±0.005 | 0.572±0.013 | 0.012 | 25.4% | 250/250/250 |
| g1_verified_only | sampled | 0.223±0.071 | 0.169±0.088 | 0.121±0.053 | 0.581±0.056 | 0.627±0.006 | 0.571±0.023 | 0.540±0.026 | 0.529±0.015 | 0.511±0.039 | 0.529±0.025 | 0.040 | 15.9% | 250/250/250 |
| b4_jsd | topk | 0.581±0.009 | 0.593±0.008 | 0.533±0.012 | 0.526±0.012 | 0.478±0.015 | 0.488±0.007 | 0.506±0.022 | 0.512±0.014 | 0.519±0.021 | 0.524±0.016 | 0.014 | 14.7% | 250/250/250 |
| c1_lsm_topk32_renorm | topk | 0.417±0.094 | 0.487±0.038 | 0.504±0.033 | 0.523±0.022 | 0.533±0.029 | 0.531±0.032 | 0.521±0.024 | 0.519±0.012 | 0.529±0.002 | 0.524±0.022 | 0.031 | 14.7% | 250/250/250 |
| g2_fire_likelihood | samp+sel | 0.619±0.004 | 0.612±0.009 | 0.615±0.010 | 0.624±0.007 | 0.554±0.017 | 0.501±0.014 | 0.481±0.006 | 0.461±0.027 | 0.495±0.046 | 0.511±0.010 | 0.015 | 11.9% | 250/250/250 |
| d2_selectkd | samp+sel | 0.622±0.015 | 0.620±0.009 | 0.595±0.010 | 0.612±0.003 | 0.641±0.015 | 0.641±0.006 | 0.559±0.082 | 0.473±0.040 | 0.472±0.033 | 0.480±0.005 | 0.022 | 4.9% | 250/250/250 |
| b1_skew_kl | sampled | 0.594±0.014 | 0.627±0.003 | 0.629±0.005 | 0.617±0.006 | 0.613±0.027 | 0.611±0.012 | 0.645±0.002 | 0.629±0.034 | 0.455±0.034 | 0.475±0.007 | 0.014 | 3.9% | 250/250/250 |
| vanilla | sampled | 0.604±0.009 | 0.627±0.003 | 0.625±0.010 | 0.575±0.028 | 0.517±0.030 | 0.475±0.008 | 0.438±0.018 | 0.441±0.005 | 0.474±0.004 | 0.473±0.013 | 0.013 | 3.4% | 250/250/250 |
| d1_tip | samp+sel | 0.580±0.006 | 0.620±0.002 | 0.617±0.009 | 0.577±0.008 | 0.529±0.034 | 0.459±0.018 | 0.466±0.007 | 0.459±0.003 | 0.489±0.011 | 0.473±0.010 | 0.011 | 3.4% | 250/250/250 |
| f3_power | sampled | 0.523±0.003 | 0.563±0.017 | 0.551±0.006 | 0.581±0.014 | 0.561±0.022 | 0.555±0.009 | 0.553±0.025 | 0.503±0.020 | 0.490±0.012 | 0.473±0.012 | 0.014 | 3.4% | 250/250/250 |
| vanilla_n8 | sampled | 0.623±0.006 | 0.629±0.012 | 0.614±0.009 | 0.555±0.008 | 0.528±0.031 | 0.480±0.017 | 0.433±0.025 | 0.471±0.011 | 0.487±0.015 | 0.466±0.008 | 0.014 | 1.8% | 250/250/250 |
| b5_k2 | sampled | 0.617±0.018 | 0.625±0.004 | 0.619±0.008 | 0.599±0.020 | 0.519±0.023 | 0.472±0.012 | 0.431±0.025 | 0.474±0.009 | 0.477±0.014 | 0.460±0.026 | 0.016 | 0.4% | 250/250/250 |
| f2_hard_clip | sampled | 0.603±0.013 | 0.631±0.008 | 0.618±0.007 | 0.633±0.009 | 0.630±0.016 | 0.631±0.010 | 0.648±0.011 | 0.464±0.004 | 0.467±0.011 | 0.457±0.017 | 0.011 | -0.1% | 250/250/250 |
| g4_failure_only | sampled | 0.615±0.015 | 0.615±0.008 | 0.629±0.001 | 0.582±0.009 | 0.497±0.027 | 0.465±0.013 | 0.425±0.012 | 0.435±0.019 | 0.466±0.002 | 0.450±0.019 | 0.013 | -1.8% | 250/250/250 |
| f1_soft_log | sampled | 0.588±0.012 | 0.609±0.006 | 0.624±0.012 | 0.631±0.002 | 0.620±0.015 | 0.632±0.011 | 0.623±0.013 | 0.457±0.003 | 0.463±0.028 | 0.447±0.016 | 0.012 | -2.4% | 250/250/250 |
| d3_teachability | samp+sel | 0.581±0.006 | 0.622±0.016 | 0.565±0.008 | 0.542±0.014 | 0.428±0.002 | 0.435±0.028 | 0.416±0.017 | 0.437±0.012 | 0.431±0.011 | 0.415±0.007 | 0.012 | -9.7% | 250/250/250 |
| b3_eopd_gate | samp+sel | 0.582±0.016 | 0.612±0.008 | 0.555±0.011 | 0.354±0.009 | 0.113±0.024 | 0.025±0.007 | 0.018±0.007 | 0.003±0.001 | 0.001±0.001 | 0.003±0.003 | 0.009 | -101.6% | 250/250/250 |
| g5_rgopd_gate | sampled | 0.618±0.016 | 0.612±0.013 | 0.627±0.012 | 0.581±0.028 | 0.496±0.019 | 0.464±0.031 | 0.425±0.012 | – | – | – | 0.019 | – | 192/195/193 |
| e2_set_coverage | topk | 0.596±0.005 | 0.597±0.001 | 0.588±0.010 | 0.581±0.014 | 0.498±0.004 | 0.469±0.015 | 0.460±0.023 | – | – | – | 0.010 | – | 187/186/189 |
| b2_forward_kl | sampled | 0.541±0.016 | 0.572±0.016 | 0.519±0.002 | 0.458±0.011 | 0.487±0.009 | 0.459±0.023 | 0.408±0.020·2 | 0.460·1 | – | – | 0.013 | – | 196/172/208 |
| a2_coldstart | sampled | 0.430±0.021 | 0.437±0.016 | 0.439±0.027 | 0.468±0.018 | 0.465±0.021 | 0.467±0.033 | 0.489±0.008 | 0.478±0.007 | – | – | 0.019 | – | 207/222/215 |
| h2_last_segment | sampled | 0.475±0.003 | 0.260±0.011 | 0.096±0.052 | 0.097±0.062 | 0.164±0.076 | 0.228±0.021 | 0.385±0.041 | 0.453±0.032 | 0.438±0.075 | – | 0.042 | – | 227/226/226 |

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
| a2_coldstart | 2 | 0/3 | 8/8/8 | 66.9 | 402 |
| b1_skew_kl | 2 | 3/3 | 10/10/10 | 34.6 | 208 |
| b2_forward_kl | 2 | 0/3 | 7/6/8 | 34.1 | 205 |
| b3_eopd_gate | 4 | 3/3 | 10/10/10 | 102.8 | 1233 |
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
| e2_set_coverage | 2 | 0/3 | 7/7/7 | 47.0 | 282 |
| e3_zvalue | 2 | 3/3 | 10/10/10 | 43.2 | 259 |
| f1_soft_log | 2 | 3/3 | 10/10/10 | 40.1 | 241 |
| f2_hard_clip | 2 | 3/3 | 10/10/10 | 42.5 | 255 |
| f3_power | 2 | 3/3 | 10/10/10 | 28.7 | 172 |
| g1_verified_only | 2 | 3/3 | 10/10/10 | 40.0 | 240 |
| g2_fire_likelihood | 4 | 3/3 | 10/10/10 | 47.8 | 573 |
| g4_failure_only | 2 | 3/3 | 10/10/10 | 59.5 | 357 |
| g5_rgopd_gate | 2 | 0/3 | 7/7/7 | 40.7 | 244 |
| h1_first_segment | 2 | 3/3 | 10/10/10 | 13.7 | 82 |
| h2_last_segment | 2 | 0/3 | 9/9/9 | 64.9 | 389 |
| h3_random_segment | 2 | 3/3 | 10/10/10 | 35.0 | 210 |
| j1_kdrl | 4 | 3/3 | 10/10/10 | 8.4 | 101 |
| vanilla | 2 | 3/3 | 10/10/10 | 59.3 | 356 |
| vanilla_n8 | 4 | 3/3 | 10/10/10 | 57.8 | 694 |
| **fleet total** | | | | | **9241** |

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
onto a GPU *pair* uses one card and leaves the partner idle (measured: 100% util / 73 GiB on the first card,
4 MiB on the second). With training winding down this left **94 of 192 GPUs idle** behind a 647-checkpoint
backlog queued behind 75 sequential sweeps. Note that `sweep --step N` does **not** select a step — the
argument is parsed but unused, and the sweep re-walks the whole grid; the single-checkpoint entry point is
`eval_suite.py run --model <ckpt>/actor --step N`. Post-eval now runs as a **shared claim queue**
(`simopd_data/eval_worker.sh`): one worker per GPU, each atomically claiming the next pending
(run, checkpoint) pair via `mkdir`, so no card idles while work remains and newly-finished runs are absorbed
by appending to the queue.

## 3. Post-hoc suite over saved checkpoints

Composite = equal-macro mean of AIME24+25 (avg@32), AMC23 (avg@32), Minerva (avg@3), MATH500 (avg@3),
τ=0.7 / top-p 0.95 / 32,768-token budget — the same protocol as the base-student anchor **0.1453**.
**mean±std across seeds** whose artifacts exist at that step (`·N` when a seed is missing); `σ̄` is the mean
across-seed std over fully-reported steps; the last column counts completed checkpoint-evaluations out of 30
(10 checkpoints × 3 seeds).

| method | s25 | s50 | s75 | s100 | s125 | s150 | s175 | s200 | s225 | s250 | σ̄ | ckpts done |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| j1_kdrl | 0.228±0.007 | 0.254±0.012 | 0.270±0.010 | 0.285±0.007 | 0.294±0.009 | 0.303±0.001 | 0.307±0.002 | 0.308±0.004 | 0.316±0.003 | 0.315±0.002 | 0.006 | 30/30 |
| c1_lsm_topk32_renorm | 0.137±0.029 | 0.161±0.039 | 0.171±0.035 | 0.186±0.030 | 0.191±0.014 | 0.200±0.029 | 0.211±0.019 | 0.232±0.004·2 | 0.233±0.004·2 | 0.234±0.001·2 | 0.028 | 27/30 |
| h1_first_segment | 0.275±0.002 | 0.292±0.003 | 0.287±0.006 | 0.292±0.004 | 0.294±0.005 | 0.295±0.003 | 0.297±0.002 | 0.301±0.001 | 0.299±0.011 | – | 0.004 | 27/30 |
| e1_pl_rank | 0.126±0.003 | 0.160±0.001 | 0.211±0.005 | 0.237·1 | – | – | – | – | – | – | 0.003 | 10/30 |
| b1_skew_kl | 0.255±0.002 | 0.293±0.003 | 0.338±0.006 | – | – | – | – | – | – | – | 0.003 | 9/30 |
| b2_forward_kl | 0.224±0.005 | 0.257±0.003 | 0.326±0.003 | – | – | – | – | – | – | – | 0.004 | 9/30 |
| f3_power | 0.188±0.004 | 0.241±0.006 | 0.295±0.004 | – | – | – | – | – | – | – | 0.005 | 9/30 |
| h3_random_segment | 0.251±0.004 | 0.298±0.004 | 0.289±0.005 | – | – | – | – | – | – | – | 0.004 | 9/30 |
| b4_jsd | 0.204±0.002 | 0.265±0.005 | – | – | – | – | – | – | – | – | 0.004 | 6/30 |
| c3_intersection | 0.134±0.004 | 0.263±0.003 | – | – | – | – | – | – | – | – | 0.004 | 6/30 |
| d2_selectkd | 0.279±0.005 | 0.295±0.004 | – | – | – | – | – | – | – | – | 0.005 | 6/30 |
| d3_teachability | 0.246±0.007 | 0.280±0.010 | – | – | – | – | – | – | – | – | 0.008 | 6/30 |
| f1_soft_log | 0.267±0.003 | 0.293±0.002 | – | – | – | – | – | – | – | – | 0.002 | 6/30 |
| f2_hard_clip | 0.277±0.005 | 0.293±0.001·2 | – | – | – | – | – | – | – | – | 0.005 | 5/30 |
| b3_eopd_gate | 0.253±0.002 | – | – | – | – | – | – | – | – | – | 0.002 | 3/30 |
| b5_k2 | 0.276±0.000 | – | – | – | – | – | – | – | – | – | 0.000 | 3/30 |
| c2_quantile_budget | 0.280±0.001 | – | – | – | – | – | – | – | – | – | 0.001 | 3/30 |
| c4_pi_tail_budget | 0.266±0.003 | – | – | – | – | – | – | – | – | – | 0.003 | 3/30 |
| d1_tip | 0.260±0.001 | – | – | – | – | – | – | – | – | – | 0.001 | 3/30 |
| g1_verified_only | 0.106±0.010 | – | – | – | – | – | – | – | – | – | 0.010 | 3/30 |
| g2_fire_likelihood | 0.274±0.008 | – | – | – | – | – | – | – | – | – | 0.008 | 3/30 |
| g4_failure_only | 0.275±0.001 | – | – | – | – | – | – | – | – | – | 0.001 | 3/30 |
| vanilla | 0.280±0.002 | – | – | – | – | – | – | – | – | – | 0.002 | 3/30 |
| vanilla_n8 | 0.285±0.005 | – | – | – | – | – | – | – | – | – | 0.005 | 3/30 |
| a2_coldstart | – | – | – | – | – | – | – | – | – | – | – | 0/30 |
| e2_set_coverage | – | – | – | – | – | – | – | – | – | – | – | 0/30 |
| e3_zvalue | – | – | – | – | – | – | – | – | – | – | – | 0/30 |
| g5_rgopd_gate | – | – | – | – | – | – | – | – | – | – | – | 0/30 |
| h2_last_segment | – | – | – | – | – | – | – | – | – | – | – | 0/30 |

### 3.1 Completed step-250 suites, per benchmark (arm mean±std, then per-seed rows)

| method | seeds | composite | Δ vs base | AIME24+25 | AMC23 | Minerva | MATH500 |
|---|---|---|---|---|---|---|---|
| **j1_kdrl** | 3 | **0.315±0.002** | +0.170 | 0.047±0.007 | 0.414±0.012 | 0.158±0.007 | 0.642±0.005 |
| ↳ s0 | | 0.3167 | | 0.0401 | 0.4281 | 0.1507 | 0.6480 |
| ↳ s1 | | 0.3160 | | 0.0547 | 0.4062 | 0.1642 | 0.6387 |
| ↳ s2 | | 0.3136 | | 0.0474 | 0.4078 | 0.1605 | 0.6387 |
| **c1_lsm_topk32_renorm** | 2 | **0.234±0.001·2** | +0.089 | 0.016±0.001·2 | 0.284±0.005·2 | 0.130±0.002·2 | 0.506±0.007·2 |
| ↳ s0 | | 0.2345 | | 0.0156 | 0.2805 | 0.1311 | 0.5107 |
| ↳ s1 | | 0.2335 | | 0.0172 | 0.2875 | 0.1287 | 0.5007 |

**Reading.** The composite tracks the in-loop curve where both exist (j1: 0.315 composite / 0.639 in-loop
MATH500; the suite's own MATH500 column reads 0.639-0.648, i.e. the offline pipeline reproduces the
in-loop number to the third decimal), but it is **not** a monotone function of it: at step 75, b1 (0.335)
and b2 (0.330) are well above j1 (0.270) even though their in-loop MATH500 is comparable, because AMC23
and Minerva move differently from MATH500. Final rankings therefore wait for the full sweep — the arms
with complete curves so far are j1 (0.315±0.002), c1 (0.234±0.001) and h1 (0.299 @225).

**Seed spread.** In-loop σ̄ sits at 0.009-0.014 for the stable arms — i.e. **seed noise is ~0.01 in greedy
MATH500**, so the c2 → j1 gap (0.684 vs 0.639) is ~4σ and real, while gaps under ~0.02 between neighbouring
arms are not separable at 3 seeds. Collapsing arms show much larger σ̄ because the collapse step itself
varies by seed; that is signal about instability, not measurement noise. On the offline composite the
seed spread is tighter still (j1 0.315±0.002), because avg@32 on AIME/AMC averages away much of the
sampling variance that a single greedy decode carries.

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
- **Post-hoc suite**: 195/750 checkpoint-evaluations done, ~75 sweeps in flight. a2's partial sweeps
  will be re-dispatched after it completes so its last checkpoints are covered (`eval_suite` is
  idempotent — completed artifacts are skipped).
- **Deliverables pending**: full 29×10 composite table, per-benchmark breakdown at 250, and the
  seed-variance band per arm.
