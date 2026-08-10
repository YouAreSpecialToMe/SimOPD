# Why the late-training score falls — a termination collapse, not a reasoning collapse

_Analysis of the 16k campaign, written 2026-08-11 from live training logs, checkpoint evals, three
purpose-run greedy diagnostics and three response-text dumps. Read-only: no arm config, loss kernel or
launcher was touched._

**Claim.** The late-training drop that `vanilla` and ~17 other arms show is not the model getting
worse at mathematics. It is the model losing the ability to **stop**. The response-length
distribution drifts right until essentially every rollout hits the 16,384-token cap; a capped
response is scored on whatever `\boxed{}` it managed to emit, so the score becomes
`P(finish) × accuracy`, and `P(finish)` is what collapses. Arms that keep their length distribution
inside the cap — `c2_quantile_budget` above all — keep their score. On the matched subpopulation
where termination behaviour does **not** change, the same `vanilla` run measurably *improves*
between step 100 and step 250.

## 1. The direct evidence: three greedy diagnostics on the same 500 problems

MATH500, greedy mean@1, 16,384-token budget — the exact protocol the in-loop curve is made of, so
these reproduce it (`vanilla` in-loop val was 0.606 @100 and 0.482 @250 against 0.604 / 0.470 here).

| checkpoint | score | P(finish) | acc \| finish | acc \| truncated | len mean | len median |
|---|---|---|---|---|---|---|
| student (untrained) | **0.458** | 0.832 | 0.550 | 0.000 | 3219 | 553 |
| teacher Qwen3-4B-Instruct | **0.906** | 0.988 | 0.913 | 0.333 | 1642 | 694 |
| vanilla@100 | **0.604** | 0.666 | 0.844 | 0.126 | 6802 | 2591 |
| vanilla@250 | **0.470** | 0.040 | 0.900 | 0.452 | 15760 | 16384 |
| c2@250 | **0.686** | 0.900 | 0.760 | 0.020 | 4505 | 2400 |

Read the `acc | finish` column down the vanilla trajectory: **0.550 → 0.844 → 0.900**. Conditional on
producing a terminated answer, the student gets monotonically better for the whole 250 steps — while its
reported score goes 0.458 → 0.604 → **0.470**. Distillation worked. The reported metric hides it because
the model stopped delivering answers in the budget it is scored in.

Two further things in that table are worth stopping on.

**The teacher is short.** Median 694 tokens, 1.2 % truncated. The base student is shorter still
(median 553). Nothing in the distillation target pulls the student towards 16,384 tokens — at step 250
`vanilla` writes **24× the teacher's median**. The length explosion is produced by the optimisation,
not imitated from the teacher.

**A truncated response is not an unanswered one.** verl's MATH scorer takes the *last* `\boxed{}`
in the text (`last_boxed_only_string`, `rfind`), so 45.2 % of `vanilla@250`'s capped responses still
score correct: the model reached its answer and then kept generating instead of emitting EOS. That
45.2 % is a lower bound on 'the answer was found' — a later, wrong box overwrites an earlier, right one.

## 2. Pairing removes the difficulty confound

The unpaired conditional accuracies above are not comparable across steps: at step 100 only the hard
problems run long, at step 250 nearly everything does, so the two 'truncated' buckets are different
populations. Both diagnostics cover the same 500 problems, so pair them.

**Truncation is a one-way ratchet.**

| | finished @250 | truncated @250 |
|---|---|---|
| finished @100 | 20 | 313 |
| truncated @100 | 0 | 167 |

Not one problem out of 500 went from truncated back to finished. The step-250 finishing set is a
strict subset of the step-100 one: 20 problems versus 333.

**And the loss lives entirely in the problems whose termination changed.**

| matched subpopulation | n | acc @100 | acc @250 | Δ |
|---|---|---|---|---|
| finished at both steps | 20 | 0.950 | 0.900 | **-0.050** |
| truncated at both steps | 167 | 0.126 | 0.251 | **+0.126** |
| finished @100 → truncated @250 | 313 | 0.837 | 0.559 | **-0.278** |
| truncated @100 → finished @250 | 0 | – | – | – |

On the 167 problems that were already running past the cap at step 100 — difficulty held fixed by
construction — the step-250 model is **better**, +0.126. On the 20 it still finishes, it is flat
(n=20, noise). All of the damage is in the 313 problems it used to finish and no longer does.

The 106 problems solved at step 100 and lost at step 250: **99.1%** are
truncated at step 250, only 8.5% were at step 100, and their median response
length goes from 1408 to 16384 tokens — a
12× blow-up on problems the model already knew
how to do in under 1,500 tokens.

## 3. Why `c2` is immune — and it is not because it reasons better

Same 500 problems, same step 250, `vanilla` vs `c2_quantile_budget`:

| subpopulation | n | vanilla | c2 |
|---|---|---|---|
| both finished | 20 | 0.900 | 0.850 |
| vanilla truncated, c2 finished | 430 | 0.481 | **0.756** |

Where both models finish, they are tied (0.900 vs 0.850, n=20).
c2's entire +0.216 advantage comes from the 430 problems where it stops and vanilla does not.

Note also that at step 100, *before* the collapse, `vanilla`'s accuracy given a finish (0.844) is
**higher** than `c2`'s at step 250 (0.760). Nothing about c2's objective makes the student a better
mathematician. It makes it a student that finishes.

## 4. What the model is actually writing

Everything above is arithmetic on lengths and scores. This section reads the responses.

Neither pipeline kept them: verl's validation path runs at `log_val_generations=0` with
`validation_data_dir=null`, so the in-loop generations were scored and discarded, and
`scripts/eval_offline.py` holds the completions only long enough to score them before writing a row
that keeps `resp_len` and `correct` but not the text. The evaluations are greedy at temperature 0,
so they regenerate: `eval_offline_textdump.py` imports `eval_offline`'s loader, `INSTRUCTION` and
template rather than copying them, and reproduces the archived step-250 diagnostic on **11 of 12**
problems (`resp_len` and `truncated` identical). The one that differs now terminates instead of
truncating — vLLM batch-composition nondeterminism at a batch of 12 versus 500, not a different model.

The 12 problems were chosen from the paired diagnostic in §2: eight that `vanilla` solved at
step 100 in **276–500 tokens** and lost at step 250, and four it truncates at step 250 yet still
scores correct. `c2` finishes all twelve.

| model | mean length | truncated | correct | distinct 20-grams | exact tail cycle |
|---|---|---|---|---|---|
| untrained student | 257 | 0/12 | **12/12** | 1.000 | 0/12 |
| vanilla OPD @250 | 15053 | 11/12 | **5/12** | 0.198 | 4/12 |
| teacher | 246 | 0/12 | **12/12** | 1.000 | 0/12 |

Both endpoints answer these problems in about **250 tokens with zero repetition and 12/12 correct**.
The model trained between them uses 60× more tokens and gets 5.

### 4.1 The failure is a terminal loop, not degraded reasoning

`test/prealgebra/2019.json` — *What is $\frac{9}{2}$ expressed as a decimal?* The model produces a clean, correct
solution with the teacher's own section headings, reaches `\boxed{4.5}` at token ~407, and then
does this until the cap, 16,384 tokens later:

```
**Final Answer:** $\boxed{4.5}$    Yes.             ---
**Final Answer:** $\boxed{4.5}$    (Yes.)           ---
**Final Answer:** $\boxed{4.5}$    (We are done.)   ---
```

A 17-word cycle, repeated 154 times. The loop body says *"(We are done.)"* — the model has the
answer, states that it is finished, and cannot emit the end-of-sequence token.

That is not one problem's quirk. Across the 11 blown-up responses the model emits the same
final-answer line **696–1271 times** (median 864).

### 4.2 It reaches the answer in the budget the untrained student already had

| | tokens |
|---|---|
| `vanilla@250` spends **reaching** its answer | 445 (median 400) |
| the untrained student needs for the whole answer | 265 (median 269) |
| `vanilla@250` spends **after** its answer | 15,939 |

**97.3% of every blown-up response is the terminal loop.** The
reasoning phase did not get longer or worse — it is roughly the length it always was. What was
appended is 16,000 tokens of a model unable to stop, and that is also where the campaign's compute
went — see `training-dynamics.md` §2.1: 916 s/step for capped arms against 450.

### 4.3 The grader then throws the correct answer away

verl's MATH scorer takes the **last** `\boxed` in the text (`last_boxed_only_string`, `rfind`). After
hundreds of loop iterations, the last one is whatever the 16,384-token cap happened to slice through.

| problem | gold | first `\boxed` | count | what went wrong at the end | scored |
|---|---|---|---|---|---|
| `algebra/1529.json` | `1` | `1` | 963 | drifted | **0** |
| `algebra/2214.json` | `4` | `4` | 980 | – | ✓ |
| `algebra/518.json` | `\frac 34` | `\dfrac{3}{4}` | 696 | cut open by the cap | **0** |
| `counting_and_probability/761.json` | `3` | `3` | 1271 | cut open by the cap | **0** |
| `prealgebra/1233.json` | `10` | `10\%` | 829 | cut open by the cap | **0** |
| `prealgebra/1298.json` | `15x - 80` | `15x - 80` | 3 | – | ✓ |
| `prealgebra/1317.json` | `4` | `4` | 864 | cut open by the cap | **0** |
| `prealgebra/192.json` | `2` | `2` | 1120 | cut open by the cap | **0** |
| `prealgebra/1922.json` | `1251` | `1251` | 836 | cut open by the cap | **0** |
| `prealgebra/1973.json` | `4` | `4` | 803 | – | ✓ |
| `prealgebra/2019.json` | `4.5` | `4.5` | 858 | – | ✓ |
| `prealgebra/2037.json` | `550` | `550` | 974 | – | ✓ |

**The first `\boxed` matches the gold answer on 11 of 12 problems. The scorer credits 5.** 6 responses score zero because the cap cut the *final* `\boxed{` open, so
`last_boxed_only_string` returns `None` — discarding several hundred correct boxed answers earlier in
the same text. One more (`algebra/1529`) drifted from `1` to `-1` during the loop: the cycle is not
always a fixed point.

This is worth separating from the model's own failure. The *model* fails by not stopping; the
*measurement* then converts a solved problem into a zero. The step-250 score is therefore a floor on
what the checkpoint knows, and the reported collapse is deeper than the capability change behind it.

### 4.4 Distillation worked — on everything except stopping

| surface signature | untrained student | vanilla@250 | teacher |
|---|---|---|---|
| fake ```python block with hallucinated output | 3/12 | 0/12 | 0/12 |
| teacher's `### Step 1` / `**Final Answer` template | 1/12 | 12/12 | 7/12 |

The base student answers by writing a Python block and hallucinating its output — a pretraining
artifact. After 250 steps of OPD that habit is gone and the teacher's format is in its place, on
**12 of 12** problems, which is more consistently than the teacher itself uses it (7 of 12). On four
problems `vanilla@250` opens with the teacher's exact first sentence; the base student never does.

So the distillation target transferred. What the student over-learned is the *closing template* —
and the closing template is precisely the string it loops on. It learned `**Final Answer:**
$\boxed{X}$` as a high-probability attractor and did not learn the one token that comes after it.

### 4.5 The supervision for stopping disappears from the training signal

Here is the mechanism that ties the text back to the cross-arm statistics. In on-policy distillation every token of the
student's own rollout carries gradient, but **a truncated rollout contains no end-of-sequence token
at all**. So the density of stop-supervision is `(1 − clip_ratio) / mean_length` — one EOS per
terminated sequence, spread over every token the batch trains on:

| arm | step 25 | step 50 | step 100 | step 150 | step 200 | step 250 |
|---|---|---|---|---|---|---|
| `c1_lsm_topk32_renorm` | 0.6881 | 0.7598 | 0.8953 | 0.7654 | 0.8104 | 0.7416 |
| `j1_kdrl` | 0.7317 | 0.9103 | 0.7357 | 0.5120 | 0.2858 | 0.2236 |
| `c2_quantile_budget` | 0.5636 | 0.0614 | 0.1029 | 0.1089 | 0.1090 | 0.1148 |
| `c4_pi_tail_budget` | 0.7506 | 0.0882 | 0.1073 | 0.1138 | 0.1142 | 0.1119 |
| `b1_skew_kl` | 0.5878 | 0.0506 | 0.1009 | 0.1111 | 0.1057 | 0.0056 |
| `d2_selectkd` | 0.5416 | 0.0217 | 0.0894 | 0.0935 | 0.0000 | 0.0002 |
| `vanilla` | 0.5336 | 0.0234 | 0.0721 | 0.0002 | 0.0002 | 0.0003 |
| `b5_k2` | 0.5760 | 0.0272 | 0.0801 | 0.0000 | 0.0002 | 0.0003 |
| `d3_teachability` | 0.5109 | 0.1325 | 0.0635 | 0.0002 | 0.0002 | 0.0000 |
| `b3_eopd_gate` | 0.5255 | 0.0716 | 0.0073 | 0.0000 | 0.0005 | 0.0000 |

EOS tokens per 1,000 training tokens. `vanilla` goes from 0.5336 to 0.000320 — a
**1670× collapse**. `c2` falls 4.9× and stops.

Across all 29 arms, r(log₁₀ terminal EOS density, final score) = **+0.653** and against the
fall from peak **-0.574** — a tighter fit than raw truncation gives.

**This is a reframing, not an independent measurement**: the quantity is an algebraic transform of
`clip_ratio` and `response_length/mean`, so it cannot confirm the account on its own. What it adds is
the right unit. A run whose rollouts stop showing the model how to stop is a run whose gradient has
nothing left to say about EOS, and the loop in §4.1 is what fills the gap. The direction is
self-reinforcing — longer rollouts mean fewer EOS demonstrations mean longer rollouts — which is what
the one-way ratchet in §2 looks like from the training side.

The step is still a hypothesis in one place: nothing here shows *why* the drift starts, only why it
does not stop once started. §6's knob (the policy-gradient form) is the best available candidate for
the trigger, and separating trigger from ratchet needs an intervention run, not more of this data.

## 5. The same story across all 29 arms

Terminal truncation rate is the single strongest cross-arm predictor of the fall:

- Pearson r(terminal truncation, fall from peak) = **+0.588**, Spearman ρ = +0.667 (n=29 arms)
- Pearson r(terminal truncation, final score) = **-0.617**, Spearman ρ = -0.750

| cap binding at the last step | arms | mean fall from peak | mean final | mean length | mean entropy |
|---|---|---|---|---|---|
| no | 11 | **0.030** | 0.586 | 7159 | 1.157 |
| yes | 18 | **0.169** | 0.439 | 16078 | 0.284 |

**The within-run test is the decisive one.** For every (arm, seed) trajectory that crossed from the
slack regime (<50 % of rollouts truncated) into the capped regime, compare the in-loop score at the
last slack step against the score at the end. That makes each run its own control.

| arm | seeds | crossing step | score before | score at end | Δ | PG form |
|---|---|---|---|---|---|---|
| `b3_eopd_gate` | 3 | 91 | 0.555 | 0.003 | **-0.553** | yes |
| `f2_hard_clip` | 3 | 188 | 0.648 | 0.457 | **-0.191** | yes |
| `g5_rgopd_gate` | 3 | 106 | 0.581 | 0.425 | **-0.157** | yes |
| `vanilla_n8` | 3 | 98 | 0.614 | 0.466 | **-0.148** | yes |
| `b5_k2` | 3 | 105 | 0.599 | 0.460 | **-0.139** | no |
| `g4_failure_only` | 3 | 106 | 0.582 | 0.450 | **-0.132** | yes |
| `d2_selectkd` | 3 | 179 | 0.610 | 0.480 | **-0.130** | yes |
| `d3_teachability` | 3 | 116 | 0.542 | 0.415 | **-0.127** | yes |
| `e2_set_coverage` | 3 | 119 | 0.581 | 0.460 | **-0.121** | no |
| `d1_tip` | 3 | 104 | 0.577 | 0.473 | **-0.104** | yes |
| `vanilla` | 3 | 105 | 0.575 | 0.473 | **-0.102** | yes |
| `g1_verified_only` | 3 | 147 | 0.597 | 0.529 | **-0.068** | yes |
| `f1_soft_log` | 3 | 198 | 0.515 | 0.447 | **-0.068** | yes |
| `g2_fire_likelihood` | 3 | 136 | 0.554 | 0.511 | **-0.043** | yes |
| `h2_last_segment` | 3 | 31 | 0.475 | 0.438 | **-0.037** | yes |
| `b1_skew_kl` | 3 | 222 | 0.493 | 0.475 | **-0.017** | yes |
| `b2_forward_kl` | 2 | 155 | 0.448 | 0.441 | **-0.007** | no |

**17 of 17 arms fell after crossing. Zero rose.** Mean Δ -0.126, median -0.121.

## 6. Which knob predicts it

`USE_POLICY_GRADIENT` defaults to `True` (`scripts/run_opd_baseline.sh:76`); ten arms set it to `False`.
In the PG form the per-token distillation loss is fed in as an **advantage** —
`advantages=-distillation_losses.detach()` in `verl/verl/trainer/distillation/losses.py:283` — through a
PPO surrogate with `loss_agg_mode=token-mean` and no baseline or whitening. In the `False` form the
distillation loss is back-propagated directly as a supervised loss.

| | cap not binding | cap binding |
|---|---|---|
| supervised form (`USE_POLICY_GRADIENT=False`) | 7 | 3 |
| PG / advantage form (default `True`) | 4 | 15 |

Fisher exact, two-sided: **p = 0.017**. The advantage form is where the length runaway lives.

`SIMOPD_KEEP_SAMPLED=1` (the five arms that add the sampled token back onto a top-k objective) is
5/5 capped with the largest mean fall of any group, but with only five
arms the exact test is not significant (p = 0.126) — suggestive, not established.

**A controlled pair.** `b5_k2` and `j1_kdrl` run the *same* loss (`k2_kdrl`) in the *same* supervised
form (`USE_POLICY_GRADIENT=False`). `b5_k2` ends at 99.5 % truncation, 16,338 tokens, −0.165 from peak.
`j1_kdrl` adds `USE_TASK_REWARDS=True` and scales the distillation coefficient down 500× (1.0 → 0.002),
and ends at 9.6 % truncation, 4,067 tokens, −0.007 from peak. A verifier reward scores a capped rollout
zero, which is exactly the pressure the pure distillation objective lacks.

## 7. What this is *not*

**Not an entropy collapse.** Capped arms do end at lower policy entropy (mean 0.284 vs 1.157), but entropy is neither necessary nor
sufficient: `h1_first_segment` (0.077, fall 0.007), `j1_kdrl` (0.156, fall 0.007), `e3_zvalue` (0.172, fall 0.049) all end at entropy *below* `vanilla`'s 0.240 while keeping their length distribution inside
the cap — and they keep their score. Low entropy with a bounded length distribution is harmless.

**Not a teacher-following artefact.** See §1 — the teacher is 24× shorter at the median.

**Not established: which moves first.** At the 25-step logging resolution the lead–lag test is
uninformative — contemporaneous r(Δtruncation, Δscore) = −0.264, but the two lagged correlations are
weak and near-symmetric (−0.074 truncation-leads vs −0.109 score-leads, n=659 step pairs). The
directional evidence comes from the paired diagnostic in §2, not from the time series.

## 8. Honest exceptions

- **1 arm falls ≥0.10 without the cap binding**: `f3_power` (fall 0.107, truncation 0.010, length 9523). A second failure mode exists and this analysis does not explain it.
- **`a2_coldstart` and `h2_last_segment` are capped from steps 31 and 37** — before they ever had a
  healthy phase — and fall only 0.011 and 0.037. They peak at 0.489 and 0.475, so there was nothing to
  lose. They are consistent with the account but carry no information for it.
- **`b2_forward_kl` is a supervised-form arm that did cap** (0.867). It is also the arm parked mid-run,
  so its curve is short and its last point is step 200.
- The diagnostics in §1–§3 are **one seed** (s0) at three checkpoints, run greedily. The cross-arm
  statistics in §5–§6 cover all 29 arms × 3 seeds. The mechanism claim rests on the former; the
  prevalence claim rests on the latter.

## 9. Per-arm reference

`lock` = first step after which ≥90 % of rollouts hit the cap for the rest of the run. `truncation`,
`length` and `entropy` here are **training-rollout** metrics at the last logged step (sampled,
temperature 1.0) — not the greedy val numbers of §1–§3, which is why e.g. `c2` reads 8,579 tokens here
and 4,505 there. `peak`/`final`/`fall` are the in-loop greedy MATH500 val, seed-averaged.

| arm | axis | PG | keep_sampled | peak | @step | final | fall | last | lock | truncation | length | entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c1_lsm_topk32_renorm` | C | yes | – | 0.533 | 125 | 0.524 | 0.009 | 250 | – | 0.008 | 1347 | 0.655 |
| `b4_jsd` | B | no | – | 0.593 | 50 | 0.524 | 0.069 | 250 | – | 0.008 | 8441 | 0.968 |
| `e1_pl_rank` | E | no | – | 0.572 | 250 | 0.572 | 0.000 | 250 | – | 0.008 | 5970 | 5.341 |
| `c3_intersection` | C | no | – | 0.617 | 50 | 0.599 | 0.018 | 250 | – | 0.008 | 6417 | 2.731 |
| `f3_power` | F | yes | – | 0.581 | 100 | 0.473 | 0.107 | 250 | – | 0.010 | 9523 | 1.301 |
| `c2_quantile_budget` | C | no | – | 0.684 | 250 | 0.684 | 0.000 | 250 | – | 0.016 | 8579 | 0.412 |
| `c4_pi_tail_budget` | C | no | – | 0.639 | 250 | 0.639 | 0.000 | 250 | – | 0.018 | 8775 | 0.608 |
| `j1_kdrl` | J | no | – | 0.646 | 200 | 0.639 | 0.007 | 250 | – | 0.096 | 4067 | 0.156 |
| `h1_first_segment` | H | yes | – | 0.632 | 200 | 0.625 | 0.007 | 250 | – | 0.151 | 5690 | 0.077 |
| `h3_random_segment` | H | yes | – | 0.649 | 50 | 0.583 | 0.066 | 250 | – | 0.182 | 9793 | 0.301 |
| `e3_zvalue` | E | no | – | 0.637 | 75 | 0.589 | 0.049 | 250 | – | 0.250 | 10152 | 0.172 |
| `g1_verified_only` | G | yes | – | 0.627 | 125 | 0.529 | 0.097 | 250 | – | 0.714 | 13928 | 0.208 |
| `b2_forward_kl` | B | no | – | 0.572 | 50 | 0.460 | 0.112 | 200 | – | 0.867 | 15329 | 0.793 |
| `b1_skew_kl` | B | yes | – | 0.645 | 175 | 0.475 | 0.169 | 250 | 247 | 0.911 | 15724 | 0.311 |
| `g2_fire_likelihood` | G | yes | yes | 0.624 | 100 | 0.511 | 0.113 | 250 | 143 | 0.969 | 16037 | 0.230 |
| `f1_soft_log` | F | yes | – | 0.632 | 150 | 0.447 | 0.185 | 250 | 208 | 0.971 | 16185 | 0.277 |
| `e2_set_coverage` | E | no | – | 0.597 | 50 | 0.460 | 0.137 | 175 | 128 | 0.977 | 16218 | 0.191 |
| `f2_hard_clip` | F | yes | – | 0.648 | 175 | 0.457 | 0.191 | 250 | 198 | 0.977 | 16196 | 0.234 |
| `d1_tip` | D | yes | yes | 0.620 | 50 | 0.473 | 0.147 | 250 | 127 | 0.990 | 16281 | 0.240 |
| `vanilla_n8` | - | yes | – | 0.629 | 50 | 0.466 | 0.163 | 250 | 117 | 0.991 | 16276 | 0.256 |
| `g4_failure_only` | G | yes | – | 0.629 | 75 | 0.450 | 0.179 | 250 | 121 | 0.992 | 16311 | 0.241 |
| `b5_k2` | B | no | – | 0.625 | 50 | 0.460 | 0.165 | 250 | 125 | 0.995 | 16338 | 0.243 |
| `vanilla` | - | yes | – | 0.627 | 50 | 0.473 | 0.153 | 250 | 122 | 0.995 | 16328 | 0.240 |
| `d2_selectkd` | D | yes | yes | 0.641 | 125 | 0.480 | 0.161 | 250 | 191 | 0.997 | 16364 | 0.194 |
| `g5_rgopd_gate` | G | yes | – | 0.627 | 75 | 0.425 | 0.202 | 175 | 131 | 0.997 | 16358 | 0.234 |
| `d3_teachability` | D | yes | yes | 0.622 | 50 | 0.415 | 0.207 | 250 | 123 | 1.000 | 16384 | 0.796 |
| `b3_eopd_gate` | B | yes | yes | 0.612 | 50 | 0.003 | 0.609 | 250 | 101 | 1.000 | 16384 | 0.001 |
| `a2_coldstart` | A | yes | – | 0.489 | 175 | 0.480 | 0.009 | 225 | 31 | 1.000 | 16384 | 0.228 |
| `h2_last_segment` | H | yes | – | 0.475 | 25 | 0.438 | 0.037 | 225 | 37 | 1.000 | 16384 | 0.188 |

---

_The response texts behind §4 are committed at [`data/textdumps/`](data/textdumps/) (three parquets,
one row per problem, full `response` string). They were produced by
[`scripts/analysis/eval_offline_textdump.py`](../scripts/analysis/eval_offline_textdump.py) over the
problem list in `scripts/analysis/textdump_pids.txt`, on two idle GPUs; no running job was preempted._

_Reproduce: `extract_metrics.py` builds the tidy metric table from `logs/*/lane*.log`;
`analyze_diag.py`, `analyze_pg.py`, `analyze_order.py` produce §2, §6 and §5 respectively;
the three diagnostics were run with `scripts/eval_offline.py --benchmarks math500 --n 1
--temperature 0 --top-p 1.0 --max-tokens 16384` against the step-100/250 actor checkpoints._
