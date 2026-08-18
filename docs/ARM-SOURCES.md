# Arm → source paper map

_One row per arm: the paper it references, what role that citation plays, and the
code witness the audit actually read (r4/r5/r6, full trail in
`arm-provenance-r4.md`). Compiled 2026-08-11 from `configs/arms.yaml`._

**How to read the "role" column.** *Audited* = the arm reproduces that paper's
claim and the verdict is about it; the house rule is **code over paper** — where
the official repo and the paper text disagree, the arm follows the code.
*Idea source* = the arm is OURS; the citation is where the mechanism or the
provocation came from, it is **not** the audited object, and the arm competes
under the same protocol as every defendant (firewall clause). *Anchor* =
protocol-defining, not compared against.

## Baseline

| arm | source | role | code witness |
|---|---|---|---|
| `vanilla` | Demystifying — [2607.13399](https://arxiv.org/abs/2607.13399); estimator lineage: thinkingmachines.ai blog on on-policy distillation | anchor (protocol) | verl `kl_penalty(k1)` + PG path (constructive: we call it) |
| `vanilla_n8` | same anchors | constructive control (j1 cell floor; reads the n knob) | — |

## A · rollout source & schedule

| arm | source | role | code witness |
|---|---|---|---|
| `a1_gkd_mix0.5` | GKD — [2306.13649](https://arxiv.org/abs/2306.13649) | audited (λ=0.5 mixing; estimator caveat registered r6: does **not** claim GKD's objective) | TRL `trl/experimental/gkd/gkd_trainer.py` (read line-by-line, r6) |
| `a2_coldstart` | Rethinking — [2604.13016](https://arxiv.org/abs/2604.13016) | audited (cold-start recipe; r6: their rejection sampling is a **validity** filter) | thunlp/OPD `scripts/infer/vllm_rollout.py` + LlamaFactory `qwen3_base_full_sft.yaml` (local clone) |
| `a3_offpolicy` | GKD — [2306.13649](https://arxiv.org/abs/2306.13649) | audited (λ=0 endpoint, the paper's sanctioned-but-unevaluated teacher-generated cell) | same as a1 |

## B · divergence

| arm | source | role | code witness |
|---|---|---|---|
| `b1_skew_kl` | DistiLLM — [2402.03898](https://arxiv.org/abs/2402.03898) | audited (SRKL α=0.1, symbol-for-symbol) | jongwooko/distillm `distillm/losses.py` |
| `b2_forward_kl` | verl as-shipped (its comment cites GKD [2306.13649](https://arxiv.org/abs/2306.13649)) | audited (implementation-is-source; known-worse direction control) | verl `forward_kl_topk` (constructive) |
| `b3_eopd_gate` | EOPD — [2603.07079](https://arxiv.org/abs/2603.07079) (ICML 2026) | audited (r5 rewrite from the official code: **additive** gated FKL, abs threshold 0.8) | WLS04/EOPD `core_algos` + `dp_actor` |
| `b4_jsd` (+ `_b0.1`, `_b0.9`) | GKD — [2306.13649](https://arxiv.org/abs/2306.13649) | audited divergence (generalized JSD-β, TRL interpolation form); the β rungs are the arm's pre-registered ablation | TRL `generalized_jsd_loss` |
| `b5_k2` | **OURS** · idea source: KDRL — [2506.02208](https://arxiv.org/abs/2506.02208) (k2 as its KD term) | idea source (estimator-package firewall arm) | — (reuses our `k2_kdrl` mode) |

## C · vocabulary support

| arm | source | role | code witness |
|---|---|---|---|
| `c1_lsm_topk32_renorm` | LSM — [2603.25562](https://arxiv.org/abs/2603.25562) | audited (k=32 both-sides renorm; r5: official code is **PG** — paper reads direct, code wins) | hhh675597/revisiting_opd `compute_opd_advantage`; numerics cross-checked against EasyOPD `kl_renorm_topk` (5e-7) |
| `c2_quantile_budget` (+ `_qb_fixed8`, `_qb_perseq`) | **OURS** · provocations: c1's fixed k; Rethinking's support-size claim ([2604.13016](https://arxiv.org/abs/2604.13016)) | idea source; the scope rungs are the arm's registered pinning-granularity ladder | — |
| `c3_intersection` | Rethinking / thunlp-OPD — [2604.13016](https://arxiv.org/abs/2604.13016) | audited (their `intersection` strategy; reduced single-epoch form, code-anchored) | thunlp/OPD `dp_actor.compute_distillation_reward` |
| `c4_pi_tail_budget` | **OURS** · source: the headline theorem's error term π(S̄)·KL (from the Rethinking line of analysis, [2604.13016](https://arxiv.org/abs/2604.13016)) | idea source (headline-constructive: theorem quantity as the knob) | — |

## D · token selection

| arm | source | role | code witness |
|---|---|---|---|
| `d1_tip` | TIP — [2604.14084](https://arxiv.org/abs/2604.14084) | audited (soft-OR, ρ=0.5; r5 removed our p98 "fix"; `TIP_MODE` decomposition pre-registered) | HJSang/OPSD_OnPolicyDistillation (selector **not in repo** — recorded; paper wins) |
| `d2_selectkd` | SelecTKD — [2510.24021](https://arxiv.org/abs/2510.24021) | audited (top-1 ∈ teacher top-**5**, β=0.01 down-weight) | no official code as of 2026-08-07 (recorded) |
| `d3_teachability` | TA-OPD — [2605.26844](https://arxiv.org/abs/2605.26844) | audited (disagreement × compatibility, Q05–Q95, 5% budget) | no official code (recorded) |

## E · within-support objective (all OURS — the 4-rung ladder with c1)

| arm | source | role | code witness |
|---|---|---|---|
| `e1_pl_rank` | **OURS** · PL machinery as in PLD — [2506.12542](https://arxiv.org/abs/2506.12542) | idea source (order-not-value + value anchor) | — |
| `e2_set_coverage` | **OURS** · casefile set-coverage substitute + c4's π_tail as a loss | internal lineage | — |
| `e3_zvalue` | **OURS** · z-form from Logit-Std KD — [2403.01427](https://arxiv.org/abs/2403.01427) | idea source (affine-invariant rung) | — |

## F · signal conditioning

| arm | source | role | code witness |
|---|---|---|---|
| `f1_soft_log` | Demystifying — [2607.13399](https://arxiv.org/abs/2607.13399) | audited (winner transform at modest gap) | odd-function equivalence verified |
| `f2_hard_clip` | Demystifying — [2607.13399](https://arxiv.org/abs/2607.13399) | audited shape; the ±10 value is **self-registered** (attribution clarified r5) | verl `loss_max_clamp` + our hit-rate panel |
| `f3_power` | PowerOPD — [2606.17199](https://arxiv.org/abs/2606.17199) | audited against the paper only (repo `EIT-NLP/PowerOPD` not public — recorded) | none available |

## G · trajectory gating

| arm | source | role | code witness |
|---|---|---|---|
| `g1_verified_only` | **OURS** (plain verified-only member) · nearest relative: RG-OPD — [2607.04037](https://arxiv.org/abs/2607.04037) (rule differs); phrasing red line: [2607.23731](https://arxiv.org/abs/2607.23731) | idea source / family member — never claimed as RG-OPD | — |
| `g2_fire_likelihood` | FiRe — [2606.02684](https://arxiv.org/abs/2606.02684) | audited (both stages: sliding-window filter + double-entropy reweight; `FIRE_MODE` pre-registered) | paper Eq.5–8 (r4 completion after truncated title) |
| `g4_failure_only` | **OURS** · g1's predicate negated; counterpoint read: [2607.23731](https://arxiv.org/abs/2607.23731) | idea source (sign-family third point) | — |
| `g5_rgopd_gate` | RG-OPD — [2607.04037](https://arxiv.org/abs/2607.04037) | audited (Eq.2 verbatim, δ=0) | repo 404 at audit time (recorded; paper wins) |

## H · supervision & rollout horizon (efficiency)

| arm | source | role | code witness |
|---|---|---|---|
| `h1_first_segment` | ESR Less-is-More — [2605.27028](https://arxiv.org/abs/2605.27028) | audited (K=100 default per r5; loss-mask form is a **declared** deformation — their original truncates the rollout, which would contaminate axis A) | no official code (recorded) |
| `h2_last_segment` | **OURS** · falsification mirror of ESR [2605.27028](https://arxiv.org/abs/2605.27028) | idea source (position vs budget deconfound) | — |
| `h3_random_segment` | **OURS** · same, position-agnostic budget control | idea source | — |
| `h4_random_scatter` | **OURS** · window-vs-scatter split of h3 | idea source (contiguity deconfound; added 2026-08-11, row backfilled 2026-08-18) | — |
| `h5_gen100` | ESR — [2605.27028](https://arxiv.org/abs/2605.27028) | audited (the paper's **own** rollout-truncation form, K=100; h1's declared form-pair, so mask-vs-truncation is a measured pair; in-loop val capped — recorded, offline suite judges) | no official code (recorded) |
| `h6_gen_sched` | POPD "Are Full Rollouts Necessary" — [2605.31490](https://arxiv.org/abs/2605.31490) + FastOPD schedule — [2602.15260](https://arxiv.org/abs/2602.15260) | audited (progressive-horizon family; **declared deviations**: continuous per-step linear ramp 128→16384 rather than their coarse staircases (+ΔH each Δk / +256 chunks) — same monotone family, finer grain; our membership-gated val exemption is house bookkeeping, not from the papers) | no official code confirmed for either (recorded, survey 2026-08-17) |
| `h7_gen512` | ESR [2605.27028](https://arxiv.org/abs/2605.27028) / TOPD [2605.31490](https://arxiv.org/abs/2605.31490) | audited (fixed-depth dose line with h5/h8; 512 = 3.1% of the window, brackets TOPD's "~10% suffices" from below) | no official code (recorded) |
| `h8_gen2048` | same pair | audited (2048 = 12.5%, brackets TOPD's claim from above; h5's exact mechanism, dose-line purity) | — |
| `h9_prune_adapt` | Prune-OPD — [2605.07804](https://arxiv.org/abs/2605.07804) | audited-lite (**declared signal swap**: their per-position top-k SET overlap does not exist at our sampled-token-k1 seam — events are teacher-lost-the-thread log-prob thresholds, the Δℓ position panels' own quantity; their supervision down-weighting deliberately NOT reproduced — the H axis moves rollout allocation only, loss surgery would confound with B/D; the budget architecture conforms: events → reliable length → adaptive budget, starts long, adapts down on evidence) | official repo yangzhch6/Prune-OPD exists — **flagged for line-audit** before any paper-facing verdict |
| `h10_task_subset` | **OURS** · provocation: PACED — [2603.11178](https://arxiv.org/abs/2603.11178) | idea source (task-allocation anchor; PACED's pass-rate weighting needs multi-rollout estimates the n=1 protocol cannot fund — recorded) | — |

_Wave-1 efficiency expansion (2026-08-18) driven by
`simopd_data/opd_rollout_efficiency_survey.md`; excluded with reasons there:
PG-OPD (needs n>1 candidates), ADWIN (async full-rollout probe infra),
Relay-OPD (wave 2; would reuse a5's online-teacher route)._

## J · KD × RL coupling

| arm | source | role | code witness |
|---|---|---|---|
| `j1_kdrl` | KDRL — [2506.02208](https://arxiv.org/abs/2506.02208) | audited (GRPO n=8 − β·k2-KL, β=2e-3; baseline is `vanilla_n8`, not the 2-card floor) | paper ruling (their ablation k2>k3) |

## I · teacher conditioning (axis shelved 2026-08-07)

| arm | source | role |
|---|---|---|
| `i0_think_scorer` / `i1_priv_cot` | **OURS** (thinking-teacher conditioning designs; i0's "confiscated scratchpad" measurement already banked: 0.708 < 0.896) | shelved; revival = status flip |

## Reference papers that shape the design without owning an arm

| paper | where it acts |
|---|---|
| OPD+ — [2606.01039](https://arxiv.org/abs/2606.01039) | estimator bias audit → `estimator-note.md` (base kept, bias recorded) |
| RSKD — [2503.16870](https://arxiv.org/abs/2503.16870) | spirit of c1's `tailbucket` internal ablation |
| SEAD [2606.28562](https://arxiv.org/abs/2606.28562) · Evidence [2606.22830](https://arxiv.org/abs/2606.22830) · Rock [2605.09253](https://arxiv.org/abs/2605.09253) · Blockwise [2606.24084](https://arxiv.org/abs/2606.24084) · Position-Bias [2606.22600](https://arxiv.org/abs/2606.22600) | selector family → shadow-panel measurement only (task #7), no training runs |
| KAT (sampling-distribution change) | discussion-only, not raced |

## Related-work mentions (coverage sweep 2026-08-11 — cite briefly; ruling: different genre, no defensive positioning needed)

| paper | one-line relation |
|---|---|
| OPD survey — [2604.00626](https://arxiv.org/abs/2604.00626) | field survey (feedback signal / teacher access / loss scope) |
| Formula-driven survey — [2606.22793](https://arxiv.org/abs/2606.22793) | taxonomy-only unification (two routes, eight axes); no experiments |
| Many Faces — [2605.11182](https://arxiv.org/abs/2605.11182) | single-part probes (teacher prefix distortion; un-renormalized top-k "+1 bias"; OPSD PI). Our renorm/detached kernels sit structurally outside its failure-2 class (`topk_losses.py:684`); it has no termination/length line |
| Geometry of OPD — [2606.07082](https://arxiv.org/abs/2606.07082) | weight-space diagnostics (subspace locking) — complementary readout |
| Temperature — [2606.00306](https://arxiv.org/abs/2606.00306) | τ ruled a global re-parameterization, not a slot coordinate (2026-08-11); protocol pins τ=1, the field-canonical value |
| AsyncOPD — [2606.24143](https://arxiv.org/abs/2606.24143) | staleness = sampler engineering, outside the loss template's object |
| [awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation) | community roster used for the 2026-08-11 coverage sweep |
