# UNIFIED-LOSS — one objective, nine coordinates

Registered 2026-08-11. Companion to `MECHANISMS.md`: that doc taxonomizes the
*effects* (mechanism classes M-I…M-IV); this one gives the *cause side* a
coordinate system. Every arm in `configs/arms.yaml` is a point in the parameter
space of ONE loss template, and every axis is a coordinate of it. This executes
the framing ruling ("mechanism-first; methods are probes") literally: a probe
*position* is now a tuple of slot values, not a metaphor.

## 1. The template

Three layers — a sampler, a per-token functional, a gradient estimator:

```
          sampler        trajectory wt        per-token functional
L(θ) = E_{y ~ π_gen} [  w(y) · (1/N) Σ_t  m_t · Φ_M( D_β( S(T_k[p_T(·|s_t)]), S(p_θ(·|s_t)) ) )  ]

gradient delivered via  b ∈ { PG (advantage = −ℓ_t.detach()),  direct }
```

| # | slot | meaning | parameters | axis | code seam |
|---|---|---|---|---|---|
| 1 | `π_gen` | who generates the trajectory | on-policy fraction λ; rollout budget T_max; group size n | A (λ), h5 (T_max), n8 cell | `gkd_mix` cache+λ; `MAX_RESPONSE_LENGTH`; `ROLLOUT_N` |
| 2 | `w(y)` | trajectory-level weight | gate predicate; quota K; normalization | G | `k1_verified_only`/`k1_failure_only`; `filter_groups` (wave 13) |
| 3 | `m_t` | token-support mask | coverage ρ; placement P ∈ {front, tail, window, scatter, criterion} | D∪H | `_window_kernel` family; d-kernels |
| 4 | `S` | statistic read off each distribution | values / z-values / rank / set membership | E | e1/e2/e3 kernels |
| 5 | `T_k` | teacher-support truncation | width k; rule (fixed / quantile / intersection / π-tail); renorm; τ scope | C | `TOPK_DISPATCH`; `SIMOPD_QB_*`; `SIMOPD_SUPPORT_MODE` |
| 6 | `D_β` | discrepancy functional | family (RKL / FKL / skew / JSD / power); β, α_skew; estimator k1/k2 | B, F (f3) | loss-mode registry |
| 7 | `Φ_M` | magnitude transform | bound M; shape (id / log / clip / power) | F | `k1_softlog`; `LOSS_MAX_CLAMP`; `k1_power` |
| 8 | `b` | gradient channel | PG / direct | crossover cells | `USE_POLICY_GRADIENT` |
| 9 | `p_T` | the reference distribution itself | scorer identity; scorer conditioning | I (shelved) | `TEACHER_MODEL`; `SIMOPD_PRIV_COT` |

Two knobs sit deliberately OFF the template, kept explicit:

- **θ₀ initialization** (a2 cold start) — where training *starts*, not what it
  optimizes;
- **objective mixing** (j1 adds J_GRPO) — the J boundary: once the added
  objective dominates (KD coef 2e-3 ≈ 500× dilution), the run is hard to claim
  as OPD at all. Archived ruling 2026-08-11; the template's *scope* is itself a
  finding-shaped statement.

The env-knob space **is** this parameterization already: the launcher
fingerprint (`run_opd_baseline.sh:229`) hashes loss mode, branch, top-k, every
`SIMOPD_*`, n, caps — a fingerprint is a coordinate tuple and always was. The
template names an existing structure; it adds no machinery.

### Stratified, not a full product

Two structural couplings keep this from being a free product space:

- `T_k` (slot 5) is only *active* for distribution-valued `D`; vanilla's
  sampled-k1 reads one column and no truncation ever fires.
- `b` (slot 8) is constrained by `D`'s family under the branch rules
  (sampled-signal → PG; divergence-valued → direct; literature arms follow
  their official code). Moving `D` therefore usually drags `b` and `T_k` — a
  structurally *forced* diagonal. The crossover cells exist to undo exactly
  this: **b5** (sampled × direct) and **c1** (top-k × PG) are the off-diagonal
  observations that let the Fisher p=0.017 PG-vs-capped signal be attributed.

## 2. Coordinate chart — all 43 arms

Move type legend: **∥** axis-parallel (single coordinate — a clean probe);
**⤢** diagonal (>1 slot; forced, registered, or deliberate); **●** literature
point (audited object, composite by design — never read as single-coordinate
evidence); **◌** off-template.

| arm | slot(s) | move (from vanilla) | type |
|---|---|---|---|
| `vanilla` | — | origin: λ=0, w≡1, m≡1, S=values(sampled), D=RKL-k1, Φ=id (M=∞), b=PG, n=1, T_max=16k, teacher=4B-Instruct | — |
| `a1_gkd_mix0.5` | 1 | λ: 0→0.5 (per-prompt coin) | ∥\* |
| `a3_offpolicy` | 1 | λ: 0→1 | ∥\* |
| `a2_coldstart` | init | θ₀: base → SFT cold start | ◌ |
| `vanilla_n8` | 1 | n: 1→8 (32 prompts/step) | ∥ |
| `h5_gen100` | 1 | T_max: 16k→100 — changes π_gen *and* supervision jointly; the registered h1↔h5 pair measures exactly that difference | ⤢ deliberate |
| `g1_verified_only` | 2 | w = 1{pass} (conditional mean) | ∥ |
| `g4_failure_only` | 2 | w = 1{fail} | ∥ |
| `g1_quota` / `g4_quota` | 2 + sampler | same gates + sample size pinned at K=16 (dynamic sampling) | ∥ vs g1/g4 |
| `g2_fire_likelihood` | 2 + 3 | Eq.4 trajectory filter + Eq.5–8 token reweight | ● |
| `g5_rgopd_gate` | 2 | directional gate (outcome sign × likelihood direction), δ=0 | ● |
| `h1` / `h2` / `h3` | 3 | window K=100 at front / tail / random offset; ρ≈K/len | ∥ ladder |
| `h4_random_scatter` | 3 | same K, non-contiguous — contiguity is the only move vs h3 | ∥ |
| `d1_tip` | 3 | placement=criterion (TIP soft-OR), ρ=0.5 | ● |
| `d2_selectkd` | 3 | placement=criterion (propose-verify), soft weight β=0.01 | ● |
| `d3_teachability` | 3 | placement=criterion (teachability), ρ=0.05 | ● |
| `b3_eopd_gate` | 3 + 6 | k1-PG base on ALL tokens + entropy-gated additive top-k FKL term (thresh 0.8, coef 1.0) | ● |
| `e1_pl_rank` | 4 | S: values→rank (PL) **+ 0.1 value anchor** | ⤢ anchor |
| `e1_pl_rank_a0` | 4 | S: values→rank, anchor 0 | ∥ purified |
| `e2_set_coverage` | 4 | S: values→set mass **+ 0.1 anchor** | ⤢ anchor |
| `e2_set_coverage_a0` | 4 | S: values→set mass alone (near-placebo past step ~50) | ∥ |
| `e3_zvalue` | 4 | S: values→z-scored values (no anchor by design) | ∥ |
| `c1_lsm_topk32_renorm` | 5 (b held) | top-32 renorm RKL, distribution-valued, **PG kept** | crossover |
| `c2_quantile_budget` | 5 | rule: fixed-k → quantile budget (batch τ, target 8, top-64 payload) | ∥ (C convention) |
| `c2_qb_fixed8` / `_perseq` | 5 | τ scope: batch → fixed / per-sequence | ∥ ladder |
| `c3_intersection` | 5 | support = student∩teacher top-32 | ● |
| `c4_pi_tail_budget` | 5 | support = smallest teacher-rank prefix with student mass ≥ 1−ε (ε=0.05) | ∥ theorem-constructive |
| `b1_skew_kl` | 6 (+7 implicit) | D: RKL→skew-KL α=0.1 ⇒ estimator bounded above at ln 10 ≈ 2.3 | ⤢ **D×M — the screening-off pair** |
| `b2_forward_kl` | 6 + 5 + 8 | RKL→FKL on UNrenormalized top-32; direct | ⤢ forced |
| `b4_jsd` | 6 + 5 + 8 | JSD β=0.5 on renorm top-32; direct | ⤢ forced |
| `b4_jsd_b0.1` / `_b0.9` | 6 | β: 0.5→0.1 / 0.9 | ∥ ladder within b4 |
| `b5_k2` | 6-est + 8 | estimator k1→k2, same RKL target; direct | crossover |
| `f1_soft_log` | 7 | Φ: id → sign·log(1+\|·\|) | ∥ |
| `f2_hard_clip` | 7 | Φ: id → clip ±10 | ∥ |
| `f3_power` | 6 + 7 | signal: Δℓ → π_T^α−π_θ^α (α=1), bounded [−1,1]; α→0 recovers vanilla | ⤢ form+bound |
| `i0_think_scorer` | 9 | scorer → 4B-Thinking | ∥ shelved |
| `i1_priv_cot` | 9 | scorer + private CoT context (control = i0) | ∥ shelved |
| `j1_kdrl` | mixing | + J_GRPO, KD coef 2e-3, n=8 cell | ◌ boundary |

\* a1/a3 are axis-parallel in λ, but the r6 estimator caveat rides along: under
the protocol's sampled-k1 PG, the off-policy half is log-ratio-weighted cloning
of teacher text, not GKD's supervised-KD term — the ladder doses **the data
source under our estimator**, and never reads as "supervised KD".

## 3. What the coordinates buy

### 3.1 Controlled variables, formalized

A qualified probe = one coordinate moved. The complaint that launched this
("很多实验没固定变量") becomes an executable predicate, and the chart sorts
every diagonal into exactly three kinds, each with its resolution:

- **forced** (family change drags b/T_k) → resolved by the crossover cells;
- **registered** (the E-axis anchor mixtures) → resolved by the a0 cells;
- **by-design** (literature points ●) → audited as objects, quarantined from
  single-coordinate claims.

### 3.2 Dose lines = measured curves along single coordinates

| coordinate | points | status |
|---|---|---|
| λ (slot 1) | {0, 0.5, 1} | run; retro-claimed as the A-axis data-source line (caveat above) |
| placement (slot 3) | front < window < scatter/criterion < tail | run; h4 + DH1 deconfound the middle |
| τ scope (slot 5) | {fixed, sequence, batch} | wave 10 |
| β (slot 6) | {0.1, 0.5, 0.9} | wave 9 |
| M (slot 7) | {∞→122, ±10→198, log→208, 2.3→247, [−1,1]→never} | run; M1 formalizes |

### 3.3 The M line spans two slots → screening-off is now testable

The compression line mixes implementations: the f-family bounds via Φ (slot 7),
b1 via D's intrinsic bound (slot 6), f3 via the form itself (6+7). The
magnitude-unification hypothesis is therefore precisely a **screening-off
claim**: collapse dynamics = g(M), independent of the implementing slot.
Registered intervention: **`f2_clip2.3`** (ledger r6 2026-08-11) — clip moved
to b1's own ln 10 ≈ 2.3, the matched-M pair. M1 is the hypothesis's
*measurement* half; f2@2.3 is its *intervention* half.

### 3.4 The endpoint: working method = per-coordinate selection

The recipe cell (Tier-3, triggered when Tier-1 verdicts land) is not "another
method in the arena": it is the point in this space whose every coordinate is
set by an adjudicated dose line, with the pre-registered prediction that it
avoids every mechanism-class failure (M-I…M-IV). Success validates the map;
failure localizes an interaction the per-coordinate analysis missed. Either
way, the mechanism paper's last chapter writes itself from this table.

## 4. Relation to prior unifications

f-divergence KD unifications (DistiLLM's skew family, GKD's generalized JSD)
parameterize slot 6 alone; GKD's λ is slot 1. The template's increment is not
the algebra — it is (i) the other seven slots, which is where this campaign's
failure modes actually live (termination is slot 1/3/7 territory, famine is
slot 2, granularity slot 5), and (ii) measured dose-response along each
coordinate instead of single-point method comparisons.
