# UNIFIED-LOSS — the unified OPD training operator

Registered 2026-08-11; **revised the same day after the formula review** (16-point
review adopted in full — ledger entry "统一框架公式评审"). Companion to
`MECHANISMS.md`. The object of study is not a scalar loss but the **operator**

```
𝒰 = ( μ ;  w, m ;  Ω, ν, S, D ;  ℰ, Φ, b )   ↦   g_θ
     └where└which data └what is compared└how it updates
```

mapping a tuple of design coordinates to an **update field**. The paper's program
in one line: coordinate change → gradient field → training dynamics. "OPD" is not
a loss; it is this whole set of coupled design choices, and every arm in
`configs/arms.yaml` is one controlled move inside it.

## 1. The operator, stage by stage

**Layer 0 — behavior policy (semi-gradient discipline).** Rollouts come from the
frozen behavior policy of the current update:

```
y ~ μ_θ̄( · | x ; λ, T_max, n ),      θ̄ = sg(θ)
```

Every downstream operator acts on the visited states `s_t = (x, y_<t)` with the
visitation measure **held fixed**: no branch differentiates through state
visitation (on-policy state sampling + fixed-state local semi-gradient). This is
what licenses the identities in §2 and pre-answers "why doesn't the direct branch
differentiate the sampling distribution".

**Per-token pipeline** (`p_t^a = π_a(·|s_t)`, `a ∈ {θ, T}`):

```
u_t^a = N_ν( Q_Ω[ p_t^a ] )          support extraction, THEN representation
z_t^a = S_σ( u_t^a )                 statistic
d_t   = D_δ( z_t^θ , z_t^T )         population comparison
c_t   = ℰ_η[ d_t ]                   realization / estimator
a_t   = Φ_φ( c_t )                   signal shaping
ℓ̃_t  = 𝒢_b[ a_t ]                   update-field constructor
```

with the two field constructors

```
𝒢_direct[a_t] = a_t                                  (pathwise at fixed s_t)
𝒢_PG[a_t]     = −sg(a_t) · log π_θ(y_t | s_t)        (score-function on the sampled
                                                      action; immediate credit, no
                                                      baseline — pinned roster-wide)
```

**Selection layer** — the finite-batch, sequence-balanced estimator (this is the
shipped normalization convention, stated as the estimator it is):

```
L̂ = ( 1 / Σ_i w_i ) Σ_i  w_i · [ ( 1 / Σ_t m_it ) Σ_t  m_it · ℓ̃_it ]
```

| stage | values in the roster | owning axis |
|---|---|---|
| `μ` | λ ∈ {0, 0.5, 1}; T_max ∈ {100, 16k}; n pinned at 1 (n=8 cell archived with the J axis, 2026-08-11) | A; h5 |
| `w` | 1 / 1[pass] / 1[fail] / quota-K / directional / likelihood | G |
| `m` | 1 / windows / scatter / criteria; ρ, placement P | D∪H |
| `Q_Ω` | {y_t} / top-k / quantile-budget / intersection / π-tail | C |
| `N_ν` | raw / renorm / tailbucket | C-internal (`SUPPORT_MODE`) |
| `S_σ` | id (values) / z-score / rank / set-mass | E |
| `D_δ` | RKL / FKL / skew_α / JSD_β / power comparator (f3) | B (+ f3, see §4) |
| `ℰ_η` | exact-sum / k1 / k2 | crossover cells |
| `Φ_φ` | id / sign·log(1+·) / clip_±M | F (f1, f2) |
| `𝒢_b` | direct / PG | branch |

**Why the Ω/ν split is load-bearing.** Renormalizing a singleton support
collapses both sides to 1 (`log 1/1 = 0`) — vanilla *requires* `(Ω={y_t},
ν=raw)`. e2's set statistic `Σ_Ω p^θ(v)` requires `ν=raw` on a top-k support —
renormalized it is identically 1. Support choice and normalization are
independent coordinates; c1's `SUPPORT_MODE=renorm|tailbucket` internal ablation
lives exactly on ν.

**Why the D/ℰ split is load-bearing.** `D` is the population discrepancy; `ℰ`
its sampled realization. k1 (`c_t = Δℓ_t`) is unbiased for RKL *in expectation*.
k2 (`c_t = ½Δℓ²`) is **not** an unbiased scalar estimator of RKL
(`E[½r²] ≠ KL`); it is a pathwise surrogate whose gradient
`r·∇log π_θ(y_t)` matches the fixed-state RKL gradient in expectation. b5's
meaning is now exact: **D fixed, ℰ moved** — the estimator firewall.

## 2. Vanilla as the alignment point

At the origin `(Ω={y_t}, ν=raw, S=id, D=RKL, ℰ=k1, Φ=id, b=PG)`, the
fixed-state identity holds:

```
∇_θ KL(p_t^θ ‖ p_t^T)
  = Σ_v ∇p^θ(v) · ( Δℓ(v) + 1 )          Σ_V ∇p^θ = 0 kills the +1
  = E_{y_t ~ p_t^θ} [ Δℓ_t · ∇_θ log π_θ(y_t|s_t) ]
```

> **Vanilla OPD is a special point where on-policy sampling, reverse-KL
> geometry, and the score-function estimator align exactly at each visited
> state; the surrounding design space breaks these identities in controlled
> ways.**

The `+1` dies because the score sums to zero over the FULL support. On a
truncated support the sum no longer closes and the +1 bias revives — unless
`ν=renorm` restores closure within Ω, or the weights are detached. The roster's
kernels are all renorm/detach (`topk_losses.py:684` witness), so that failure
class is structurally absent here (cf. Many Faces 2605.11182, its failure-2).

## 3. The update field is the object, not the scalar

`𝒢_PG[a_t]` is in general **not** the gradient of any scalar built from `a_t`:
for a distribution-valued comparison (c1's renorm-RKL), `−sg(d_t)∇log π_θ(y_t)`
is a different vector field from `∇d_t`. They coincide only at special points —
§2's origin is one, and that is vanilla's derivational status. The crossover
cells are the off-diagonal observations of 𝒢: b5 (sampled × direct), c1
(top-k × PG), and the pending **C1 = c1-direct**, the fourth corner of the
(support × field) 2×2.

**Framework-derived implication (registered analytic note, M-task candidate).**
At PPO ratio = 1 the per-sample fields of k1-PG and k2-direct coincide
*exactly*: `∇(½r²) = r·∇log π_θ(y_t)` vs surrogate gradient
`sg(r)·∇log π_θ(y_t)` — numerically identical. The framework therefore predicts
**b5 ≈ vanilla wherever the ratio stays at 1**, and any observed b5↔vanilla
divergence measures the *surrogate machinery* (within-step micro-batch ratio
drift + clipping), not the estimator per se. This sharpens b5's registered
reading ("any difference is an estimator effect") into "any difference is a
surrogate-machinery effect" — to be checked against the 16k curves before the
verdict wording.

## 4. Design coordinates ≠ emergent mechanism coordinates

- **b1** (skew α=0.1) is a **D-move**; its positive-side bound
  `c_t ≤ −log α = ln 10` is an *emergent* property of the mixture, not a Φ
  setting.
- **f3** (`p_T^α − p_θ^α`) is **not** `Φ(Δℓ)` — Δℓ does not determine
  `p_T − p_θ`. It swaps log-ratio geometry for probability geometry: an
  out-of-family **bounded comparator** (a D-move with Φ=id). f1/f2 are the true
  amplitude transforms.

The compression curve therefore lives on an **emergent coordinate**
`M(𝒰) = effective signal bound`, onto which arms from *different design
coordinates* project: {vanilla ∞, f2 ±10, f1 log, b1 ln10, f3 [−1,1]} →
locks {122, 198, 208, 247, never}. The magnitude-unification hypothesis is
precisely: training dynamics factor through `M(𝒰)`, screening off the design
coordinate that produced it. `f2_clip2.3` vs b1 is the matched-M intervention;
M1 the measurement. This also settles "merge B and F?": **no** — distinct
design axes that project onto one mechanism coordinate, and the projection
itself is the finding.

## 5. Global origin and axis-local carriers

"Each arm moves exactly one coordinate from vanilla" is not achievable: moving
S or D off the sampled point mechanically drags Ω, ν, b along (the space is a
stratified product, §1). The actual design discipline is:

> **Each axis varies one scientific coordinate relative to an axis-specific
> carrier configuration; mechanically required carrier coordinates are held
> fixed within the axis.**

| axis | carrier (held fixed) | scientific coordinate |
|---|---|---|
| A | vanilla pipeline | μ.λ |
| h5 | vanilla pipeline | μ.T_max |
| G | vanilla local pipeline | w |
| D∪H | vanilla local pipeline | m (ρ, placement) |
| C | b=direct, D=renorm-RKL | Ω rule / ν / τ-scope |
| B | Ω=top-32 renorm, b=direct (b1 rides the sampled carrier) | D (family, β) |
| E | Ω=top-32, b=direct; ν as S requires (rank/z shift-invariant; set needs raw) | S |
| F | vanilla sampled carrier (PG) | Φ (f1, f2); f3 = D-move (§4) |

Cross-axis reads must route through carriers. Example: the S-ladder's "values"
rung *on the E carrier* is c1-**direct** — gap cell C1 — because shipped c1 is
PG (official code). That is the third of C1's three identities (fourth 2×2
corner; S-ladder branch repair; cheapest verdict on the roster).

Placement notes: `h5_gen100` is a **μ-move** (T_max 16k→100, m≡1);
`h1` is the m-move (T_max=16k, m=1[t≤100]) — the registered bridge pair
between supervision locality and rollout locality. `h4_random_scatter` stays an
m-move (scatter placement). `j1` sits outside the operator (§7).

## 6. Selection layer: conditioning, not reweighting

The double conditional-mean estimator makes the G/H algebra automatic:

- `w_i = 1[R_i > 0]` yields `(1/B_pass) Σ_{pass} L_i` — the shipped
  mask + `T/T_keep` rescale + token-mean is this conditional mean's *algebraic
  implementation*, not a method coordinate.
- `m_it = 1[t ≤ 100]` yields `(1/100) Σ_{t≤100} ℓ_it` — same algebra at token
  level.
- **g1_quota changes no objective.** The estimand stays `E[L | R > 0]`; the
  quota freezes `μ_θ̄` until `B_pass = K`, pinning the estimator's sample size —
  a **variance intervention on the same estimand**. M4's famine-amplifier
  account is exactly "the conditional-mean estimator at B_pass ≈ 1–5 has
  catastrophic variance"; the quota cell is its control.

## 7. Outside the operator (pinned and disclosed)

Initialization `θ₀` (a2). Objective mixing `L_total = L_task + c·L_OPD` — the
J boundary asks whether the local distillation operator needs a global task
objective to close the loop. The J axis is archived **in full** (2026-08-11):
`j1_kdrl` and its n=8 control — a reward-dominated cell is hard to claim as
OPD. The boundary question stands; the completed cells are banked as appendix
material, and `μ.n` is pinned at 1 roster-wide. Execution:
synchronous, zero staleness. Temperature τ=1 (a global re-parameterization that
composes with every stage — not a coordinate; ruling 2026-08-11). Temporal
credit: immediate, no baseline (the return-to-go / GAE family is a disclosed
pinned coordinate, cf. survey 2606.22793).

## 8. Insight machinery

- **Dose lines** = sections along one design coordinate at a fixed carrier
  (λ line, β ladder, τ-scope ladder, position line).
- **Screening-off tests** live on emergent coordinates (M; §4).
- **Effect decomposition** via off-diagonal cells (the field 2×2; Δlock:
  estimator/field swap ~3 steps vs bound swap +125).
- **Attractor universality** via the near-placebo (e2_a0: is the length
  attractor reachable with almost no training pressure — a statement about the
  objective landscape, not any method).
- **Composability** (the recipe cell): whether per-coordinate optima compose —
  the framework's final falsifiable prediction.

## 9. Relation to prior unifications

f-divergence unifications (DistiLLM, GKD's JSD) parameterize `D` alone; GKD's λ
is `μ`. The formula-driven survey (2606.22793) taxonomizes similar axes with no
experiments; this framework differs in being an *operator* account (𝒰 ↦ update
field, with the semi-gradient discipline explicit) instantiated by controlled
single-coordinate probes with measured dose-response. Related-work list:
`ARM-SOURCES.md` final section.
