# UNIFIED-LOSS — the unified OPD training operator

Registered 2026-08-11; revised three times the same day after the formula
review rounds (ledger: "统一框架公式评审", "ℰ 再拆", "定稿书写形式").
Companion to `MECHANISMS.md`. The object of study is not a scalar loss but the
**Unified OPD Training Operator** — the update field written as one composed
expression:

```
g(θ) = E_{x~𝒟, y~μ_θ̄} [  w(y) · (1/Σ_t m_t) · Σ_t  m_t ·
           ℛ( Φ( 𝒬[ D( S(N_ν(Q_Ω[p_t^θ])),  S(N_ν(Q_Ω[p_t^T])) ) ] ) )  ]

p_t^θ = π_θ(·|s_t),    p_t^T = π_T(·|s_t),    s_t = (x, y_<t)
```

Seven layers — outside-in over the data, inside-out at the token:

```
μ  →  w, m  →  Ω, ν, S  →  D  →  𝒬  →  Φ  →  ℛ
states  what to teach  what information  comparison  measurement  signal  update
```

The paper's program in one line: coordinate change → gradient field → training
dynamics. "OPD" is not a loss; it is this whole set of coupled design choices,
and every arm in `configs/arms.yaml` is one controlled move inside it.

| slot | role | values |
|---|---|---|
| `μ_θ̄` | trajectory/state source | on-policy / mixed λ / rollout horizon T_max |
| `w(y)` | trajectory-level selection | 1 / pass-only / fail-only / quota-K / directional / likelihood |
| `m_t = M_{ρ,P}(t, s_t, p^θ, p^T)` | token-level supervision allocation | ρ = how much; P = which tokens (position / entropy / disagreement); soft ∈ [0,1] |
| `Q_Ω` | vocabulary support | {y_t} / top-k / quantile-budget / intersection / π-tail |
| `N_ν` | support normalization / mass representation | raw / renorm / tail-bucket |
| `S` | teacher information resolution | value / z-shape / rank / set |
| `D` | teacher–student discrepancy geometry | RKL / FKL / skew / JSD (f3's power comparator: recorded out-of-family, §4) |
| `𝒬` | finite evaluation | MC at the sampled token / exact support sum |
| `Φ` | effective signal shaping | id / soft-log / clip / positive-only clip / smooth bound (tanh) |
| `ℛ` | update realization | score-function (PG) / direct gradient / k2-direct surrogate |

The J axis's RL coupling `L_task + β·L_OPD` sits **outside** this core operator
(§7) — it wraps the operator in a second objective rather than setting one of
its slots.

## 1. The operator, stage by stage

**Layer 0 — behavior policy (semi-gradient discipline).** Rollouts come from
the frozen behavior policy of the current update:

```
y ~ μ_θ̄( · | x ; λ, T_max, n ),      θ̄ = sg(θ)
```

Every downstream operator acts on the visited states `s_t = (x, y_<t)` with the
visitation measure **held fixed**: no stage differentiates through state
visitation (on-policy state sampling + fixed-state local semi-gradient). This
licenses the identities in §2 and pre-answers "why doesn't the direct
realization differentiate the sampling distribution".

**Per-token pipeline** (`p_t^a = π_a(·|s_t)`, `a ∈ {θ, T}`):

```
u_t^a = N_ν( Q_Ω[ p_t^a ] )     support restriction (Q_Ω), then representation (N_ν)
z_t^a = S_σ( u_t^a )            statistic
D_δ                             discrepancy geometry on (z^θ, z^T),
                                with pointwise integrand d(v) where one exists
c_t   = 𝒬[ D_δ ]                evaluation: exact support sum, or MC at the
                                sampled token   (Q_Ω restricts; 𝒬 evaluates —
                                distinct operators)
a_t   = Φ_φ( c_t )              signal shaping
g_t   = ℛ[ a_t ]                gradient realization (a descent field)
```

The three realizations in the roster:

```
ℛ_PG[a]     =  sg(a) · ∇_θ log π_θ(y_t|s_t)      score-function on the sampled action
ℛ_direct[a] =  ∇_θ a                              pathwise through a's θ-dependence
ℛ_k2[r]     =  ∇_θ ½r²  =  r · ∇_θ log π_θ(y_t)   potential-mediated: a surrogate whose
                                                   pathwise gradient equals the
                                                   score-function field at ratio = 1
```

(Code realizes these as surrogate losses — the PPO-clipped surrogate for ℛ_PG —
which coincide with the fields above at ratio = 1; immediate credit, no
baseline, pinned roster-wide.)

**Selection layer** — the finite-batch, sequence-balanced estimator (the
shipped normalization convention, stated as the estimator it is; fields
aggregate linearly):

```
ĝ = ( 1 / Σ_i w_i ) Σ_i  w_i · [ ( 1 / Σ_t m_it ) Σ_t  m_it · g_it ]
```

| stage | values in the roster | owning axis |
|---|---|---|
| `μ` | λ ∈ {0, 0.5, 1}; T_max ∈ {100, 16k}; n pinned at 1 (n=8 cell archived with the J axis, 2026-08-11) | A; h5 |
| `w` | 1 / 1[pass] / 1[fail] / quota-K / directional / likelihood | G |
| `m` | m_t = M_{ρ,P}(t, s_t, p_t^θ, p_t^T) ∈ [0,1]; budget ρ, selection rule P | D∪H |
| `Q_Ω` | {y_t} / top-k / quantile-budget / intersection / π-tail | C |
| `N_ν` | raw / renorm / tailbucket | C-internal (`SUPPORT_MODE`) |
| `S_σ` | id (values) / z-score / rank / set-mass | E |
| `D_δ` | RKL / FKL / skew_α / JSD_β / power comparator (f3) | B (+ f3, see §4) |
| `𝒬` | MC at sampled token / exact support sum | roster-confounded with Ω (§3) |
| `Φ_φ` | id / soft-log / clip_±M / positive-only clip / tanh_M | F (f1, f2, f4, f5, f2@2.303) |
| `ℛ` | PG / direct / k2-potential | crossover pairs (§3) |

**m_t is token-level supervision allocation, one coordinate for D and H.**

```
m_t = M_{ρ,P}( t, s_t, p_t^θ, p_t^T ) ∈ [0, 1]
```

with ρ the supervision budget (coverage) and P the selection rule. P may read
**position only** — h1/h2/h3 first/last/random window, h4 scatter — or the
**distributions themselves** — d1 TIP high-entropy, d2 propose-verify, d3
teachability (disagreement/compatibility); soft weights (d2's rejected-token
β=0.01) are the continuous case. D and H therefore share one design
coordinate. They stay separate *experimental* axes because they ask different
scientific questions — **position locality** (does WHERE supervision lands
matter) vs **information locality** (does WHICH tokens, by content, matter) —
and the DH1 bridge cells read the two against each other at matched ρ.

**Why the Ω/ν split is load-bearing.** Renormalizing a singleton support
collapses both sides to 1 (`log 1/1 = 0`) — vanilla *requires* `(Ω={y_t},
ν=raw)`. e2's set statistic `Σ_Ω p^θ(v)` requires `ν=raw` on a top-k support —
renormalized it is identically 1. Support choice and normalization are
independent coordinates; c1's `SUPPORT_MODE=renorm|tailbucket` internal
ablation lives exactly on ν.

**Why D / 𝒬 / ℛ are three different things.**

- **k1 is not "an estimator choice parallel to k2".** It is D_RKL's pointwise
  integrand `r(v) = log p(v) − log q(v)` observed at the sampled token — i.e.
  `𝒬 = MC`. It happens that `E_{y~p}[r] = KL(p‖q)`, so the same object is also
  a single-sample scalar estimator; that is a property, not a definition.
- **k2 is not an RKL estimator at all** (`E[½r²] ≠ KL`). It is a *potential*
  inside ℛ, designed so that `∇½r² = r·∇log π` — a gradient surrogate, living
  at the realization layer.
- **C1 is the existence proof that 𝒬 ≠ ℛ.** c1 evaluates its renorm-KL exactly
  (`𝒬 = exact support sum`, fully pinned) and *still* admits two realizations:
  ℛ_PG (`sg(D̂_t)·∇log π(y_t)`, the official code) and ℛ_direct (`∇D̂_t`, the
  gap cell). Same evaluated scalar, genuinely different fields.

## 2. Vanilla as the alignment point

Vanilla OPD is the operator's origin, slot by slot:

```
μ = π_θ̄ (pure on-policy),  w ≡ 1,  m_t ≡ 1,  Ω_t = {y_t},  ν = raw,  S = id,
D = KL(p^θ ‖ p^T),   𝒬: r_t = log p_t^θ(y_t) − log p_t^T(y_t),   Φ(r) = r,
ℛ_PG(r) = r_t · ∇_θ log π_θ(y_t|s_t)
```

At this point the fixed-state identity holds:

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
`ν=renorm` restores closure within Ω, or the weights are detached. The
roster's kernels are all renorm/detach (`topk_losses.py:684` witness), so that
failure class is structurally absent here (cf. Many Faces 2605.11182,
failure-2).

```
E_{y_t ~ p_t^θ} [ r_t · ∇_θ log p_t^θ(y_t) ]  =  ∇_θ KL(p_t^θ ‖ p_t^T)
```

**This identity is the origin of the whole unified framework.** Every other
arm breaks exactly one link of the chain (relative to its carrier, §5), and
the paper reads what each break does to the update field and the dynamics.

## 3. The update field is the object, not the scalar

`ℛ_PG[a]` is in general **not** the gradient of any scalar built from `a`: for
a distribution-valued comparison (c1's renorm-RKL), `sg(d_t)·∇log π_θ(y_t)` is
a different vector field from `∇d_t`. They coincide only at special points —
§2's origin is one, and that is vanilla's derivational status.

**Identifiability (revised under the 𝒬/ℛ split).**

- **ℛ is identifiable**, via two matched pairs, one at each 𝒬 value:
  vanilla ↔ b5 (ℛ swap at `𝒬 = MC`) and c1 ↔ c1-direct (ℛ swap at
  `𝒬 = exact`; the latter is gap cell **C1**, the fourth corner of the
  (evaluation × realization) 2×2).
- **𝒬 is roster-confounded with Ω**: sampled evaluation ⟺ singleton support in
  every existing arm. The decoupling cell — MC evaluation *within* a top-k
  support (`z ~ q̃_Ω`) — exists in principle and is not instantiated;
  disclosed, not claimed.

**The b5 walk-through (framework-derived, registered analytic note).** b5
shares D (RKL), 𝒬 (MC at the sampled token), and the effective signal `r` with
vanilla; **only ℛ moves** (PG → k2-potential). Since
`∇½r² = r·∇log π = sg(r)·∇log π` numerically, the two fields coincide exactly
at ratio = 1, and the framework predicts b5 ≈ vanilla wherever the ratio stays
at 1. The banked readings agree: lock 122 (vanilla) → 125 (b5) — the 3-step Δ
the Fisher note recorded — now read as the *surrogate-machinery residual*
(within-step micro-batch ratio drift + clipping), not an estimator effect.
Full-curve check remains an M-task before the verdict wording.

## 4. Design coordinates ≠ emergent mechanism coordinates

- **b1** (skew α=0.1) is a **D-move**; its positive-side bound
  `c_t ≤ −log α = ln 10` is an *emergent* property of the mixture, not a Φ
  setting.
- **f3** (`p_T^α − p_θ^α`) is **not** `Φ(Δℓ)` — Δℓ does not determine
  `p_T − p_θ`. It swaps log-ratio geometry for probability geometry: an
  out-of-family **bounded comparator** (a D-move with Φ=id). f1/f2 are the
  true amplitude transforms.

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
S or D off the sampled point mechanically drags Ω, ν, 𝒬, ℛ along (the space is
a stratified product). The actual design discipline is:

> **Each axis varies one scientific coordinate relative to an axis-specific
> carrier configuration; mechanically required carrier coordinates are held
> fixed within the axis.**

| axis | carrier (held fixed) | scientific coordinate |
|---|---|---|
| A | vanilla pipeline | μ.λ |
| h5 | vanilla pipeline | μ.T_max |
| G | vanilla local pipeline | w |
| D∪H | vanilla local pipeline | m (ρ; P positional ↔ informational) |
| C | 𝒬=exact, ℛ=direct, D=renorm-RKL | Ω rule / ν / τ-scope |
| B | Ω=top-32 renorm, 𝒬=exact, ℛ=direct (b1 rides the sampled carrier) | D (family, β) |
| E | Ω=top-32, 𝒬=exact, ℛ=direct; ν as S requires (rank/z shift-invariant; set needs raw) | S |
| F | vanilla sampled carrier (𝒬=MC, ℛ=PG) | Φ (f1, f2); f3 = D-move (§4) |

Cross-axis reads must route through carriers. Example: the S-ladder's "values"
rung *on the E carrier* is c1-**direct** — gap cell C1 — because shipped c1 is
ℛ=PG (official code). That is the third of C1's three identities (2×2 fourth
corner; S-ladder branch repair; cheapest verdict on the roster).

Placement notes: `h5_gen100` is a **μ-move** (T_max 16k→100, m≡1); `h1` is the
m-move (T_max=16k, m=1[t≤100]) — the registered bridge pair between
supervision locality and rollout locality. `h4_random_scatter` stays an m-move
(scatter placement).

## 6. Selection layer: conditioning, not reweighting

The double conditional-mean estimator makes the G/H algebra automatic:

- `w_i = 1[R_i > 0]` yields `(1/B_pass) Σ_{pass} L_i` — the shipped
  mask + `T/T_keep` rescale + token-mean is this conditional mean's *algebraic
  implementation*, not a method coordinate.
- `m_it = 1[t ≤ 100]` yields `(1/100) Σ_{t≤100} ℓ_it` — same algebra at token
  level.
- **g1_quota changes no objective.** The estimand stays `E[L | R > 0]`; the
  quota freezes `μ_θ̄` until `B_pass = K`, pinning the estimator's sample
  size — a **variance intervention on the same estimand**. M4's
  famine-amplifier account is exactly "the conditional-mean estimator at
  B_pass ≈ 1–5 has catastrophic variance"; the quota cell is its control.

## 7. Outside the operator (pinned and disclosed)

Initialization `θ₀` (a2). Objective mixing `L_total = L_task + c·L_OPD` — the
J boundary asks whether the local distillation operator needs a global task
objective to close the loop. The J axis is archived **in full** (2026-08-11):
`j1_kdrl` and its n=8 control — a reward-dominated cell is hard to claim as
OPD. The boundary question stands; the completed cells are banked as appendix
material, and `μ.n` is pinned at 1 roster-wide. Execution: synchronous, zero
staleness. Temperature τ=1 (a global re-parameterization that composes with
every stage — not a coordinate; ruling 2026-08-11). Temporal credit:
immediate, no baseline (the return-to-go / GAE family is a disclosed pinned
coordinate, cf. survey 2606.22793).

## 8. Insight machinery

- **Dose lines** = sections along one design coordinate at a fixed carrier
  (λ line, β ladder, τ-scope ladder, position line).
- **Screening-off tests** live on emergent coordinates (M; §4).
- **Effect decomposition** via off-diagonal cells (the (𝒬 × ℛ) matched pairs;
  Δlock: realization swap 3 steps vs bound swap +125).
- **Attractor universality** via the near-placebo (e2_a0: is the length
  attractor reachable with almost no training pressure — a statement about the
  objective landscape, not any method).
- **Composability** (the recipe cell): whether per-coordinate optima compose —
  the framework's final falsifiable prediction.

## 9. Relation to prior unifications

f-divergence unifications (DistiLLM, GKD's JSD) parameterize `D` alone; GKD's
λ is `μ`. The formula-driven survey (2606.22793) taxonomizes similar axes with
no experiments; this framework differs in being an *operator* account (𝒰 ↦
update field, with the semi-gradient discipline explicit) instantiated by
controlled single-coordinate probes with measured dose-response. Related-work
list: `ARM-SOURCES.md` final section.
