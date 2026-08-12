# UNIFIED-LOSS — the unified OPD training operator

Registered 2026-08-11; revised three times the same day after the formula
review rounds (ledger: "统一框架公式评审", "ℰ 再拆", "定稿书写形式").
**v2, 2026-08-12 (user formula review): the seven slots are grouped into four
macro-coordinates Γ/𝒜/𝒞/𝒰 (§0); the boxed operator now carries the double
conditional mean the code actually ships (the earlier draft normalized m but
not w — at G-axis pass rates of 0.5–2% that is a 50–200× factor, not a
formality); the support-restriction operator is renamed `Res_Ω` to break the
visual collision with the evaluation operator 𝒬; and the emergent magnitude
coordinate is renamed M(𝒰) → M[g], because the bound is NOT a function of the
(Φ,ℛ) pair alone — b1 and f3 induce theirs from 𝒞 (§4).**
Companion to `MECHANISMS.md`; measured early readings against this frame in
`expansion-early-readings.md`. The object of study is not a scalar loss but the
**Unified OPD Training Operator** — the update field written as one composed
expression.

## 0. Canonical form: four macro-coordinates

For prompts `x ~ 𝒟`, a frozen behavior policy generates `y ~ μ_θ̄(·|x)`, with
visited states `s_t = (x, y_<t)`, student `p_t^θ = π_θ(·|s_t)` and teacher
`p_t^T = π_T(·|s_t)`. The update field is

```
         E_{x,y} [ w(y) · (1/Σ_t m_t) · Σ_t  m_t ·
g(θ)  =  ────────  ℛ( Φ( 𝒬[ D( S(N_ν(Res_Ω p_t^θ)),  S(N_ν(Res_Ω p_t^T)) ) ] ) ) ]
         E_{x,y}[ w(y) ]
```

The `/E[w]` is not decoration: the shipped estimator is the **double conditional
mean** (§6) — `ĝ = (1/Σ_i w_i) Σ_i w_i [ (1/Σ_t m_it) Σ_t m_it g_it ]` — whose
estimand is `E[·|w=1]`, not `E[w·(·)]`. At g1's measured pass rates (0.5–2%)
the two differ by 50–200×, and the g1_quota cell's entire variance-intervention
reading (§6) lives on the conditional-mean form. Both selection weights are
normalized, symmetrically.

The design space is summarized by four macro-coordinates:

```
g(θ) = g(θ; Γ, 𝒜, 𝒞, 𝒰)

Γ = (μ, w, m)      where and what to teach
𝒜 = (Ω, ν, S)      what information is available
𝒞 = (D, 𝒬)         how teacher and student are compared
𝒰 = (Φ, ℛ)         how the comparison becomes an update
```

- **Γ — where and what to teach.** `μ` fixes the visited states (λ mixing,
  T_max, n); `w(y)` selects whole trajectories; `m_t` allocates token-level
  supervision — positional windows, random scatter, entropy/disagreement
  criteria (§1's m-coordinate, shared by the D and H axes).
- **𝒜 — what information is available.** `Res_Ω` restricts the vocabulary
  support; `ν` fixes how support/tail mass is represented (raw / renorm /
  tail-bucket); `S` fixes the information resolution kept from each side
  (value / z-shape / rank / set membership).
- **𝒞 — how they are compared.** `D` is the discrepancy geometry (RKL / FKL /
  skew / JSD / power comparator); `𝒬` its finite evaluation (exact support sum
  vs Monte-Carlo at the sampled token — "student-MC").
- **𝒰 — how the comparison becomes an update.** `Φ` shapes the effective
  signal (id / soft-log / clip / positive-only clip / tanh); `ℛ` realizes the
  parameter update (score-function/PG, direct differentiation, k2-potential).

Two facts keep the macros honest:

1. **They are not independently settable.** The space is a stratified product:
   moving `S` off the sampled point mechanically drags `Ω, ν, 𝒬, ℛ` along —
   i.e. an 𝒜 move forces carrier choices inside 𝒞 and 𝒰. The experimental
   discipline is therefore per-axis carriers (§5), never a free four-factor
   grid, and cross-macro reads route through carriers.
2. **Design macros are not mechanism coordinates.** Two nominally different
   methods can be mechanically equivalent if they induce the same update
   geometry, and two implementations of one nominal objective can behave
   differently if they differ in ℛ. Both directions now exist as
   *measurements*, not just claims: b5 ↔ vanilla (fields coincide at ratio = 1;
   banked Δlock of 3 steps) and c1 ↔ c1_direct (same evaluated scalar,
   +0.221 val@25, 47× seed-spread change, stability flip —
   `expansion-early-readings.md` §1).

Outside the operator, pinned and disclosed (§7): the J axis's `L_task + c·L_OPD`
coupling, initialization θ₀ (a2), temperature τ=1 (a global re-parameterization,
not a coordinate), `μ.n` pinned at 1 roster-wide, synchronous execution,
immediate credit with no baseline.

**Sign convention.** `g` as written is the (per-slot generalized) KL gradient;
the parameter update is `θ ← θ − ηg`. Code-side, verl feeds the *advantage*
`Δℓ_t = −r_t` into its PPO surrogate (ascent on Δℓ = descent on KL). All
shipped loss-side transforms are odd functions, so the two conventions commute
— the f1 argument, and the reason it must stay true for every Φ member.

The paper's program in one line: coordinate change → gradient field → training
dynamics. "OPD" is not a loss; it is this whole set of coupled design choices,
and every arm in `configs/arms.yaml` is one controlled move inside it. The
per-slot detail follows.

| slot | macro | role | values |
|---|---|---|---|
| `μ_θ̄` | Γ | trajectory/state source | on-policy / mixed λ / rollout horizon T_max |
| `w(y)` | Γ | trajectory-level selection | 1 / pass-only / fail-only / quota-K / directional / likelihood |
| `m_t = M_{ρ,P}(t, s_t, p^θ, p^T)` | Γ | token-level supervision allocation | ρ = how much; P = which tokens (position / entropy / disagreement); soft ∈ [0,1] |
| `Res_Ω` | 𝒜 | vocabulary support | {y_t} / top-k / quantile-budget / intersection / π-tail |
| `N_ν` | 𝒜 | support normalization / mass representation | raw / renorm / tail-bucket |
| `S` | 𝒜 | teacher information resolution | value / z-shape / rank / set |
| `D` | 𝒞 | teacher–student discrepancy geometry | RKL / FKL / skew / JSD (f3's power comparator: recorded out-of-family, §4) |
| `𝒬` | 𝒞 | finite evaluation | student-MC at the sampled token / exact support sum |
| `Φ` | 𝒰 | effective signal shaping | id / soft-log / clip / positive-only clip / smooth bound (tanh) |
| `ℛ` | 𝒰 | update realization | score-function (PG) / direct gradient / k2-direct surrogate |

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
u_t^a = N_ν( Res_Ω[ p_t^a ] )   support restriction (Res_Ω), then representation (N_ν)
z_t^a = S_σ( u_t^a )            statistic
D_δ                             discrepancy geometry on (z^θ, z^T),
                                with pointwise integrand d(v) where one exists
c_t   = 𝒬[ D_δ ]                evaluation: exact support sum, or MC at the
                                sampled token   (Res_Ω restricts; 𝒬 evaluates —
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
aggregate linearly). This is the batch form of §0's double conditional mean:

```
ĝ = ( 1 / Σ_i w_i ) Σ_i  w_i · [ ( 1 / Σ_t m_it ) Σ_t  m_it · g_it ]
```

| stage | values in the roster | owning axis |
|---|---|---|
| `μ` | λ ∈ {0, 0.5, 1}; T_max ∈ {100, 16k}; n pinned at 1 (n=8 cell archived with the J axis, 2026-08-11) | A; h5 |
| `w` | 1 / 1[pass] / 1[fail] / quota-K / directional / likelihood | G |
| `m` | m_t = M_{ρ,P}(t, s_t, p_t^θ, p_t^T) ∈ [0,1]; budget ρ, selection rule P | D∪H |
| `Res_Ω` | {y_t} / top-k / quantile-budget / intersection / π-tail | C |
| `N_ν` | raw / renorm / tailbucket | C-internal (`SUPPORT_MODE`) |
| `S_σ` | id (values) / z-score / rank / set-mass | E |
| `D_δ` | RKL / FKL / skew_α / JSD_β / power comparator (f3) | B (+ f3, see §4) |
| `𝒬` | student-MC at sampled token / exact support sum | roster-confounded with Ω (§3) |
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
ablation lives exactly on ν — and its execution (`c1_tailbucket`, wave 15) is
tracking shipped c1 on every panel so far, the first direct measurement that ν
is not load-bearing (`expansion-early-readings.md` §2).

**Why D / 𝒬 / ℛ are three different things.**

- **k1 is not "an estimator choice parallel to k2".** It is D_RKL's pointwise
  integrand `r(v) = log p(v) − log q(v)` observed at the sampled token — i.e.
  `𝒬 = student-MC`. It happens that `E_{y~p}[r] = KL(p‖q)`, so the same object
  is also a single-sample scalar estimator; that is a property, not a
  definition.
- **k2 is not an RKL estimator at all** (`E[½r²] ≠ KL`). It is a *potential*
  inside ℛ, designed so that `∇½r² = r·∇log π` — a gradient surrogate, living
  at the realization layer.
- **C1 is the existence proof that 𝒬 ≠ ℛ.** c1 evaluates its renorm-KL exactly
  (`𝒬 = exact support sum`, fully pinned) and *still* admits two realizations:
  ℛ_PG (`sg(D̂_t)·∇log π(y_t)`, the official code) and ℛ_direct (`∇D̂_t`, the
  gap cell). Same evaluated scalar, genuinely different fields — and as of
  wave 15 the two fields are *measured* to differ enormously (§0, fact 2).

## 2. Vanilla as the alignment point

Vanilla OPD is the operator's origin, macro by macro:

```
Γ₀ = (π_θ̄, 1, 1)          pure on-policy; every trajectory; every token
𝒜₀ = ({y_t}, raw, id)      sampled-token support; raw mass; full value resolution
𝒞₀ = (RKL, student-MC)     reverse KL, evaluated at the sampled token
𝒰₀ = (id, score-function)  unshaped signal, PG realization
```

equivalently, slot by slot:

```
μ = π_θ̄,  w ≡ 1,  m_t ≡ 1,  Ω_t = {y_t},  ν = raw,  S = id,
D = KL(p^θ ‖ p^T),   𝒬: r_t = log p_t^θ(y_t) − log p_t^T(y_t),   Φ(r) = r,
ℛ_PG(r) = r_t · ∇_θ log π_θ(y_t|s_t)
```

At this point the fixed-state identity holds:

```
∇_θ KL(p_t^θ ‖ p_t^T)
  = Σ_v ∇p^θ(v) · ( Δℓ(v) + 1 )          Σ_V ∇p^θ = 0 kills the +1
  = E_{y_t ~ p_t^θ} [ r_t · ∇_θ log π_θ(y_t|s_t) ]
```

> **Vanilla OPD is a special point where on-policy sampling, reverse-KL
> geometry, the sampled log-ratio, and the score-function realization align
> exactly at each visited state; the surrounding design space breaks these
> identities in controlled ways.**

The `+1` dies because the score sums to zero over the FULL support. On a
truncated support the sum no longer closes and the +1 bias revives — unless
`ν=renorm` restores closure within Ω, or the weights are detached. The
roster's kernels are all renorm/detach (`topk_losses.py:684` witness), so that
failure class is structurally absent here (cf. Many Faces 2605.11182,
failure-2).

**This identity is the origin of the whole unified framework.** Every other
arm breaks exactly one link of the chain (relative to its carrier, §5), and
the paper reads what each break does to the update field and the dynamics —
the measured mechanism coordinates being support mass, effective sample count,
admitted signal tails, entropy, sequence length, truncation, and time-to-lock.

## 3. The update field is the object, not the scalar

`ℛ_PG[a]` is in general **not** the gradient of any scalar built from `a`: for
a distribution-valued comparison (c1's renorm-RKL), `sg(d_t)·∇log π_θ(y_t)` is
a different vector field from `∇d_t`. They coincide only at special points —
§2's origin is one, and that is vanilla's derivational status.

**Identifiability (revised under the 𝒬/ℛ split).**

- **ℛ is identifiable**, via two matched pairs, one at each 𝒬 value:
  vanilla ↔ b5 (ℛ swap at `𝒬 = student-MC`) and c1 ↔ c1_direct (ℛ swap at
  `𝒬 = exact`; the latter is gap cell **C1**, the fourth corner of the
  (evaluation × realization) 2×2 — enlisted as wave 15 and, in its first 40
  steps, showing the largest single-coordinate effect measured on the roster:
  +0.221 val@25, 47× tighter seed spread, and the loss of c1's
  never-excursion property; the recovery-vs-lock fork is registered in
  `expansion-early-readings.md` §1).
- **𝒬 is roster-confounded with Ω**: sampled evaluation ⟺ singleton support in
  every existing arm. The decoupling cell — MC evaluation *within* a top-k
  support (`z ~ q̃_Ω`) — exists in principle and is not instantiated;
  disclosed, not claimed.

**The b5 walk-through (framework-derived, registered analytic note).** b5
shares D (RKL), 𝒬 (student-MC), and the effective signal `r` with vanilla;
**only ℛ moves** (PG → k2-potential). Since
`∇½r² = r·∇log π = sg(r)·∇log π` numerically, the two fields coincide exactly
at ratio = 1, and the framework predicts b5 ≈ vanilla wherever the ratio stays
at 1. The banked readings agree: lock 122 (vanilla) → 125 (b5) — the 3-step Δ
the Fisher note recorded — now read as the *surrogate-machinery residual*
(within-step micro-batch ratio drift + clipping), not an estimator effect.
Full-curve check remains an M-task before the verdict wording.

## 4. Design coordinates ≠ emergent mechanism coordinates

- **b1** (skew α=0.1) is a **𝒞 move** (D-family); its positive-side bound
  `c_t ≤ −log α = ln 10` is an *emergent* property of the mixture, not a Φ
  setting.
- **f3** (`p_T^α − p_θ^α`) is **not** `Φ(Δℓ)` — Δℓ does not determine
  `p_T − p_θ`. It swaps log-ratio geometry for probability geometry: an
  out-of-family **bounded comparator** (a 𝒞 move with Φ=id). f1/f2 are the
  true amplitude transforms (𝒰 moves).

The compression curve therefore lives on an **emergent mechanism coordinate**

```
M[g] = the admitted-signal bound of the update field g
```

written `M[g]` and NOT `M(𝒰)`, because the bound is not a function of the
(Φ,ℛ) macro alone: arms from **different macros** project onto it —
{vanilla ∞ (origin), f2 ±10 (𝒰), f1 log (𝒰), b1 ln10 (𝒞), f3 [−1,1] (𝒞)} →
locks {122, 198, 208, 247, never}, strictly monotone 5/5 (M1 first harvest).
The magnitude-unification hypothesis is precisely: **training dynamics factor
through M[g], screening off which macro-coordinate induced the bound**.
`f2_clip2.3` — a 𝒰-induced bound pinned at b1's 𝒞-induced value 2.303 — is the
calibrated intervention (registered lock ≈ 247; running as wave 14), and M1 the
measurement. This also settles "merge B and F?": **no** — distinct design
macros that project onto one mechanism coordinate, and the projection itself is
the finding.

## 5. Global origin and axis-local carriers

"Each arm moves exactly one coordinate from vanilla" is not achievable: moving
S or D off the sampled point mechanically drags Ω, ν, 𝒬, ℛ along (the space is
a stratified product — §0 fact 1, restated here where the axis table lives).
The actual design discipline is:

> **Each axis varies one scientific coordinate relative to an axis-specific
> carrier configuration; mechanically required carrier coordinates are held
> fixed within the axis.**

| axis | carrier (held fixed) | scientific coordinate |
|---|---|---|
| A | vanilla pipeline | Γ.μ.λ |
| h5 | vanilla pipeline | Γ.μ.T_max |
| G | vanilla local pipeline | Γ.w |
| D∪H | vanilla local pipeline | Γ.m (ρ; P positional ↔ informational) |
| C | 𝒬=exact, ℛ=direct, D=renorm-RKL | 𝒜.Ω rule / 𝒜.ν / τ-scope |
| B | Ω=top-32 renorm, 𝒬=exact, ℛ=direct (b1 rides the sampled carrier) | 𝒞.D (family, β) |
| E | Ω=top-32, 𝒬=exact, ℛ=direct; ν as S requires (rank/z shift-invariant; set needs raw) | 𝒜.S |
| F | vanilla sampled carrier (𝒬=student-MC, ℛ=PG) | 𝒰.Φ (f1, f2, f4, f5); f3 = 𝒞 move (§4) |

Cross-axis (and hence cross-macro) reads must route through carriers. Example:
the S-ladder's "values" rung *on the E carrier* is c1-**direct** — gap cell C1
— because shipped c1 is ℛ=PG (official code). That is the third of C1's three
identities (2×2 fourth corner; S-ladder branch repair; cheapest verdict on the
roster — all three now funded and running as wave 15).

Placement notes: `h5_gen100` is a **Γ.μ move** (T_max 16k→100, m≡1); `h1` is
the Γ.m move (T_max=16k, m=1[t≤100]) — the registered bridge pair between
supervision locality and rollout locality. `h4_random_scatter` stays a Γ.m
move (scatter placement).

## 6. Selection layer: conditioning, not reweighting

The double conditional-mean estimator (§0's boxed form; batch form in §1)
makes the G/H algebra automatic:

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
  (This is why §0's population form carries the `/E[w]`: without it the
  operator would claim the `E[w·L]` estimand, which is not what ships and not
  what the quota cell controls.)

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
- **Screening-off tests** live on emergent mechanism coordinates (M[g]; §4).
- **Effect decomposition** via off-diagonal cells (the (𝒬 × ℛ) matched pairs;
  Δlock: realization swap 3 steps vs bound swap +125 — and wave 15's c1
  triangle decomposing speed/variance/stability across ℛ and ν).
- **Attractor universality** via the near-placebo (e2_a0: is the length
  attractor reachable with almost no training pressure — a statement about the
  objective landscape, not any method).
- **Composability** (the recipe cell): whether per-coordinate optima compose —
  the framework's final falsifiable prediction.

## 9. Relation to prior unifications

f-divergence unifications (DistiLLM, GKD's JSD) parameterize `D` alone — one
slot of 𝒞; GKD's λ is Γ.μ. The formula-driven survey (2606.22793) taxonomizes
similar axes with no experiments; this framework differs in being an *operator*
account (design tuple ↦ update field, with the semi-gradient discipline
explicit) instantiated by controlled single-coordinate probes with measured
dose-response. Related-work list: `ARM-SOURCES.md` final section.
