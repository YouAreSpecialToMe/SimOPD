# Expansion waves 9–15 — early readings across every panel, and the registered forks

_Written 2026-08-12 during the fleet outage, from three frozen sources: the completed
16k campaign's per-step tables (`training-dynamics.md`, `late-training-collapse.md`,
`campaign_16k_report.md`), the auto-published step-25 expansion table (commit
`cb1b90b`), and the last live `watch.py` snapshot before connectivity dropped
(2026-08-12 06:04Z; runs then at steps ~35–70). Nothing here is a verdict: verdicts
wait for step 250 + the offline suite. What this document does is (a) read every
panel together — val, length, truncation, entropy, seed spread, s/it — rather than
accuracy alone, and (b) **register the forks now**, before the cluster comes back
with the deciding steps._

---

## 0. The lens everything must be read through: the step-50 excursion is universal

The completed campaign's own curves, at matched steps (mean rollout length / truncation):

| arm | @25 | @50 | @75 | @100 | eventual lock |
|---|---|---|---|---|---|
| `vanilla` | 1853 / 0.013 | **12872 / 0.701** | 9863 / 0.240 | 10711 / 0.232 | 122 |
| `f2_hard_clip` | 1792 / 0.013 | **12429 / 0.646** | 9435 / 0.193 | 9398 / 0.099 | 198 |
| `b1_skew_kl` | 1688 / 0.013 | **10728 / 0.458** | 8938 / 0.102 | 9195 / 0.073 | 247 |
| `c2_quantile_budget` | 1762 / 0.010 | **10082 / 0.385** | 9239 / 0.122 | 8990 / 0.076 | never |
| `c1_lsm_topk32_renorm` | 1436 / 0.013 | **1318 / 0.010** | 1267 / 0.016 | 1132 / 0.008 | never |

**Every arm on the roster except c1 blows up at step ~50 and recovers by 75.** Even
never-locking c2 hit 0.385 truncation at 50. So an expansion arm showing 0.3–0.6
truncation at steps 40–50 is, by itself, **exactly on the universal excursion and
diagnostic of nothing**. What is diagnostic:

1. **whether it excursions at all** (only c1 never did — so a c1-derived arm
   excursing IS a signal even at step 40);
2. **whether it recovers by ~75** (all originals did);
3. **when it relapses** — the ratchet stat: of excursion-recoveries, **40/46
   later relapsed into lock** (`MECHANISMS.md` M-I). Recovery is the norm;
   *staying* recovered is the rare property (c2/c4/c3/e1/j1/h1's ~6 exceptions).

Every claim below is phrased against this lens. An earlier live read of these same
numbers ("f2_clip2.3 is running away far earlier than b1") was wrong for ignoring
it — b1's own step-50 point was 10.7k/0.458, indistinguishable from what
f2_clip2.3 shows now.

---

## 1. The realization axis ℛ is live, large, and two-faced — the c1 triangle

Three arms, one loss, one support (teacher top-32, renorm), one exact evaluation.
Only the marked coordinate moves:

| panel | `c1` (ℛ=PG, shipped; complete) | `c1_direct` (ℛ moved; @~40) | `c1_tailbucket` (ν moved; @~60) |
|---|---|---|---|
| val @25 | 0.417 ± **0.094** | **0.638 ± 0.002** | 0.498 ± 0.065 |
| val @latest | 0.487±0.038 @50 → 0.524 @250 | 0.636–0.640 @~40 | 0.528–0.566 @~60 |
| rollout len | 1318 @50; **never leaves ~1.3k** | **9.8k–11.7k** | 1.2k–1.6k |
| truncation | 0.010 @50; max 0.016 all run | **0.37–0.48** | 0.02 |
| s/it | ~109 (7.6 wall-h/seed, 45 GPU·h/arm) | **601–738** | 72–122 |
| entropy | 1.9–2.6 early (roster's only high-entropy arm) | (not in snapshot) | (not in snapshot) |
| EOS-density proxy (1−trunc)/len, per ktok | ~0.75 | **~0.05** | ~0.70 |

Three readings, each on a different panel — none available from accuracy alone:

- **Speed and variance are branch properties.** At step 25 the direct form is
  +0.221 over shipped c1 with a **47× tighter seed spread** (±0.002 vs ±0.094 —
  and c1's full-campaign σ̄ 0.031 was the roster's worst). The PG realization on a
  divergence-valued loss is not just slower; it is *noisier across seeds*, which
  the campaign had been attributing to the arm rather than to its estimator.
- **Stability was also a branch property.** c1 is the single arm in 29 that never
  excursions. c1_direct excursed by step 40 — under lens rule (1), the excursion's
  *presence* is already a branch effect. The 16k campaign's "renorm-safe family"
  reading (`MECHANISMS.md` M-I probe list) conflated support discipline with the
  PG brake; the triangle now separates them.
- **The stop-supervision account applies forward.** c1_direct's EOS-density proxy
  is already ~14× thinner than its two siblings'. If M-I's starvation loop is
  right, its fate is decided by whether the keep-rule-less renorm objective lets
  lengths re-contract (c2-class) or not (b5-class).

**Registered fork (before data):**
- `P-c1d-lock` — c1_direct locks ≤150 and in-loop val falls ≥0.10 from peak.
  Verdict wording: *"PG was simultaneously c1's handbrake and its seatbelt: the
  literature configuration (LSM's official code) buys its no-collapse stability by
  an estimator artifact, at the price of +0.2 val and 47× seed variance."*
- `P-c1d-c2class` — lengths re-contract ≤ ~8.5k, val ≥0.60 at 250, no lock.
  Verdict wording: *"the literature default is strictly dominated: the direct form
  is faster, tighter, AND stable — fixed-k truncation was never the problem."*
- Lean, stated for the record: **P-c1d-lock**, weakly — its truncation at 40
  (0.37–0.48) already exceeds c2's excursion peak (0.385 @50) with no keep-rule
  brake present, and 87% of recoveries relapse. Either branch of the fork is a
  headline-grade sentence about the (𝒬×ℛ) 2×2.

## 2. The ν coordinate is not load-bearing — tailbucket tracks c1 on every panel

Through step ~60, `c1_tailbucket` matches shipped c1 on val trajectory (0.53–0.57
vs c1's 0.49–0.50 over the same window), length (1.2–1.6k vs 1.3k), truncation
(0.02 vs 0.01), and cost (72–122 s/it vs ~109). **P-identity leads on all four
panels**: what the support *keeps* (the k identities) appears to carry the signal;
how the discarded mass is *represented* (renormalized away vs kept as one bucket)
moves nothing so far. If this holds to 250 + suite, the ν question closes at
~50 GPU·h and full-vocab C3 stays deferred exactly as the identification-hole
ruling intended.

## 3. The magnitude account's keystone test is on track, not refuted

`f2_clip2.3` (hard clip at b1's measured bound 2.303; M1's calibrated prediction:
**lock ≈ 247**) sits at 8.9–11.5k / 0.30–0.59 truncation at step ~44 — which is
**b1's own step-50 excursion** (10.7k / 0.458) to within noise. Under the lens,
nothing has been tested yet. The deciding legs:

- recovery by ~75 (b1-like) — else P-M fails early;
- relapse timing: **lock in the 200–250 window confirms M1's dose→lock curve**
  transfers across design coordinates (skew-mixture → hard clip at matched bound);
  lock ≪200 falsifies the strong screening-off claim (dynamics do NOT factor
  through the emergent bound M(𝒰) alone).

`f4_posclip` (positive tail only) at 0.44–0.64 @~42 and `f5_tanh` (both tails,
smooth) at ~0.22 @~36 are likewise inside the universal excursion band. The
which-tail and kink-vs-smooth readings live in their recovery legs and lock steps,
not here. Registered leans: `P-f4-lock<f2` (the un-clipped negative tail is the
harmful one — consistent with M1's "the dose lives beyond p99.7 of the signed
tail"), `P-f5≈f2` (boundedness, not the kink, is the stabilizer).

## 4. The c2 ladder already decomposes c2's headline number

At step 25, with everything else equal to c2 except the keep rule:

| arm | keep rule | val @25 | Δ |
|---|---|---|---|
| `c1` (reference) | fixed top-32, renorm, **PG** | 0.417±0.094 | — |
| `c2_qb_fixed8` | **fixed top-8** of the k=64 payload, direct | 0.607±0.002 | +0.190 vs c1 |
| `c2_qb_perseq` | adaptive τ per sequence | 0.630±0.009 | +0.023 vs fixed8 |
| `c2` (shipped) | adaptive τ per batch | 0.635±0.001 | +0.005 vs perseq |

Read jointly: **the budget-matched fixed control captures ~85% of the c2−c1 gap**
— the adaptive allocation itself is worth ~+0.02–0.03 early, and the granularity
of its pinning (batch vs per-sequence, the recorded saturation-drift concern)
~+0.005, i.e. `P-harmless` leading for the wave-10 question. The big money was
never "adaptivity"; it is wherever fixed8 differs from c1 — branch and/or budget
size — which is exactly the decomposition c1_direct completes from the other end.
(Truncation panels: fixed8 0.12–0.20 @42, perseq 0.26–0.34 @35 — excursion band;
registered lean: both recover, c2-class.)

## 5. β ladder, one sentence

Early val orders b0.1 (FKL-lean) 0.549 < b0.9 (RKL-lean) 0.60 ≈ b4@0.5's own
early curve — a direction-matters hint (H2) — but both rungs are mid-excursion
and the H1/H2 adjudication is a full-curve question. No lean registered.

## 6. Cost is an outcome, again

Within one wave at one moment, s/it spans **50 → 796 (16×)**, tracking realized
length exactly as the cost-anatomy harvest found for the 16k campaign. Two
consequences: (a) any GPU·h forecast for the domain campaigns must be re-based on
*realized* lengths (the 4k cap makes code/IF structurally immune to the worst of
this); (b) `h5_gen100` measured the other extreme — three seeds to 250 in ~75 min
wall each, ≈ **7 GPU·h for the whole arm vs h1's 82** — the pre-registered "~50×
less generation" claim, now a measurement. Its in-loop val is structurally 0
(greedy@100-token budget); the h1↔h5 form-pair verdict is the offline suite's
alone, and its 30 checkpoints were first in the eval queue when the fleet went
dark.

## 7. What the outage does and does not cost

Checkpoints land every 25 steps on the shared filesystem: worst case ≤24 steps
per run re-trained, zero verdict impact. All 39 runs' claims, fingerprints, logs
and the two prepared domain datasets (code: 12,826 self-verified rows, all
surfaces CLEAN) are on `/mgfs`. Dead with the pods: campaign daemons, 20 eval
workers (their `evalq_exp` claims must be reaped before relaunch — that queue has
no refill reaper), and the patrol/publish daemon on dsw243.

## 8. Registered-readings checklist for the restart

| # | reading | criterion | settles |
|---|---|---|---|
| R1 | c1_direct recovery vs lock | trunc curve 75–150; val fall from peak | P-c1d fork (§1) |
| R2 | c1_tailbucket flat to 250 | no lock; val within c1±0.03; suite ≈ c1's 0.230 | ν closed; C3 stays deferred |
| R3 | f2_clip2.3 lock step | lock ∈ [200, 250] vs ≪200 | M1 screening-off, keystone |
| R4 | f4 vs f2 lock order; f5 vs f2 | lock steps | which-tail; kink-vs-smooth |
| R5 | fixed8/perseq recovery + finals | c2-class curves; Δ(c2, perseq) < noise | wave-10 granularity + adaptivity share |
| R6 | h5 suite composite vs h1's 0.303 | within ±0.02 | form-pair equivalence at 1/50 cost |
| R7 | β ladder full curves | H1 vs H2 pattern | b4's verdict wording |

_Every criterion above is stated before the deciding data exists; the live table
in `campaign_16k_report.md` (auto-published) will carry the numbers as they land._

---

## Addendum — the 06:19Z auto-publish (found in the push log after this document
was drafted; the outage began between 06:19 and ~07:30)

Four s50 readings landed that this document's snapshot predates:

| arm | s50 reading | bears on |
|---|---|---|
| `c1_tailbucket` | **0.549±0.019 @50, 0.540 @75·1** — vs shipped c1's 0.487±0.038 / 0.504±0.033 at the same steps | R2 strengthens: through 75 the ν probe now *leads* c1 by ~+0.04–0.05 at matched steps with a tighter spread, lengths still pinned. "Not load-bearing" may under-claim; "renorm is, if anything, slightly worse than keeping the tail bucket" becomes writable if this holds |
| `h4_random_scatter` | **0.639±0.004 @50** (from 0.561 @25) | the scatter member is climbing through its excursion — at 50 it already matches h3's own peak (0.649 @50); the window-vs-scatter reading (wave 12's tail-dose test) will be a live contrast, not a one-sided rout |
| `b4_jsd_b0.1` | 0.506 @50·1 (from 0.549 @25) | the FKL-leaning rung is degrading first — an H2-flavored (direction-matters) early point, still one seed |
| `e1_pl_rank_a0` | 0.475±0.004 @50 — vs anchored e1's 0.499 @50 | the pure-order rung sits ≈0.02 below its anchored parent early: the anchor's contribution is visible but small, exactly the ladder-purity question wave 11 exists to price |
