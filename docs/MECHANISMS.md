# The mechanism reorganization — same arms, same results, new shelving

_2026-08-11. This document re-sorts the EXISTING campaign — the A–J axes and every
banked result — under a mechanism-first taxonomy. Nothing is re-run, no protocol
changes, no verdict changes: the axes remain the method-space coordinates and the
arm-vs-vanilla paired tests remain the practice-level verdicts. What changes is
the organizing structure: mechanisms are the objects of study, methods are probes
at positions in mechanism space. Sources: `late-training-collapse.md`,
`training-dynamics.md`, `campaign_16k_report.md`, ledger entries in
`arm-provenance-r4.md` (375df7f, 2f8f3d5)._

## 0. The four mechanism classes

| class | object of study | one-line state |
|---|---|---|
| **M-I** | Termination / length dynamics — whether the student can stop, and why it loses the ability | richest evidence; missing its causal keystone |
| **M-II** | Entropy / sharpening dynamics — where the policy distribution collapses to | two documented pathologies; one armed ablation |
| **M-III** | Signal geometry — what information each token actually receives | instruments installed, several readings never taken |
| **M-IV** | Capability vs delivery — what the score measures | decomposition done once; needs the rescore + the suite |

The axes map onto the classes, not one-to-one: A (source/schedule) feeds M-I
(init, data dose) and M-II; B/F (divergence/conditioning) feed M-I (magnitude)
and M-II (estimator); C/D/E/G/H are M-III's coordinate system with M-I side
effects; the eval protocol is M-IV.

## M-I · Termination / length dynamics

**Banked evidence** (all read-only analyses of existing runs):
- The termination collapse itself: score = P(finish) × acc|finish; vanilla's
  acc|finish rises 0.550→0.900 while P(finish) falls 0.832→0.040; the loss is
  entirely in problems whose termination flipped; truncation is a one-way ratchet
  (0/500 returns).
- The terminal loop: 97.3% of a blown-up response is the teacher's own closing
  template repeated 696–1271×; the reasoning phase never got longer.
- The causal-loop candidate: truncated rollouts carry **no EOS token**, so
  stop-supervision density self-starves — vanilla 1670× collapse, c2 4.9× then
  stable; r(log EOS density, final) = +0.653, tighter than truncation itself.
- The compression dose-response (registered, falsifiable): five same-support
  same-branch arms, magnitude bound ∞/10/log/2.3/1 → lock 122/198/208/247/never.
  Correction 2026-08-13: lock LAGS harm — by the val-collapse criterion
  (≥0.10 off peak) the intervals are (100,125]/(175,200]/(175,200]/(200,225]/
  never, so f1/f2 tie at eval resolution and the strict 5/5 belongs to the lock
  metric only. b1 dies on BOTH scoreboards at 225 (suite 0.338→0.246; τ0.7/32k
  does not rescue it) while lock waits until 247: the damage is concurrent with
  the mid-ratchet (training on degenerate rollouts), and the harm→lock lag
  grows as the bound shrinks (≈0 / ≈0 / ≈8 / ≈22+).
- The magnitude-unification hypothesis (registered): the only two capped top-k
  arms (e2, b2) are exactly the unbounded losses.
- Termination pressure: b5↔j1 controlled pair (verifier zeroes a capped rollout).
- Initialization: a2's SFT stage delivers a termination-broken init (capped from
  step 31; teacher median 694 tokens — the length is trained, not imitated).
- **The position dose line** (registered, D×H cross-reading): where the
  supervised span sits controls termination; how much is supervised does not.
  Front window (h1, trunc 0.151) < random contiguous window (h3, 0.182) <
  criterion-scattered selection (d1/d2/d3, 0.990–1.000) < tail window (h2,
  locked from step 37). The budget refutation is d3 vs h1: 5% of tokens
  criterion-scattered caps at 1.000 while 0.6% front-windowed is safe. The
  ordering's middle step is confounded (scatter-vs-window × criterion ×
  budget) — the DH1 bridge cells split it.
- The early-excursion ratchet: spike-before-80 is close to decisive (40/46
  recoveries relapse); a free early-stopping signal for any follow-up campaign.

**Probe positions** (existing arms re-labeled):
{vanilla, f2, f1, b1, f3} = the magnitude dose ladder · {b4 vs b2; e2} = bounded
vs unbounded at distributional support · c1–c4 = the renorm-safe family (c1 never
excursions at all) · j1 = explicit termination pressure · a2 = init probe ·
h1/h3 = supervised-span probe · wave-9 β ladder (H3 = boundedness) and wave-10 QB
ladder = the two registered adjudicators.

**Missing**:
- **N1 (the keystone): a minimal causal intervention cell.** Every M-I claim is
  correlational-plus-mechanistic; j1 fixes termination but moves three knobs at
  once (reward, 500× coefficient, n=8 groups). The clean cell is
  `vanilla + a truncation-zeroing reward term and nothing else` — if that alone
  pins the length distribution, the EOS-starvation loop is causally closed.
  Amendment candidate (2 cards × 3 seeds, ~350 GPU·h at vanilla's own pace).
- M1 measured-magnitude harvest (registered) and its top-k extension.
- Waves 9/10 execution.
- a1/a3 length predictions: registered in spirit (teacher-short ⇒ a3 should be
  termination-safe by construction); formalize in the ledger at unlock time.

## M-II · Entropy / sharpening dynamics

**Banked evidence**:
- c1's late drift: entropy cliff after 175 (1.9→0.66), the roster's cleanest
  monotone distill-loss rise (0.25→0.45), top-32 mass climbing in lockstep — the
  slow-motion signature of the PG pathology (uniformly non-positive advantages).
- b3's degenerate fixed point: transition window 50–100, the campaign's only
  collapse-coincident grad spike (18.5@100), then a near-zero-loss attractor the
  gate itself stops seeing (entropy 0.001, mass 0.9997, verifier 0.000).
- The exoneration: low entropy with bounded length is harmless (h1 0.077,
  j1 0.156, e3 0.172 — all below vanilla's 0.240, none falls).
- The spread: three orders of magnitude at step 250 (0.001 → 5.34), with e1's
  anchor term *inflating* entropy instead.
- g1/h2's deep-U: entropy crashes and recovers (g1 0.43→0.10→0.21) with the
  score — mechanism unexplained.

**Probe positions**: c1 (+ the armed C1 direct ablation, P-PG vs P-K registered)
· b3 (+ M2 gate open-rate) · g1/g4 sign family · e1 (anchor) · h2.

**Missing**:
- C1 execution (~50 GPU·h; highest decision-per-GPU·h on the roster).
- M2 harvest (b3_gate open-rate in the transition window).
- **N3: the deep-U mechanism** — g1/h2 crater-and-recover is unexplained;
  `gate_keep_frac` (g1/g4) and the h2 panels already logged should say whether
  the crater is a gated-data famine (verifier passes ~nothing early) that heals
  as accuracy rises. Analysis only.

## M-III · Signal geometry

**Banked evidence**: the C-axis in-ladder ordering (adaptive > fixed, student
edge > teacher edge, budget-pinning > error-pinning — all pending C1 and compute
normalization); the E-axis 4-rung ladder ran (values 0.524 / z 0.589 / order
0.572 / set 0.460-parked — note the non-monotone rung order, unanalyzed); D/G/H
families complete with selector, gating, and window results.

**The supervision-support family (D ∪ H, merged reading — presentation level
only; run_ids, axis letters, citations and fingerprints unchanged).** Both axes
are the same mechanical operation — `loss_t = 1[t∈Keep]·Δℓ_t·rescale` — and
differ only in how Keep is chosen. Merged, the family is a three-factor space
the current arms sample sparsely:

| factor | sampled at | by |
|---|---|---|
| position (WHERE) | front / random-window / tail / everywhere | h1 / h3 / h2 / d-family |
| criterion (BY WHAT) | entropy∨wrong / verify / teach / none | d1 / d2 / d3 / h-family |
| budget (HOW MUCH) | 0.6% (K=100) / 5% / 50% / TAR / 100% | h* / d3 / d1 / d2 / vanilla |

The merge exposes a real design hole: **the D axis never had its
random-at-matched-budget control.** d1-vs-vanilla answers "does selecting half
work", not "does TIP's criterion beat a random half" — and the latter is the
selector literature's actual claim. h3 is random but contiguous at K=100, a
different budget and shape. The DH1 bridge cells (random-scatter @50% and @5%)
close this; they need no criterion, hence no KEEP_SAMPLED payload, hence 2-card
lanes — which also prices the criterion itself: the d-family pays 4-card lanes
for its criteria, and if random-scatter matches them, that entire cost bought
nothing. H's "less-is-more" budget claim survives inside the family as the
budget factor's extreme sample point (deliberately not budget-matched).

**The reorganization's sharpest finding about M-III: the instruments are
installed but several readings were never taken.** Every top-k arm has been
logging, at zero cost, the panels that this class's claims need:
- `pi_tail_k{8,16,32}` — the headline theorem's error term, a full K-scan per run;
- overlap **mass** panels — the direct test of the literature's "intersection
  holds 97–99% of the mass" claim (the count version cannot test it);
- `rank_kendall_tau`; the D-axis shadow masks (redundancy prediction #4 —
  whether TIP/SelecTKD/Teachability select the same tokens — answerable from any
  single run's logs, never computed).

**Missing**:
- **N4: the panel harvests** — overlap-mass claim check, pi_tail K-scan readout,
  shadow-mask Jaccard. Analysis only, data already on disk.
- C2 (tailbucket), C3 (full-vocab upper bound), C4/wave-10 (QB ladder) — registered.
- D-axis decomposition runs (`TIP_MODE` / `FIRE_MODE`, pre-registered V-wave).
- An E-axis second pass (the 4-rung ladder has results but no analysis pass yet;
  same for D/G/H).

## M-IV · Capability vs delivery

**Banked evidence**: the P(finish)×acc decomposition; the grader floor (first
`\boxed` matches gold on 11/12 while last-boxed scores 5 — the reported collapse
is deeper than the capability change); the suite's 32,768 budget makes it the
arbiter, not the rubber stamp; the teacher profile (composite ceiling 0.6482 is
set by AIME 0.452 / Minerva 0.289 — gains on AIME are gains where headroom is).

**Missing**:
- **N2: the capability-vs-delivery rescore** — re-score the existing greedy
  diagnostics (and, cheaply, any checkpoint) under first-boxed / any-boxed
  rules to produce the *capability curve* next to the *delivery curve*. The
  textdump infrastructure (`eval_offline_textdump.py`) already exists. Analysis
  only; protocol scoring stays last-boxed — this is a reading, not a rule change.
- S: the 555 remaining suite cells + per-bench paired tests (the verdicts).
- a2's suite sweep (the 2× budget may materially re-rank it).

## Gap synthesis — what the reorganization says we actually lack

Already registered (pointers): C1, C2, C3, C4/wave-10, B1/wave-9, A1–A3, M1, M2,
S — see `RESULTS-GAPS.md`.

Newly exposed by the re-shelving:

| id | class | what | cost |
|---|---|---|---|
| N1 | M-I | minimal causal intervention: vanilla + truncation-zeroing reward, nothing else | amendment; ~350 GPU·h |
| N5 | M-I | **the EOS supply audit, token-level** (registered 2026-08-12). Everything M-I knows about EOS is sequence-level algebra — density ≔ (1−clip)/len is a reframing of two logged metrics, disclosed as such in `late-training-collapse.md` §4.5. Nothing token-level was ever measured. Three probes, each a minutes-per-checkpoint forward pass over the textdumps' known should-stop positions (the "answer already written" offsets): (a) **student-side** π(EOS) trajectory across training at those positions — the starvation loop's direct observable, for a collapsing arm (vanilla), an immune one (c2), and the ℛ pair (c1 vs c1_direct); (b) **teacher-side** rank/mass of EOS in the top-k payload at the same positions — the EOS-via-support hypothesis: distributional top-k losses keep supplying stop-gradient WITHOUT the student sampling EOS, which would explain why the top-k family mostly never locks and why the sampled family starves; predicts safe-vs-capped splits WITHIN top-k (c1/c2/b4 vs b2/e2) by whether the objective family lets that supply act; (c) **gradient attribution**: the EOS column's share of the per-position update, sampled family vs support family. Needs GPUs (checkpoint forward passes), not training — first compute task when the fleet returns, before any new wave | probe, ~10–20 GPU·h |
| N2 | M-IV | first-boxed/any-boxed rescore → capability curve vs delivery curve | analysis only |
| N3 | M-II | deep-U mechanism via `gate_keep_frac` + h2 panels | analysis only |
| N4 | M-III | the never-taken panel readings: overlap mass, pi_tail K-scan, shadow Jaccard | analysis only |

Three of the four new gaps are pure log-harvest. The only new compute ask in the
entire reorganization is N1 — and it is the causal keystone of the class with
the most evidence.
