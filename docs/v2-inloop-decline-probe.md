# The v2 wave declines the same way — and a two-temperature probe separates two stopping pathologies

_2026-08-20, from six greedy + two tau=1 diagnostic cells on gpu252 (dsw-279), single seed (s0),
two checkpoints per arm. Artifacts (full response text) in `simopd_data/evals_v2probe/`;
tools `scripts/analysis/v2probe_launch.sh`, `v2probe_tau_launch.sh`, `v2probe_paired.py`.
Protocol = the in-loop val protocol exactly: MATH500, greedy, 16,384 budget, stop set auto-resolved
from each checkpoint's own contract pin (v2 dual-stop). Extends `late-training-collapse.md`
(legacy wave) to the stop-contract-v2 wave._

**Claim.** The v2 wave's in-loop val declines are termination collapse, not reasoning collapse —
same as legacy, which is expected: the dual-stop contract fixed the *sampler* seam, and the collapse
lives in the *gradient*. Conditional on finishing, every probed checkpoint got better with training.
New here: a tau=1.0 supplement splits the failure into two distinct pathologies — **suppression**
(stop channel dead at every temperature; the terminal-gradient arms) and **demotion** (stop mass
alive in the tail but lost from argmax; the pure off-policy arm, whose in-loop length telemetry
is teacher-cache constants and therefore never showed it).

## 1. Three paired greedy diagnostics (same 500 problems)

| pair | score | P(finish) | acc\|finish | fin→trunc | trunc→fin | fin@both acc |
|---|---|---|---|---|---|---|
| a1_gkd_mix0.5 50→125 | 0.582→0.308 | 0.868→**0.000** | 0.666→ – | 434 | 0 | – |
| a3_offpolicy 50→250 | 0.584→0.416 | 0.848→**0.034** | 0.682→0.882 | 410 | 3 | 0.786→1.000 (n=14) |
| h6_gen_sched 25→175 | 0.630→0.444 | 0.950→**0.060** | 0.663→0.900 | 446 | 1 | 0.793→0.897 (n=29) |

All of the loss is in the fin→trunc bucket (a1 0.666→0.336, a3 0.678→0.444, h6 0.655→0.426 — the
survivors of last-boxed salvage). Problems that finished at both steps got **better**. "Finished but
went from right to wrong": 0, 0 and 1 problems respectively. The one-way ratchet replicates
(4 trunc→fin out of 1,500 pairings).

Late truncated responses are answer-then-loop, as in legacy §4: a1 median first `\boxed{}` at
character 1,766 of 16k-token responses; h6 repeats its final-answer line a median of **631 times**;
tail-cycle rates 458-484/500.

## 2. The two-temperature signature (tau=1.0, math500_sub100, n=100)

| checkpoint | greedy P(finish) | tau=1 P(finish) | tau=1 len p10/p50/p90 |
|---|---|---|---|
| a1@125 | 0.000 | **0.000** | 16384 / 16384 / 16384 |
| a3@250 | 0.034 | **0.460** | 2414 / 16384 / 16384 |

- **a1 = suppression.** No stop at any temperature: 100/100 sampled rollouts fill the entire 16,384
  budget. The stop channel is annihilated, consistent with the measured p(eot) crush
  (0.92 → 6e-5 by vanilla@125, `eos_stop_audit`). a1's on-policy half carries the k1 sampled-column
  gradient — the terminal token eats a ~-25 nat advantage every time the student stops.
- **a3 = demotion.** 46% of sampled rollouts terminate (p10 length 2,414) while greedy essentially
  never does. The terminator survives in the tail mass — pure off-policy training puts no gradient
  on the student's own stop positions — but per-position CE normalization drain plus state drift
  demote it below argmax at the student's own reachable states. tau=1 score is 0.040 (noisy decoding
  is bad for a base-size model; the reading here is length, not accuracy).

**Telemetry blind spot (correction).** a3 is `gkd_mix` at lambda=1.0: its rollouts are *served from
the teacher cache*, so the in-loop `response_length/clip_ratio` curves are cache constants
(~6.9k / 0.10, exactly flat) and describe the teacher data, not the student. An earlier read of this
session ("a3's decline is not length-driven, r(val,clip)=+0.29") correlated the greedy val against
those constants and is **retracted** — the probe shows a3's decline is also termination, in the
demotion form. For lambda=1 arms the only student-behavior readings are offline probes like this one.

## 3. What the loops look like

a1@125 (correct=1, still scored by salvage):
`**Final Answer:** \boxed{9}` repeated to the cap — the teacher's closing template as attractor.

a3@250 (correct=1): `We are done. ✅\n\n\boxed{9}\tNdrFcShort` cycling — the model declares
completion, then emits a corrupted token where a terminator should be: the argmax at its stop
states has fallen onto junk. (Matches the audit rows where the teacher's top-1 at late-vanilla stop
positions surfaces as `✅`.)

h6@175: doubt-wander loops ("But the graph is not given. Wait —") alongside final-answer loops.

## 4. The terminal-gradient-exposure account (what the corr wave tests causally)

Reclassifying the legacy 29-arm ledger by whether the objective's gradient can reach the student's
terminator — sampled-column losses and support-mass-drain losses (CE/coverage) = exposed;
support-internal renormalized losses and windows that exclude the terminal = not exposed:

|  | capped at end | not capped |
|---|---|---|
| exposed (21) | 18 | 3 |
| not exposed (8) | **0** | 8 |

Fisher two-sided p = 3.8e-5 (the PG-form split of `late-training-collapse.md` §6 gives 0.017 and
leaves c1/b5/b2/e2 as exceptions; this classification absorbs all four). The three exposed-but-safe
arms are dose attenuations, and the dose line is monotone in penalty magnitude at the terminal:
k1/k2 unbounded ~25 nats → lock ~101-143; f1/f2 softened → 198-208; b1 skew-KL bounded 2.3 nats →
lock 247; f3 probability-scale bounded 1 → never locks; j1 coef 0.002 + verifier reward → never.
Position dose: h1 (first-100 window, terminal never covered) 0.151 truncation vs h2 (last-100,
terminal always covered) 0.997, lock@37; h3 (random window, ~3% coverage) 0.182. b4's beta=0.9 rung
(mode-seeking limit on a renorm support) does not cap — mode-seeking per se is not the driver.
c2/c4's immunity is support geometry: the student's terminator sits at teacher rank ~2.5e4
(q~1e-11), outside any sane support, and two-sided renormalization gives out-of-support tokens
exactly zero gradient — receipts: p(eot at stop states) stays 0.95-1.0 through c4@100 while
vanilla@125 crushes it to 6e-5. This classification is post-hoc (desc/code-read); the causal test
is `vanilla_corr` + the N0 combos (registered 2026-08-19), which replace the terminal token-ratio
with the event-level ratio and change nothing else.

**Pre-registered predictions for vanilla_corr on THIS probe protocol**: P(finish)@125 stays >0.8
(vs a1's 0.000 / legacy vanilla's 0.666@100), no 50%-clip crossing before step 250, val does not
fall from peak by more than 0.03, p(eot at stop states) stays within 10x of base. Suppression
signature (tau=1 all-cap) must not appear; if a demotion-shaped residual remains, it bounds the
non-spelling share (placement disagreement + drift) of the collapse.

## 5. Honest limits

- Single seed, two checkpoints per arm, one benchmark; mechanism claims lean on the audit +
  ledger cross-classification, prevalence on the legacy 29x3 table.
- a3's demotion mechanism (drain vs state drift) is not separated here; the scaled audit
  (p(eot|stop states) across the checkpoint grid) is the cheap discriminator and also settles
  the lead-lag question `late-training-collapse.md` §7 left open.
- a5's composite telemetry (AggreVaTe mixing) has the same blind-spot risk as a3; not yet probed.
- The a2-v2 bootstrap (SFT-primed p(im_end)~1e-3 + dual-stop sampler) is the live positive-direction
  test: fr_stop should grow if the teacher's reward for well-placed im_end is real. In flight.

## 6. Teacher intent at the student's answer positions (post-answer probe, 2026-08-20)

At every k-th "answer line" inside 36 truncated (looping) late-checkpoint responses, the teacher's
next-token distribution (170 positions; `scripts/analysis/post_answer_probe.py`, receipt
`docs/data/post_answer_probe.csv`):

| arm | after 1st answer | 2nd | 3rd | 10th |
|---|---|---|---|---|
| a1@125 q(im_end) med | 0.107 | 0.824 | 0.970 | 0.896 |
| a3@250 q(im_end) med | 0.011 | 0.109 | 0.939 | 0.369 |
| h6@175 q(im_end) med | 0.005 | 0.060 | 0.136 | 0.158 |
| q(eot) med, all arms | ~1e-12 | ~1e-12 | ~1e-12 | ~1e-13 |

Three readings. (1) At the FIRST answer the teacher usually wants a few more tokens (units, a
closing `$$`, a newline) — greedy takeover emits `<|im_end|>` within ~24 tokens in 8/9 cases —
so the first-answer position carries genuine, mild placement disagreement (log q_stop ~ -2.3).
(2) From the second repetition on, the teacher wants to stop NOW (q(im_end) 0.82-0.97 for a1;
top1 = im_end in 67-83%). (3) That desire lives entirely on `<|im_end|>`; q(eot) stays ~1e-12 at
every loop depth, so under the token-level k1 read the student's own stop is punished ~-25 nats
even while the teacher is begging it to stop — the scream is in a spelling the loss never checks.
h6 caveat: in degenerate wander-loops the teacher itself is lost (q_stop 0.14-0.16, top-1 often
mojibake) — stop-desire is only legible in clean final-answer loops, which is fine for the N0 fix
(it prevents entering the ratchet; it does not need to rescue already-degenerate states).
Corollary for `vanilla_corr`: the event-level read converts terminal positions into honest,
bounded placement supervision — log q_T(E_T) is ~-2.3 at "not quite yet" and ~-0.1 at "stop now",
instead of a flat -25 spelling artifact.

## 7. Style anatomy: teacher, base student, healthy-trained, sick-trained (2026-08-20)

Four fresh greedy math500 cells with full text (`evals_style/`; tools `scripts/analysis/style_cells.sh`
+ `style_anatomy.py`), vanilla@250 from the sweep as the sick contrast:

| | score | trunc% | len med | scaffold (###/$$/---) | ✅ | Wait /resp | verify /resp | "We are done" /resp | boxed med | post-answer tail |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher 4B | .900 | 1.0 | 694 | 92/95/92% | **84%** | 1.7 | 0.6 | 0.0 | 1 | 2.4% |
| base 1.7B | .458 | 17.0 | 553 | 6/0/0% | 0 | **0.0** | 0.1 | 0.0 | 1 | 1.3% |
| c2@250 | .674 | 10.8 | 2400 | 97/97/98% | 2% | 8.0 | 2.3 | 0.0 | 2 | 9.5% |
| c4@250 | .656 | 21.6 | 2810 | 98/95/97% | 2% | 12.0 | 2.5 | 0.0 | 2 | 19.3% |
| vanilla@250 | .462 | 95.8 | 16384 | 97/91/98% | 1% | 9.5 | 3.2 | **142.0** | **712** | (loop) |

Readings. (1) The base student is terse numbered-list prose with a 27.6% hallucinated-python-block
rate, zero verification vocabulary (Wait 0.00/resp), and stops within 1.3% of its answer — but
already blows the 16k cap on 17% of problems untrained. (2) The teacher builds to its answer at the
97.5% position of the text and closes within 2.4%. (3) c2/c4 adopted the teacher's full scaffold
(97-98% marker rates; the ✅ signature notably did NOT transfer, 2%) and AMPLIFIED the verification
habit 5-7x beyond the teacher (Wait 8-12 vs 1.7; second-method re-solves; dedicated Verification
sections; the answer stated twice) — hence 3.5-4x the teacher's length. The wander-loop raw material
is therefore measured as teacher-descended vocabulary amplified by the optimization, not present in
the base model. (4) vanilla@250 is style-identical to c2/c4 (same scaffold, same amplified checking)
— the ONLY stylistic fork is that its terminator is dead: "We are done" x142, boxed x712,
distinct-20gram 0.037. Healthy and sick trained students are the same species stylistically;
the disease is only the full stop. (5) Length growth decomposes into a benign component
(verification amplification, +3.5x, shared by healthy arms) and the malignant one (terminator
death, unbounded, vanilla only).

### 7.1 Training-parameter rows: temperature is NOT the teacher-length confound — the prompt distribution is

Two further cells at the real training sampling params (tau=1.0, top_p=1.0; run_opd_baseline.sh:447,
gen_offpolicy.py:116), plus the decoded actual teacher cache (1,500 of 14,467 rows):

| | score | trunc% | len med | Wait/resp | verify/resp | "We are done"/resp | note |
|---|---|---|---|---|---|---|---|
| teacher @tau1, math500 | .914 | 0.6 | **722** | 3.6 | 1.4 | 0.000 | == greedy teacher (694): temperature does NOT lengthen it |
| student @tau1, math500 | .116 | 2.6 | 318 | 0.8 | 0.2 | 0.000 | 68% produce NO boxed; short drifty non-answerer, not a looper |
| **teacher cache (real demos, train prompts)** | – | **8.6** | **6449** | **21.8** | 6.4 | **0.170** | the wander-style corpus |

The 9x teacher length gap (722 vs 6449) is prompt-distribution, not temperature: the train parquet's
`extra_info.orig_source` is **acereason_math** (competition-hard; the `data_source` string
DigitalLearningGmbH/MATH-lighteval is scorer routing, not provenance). On genuinely hard prompts the
free-running teacher deliberates at length: Wait 21.8/resp (6x its math500 rate), 8.6% of demos
hit the 16k cap with no terminator, and "We are done" — the exact phrase of the late loop body —
appears in ~1 of 6 real demos. Earlier session claims that "the teacher is verbose at temperature 1"
are corrected accordingly: it is verbose on the training distribution.

Closure numbers: every arm's training-rollout length settles at 8.4-9.5k ~ the free-run teacher
equilibrium on these prompts; the healthy arms' permanent clip floor (0.02-0.08) is the same order
as the teacher's own 8.6% cap rate. The first clip hump (steps 35-55, peak 0.4-0.8, universal across
vanilla/c2/c4/corr) is the style tide toward that equilibrium; only the second hump (terminator
death) is pathology. Style trajectory on the vanilla sweep dates the tide: Wait bursts 0 -> 13.8/resp
at step 50 (entropy min 0.11 @35-45 turns upward exactly there); scaffold completes 19% -> 91% by
step 100 with boxed=1 and 1-3% post-answer tail (perfect stop discipline on the eve of the cliff);
the death markers (boxed 538, distinct-20gram 0.09, tail 19%) all flip together in the 100->125
window, and "We are done" enters the loop body only at 150+ (2.9 -> 147/resp by 225) — the
teacher-corpus phrase k1-amplified into the dominant cycle.

### 7.2 Candidate aligned student: Qwen3-1.7B (instruct) — terminator and style pre-aligned

Config: eos_token_id = [151645, 151643], identical to the teacher (the Base student has 151643 only).
Real rollouts (120 math500 prompts x {greedy, tau=1}, `scripts/analysis/probe_q17b.py`): **235/235
stopped rollouts end on `<|im_end|>`, zero on `<|endoftext|>`**; greedy 115 stop + 5 length,
tau1 120/120 stop, len p50 ~480. Style row (full greedy cell, `evals_style/q17b_it_greedy_text*`):
score 0.702, trunc 2.6% (Base: 17%), len p90 1992 (Base: 16384), teacher scaffold 96-100% INCLUDING
the ✅ signature at 79.8% (which 250 steps of OPD never taught c2/c4: 2%), Wait 1.4/resp,
python-block artifact 0%, single boxed, 2.0% post-answer tail. It is a miniature teacher — same
Qwen alignment recipe, officially distilled from the flagship — so swapping it in dissolves the
terminator mismatch AND the style-transfer task simultaneously, changing the experiment from
cold-start base->instruct distillation (the campaign setting) to instruct->instruct polish
(0.702 -> 0.900 headroom). Registered implication: a `q17b_vanilla` control cell (aligned student x
UNCORRECTED k1, seed 0) pre-registers as no-collapse; paired with vanilla_corr (misaligned student x
corrected loss) it pincers the causal claim "collapse = spelling mismatch x sampled-column gradient"
from both sides.

## 8. Two adjudications: b2's late collapse is drain (not drift); the N0 event read teaches events, not spellings

**b2_forward_kl autopsy** (`clock_b2.py` + `probe_b2.py`, banks 25-175). Two-spelling clock at the
15 fixed answer-complete states: p(im_end) NEVER rises above 5.7e-7 at any bank — the forward-KL
stop-position teaching (1 position/rollout) is outgunned ~1:6000 by body-position normalization
drain, which crushes p(eot) 0.997 -> 0.022 (@125) -> 6.6e-5 (@175). Dual-contract rollouts: zero
im_end tokens anywhere (mid-text or terminal) in 240 rollouts; dual-stop rescues nothing
(b2@50: 75 vs 77 stops; b2@175: 120/120 truncated under both contracts). VERDICT: pure
suppression via mass drain — the P-drift hypothesis is REFUTED for b2 (contrast legacy a1, R5 P3,
where dual-stop re-eval rescued .383 -> .118). The exposure taxonomy's two kill paths now each have
an anatomical specimen: sampled-column spelling tax (vanilla; clock cliff inside the behavioral
window) and normalization drain (b2; fixed-state clock leads the behavioral cliff, 75-125 vs
125-150). b2_corr (unnormalized + collapsed support + N0) runs at literal-zero truncation
(clip 0.0078 = 1/batch floor) — the drain path, plugged, leaves forward KL healthy.

**n0 A-axis spelling verdict** (`clock_n0.py`, banks 25/50/75 x {a1,a4,a5,a3}_n0 + dual-stop
rollouts @75). All four arms hold p(eot) at 0.90-1.0 with stop=top1 15/15 (the N0 fix working:
no suppression, no drain) while p(im_end) stays at noise (1e-13..2.3e-7) — and 125/125 stopped
rollouts end on eot, zero on im_end. Under the OLD token-level read the same demos taught the
teacher's spelling outright (legacy a1 emitted im_end by step 50, R5); under the event-level read
the demo ending credits log p(E_S) proportionally to current shares — rich-get-richer lands on the
student's own spelling. Off-policy demos under N0 teach WHERE to stop, not HOW to spell it; the
demo-token im_end -> eot swap question is thereby empirically closed (no-op under termfix, and the
students have already done the translation themselves). Watch item: a3_n0 is the only arm with
p(eot) softening (0.995 -> 0.905) and p(im_end) rising (23x) — the pure-off-policy body-position
pull still mildly dilutes non-demo tokens; same direction as its v2 demotion pathology, much milder.
