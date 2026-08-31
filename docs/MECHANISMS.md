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
- Initialization: a2's SFT stage delivers a termination-broken init — 65% of
  training rollouts capped at step 1 (mean 14.4k tokens), 100% from step 35.
  **Corrected reading (2026-08-19 CPU audit, `scripts/analysis/a2_coldstart_probe.py`,
  `docs/data/a2_coldstart_probe.txt`): the terminator, not the length.** The
  SFT target is the student-template render, so its supervised span ends
  `…$$<|im_end|>\n` — the TEACHER's terminator — and never contains
  `<|endoftext|>` (0 of 6,358 rows; `<|endoftext|>` appears only as pad with
  loss_mask 0). After 98 SFT steps the model has p(`<|im_end|>`) ≈ 1e-3 at the
  end of a teacher-CoT answer (from 1e-11 — one target token per ~2.3k), has
  crushed the base's own stop p(`<|endoftext|>`) 0.98 → 1e-3 (never a target,
  the main competitor at exactly those positions under CE), and puts the mass on
  continuation (`\n\n` 0.52 → "This is the smallest…", "---\n\n**Answer:**
  $\boxed{}$…" — the teacher's closing template, which the SFT data legitimately
  contains as mid-response restatements: this is the terminal loop's origin).
  And even a perfect SFT would not have produced a stopping student: the
  rollout honors only `<|endoftext|>` (the SFT export's generation_config eos is
  151643), whereas Rethinking's own recipe runs LlamaFactory's `qwen3` template
  with `replace_eos=True` (`stop_words=["<|im_end|>"]`), so THEIR SFT'd student's
  eos becomes `<|im_end|>` and their OPD rollouts stop on it. Our port (verl
  sft_trainer, Base tokenizer untouched) lost that; a2 as run tests nothing about
  the cold-start recipe — it tests an init that cannot terminate by construction.
  Corrected cell: terminate the SFT target as `…<|im_end|><|endoftext|>` (both
  models put their next-token mass on `<|endoftext|>` after `<|im_end|>`: teacher
  1.0, base 0.59), keep everything else, rerun (RESULTS-GAPS A1').
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
- **The stop-token identity mismatch (2026-08-19 CPU audit,
  `scripts/analysis/eos_stop_probe.py` / `eos_stop_audit.py`, receipts in
  `docs/data/eos_stop_*.txt`).** The student and the teacher end a response with
  DIFFERENT tokens. Qwen3-1.7B-Base under verl stops a rollout only on
  `<|endoftext|>` 151643 (tokenizer eos = generation_config eos = vLLM's sole
  stop, training and eval alike; of stopped eval responses 99% end on it directly
  with no `<|im_end|>` before it — resp_len − text-token count = 1). The Instruct
  teacher ends on `<|im_end|>` 151645. At the position where a real student
  rollout stopped (30 stopped math500 responses from vanilla@125/225 and c4@100):
  q_T(`<|im_end|>`) median 0.95, q_T(`<|endoftext|>`) median 1.4e-11 (teacher
  rank ~2×10⁴, never inside any top-k); the base student has p(`<|endoftext|>`)
  median 0.98 and p(`<|im_end|>`) ~1e-11. Same event, disjoint tokens.
  Consequences, all measured: (i) vanilla's sampled-column k1 gives the student's
  stop token Δℓ = log q − log p = **−25 nats** (median; −28…−17) at EVERY stop
  event — the single most negative reward of the response in 25/30, beyond
  p99.9 in all (body median 0.00, p1 −9.4) — and the teacher's own terminator is
  never sampled (p ~1e-11), so no positive stop signal ever arrives; (ii)
  vanilla@250 has p(`<|endoftext|>`) median 6e-5 at natural stop positions (base
  0.98) and its greedy continuation there is the closing template ("  \n\n---
  \n\n**Final Answer**…") — the erosion is at the stop decision itself; (iii)
  c2/c4@250 keep 0.997: renormalized support KL puts zero gradient on
  out-of-support tokens, so the base's stop survives untouched — and neither
  ever learns `<|im_end|>` (~1e-14 → 1e-19). **The exact-top-k arms do not
  SUPPLY stopping; they stop PUNISHING it.** The reading of the dose ladder
  changes accordingly: |Δℓ| ∞/10/log/2.3/1 at the stop event is −25/−10/−3.3/
  −2.3/−1 — the same ordering as lock 122/198/208/247/never — and the
  "dose beyond p99.7" is, in every stopped response, the terminal token; d/h2
  (sampled column, terminal included) lock, h1 (front window, terminal excluded)
  does not; g4 (correct-only) locks because the correct traces are exactly the
  ones that stop. Whether the mismatch is the WHOLE of the continuation drive or
  only its trigger is the open causal question — the N0/N2 cells below split it.
  Not a tokenization difference (`scripts/analysis/tokenizer_parity.py`,
  `docs/data/tokenizer_parity.txt`): the two tokenizers have identical vocab
  (151669), identical merges/normalizer/pre-tokenizer, identical ids and special
  flags for `<|endoftext|>`/`<|im_start|>`/`<|im_end|>` (special) and
  `<think>`/`</think>` (151667/151668, special=False in both), and encode 1500 real
  responses identically; the teacher never tokenizes in training anyway (it scores
  the student's ids). What differs is only what each model was TRAINED to end
  with (`tokenizer_config.eos_token`: `<|endoftext|>` vs `<|im_end|>`;
  `generation_config.eos_token_id`: 151643 vs [151645, 151643]) and the chat
  template — the Base template emits `<think>\n\n</think>\n\n` under
  `enable_thinking=False`, the Instruct-2507 template emits nothing; that prefix
  reaches the teacher in every training/eval prompt but is benign: teacher mean
  log q per response token −0.806 (student template) vs −0.824 (its own),
  q(`<|im_end|>`) at the end 0.94 vs 0.88, 0.1% of tokens move by >2 nats.

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
- **N2 (registered 2026-08-19, re-registered the same day as `n2_termcal`,
  termination-marginal calibration): the supply-side twin of N1.**
  N1 closes the loop with a SIGN (RL pressure on truncated rollouts); N2 closes
  it with SUPPLY: vanilla unchanged plus an exact, full-softmax stop-vs-continue
  term at every visited state, BCEWithLogits(m_t, q_t) with q_t = Σ_{e∈E_T}
  p_T(e|s_t) over the TEACHER's terminators E_T = {151643, 151645} and m_t = the
  log-odds of Σ_{e∈E_S} p_θ(e|s_t) over the tokens that actually END the
  student's rollout E_S = {151643}; gradient p−q on the log-odds, bounded, and
  the push lands inside E_S by the student's own conditional (so on
  `<|endoftext|>`) while student mass on a teacher-only terminator counts as
  CONTINUE — the second-layer "which terminator" KL is deliberately absent, the
  channel moves the termination hazard and nothing else. The stop-token audit is
  what forced two sets: the original single-token registration (E = {151643} on
  both sides) would have calibrated p(stop) toward q_T(`<|endoftext|>`) = 1e-11 —
  a manufactured anti-stop channel from a TRUE q, the very failure the design
  review guarded against for top-k misses. Payload contract (`simopd.eos_gather`):
  every id is force-gathered, never inferred; a miss is refused. Live receipt:
  panel `eos_dl_at_stop` logs vanilla's Δℓ at the student's own stop tokens (the
  audit's −25) on every step. λ ladder now opens UPWARD ({1, 3, 10, 30}) — the PG
  kick on the stop logit is ~25·(1−p) per stop event. Both N1 and N2 stable ⇒
  two independent routes; one stable ⇒ the two forces are asymmetric and collapse
  answers only that remedy.
- **N0 (registered 2026-08-19, `vanilla_corr`, wave 17 — run FIRST): semantic
  alignment of the target representation — STOP token ratio → STOP event
  ratio; no new loss, no new gradient channel.** As a knob (`SIMOPD_TERM_EVENT=1`)
  it repairs both routes of the mismatch: **Path 1** (sampled signal: e_S sampled
  → −25) by reading the event-level sampled log-probs at the student's stops, and
  **Path 2** (support: e_S ∉ S_T → the student's STOP mass squeezed out by any
  support-restricted objective, and selectors fed a fake terminator divergence) by
  collapsing the terminator coordinate of the teacher's top-k onto the student's
  stop id, rank-inserted (`_collapse_terminator_support`). Measured victims of
  Path 2: b2 1.6e-4 / e2 0.005 / c3 0.56 p(eot) at natural stops; d1's TIP
  normalization pinned by the outlier (`docs/data/selector_stop_audit.txt`). User's formulation: "when the student
  samples `<|endoftext|>`, read the teacher's `<|im_end|>` probability" — made
  exact as the event level: vanilla with ONE change, at the sampled stop token
  the per-token signal is Δℓ_T = log Σ_{E_T} q − log Σ_{E_S} p instead of the
  token-level log q(`<|endoftext|>`) − log p(`<|endoftext|>`); every other token
  untouched, no coefficient, no channel. It removes the −25 without adding
  supply, so it isolates the artifact: **P-artifact** — N0 holds an interior
  fixed point ⇒ vanilla's collapse is (mostly) the identity mismatch and the
  exact-top-k family's stability is the exemption, the practitioner finding of
  the chapter (Base student + Instruct teacher under vLLM/verl defaults);
  **P-drive** — N0 locks like vanilla ⇒ the continuation drive is real beyond the
  terminal token and N2's supply is the live question; together they read the
  2×2 {V, N0, N2, N0+N2}. The teacher's judgement about WHEN to stop survives
  (q_T(E_T) is small mid-answer, so a premature stop is still punished). Live
  receipt: `eos_dl_at_stop` (applied, event-level; expect ≈ log 0.95 − log 0.98)
  next to `eos_dl_at_stop_raw` (vanilla's, expect ≈ −25). Watched corner: a
  sampled `<|im_end|>` is not in E_S and keeps its token-level signal (+25 if it
  ever happens; `eos_pm_1` is the watch panel; at p ~1e-11 it does not). Same
  payload and lane as N2 (`ARM=vanilla_corr bash deploy/dsw/rehearse_n2.sh 0,1`).
  Not the environment fix (letting the rollout stop on `<|im_end|>` too) — that
  changes the environment for every arm in the campaign and is a follow-up
  campaign's opening question, together with whether other Base/Instruct pairs
  carry the same mismatch.
  What N0 does NOT do (collaborator framing 2026-08-19): stop supervision
  reaches the student only when it actually samples STOP, so if p_S(STOP)
  keeps falling for any other reason N0's correction opportunities fall with
  it — N0 fixes the SIGN of the stop event and keeps sampled-OPD's starvation
  of DENSITY; it cannot pull a locked policy back. N2 is the complement: the
  dense channel KL(Bern(q_T(STOP)) ‖ Bern(p_S(STOP))) at every visited state
  restores stopping mass even where STOP is never sampled. **N0 = semantic
  alignment, N2 = non-starving supply**, and the 2×2 {V, N0, N2, N0+N2} reads:
  N0 alone holds ⇒ the artifact was the whole engine (density starvation never
  bit once the sign was right); N2 alone holds but N0 does not ⇒ density is
  load-bearing beyond the sign; both hold ⇒ two sufficient routes; neither ⇒
  the continuation drive lives outside the termination coordinate. Live
  receipts: `eos_n_stop` (N0's correction-opportunity count per micro-batch)
  next to the truncation trajectory; `eos_ravail_500up` (N2's available
  restoration).
  Contract: both cells run under the legacy single-eos rollout contract
  (`SIMOPD_STOP_IDS: off` pinned in their env — E_S = {151643} must equal what
  ends a rollout; `eos_gather` refuses otherwise) so they stay single-knob
  against the banked vanilla rows. The dual-terminator contract v2
  (`simopd.stop_set`, A-AXIS R5 appendix) is the launcher's default for NEW
  runs and the A-axis v2 restart wave runs under it; a v2 pair of N0/N2 needs
  a vanilla_v2 control first, and then E_S = {151643, 151645} — the symmetric
  union (under v2 the sampled-`<|im_end|>` corner disappears because it stops
  the rollout). Eval follows the run's own contract (`--stop-token-ids auto`).
- M1 measured-magnitude harvest (registered) and its top-k extension.
- Waves 9/10 execution.
- a1/a3 length predictions: registered in spirit (teacher-short ⇒ a3 should be
  termination-safe by construction); formalize in the ledger at unlock time.

### M-I generality: the collapse reproduces on a 4.7×/8× larger pair (2026-08-21)

Found in already-banked runs, no new compute. The `w` cell is **Qwen3-8B-Base ←
Qwen3-32B** (wandb group `Qwen3-8B-Base__from__Qwen3-32B__s0`, cap 8192) — a
different pair carrying the SAME terminator mismatch, because the split is
Base-vs-chat, not small-vs-large: 8B-Base's generation_config eos is 151643
alone, 32B's is [151645, 151643], exactly the mainline asymmetry.

`vanilla_s0_w` truncation: 0.011 @21 → 0.030 @31 → **0.834 @41** → 0.973 @51 →
1.000 @75, length 1093 → 8192. The same U: down to a trough, then the whole ramp
inside ~10 steps.

The comparison must be made at MATCHED cap — the 16k mainline runs ramp later
(~103) only because a bigger cap takes longer to register as truncation. Against
the mainline's own 8k-era vanilla seeds (trunc .037/.036/.055 @26, .811/.770/.791
@51), the w cell is **the same curve, ~10 steps earlier**. A 4.7× student and an
8× teacher move the onset by ten steps and not by its shape: this is not a
small-model artifact.

Two more w-cell rows say the same thing about mitigations as the mainline does:
`f2_hard_clip_s0_w` pushes onset 41 → 51-75 and `h1_first_segment_s0_w` pushes it
to 51 → saturated by 125. **Both delay, neither prevents** — the same verdict the
mainline F/H family earns, and the same shape as h2's N0 (delayed 54 steps, then
exploded). Whatever prevents the collapse has to act on the terminator itself.

Not yet answered here: no N0 cell exists on the w pair, so this replicates the
DISEASE across pairs, not the cure. The cheapest test of the cure's generality is
a `vanilla_corr`-equivalent on the w pair (8 GPUs, 81 s/step banked — ~6 h to 250).

### M-I cure: the terminator fix holds to 250 on the mainline pair (2026-08-24)

The generality section above closed on "this replicates the DISEASE across pairs,
not the cure." The mainline cure now has its 250-step curve. Single-variable
comparison, same student/teacher/data/protocol, only the kernel differs:
`vanilla` = `k1_rec`, `vanilla_corr` = `k1_termfix`. Both cells were evaluated
under the SAME measurement contract (`stop_set=off`, checked in
post_eval_cells.csv) — mixing contracts here would have made the whole comparison
meaningless.

| step | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy composite (mean of s0/s1/s2) | .280 | .290 | **.328** | .313 | .254 | .248 | — | — | .247 | .247 |
| legacy truncation | .12 | .52 | .42 | .41 | .93 | 1.00 | — | — | .98 | .97 |
| **corr** composite (s0) | .285 | .290 | .332 | .326 | **.329** | .332 | .347 | **.348** | .343 | pending |
| **corr** truncation | .12 | .53 | .38 | .23 | .19 | .14 | .11 | .10 | .15 | pending |

The two curves are **indistinguishable through step 75** and split exactly where
the collapse starts. Legacy runs to the 32k eval cap (length 30.3k–32.5k from
step 125 on, truncation .92–1.00 across all three seeds); corrected turns around
at 75–100 and settles at ~9k with truncation .10–.15. So the late-training score
drop is delivery, not capability — and removing the terminator mismatch removes
it. n=1 seed on the corrected side (the legacy side is 3/3 consistent); the
250-step evaluation is queued, not run.

**The medicine is not what cures it.** `n2_termcal` is raw N2 (`k1_termcal`, no
`SIMOPD_TERM_EVENT`): it collapses on schedule — eos_p_at_stop .24 → **.012 @100**,
truncation .85, length 15.2k and still climbing when the job died at step 112.
`n2_corr` (same kernel + the fix) does not collapse, but it is worse than the plain
corrected carrier on the very instrument N2 targets: late truncation .23 vs .11,
eos_p_at_stop .23 vs .50. On present evidence the fix is sufficient and N2 adds
nothing; n2_corr has no complete offline cell yet, so this is instrument-level only.

**The fix is not universal.** Two arms carry `SIMOPD_TERM_EVENT=1` and collapse
anyway: `e2_set_coverage_a0_corr` (`set_coverage_anchor`; truncation .86, entropy
**.03**, length 15.1k @229) and `h2_last_segment_corr` (`k1_lastseg`; .84, 14.7k
@142 — the same arm whose N0 version delayed 54 steps and then blew up). Six more
sit in the danger band (g1 .53, h4 .46, b2 .36, c4_pi_tail_budget_corr .34,
c5_union_rkl .30, and legacy a1_gkd_mix0.5 .53).

Of **51 runs** in the export window: 34 healthy, 6 danger, 6 collapsed, 5
budget-capped. The three collapsed runs on the corrected side are the two above
plus raw `n2_termcal`; **the other three collapsed runs are all legacy-carrier**
(`h6_gen_sched` 1.00, `h10_task_subset` .98, `a4_dagger_anneal` .71). Budget-capped
means h7_gen512 / h8_gen2048 / h9_prune_adapt (and the _n0 twins of the first and
third) sitting at their OWN small caps — length 511/1350/2039, design rather than
collapse, and any classifier reading truncation alone calls them collapsed.

**Four more within-arm pairs point the same way — but they are NOT single-knob.**
The window contains both the legacy and the `_n0` version of six A/H-axis arms.
Read the envs before reading the table: `_n0` here means `k1_termfix` + student-side
`EOS_IDS=151643,151645` + **`STOP_IDS=151645`, i.e. the v2 stop contract where
im_end also ENDS a rollout**. So each pair differs in two knobs at once — the
carrier AND the contract — and part of the length/truncation gap below is
mechanical: a rollout that may end on im_end ends sooner. These pairs corroborate
the direction; only the vanilla vs vanilla_corr comparison above (both
`STOP_IDS=off`, kernel the only difference) isolates the carrier.

| arm | legacy carrier | corrected (`_n0`) |
|---|---|---|
| a4_dagger_anneal | **collapsed** 13.3k / .71 | healthy 9.9k / .22 |
| h6_gen_sched | **collapsed** 14.6k / 1.00 | healthy 9.0k / .15 |
| h10_task_subset | **collapsed** 16.2k / .98 | healthy 9.4k / .12 |
| a1_gkd_mix0.5 | **danger** 11.4k / .53 | healthy 8.3k / .12 |
| a3_offpolicy | healthy 6.9k / .08 | healthy 6.8k / .08 |
| a5_aggrevate | healthy 10.1k / .24 | healthy 9.7k / .21 |

Four flip from collapsed/danger to healthy; the two that never collapsed (pure
off-policy, aggrevate) are unchanged — whatever the `_n0` change does, it moves the
arms that had the disease and leaves the others alone. What it does NOT license is
an offline comparison: those four `_n0` arms are evaluated under `stop_set=151645`
while vanilla/vanilla_corr/c4 are under `stop_set=off` (post_eval_cells.csv), so
their composites are a different protocol and must never be differenced against the
`off` arms — the same rule the cells table enforces within a cell, applied across
arms.

**Negative result: there is no early-warning threshold.** The obvious idea —
watch eos_p_at_stop, entropy or truncation cross a line and call the collapse
early — does not survive contact with the healthy arms. `vanilla_corr` trips all
three (eos_p<.3 @39, entropy<.2 @35, truncation>.6 @44) during the ordinary
length ramp and then runs healthy to 250; `f1_soft_log_corr` and `n2_corr` trip
eos_p<.3 at step **5** and never collapse. Discrimination exists only in the late
window. This is the data-side argument for the pre-registered stance: judge by
slope in the late window, not by position or by an early threshold, and do not
call a cell before 250.

Reproduce: `python scripts/analysis/collapse_status.py --write` (reads
docs/data/inloop_wave_dynamics.csv + post_eval_bystep.csv, writes
docs/data/collapse_status.csv; no cluster, no wandb).

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
