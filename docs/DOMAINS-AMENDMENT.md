# Dated amendment (2026-08-13): the IF and code domain campaigns

_Per `EXPANSION-PLAN.md`'s own discipline ("改动本文件走 dated amendment"), this
registers the two domain campaigns BEFORE their first run. Ruling being executed:
all arms train on IF and code (user, 2026-08-12), at fleet scale via the DLC
shape (`deploy/dlc/`). The plan's original Phase-3 scoping ("训练域扩展只配最终
配方") is superseded for these two domains by that ruling._

## 1. What a domain is, mechanically

A domain = one (manifest, claim namespace, env triple). `campaign.sh` is
untouched — it already reads `MANIFEST` and `CLAIM_DIR` from the environment:

| domain | manifest | claim dir / BATCH_TAG | env deltas (everything else = locked protocol) |
|---|---|---|---|
| math | `configs/campaign.tsv` | `.campaign` / `16k` | — (byte-identical to today) |
| IF | `configs/campaign_if.tsv` | `.campaign_if` / `if4k` | `DATA_DIR=simopd_if`, `VAL_FILE_BASENAME=ifeval.parquet`, `MAX_RESPONSE_LENGTH=4096`, `CUSTOM_REWARD_PATH=src/simopd/domain_reward.py` |
| code | `configs/campaign_code.tsv` | `.campaign_code` / `code4k` | `DATA_DIR=simopd_code`, `VAL_FILE_BASENAME=val_holdout.parquet`, `MAX_RESPONSE_LENGTH=4096` (reward = stock verl dispatch: `data_source=codecontests` → prime_code) |

Run names become `<arm>_s<seed>_<tag>`, so checkpoints, fingerprints and eval
artifacts never collide across domains. The two launcher seams added for this
(`VAL_FILE_BASENAME`, `CUSTOM_REWARD_PATH`) are absent-by-default: every
existing math run is byte-identical, and both ride the fingerprint (hydra extra
args / the `data=` line), so domain runs split batches by construction.

## 2. Datasets (prepared, decontaminated, self-verified)

- **code**: DeepCoder-Preview taco+primeintellect pooled (lcbv5 excluded —
  LiveCodeBench is a public leaderboard), deduped, fn_name-style rows dropped
  (call-convention tests structurally mismatch the stdin/stdout prompt), and
  **self-verified**: a row survives only if an official solution scores 1.0
  through our exact prime_code harness on the CAPPED tests (77% survive).
  `train.parquet` **12,826 rows** — deliberately below the math set's 14,476;
  reward reachability outranked size matching. Val = 200 held-out same-dist
  rows. All 8 surfaces CLEAN at 13-gram (incl. HumanEval+/MBPP+/IFEval).
- **IF** _(corrected 2026-08-13, the day the set was actually built)_: the
  planned `nvidia/Nemotron-Cascade-RL-Instruction-Following` id was a phantom
  (404 — and the real dataset is public, no license token needed). Real source:
  **`nvidia/Nemotron-RL-instruction_following`**, 46,391 prompts, of which 77%
  use nvidia's EXTENDED constraint taxonomy (bigram_wrapping, last_word_answer,
  …) that the vendored Google checker cannot instantiate — those score 0
  forever and read as "hard prompts", so they are dropped by the registry
  validation. Kept: the **~10.7k fully checkable rows** (deduped) — same ruling
  as the code set: reward reachability outranks size matching. Loaded by raw
  jsonl parse (the shard's list offsets trip datasets' arrow builder). Val =
  google/IFEval 541, strict-prompt-acc (built, registry-asserted). The public
  fallback (allenai/RLVR-IFeval) was probed and rejected: Tulu's `validate_*`
  taxonomy is not registry-compatible.

## 3. Roster

All stock arms × 3 seeds = 42 × 3 = 126 rows per domain (`gen_domain_manifest.py`,
manifests committed). Two disciplines encoded in the rows:

- **vanilla ×3 pinned to `d0`** — each domain's noise floor comes from one box
  (the m1 principle survives the pool);
- **a2_coldstart fenced** — its SFT stage-1 is per-domain and does not exist;
  math-SFT init on a domain stream answers no registered question. Unlock =
  build the domain's stage-1, a pin move.

Arm-env overrides compose as always (h5 keeps its own `MAX_RESPONSE_LENGTH=100`
because arm env is applied after domain env; j1's verifier rides the domain's
reward path — IF's checker is milliseconds, code's prime_code is capped at 6
tests/row by the prep).

## 4. Gates before the first row (in order)

1. **D6 domain probe** — teacher (4B-2507) and student (1.7B-Base) zero points
   on HumanEval+/MBPP+/IFEval via the existing transfer harness (two
   invocations, no new code). If the teacher's ceiling on a domain is not
   usefully above the student's, that domain's campaign is not worth its bill —
   measured, not assumed, same as the math D6.
2. **3-step rehearsal per domain** (`STEPS=3`, one lane): exercises the val
   swap, the custom-reward seam, and preflight against the new parquets. The
   probe-free "快败+熔断" discipline resumes after one green rehearsal.
3. **Repin with REASON** — the launcher gained two seams (run-defining file);
   the pin moves once, before the first domain row, never mid-batch.

## 5. Registered predictions (before data)

- **P-domain-no-collapse** (M-I cross-domain test): natural answer lengths in
  both domains sit far below the 4096 cap, so EOS supervision density cannot
  self-starve → vanilla should NOT show the late collapse in either domain.
  If it collapses anyway at 4k with short answers, the EOS-starvation account
  is in trouble. Either way M-I gains its cheapest cross-domain evidence.
- **P-revive / P-flat** (the headline theorem's third interval): code's
  candidate-identifier space is large → support/token design (C/D axes) should
  show LARGER gaps over vanilla than they did on math. P-flat = the math
  hierarchy (𝒰 ≳ Γ > 𝒞 > 𝒜) transfers unchanged, and the theorem's revival
  clause loses its best remaining habitat.
- **Per-arm cross-domain verdict table** — the audit's promised
  literature-absent column: the same trick judged independently on math, IF,
  and code, one protocol.

## 6. Cost, honestly

Short rollouts (0.3–1.5k natural) at a 4096 cap: per-row estimate 7–12 GPU·h →
per domain ≈ **0.9–1.5k GPU·h**, both ≈ 2–3k, roughly a quarter of the 16k
campaign. Cost is a treatment outcome (findings 定律三): these numbers are
re-based the moment realized lengths exist, and the 4k cap bounds the blowup
the 16k campaign paid for.

---

## 7. Pair scale-out (amended 2026-08-13, same submission)

The user ruling extends the one-job fleet to two teacher–student pairs. A pair
is the same namespace mechanism as a domain — bigger lanes, its own anchors:

| namespace | pair | lane (GPUs) | shape | tag | status |
|---|---|---|---|---|---|
| `w8b` | **Qwen3-8B-Base ← Qwen3-32B** | 8 (whole box) | student FSDP-4 + teacher TP2 ×2 replicas, mem 0.40, cap 8192 | `w` — **joins the banked cell**: `vanilla_s0_w` already ran this exact shape (81 s/step) and minted the step-0 anchor **0.664** | ungated; roster = §6.5 first batch (vanilla ×2 + c1 + f1; d2/b3 fenced, see below) |
| `p4b` | **Qwen3-4B-Base ← Qwen3-14B** | 4 (half box) | student FSDP-2 + teacher TP1 ×2 replicas, mem 0.40, cap 8192 | `p4b` | **fully gated** on `gates/p4b_ok`, written by: P1b teacher-ceiling probe + both models fetched + one 3-step rehearsal with `VAL_BEFORE_TRAIN=True` (mints THIS cell's anchor — 0.468 and 0.664 both say nothing about a 4B-Base) |

Notes on the record:

- **There is no Qwen3-3B.** The family is 0.6/1.7/4/8/14/32B; the "14B → small"
  rung is **4B-Base**, which is also the cell the plan already registered
  (§3, "配方向上迁移"). If a 3B-class student is genuinely wanted, it means
  leaving the Qwen3 family — a protocol change, not a pair addition.
- **Teacher TP needs no launcher change**: the proven route is `EXTRA_HYDRA`
  (`distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2`),
  which already rides the fingerprint.
- **Topology-carrying arms are fenced in pair manifests, not dropped.** The
  KEEP_SAMPLED family (b3/d1/d2/d3/g2) carries `NGPUS_PER_NODE=2 +
  TEACHER_WORLD_SIZE=2` in its arm env, and arm env is applied AFTER namespace
  env — on an 8-GPU pair lane it would silently downgrade the topology to a
  shape the 8B student cannot fit (FSDP-2 = 61 G/GPU static). §6.5's first
  batch loses d2 and b3 to this fence until a lane-level topology re-assert
  lands (run-defining; goes through the pin).
- **Eviction is width-aware**: a pair row needs a whole/half box, so the
  worker's eval-eviction threshold follows the widest startable namespace —
  without this, eval backfill would starve pair lanes forever at 2-GPU
  granularity.

Fleet budget with everything on (steady state), sized by `TOTAL_GPUS` (the
quota is ~500, not a clean 512 — `submit_fleet.sh` derives the worker count):
math resume 78 + IF 126×2 + code 126×2 + w8b 4×8 + p4b 4×4 ≈ **410 GPUs of
training**, remainder + every draining lane = eval. At 500 → 62 workers = 496
usable; the eval share absorbs the difference. One submission.

---

## 8. Correctness audit before the first row (2026-08-13, CPU-only — all green)

Seven mechanical checks, no GPUs (`scripts/audit_domains_cpu.sh`, reproducible):

1. **Reward unit battery** 14/14 at current HEAD.
2. **Call contract**: the naive/dapo/batch reward managers invoke
   `compute_score(data_source=, solution_str=, ground_truth=, extra_info=)` —
   the dispatcher's signature matches keyword-for-keyword, and the manager's
   per-sample timeout guard scores 0.0 instead of killing a run.
3. **Routing**: `codecontests` → prime_code in THIS verl checkout, proven
   empirically (a correct program scores 1.0 through `default_compute_score`).
4. **Schema**: code parquet ≡ math parquet, column-for-column and
   type-for-type (prompt = chat ndarray, reward_model = {style, ground_truth}).
5. **Prompt lengths** (1.7B tokenizer): code train p50=521 / p99=1377 /
   max=2373 → **3.8% exceed the 1024 cap**; val 5/200. **Ruling: the cap stays
   1024.** The dataloader's `filter_overlong_prompts` drop is identical for
   every arm (effective set ≈12,287 train / ≈196 val for all of them), so arm
   comparisons stay internally valid; raising the cap inflates every code
   row's packing ceiling and still cannot keep max=2373. Recorded here so the
   row counts read as protocol, not surprise.
6. **Official preflight** on the code parquets with the domain's exact knobs:
   `preflight ok` (k1_rec registered, 6 hook modules importable).
7. **Roster**: all 42 stock arms present in each of the three manifests; the
   late additions (f5_tanh, c1_direct, c1_tailbucket, h5_gen100, c2_qb_fixed8,
   c2_qb_perseq, f2_clip2.3, e1_pl_rank_a0, e2_set_coverage_a0, b4_jsd_b0.1,
   b4_jsd_b0.9) verified individually: **math ✓ / if ✓ / code ✓** each.

What CPU cannot prove stays on the smoke job: the 3-step rehearsal per domain
(val swap + custom-reward seam, live), and IF's own length audit the day its
parquet exists (the prep re-runs section 5 by hand).
