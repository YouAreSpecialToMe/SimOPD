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
- **IF**: `nvidia/Nemotron-Cascade-RL-Instruction-Following` sampled to 14,476
  (anchor family = the math set's own), constraint ids validated against the
  vendored registry, kwargs None-stripped. **Gated on an HF license token —
  prep is one command once the token lands.** Val = google/IFEval 541,
  strict-prompt-acc. The public fallback (allenai/RLVR-IFeval) was probed and
  rejected: Tulu's `validate_*` taxonomy is not registry-compatible.

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
