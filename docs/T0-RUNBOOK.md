# T-0 runbook: bringing the suspended math campaign back on new cards

_Written 2026-08-13 while the fleet is down, so the day cards arrive is
mechanical. Every claim below was verified on that date, not assumed._

## What is waiting

**39 unfinished runs** (13 arms × 3 seeds, steps 25–200 of 250) with banks in
`$CKPT_ROOT/simopd/*_16k`; the other 87 of 126 are done at step 250. Every
one of the 126 has `simopd_fingerprint.txt` + `latest_checkpointed_iteration.txt`.

| arm | seeds' steps | | arm | seeds' steps |
|---|---|---|---|---|
| b2_forward_kl | 175/150/200 | | e2_set_coverage_a0 | 50/25/25 |
| b4_jsd_b0.1 | 50×3 | | f2_clip2.3 | 50×3 |
| b4_jsd_b0.9 | 50×3 | | f4_posclip | 25×3 |
| c1_direct | 25×3 | | f5_tanh | 25×3 |
| c1_tailbucket | 100×3 | | h4_random_scatter | 50×3 |
| c2_qb_fixed8 | 50×3 | | c2_qb_perseq | 25×3 |
| e1_pl_rank_a0 | 75×3 | | | |

## Why resume survives everything that changed (verified, not argued)

1. **The fingerprint contains no git commit** — it is a sha1 of config only
   (models, loss, pg, topk, rollout_n, topology, ARM_ARGS, extra hydra,
   `SIMOPD_*` env minus diagnostics, batch/lr, lengths, `data=`, vllm mem,
   token packing). So the coming repin cannot break it.
2. **Every fingerprint input is stable across the m-fleet → DLC transition**:
   math args render byte-identically under the domain seams (measured);
   arms.yaml has zero diff since the launch pin 708f8bb; `DATA_DIR` and the
   four `SIMOPD_*` path vars come from the same `simopd_env.sh` both
   generations of worker source.
3. **run_opd_baseline.sh auto-resumes**: bank pointer + matching fingerprint →
   `=== resuming <run> from step N`; mismatch → FATAL (with `RESUME=force`
   escape); `RESUME=fresh` refuses to clobber. A resumed run loses at most 24
   steps (SAVE_FREQ=25) and skips the redundant val_before_train.

## The three ledgered moves, in order

1. **Repin math** (the only preflight MISS):
   `REASON="domain seams absent-by-default for math (proven byte-identical args); domain_reward.py never imported by math" MACHINE=<id> bash deploy/campaign.sh --repin`
2. **Manifest re-dispatch, committed** (assignments live in git — the 91af7de
   lesson): the 39 rows' machine column `m25–m37` → `any`. DLC workers walk
   `$2==m || $2=="any"`, so rows named for dead boxes are invisible to them
   until this edit.
3. **Release the 27 stale pool claims** held by dead m25–m34 (verified list =
   claims ∩ unfinished; pinned rows — f5_tanh, b2, the odd s2 seeds — never
   had claims, campaign.sh only claims pool rows). `rm -r .campaign/claims/<name>`
   for each, named in the same commit message as move 2. Safe in both
   futures: owners are dead pods; if the same pods somehow revive, their
   daemons simply re-claim and resume identically.

**Branch: if the zzx pods revive as-is instead of DLC cards arriving** — do
nothing except move 1; their daemons hold their pins/claims and resume their
own rows from the banks.

## After the moves

DLC workers claim freed rows → see bank + matching fingerprint → resume.
Verify with the boot logs (`=== resuming ... from step N`) and one
`pass: evalw=... startable=...` line showing math rows counted. The 3-step
domain rehearsals and D6 probes ride the same smoke job (DOMAINS-AMENDMENT §4).
