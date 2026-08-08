# Running SimOPD arms as a remote collaborator

SimOPD is a pre-registered audit of on-policy distillation (OPD) variants: one
frozen protocol, 33 "arms" (one trick per arm), verdicts per arm against a vanilla
floor. Target: ICLR 2027. Everything that defines a run is pinned in this repo;
your job is compute, not design. The three local machines (m1-m3, 24×A100) run
the vanilla floor and a handful of arms; **everything labeled `remote` in
`configs/campaign.tsv` is open for collaborators** -- 11 main-batch arms and a
10-arm supplement cohort, 3 seeds each.

## The contract (read first)

1. **Never modify run-defining files**: `src/`, `configs/arms.yaml`,
   `scripts/run_opd_baseline.sh`, `scripts/arm.py`. If a script *refuses* to run,
   that is the experiment protecting itself -- open an issue with the message; do
   not patch around it. Assignment files (`configs/campaign.tsv`) are the only
   thing your PRs touch.
2. **Run at the pin.** Your claim PR records the commit you run at; `campaign.sh`
   refuses if run-defining files drifted from the pin. Every run stamps a config
   fingerprint automatically.
3. **Defaults are the protocol.** `STEPS=250`, `SAVE_FREQ=25` (keep every ckpt,
   ~310GB/run), 16384-token cap, pinned optimizer and sampling. Do not override
   anything; arm-specific knobs enter only through `scripts/arm.py env <arm>`,
   which the lane runner calls for you.
4. **Three seeds** (0/1/2) per arm; each row in the manifest is one run.
5. **Cluster-local floor**: before your first arm, run `vanilla` seeds 0-2 on YOUR
   cluster. Verdicts for your arms are computed against *your* floor -- our pilot
   showed cross-cluster comparisons need caveats, so every site mints its own.
   Add these three rows in your claim PR (they are expected extras, not stolen
   from the pool).

## Hardware per run

2×A100-80G (1 actor + 1 teacher GPU; the teacher is a separate vLLM server the
lane starts for you). ~14-28h per run at the 16k cap. The n8 mini-cell
(`vanilla_n8`, `j1_kdrl`) is deferred (wave 8, `hold`) -- not claimable.

## Site bring-up (once)

```bash
git clone git@github.com:YouAreSpecialToMe/SimOPD.git && cd SimOPD
# python env: verl + vLLM, see setup notes in docs/ (same pins as the paper)
# models: Qwen/Qwen3-1.7B-Base (student), Qwen/Qwen3-4B-Instruct-2507 (teacher)
# training data: ~14.4k-row train.parquet, distributed out-of-band -- open an
#   issue titled "data access: <site>" and we send the parquet + sha256; place it
#   at $DATA_DIR/train.parquet (default ~/data/simopd_math/)
MACHINE=site:<yourlabel> bash deploy/campaign.sh --dry     # registers this box
bash deploy/campaign.sh --fingerprint                      # hardware/software record; paste into your claim PR
MACHINE=site:<yourlabel> bash deploy/campaign.sh --machine-control
#   ^ 50-step vanilla probe (~4-5h): validates memory at the 16k token budget and
#     gives a val@25/50 point to sanity-check against the numbers in your claim PR thread
```

## Claiming arms

Open a PR that edits `configs/campaign.tsv` only:
- change `remote` to `site:<yourlabel>` on the rows you take (whole arms -- all
  3 seeds -- preferred);
- add your three `vanilla` floor rows with the same label;
- PR description: commit you will run at + the `--fingerprint` output.
`bash deploy/campaign.sh --plan` must stay clean (it enforces single ownership of
every (arm, seed)). Wave 6 first; wave 7 (supplement) only after your wave-6 rows
are running. `a1/a2/a3` are gated on our side -- never claimable.

## Running

```bash
MACHINE=site:<yourlabel> bash deploy/campaign.sh           # fills free lanes once
nohup bash deploy/campaign_daemon.sh > logs/$(hostname)_daemon.log 2>&1 &   # keeps them fed
python scripts/progress.py                                 # live view
```
Failures: read `scripts/triage.py logs/*/lane*.log` before relaunching; a run that
failed 3× quarantines itself -- report it in your claim PR thread, do not force.

## Returning results

Per finished run:
1. **Offline suite eval on your GPUs** (the official curve is evaluated
   post-training over the saved ckpts):
   `python scripts/eval_suite.py sweep --run-id <arm>_s<seed> --ckpt-dir <ckpt root>`
   (AIME24/25@32 + AMC23@32 + Minerva@3 + MATH500@3; sampling τ0.7/top_p0.95 is
   baked in -- do not override).
2. **Deliver back** in the claim PR thread: the eval output dir
   (`$SIMOPD_EVALS/<run>/`), the run's lane log, its fingerprint file, and your
   logger export (wandb csv or equivalent). Small files in the PR; anything big
   as a shared-drive link.
3. **Checkpoints stay on your storage** (all of them). We may later request
   specific steps for mechanism-wave analysis.

Questions → GitHub issues. Protocol details → `docs/PROTOCOL-unified.md`; what
each arm is and why → `docs/arm-provenance-r4.md` + `configs/arms.yaml` notes.
