# SimOPD scaling campaign ("0921") -- results snapshot

Built 2026-09-25 10:41:44 server time (CST = PT + 15 h) by `tools/export_0921.sh` on w0 from the four Merlin nodes; refreshed periodically while
the campaign runs, so a snapshot may be partial. Recipe: `docs/SCALING-RECIPE.md` of branch `codex/scaling-selectkd-skl` @ 9e9076d (teachers
Qwen3-4B/8B/32B x students Qwen3-1.7B/4B/8B-Base x {vanilla, ours}, seed 0, 250 iterations, checkpoints every 50). Offline suite = the
branch's locked one: AIME24/25 + AMC23 avg@32, Minerva + MATH500 avg@3, T 0.7, top-p 0.95, 32,768 tokens; the composite is the equal macro
average of scripts/eval_suite.py (AIME24+25 pooled by problem, AMC23, Minerva, MATH500 -- 0.25 each). The three untrained students are
evaluated once each as `student_base_<size>` at step 0.

State of this snapshot: 18 training runs archived, 465 suite artifacts, 93/93 fully evaluated (run, step) cells.

Files
- `eval_cells.csv` -- one row per suite artifact (newest per run/step/benchmark): avg@k = mean over problems of the per-problem mean of `correct`;
  pass@k = share of problems with at least one correct sample; lengths in tokens; trunc_rate = share of samples that hit the token budget.
- `eval_composites.tsv` -- per (run, step): the four components and the composite (blank until all five benchmarks exist).
- `eval_grid_status.tsv` -- the 93 cells the campaign needs (18 runs x 5 steps + 3 untrained students) and what is done.
- `per_sample/` -- every artifact without its `response` / `token_ids` columns: per problem and sample `correct`, `resp_len`, `truncated`,
  `finish_reason`, `last_token_id`, plus the artifact's provenance columns. The full responses stay on the nodes (`simopd_data/store_0921/evals/`).
- `train/runs.tsv` -- per run: node, wall time, exit code, steps, mean s/step, stop contract, W&B id, in-loop MATH500 greedy at 50..250.
- `train/inloop_val.tsv` -- the in-loop validation curve (verl's MATH500 greedy mean@1; health telemetry, not the offline suite).
- `train/metrics/<run>/launch_*.jsonl.gz` -- the verl file logger: every scalar per step, one file per launch (a resume = a later launch).
- `train/manifests/<run>/` -- run_manifest.json, the launch manifests, stop contract, fingerprint, latest checkpoint step.
- `traj_samples/<run>/stage_<step>.parquet|json` -- 100 sequences drawn uniformly from every 25th training step's batch (tools/traj_sample.py).
- `logs/` -- gzipped training / eval-farm / guard logs, queue files, exit records.
- `MANIFEST.tsv` -- path, bytes, md5 of every file in this snapshot (written last).
