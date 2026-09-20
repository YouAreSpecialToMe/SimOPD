# Verdict ledger -- amc23 @ step 250 (evals: /home/tiger/opd/wandb_inbox/evals)

## Noise floor (vanilla seeds, range of accuracy)
- seed 0: acc 0.4188  (40 problems)
- floor: AWAIT-FLOOR (1/3 seeds evaluated)

## Arms vs vanilla_corr_s0
| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |
|---|---|---|---|---|---|
| vanilla_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1 |
| n2_corr | +0.0156 | 2 | 2 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right |
| vanilla_te | +0.0344 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |

## Side-effect panel (transfer deltas vs vanilla_corr_s0; informational until per-bench floors exist)
- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)

# Verdict ledger -- aime24 @ step 250 (evals: /home/tiger/opd/wandb_inbox/evals)

## Noise floor (vanilla seeds, range of accuracy)
- seed 0: acc 0.0750  (30 problems)
- floor: AWAIT-FLOOR (1/3 seeds evaluated)

## Arms vs vanilla_corr_s0
| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |
|---|---|---|---|---|---|
| vanilla_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1 |
| n2_corr | -0.0375 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right |
| vanilla_te | -0.0083 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |

## Side-effect panel (transfer deltas vs vanilla_corr_s0; informational until per-bench floors exist)
- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)

# Verdict ledger -- aime25 @ step 250 (evals: /home/tiger/opd/wandb_inbox/evals)

## Noise floor (vanilla seeds, range of accuracy)
- seed 0: acc 0.0625  (30 problems)
- floor: AWAIT-FLOOR (1/3 seeds evaluated)

## Arms vs vanilla_corr_s0
| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |
|---|---|---|---|---|---|
| vanilla_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1 |
| n2_corr | +0.0292 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right |
| vanilla_te | +0.0167 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |

## Side-effect panel (transfer deltas vs vanilla_corr_s0; informational until per-bench floors exist)
- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)

# Verdict ledger -- amc23 @ step 200 (evals: /home/tiger/opd/wandb_inbox/evals)

## Noise floor (vanilla seeds, range of accuracy)
- seed 0: acc 0.4406  (40 problems)
- floor: AWAIT-FLOOR (1/3 seeds evaluated)

## Arms vs vanilla_corr_s0
| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |
|---|---|---|---|---|---|
| vanilla_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1 |
| n2_corr | +0.0094 | 2 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right |
| f2_hard_clip_corr | -0.0219 | 2 | 2 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f2_hard_clip |
| f3_power_corr | -0.0094 | 2 | 2 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f3_power |
| b1_skew_kl_corr | -0.0094 | 1 | 4 | 0.3750 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked b1_skew_kl |
| g1_verified_only_corr | -0.1531 | 7 | 1 | 0.0703 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked g1_verified_only |
| h1_first_segment_corr | +0.0000 | 4 | 3 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h1_first_segment row, seed-paired |
| h3_random_segment_corr | -0.0156 | 0 | 2 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h3_random_segment row, seed-paired |
| h4_random_scatter_corr | -0.0500 | 2 | 2 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h4_random_scatter row, seed-paired |
| d2_selectkd_corr | -0.0094 | 2 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d2_selectkd row, seed-paired |
| d3_teachability_corr | -0.0375 | 3 | 1 | 0.6250 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d3_teachability row, seed-paired |
| b2_forward_kl_corr | +0.0031 | 2 | 3 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked b2_forward_kl row, seed-paired |
| c3_intersection_corr | -0.0094 | 2 | 2 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked c3_intersection row, seed-paired |
| vanilla_te | +0.0094 | 0 | 2 | 0.5000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_fixed8_corr | -0.0125 | 2 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_perseq_corr | -0.0250 | 3 | 2 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a1_gkd_mix0.5_n0 | -0.0938 | 3 | 0 | 0.2500 | AWAIT-FLOOR (p=≥0.05) |
| a3_offpolicy_n0 | -0.2062 | 10 | 0 | 0.0020 | AWAIT-FLOOR (p=<0.05) |
| a4_dagger_anneal_n0 | -0.0656 | 4 | 1 | 0.3750 | AWAIT-FLOOR (p=≥0.05) |
| a5_aggrevate_n0 | -0.0312 | 4 | 2 | 0.6875 | AWAIT-FLOOR (p=≥0.05) |

## Side-effect panel (transfer deltas vs vanilla_corr_s0; informational until per-bench floors exist)
- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)

# Verdict ledger -- aime24 @ step 200 (evals: /home/tiger/opd/wandb_inbox/evals)

## Noise floor (vanilla seeds, range of accuracy)
- seed 0: acc 0.0750  (30 problems)
- floor: AWAIT-FLOOR (1/3 seeds evaluated)

## Arms vs vanilla_corr_s0
| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |
|---|---|---|---|---|---|
| vanilla_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1 |
| n2_corr | -0.0125 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right |
| f2_hard_clip_corr | +0.0083 | 1 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f2_hard_clip |
| f3_power_corr | -0.0125 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f3_power |
| b1_skew_kl_corr | +0.0042 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked b1_skew_kl |
| g1_verified_only_corr | -0.0500 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked g1_verified_only |
| h1_first_segment_corr | -0.0375 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h1_first_segment row, seed-paired |
| h3_random_segment_corr | -0.0125 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h3_random_segment row, seed-paired |
| h4_random_scatter_corr | -0.0042 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h4_random_scatter row, seed-paired |
| d2_selectkd_corr | -0.0083 | 1 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d2_selectkd row, seed-paired |
| d3_teachability_corr | -0.0083 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d3_teachability row, seed-paired |
| b2_forward_kl_corr | -0.0458 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked b2_forward_kl row, seed-paired |
| c3_intersection_corr | -0.0042 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked c3_intersection row, seed-paired |
| vanilla_te | -0.0167 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_fixed8_corr | -0.0250 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_perseq_corr | -0.0375 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) |
| a1_gkd_mix0.5_n0 | -0.0375 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) |
| a3_offpolicy_n0 | -0.0667 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) |
| a4_dagger_anneal_n0 | -0.0208 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a5_aggrevate_n0 | -0.0250 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |

## Side-effect panel (transfer deltas vs vanilla_corr_s0; informational until per-bench floors exist)
- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)

# Verdict ledger -- aime25 @ step 200 (evals: /home/tiger/opd/wandb_inbox/evals)

## Noise floor (vanilla seeds, range of accuracy)
- seed 0: acc 0.0792  (30 problems)
- floor: AWAIT-FLOOR (1/3 seeds evaluated)

## Arms vs vanilla_corr_s0
| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |
|---|---|---|---|---|---|
| vanilla_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1 |
| n2_corr | +0.0083 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right |
| f2_hard_clip_corr | +0.0042 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f2_hard_clip |
| f3_power_corr | +0.0167 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f3_power |
| b1_skew_kl_corr | +0.0042 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked b1_skew_kl |
| g1_verified_only_corr | -0.0500 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked g1_verified_only |
| h1_first_segment_corr | -0.0125 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h1_first_segment row, seed-paired |
| h3_random_segment_corr | +0.0125 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h3_random_segment row, seed-paired |
| h4_random_scatter_corr | -0.0042 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h4_random_scatter row, seed-paired |
| d2_selectkd_corr | -0.0250 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d2_selectkd row, seed-paired |
| d3_teachability_corr | -0.0250 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d3_teachability row, seed-paired |
| b2_forward_kl_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked b2_forward_kl row, seed-paired |
| c3_intersection_corr | -0.0083 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked c3_intersection row, seed-paired |
| vanilla_te | +0.0042 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_fixed8_corr | -0.0292 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_perseq_corr | -0.0083 | 1 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a1_gkd_mix0.5_n0 | -0.0167 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a3_offpolicy_n0 | -0.0542 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a4_dagger_anneal_n0 | -0.0125 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a5_aggrevate_n0 | -0.0083 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |

## Side-effect panel (transfer deltas vs vanilla_corr_s0; informational until per-bench floors exist)
- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)

# Verdict ledger -- amc23 @ step 100 (evals: /home/tiger/opd/wandb_inbox/evals)

## Noise floor (vanilla seeds, range of accuracy)
- seed 0: acc 0.4125  (40 problems)
- floor: AWAIT-FLOOR (1/3 seeds evaluated)

## Arms vs vanilla_corr_s0
| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |
|---|---|---|---|---|---|
| vanilla_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1 |
| n2_corr | +0.0094 | 0 | 3 | 0.2500 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right |
| f2_hard_clip_corr | +0.0312 | 2 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f2_hard_clip |
| f3_power_corr | +0.0094 | 3 | 1 | 0.6250 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f3_power |
| h2_last_segment_corr | -0.1781 | 10 | 0 | 0.0020 | AWAIT-FLOOR (p=<0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked h2_last_segment |
| b1_skew_kl_corr | -0.0187 | 3 | 0 | 0.2500 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked b1_skew_kl |
| g1_verified_only_corr | -0.0187 | 3 | 1 | 0.6250 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked g1_verified_only |
| g6_seqmean_corr | -0.0187 | 2 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked g6_seqmean |
| h1_first_segment_corr | -0.0344 | 5 | 2 | 0.4531 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h1_first_segment row, seed-paired |
| h3_random_segment_corr | -0.0594 | 5 | 1 | 0.2188 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h3_random_segment row, seed-paired |
| h4_random_scatter_corr | -0.0562 | 4 | 1 | 0.3750 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h4_random_scatter row, seed-paired |
| d2_selectkd_corr | +0.0625 | 1 | 2 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d2_selectkd row, seed-paired |
| d3_teachability_corr | -0.0281 | 5 | 1 | 0.2188 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d3_teachability row, seed-paired |
| b2_forward_kl_corr | -0.0031 | 3 | 1 | 0.6250 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked b2_forward_kl row, seed-paired |
| e2_set_coverage_a0_corr | -0.0969 | 6 | 1 | 0.1250 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked e2_set_coverage_a0 row, seed-paired |
| c3_intersection_corr | +0.0188 | 2 | 2 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked c3_intersection row, seed-paired |
| vanilla_te | +0.0375 | 1 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_fixed8_corr | -0.0156 | 2 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_perseq_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a1_gkd_mix0.5_n0 | -0.0656 | 5 | 2 | 0.4531 | AWAIT-FLOOR (p=≥0.05) |
| a3_offpolicy_n0 | -0.1562 | 10 | 0 | 0.0020 | AWAIT-FLOOR (p=<0.05) |
| a4_dagger_anneal_n0 | -0.0719 | 5 | 1 | 0.2188 | AWAIT-FLOOR (p=≥0.05) |
| a5_aggrevate_n0 | -0.0844 | 5 | 0 | 0.0625 | AWAIT-FLOOR (p=≥0.05) |

## Side-effect panel (transfer deltas vs vanilla_corr_s0; informational until per-bench floors exist)
- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)

# Verdict ledger -- aime24 @ step 100 (evals: /home/tiger/opd/wandb_inbox/evals)

## Noise floor (vanilla seeds, range of accuracy)
- seed 0: acc 0.0708  (30 problems)
- floor: AWAIT-FLOOR (1/3 seeds evaluated)

## Arms vs vanilla_corr_s0
| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |
|---|---|---|---|---|---|
| vanilla_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1 |
| n2_corr | -0.0208 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right |
| f2_hard_clip_corr | -0.0125 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f2_hard_clip |
| f3_power_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f3_power |
| h2_last_segment_corr | -0.0667 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked h2_last_segment |
| b1_skew_kl_corr | -0.0083 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked b1_skew_kl |
| g1_verified_only_corr | -0.0208 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked g1_verified_only |
| g6_seqmean_corr | -0.0167 | 2 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked g6_seqmean |
| h1_first_segment_corr | -0.0292 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h1_first_segment row, seed-paired |
| h3_random_segment_corr | -0.0292 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h3_random_segment row, seed-paired |
| h4_random_scatter_corr | -0.0375 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h4_random_scatter row, seed-paired |
| d2_selectkd_corr | -0.0167 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d2_selectkd row, seed-paired |
| d3_teachability_corr | -0.0375 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d3_teachability row, seed-paired |
| b2_forward_kl_corr | -0.0083 | 1 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked b2_forward_kl row, seed-paired |
| e2_set_coverage_a0_corr | -0.0333 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked e2_set_coverage_a0 row, seed-paired |
| c3_intersection_corr | +0.0000 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked c3_intersection row, seed-paired |
| vanilla_te | -0.0167 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_fixed8_corr | +0.0042 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_perseq_corr | +0.0000 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a1_gkd_mix0.5_n0 | -0.0500 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) |
| a3_offpolicy_n0 | -0.0708 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) |
| a4_dagger_anneal_n0 | -0.0458 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) |
| a5_aggrevate_n0 | -0.0292 | 2 | 0 | 0.5000 | AWAIT-FLOOR (p=≥0.05) |

## Side-effect panel (transfer deltas vs vanilla_corr_s0; informational until per-bench floors exist)
- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)

# Verdict ledger -- aime25 @ step 100 (evals: /home/tiger/opd/wandb_inbox/evals)

## Noise floor (vanilla seeds, range of accuracy)
- seed 0: acc 0.0583  (30 problems)
- floor: AWAIT-FLOOR (1/3 seeds evaluated)

## Arms vs vanilla_corr_s0
| arm | Δacc | b (van✓ arm✗) | c (van✗ arm✓) | p (exact McNemar) | verdict |
|---|---|---|---|---|---|
| vanilla_corr | +0.0000 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠single-knob vs vanilla; verdict reads the termination fixed point (trunc/len trajectory, interval not endpoint) before composite; P-artifact (interior fixed point) vs P-drive (locks like vanilla); eos_dl_at_stop (applied, event-level) vs eos_dl_at_stop_raw (vanilla's -25) is the live receipt; the sampled-<|im_end|> corner is watched on eos_pm_1 |
| n2_corr | +0.0125 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr (carrier, no channel) and n2_termcal (channel, no fix): the 2x2 with vanilla reads whether dense supply has value once the sign is right |
| f2_hard_clip_corr | +0.0375 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f2_hard_clip |
| f3_power_corr | +0.0167 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked f3_power |
| h2_last_segment_corr | -0.0375 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked h2_last_segment |
| b1_skew_kl_corr | +0.0083 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked b1_skew_kl |
| g1_verified_only_corr | -0.0083 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked g1_verified_only |
| g6_seqmean_corr | +0.0042 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and banked g6_seqmean |
| h1_first_segment_corr | -0.0042 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h1_first_segment row, seed-paired |
| h3_random_segment_corr | -0.0042 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h3_random_segment row, seed-paired |
| h4_random_scatter_corr | +0.0042 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1: judged against vanilla_corr and the banked h4_random_scatter row, seed-paired |
| d2_selectkd_corr | +0.0083 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d2_selectkd row, seed-paired |
| d3_teachability_corr | -0.0125 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 2 (D, both paths): judged against vanilla_corr and the banked d3_teachability row, seed-paired |
| b2_forward_kl_corr | -0.0083 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked b2_forward_kl row, seed-paired |
| e2_set_coverage_a0_corr | -0.0250 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked e2_set_coverage_a0 row, seed-paired |
| c3_intersection_corr | +0.0375 | 0 | 1 | 1.0000 | AWAIT-FLOOR (p=≥0.05) ⚠corrected-rerun wave 1 (support-corrected, Path 2): judged against vanilla_corr and the banked c3_intersection row, seed-paired |
| vanilla_te | +0.0458 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_fixed8_corr | +0.0042 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| c2_qb_perseq_corr | +0.0208 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a1_gkd_mix0.5_n0 | +0.0083 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a3_offpolicy_n0 | -0.0458 | 1 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a4_dagger_anneal_n0 | -0.0250 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |
| a5_aggrevate_n0 | -0.0125 | 0 | 0 | 1.0000 | AWAIT-FLOOR (p=≥0.05) |

## Side-effect panel (transfer deltas vs vanilla_corr_s0; informational until per-bench floors exist)
- (no transfer artifacts at this step yet; scripts/eval_transfer.sh per finished arm)

