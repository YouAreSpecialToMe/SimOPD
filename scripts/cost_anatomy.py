"""Cost anatomy of the 16k campaign: length-driven vs method-intrinsic.

    python scripts/cost_anatomy.py [--csv docs/data/training_metrics_16k_allkeys.csv.gz]

Three readings off the shipped metrics (first harvested 2026-08-11, RESULTS-GAPS
section 1.4):

  1. step-time decomposition  -- generation is ~86% of the step and tracks the
                                 arm's LENGTH REGIME: the GPU-hour bill is a
                                 treatment outcome, not a method overhead
  2. per-token update cost    -- the kernel-intrinsic FLOP premium after length
                                 is normalized away: 0.06-0.08 ms/tok roster-wide
                                 (quantile / renorm / PL / z are all inside it)
  3. peak-memory budget       -- static 25.6 GB (16 B/param x 1.72B) + 1.9 GB
                                 checkpointed activations + n x 4.93 GB full-vocab
                                 [T,V] copies at dense packing. Implied n clusters
                                 by family: sampled ~2 (3.6 at 16k lengths),
                                 top-k distributional ~7 (consistent with an
                                 unchunked fp32 log-softmax chain: 1 bf16 + 3 fp32
                                 = 7 bf16-equivalents), +full-vocab-criteria ~9
                                 (the 2-card OOM wall). The top-k premium is
                                 IMPLEMENTATION-CONTINGENT, not method-intrinsic:
                                 every kernel's required output is O(T*k).
"""

import argparse

import pandas as pd

V, H, LYR, CAP = 151_936, 2048, 28, 17_408
N = V * H + LYR * ((2048 * 2048 + 2 * 2048 * 1024 + 2048 * 2048) + 3 * 2048 * 6144)
STATIC = N * 16 / 2**30            # bf16 params + fp32 master + Adam m,v + bf16 grads
CKPT = LYR * H * 2 * CAP / 2**30   # checkpointed layer inputs at dense packing
COPY = V * 2 * CAP / 2**30         # one bf16 [T,V] materialization

FAMILIES = {
    "sampled (k1 path)": ["vanilla", "b1_skew_kl", "b5_k2", "f1_soft_log", "f2_hard_clip",
                          "f3_power", "g1_verified_only", "g4_failure_only", "g5_rgopd_gate",
                          "h1_first_segment", "h2_last_segment", "h3_random_segment"],
    "top-k distributional": ["c1_lsm_topk32_renorm", "c2_quantile_budget", "c3_intersection",
                             "c4_pi_tail_budget", "b2_forward_kl", "b4_jsd", "e1_pl_rank",
                             "e2_set_coverage", "e3_zvalue"],
    "wall cohort (4-card DP-2; per-GPU geometry differs)": ["b3_eopd_gate", "d1_tip",
                                                            "d2_selectkd", "d3_teachability",
                                                            "g2_fire_likelihood"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="docs/data/training_metrics_16k_allkeys.csv.gz")
    df = pd.read_csv(p.parse_args().csv)

    print(f"pinned: static {STATIC:.1f} GB | ckpt {CKPT:.1f} GB | one bf16 [T,V] copy {COPY:.2f} GB\n")
    for fam, arms in FAMILIES.items():
        print(f"== {fam}")
        for a in arms:
            g = df[df.arm == a]
            if not len(g):
                continue
            step = g["timing_s/step"].median()
            gen = g["timing_s/gen"].median()
            upd_tok = g["timing_per_token_ms/update_actor"].median()
            mem = g["actor/perf/max_memory_allocated_gb"].median()
            resp = g["response_length/mean"].median()
            copies = (mem - STATIC - CKPT) / COPY
            print(f"  {a:24s} resp {resp:6.0f}  step {step:6.0f}s (gen {gen / step:4.0%})"
                  f"  upd {upd_tok:.2f} ms/tok  peak {mem:5.1f} GB  -> [T,V] copies {copies:4.1f}")
        print()
    print("caveats: copy counts are allocator-level inference under the dense-packing")
    print("T_mb assumption (+-1 for mid-length arms); wall-cohort rows ran DP-2 and are")
    print("not directly comparable; the fp32-chain reading of n~7 awaits a dtype check")
    print("at verl's log_softmax call site.")


if __name__ == "__main__":
    main()
