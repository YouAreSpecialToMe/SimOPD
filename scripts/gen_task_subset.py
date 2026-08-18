#!/usr/bin/env python3
"""Deterministic prompt-subset builder for h10_task_subset (H-axis wave 1).

Keeps a sha1-parity fraction of training prompts, schema byte-identical to the
source parquet (verl reads it directly via TRAIN_FILE_BASENAME); provenance
goes in a SIDECAR json, never as extra columns -- adding columns would change
the training schema every other arm reads.

Selection is content-keyed (sha1 of the serialized prompt field + salt), so it
is reproducible across machines/pandas versions and independent of row order.
The 9 known duplicate prompt pairs in the campaign train set are inherited
as-is: vanilla trains on them, the subset stays distribution-faithful.

    python scripts/gen_task_subset.py \
        --train-parquet $D/simopd_math/train.parquet \
        --frac 0.5 --out $D/simopd_math/train_sub50.parquet
"""

import argparse
import hashlib
import json
import os
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--salt", default="h10",
                    help="hash salt; a different salt draws an independent subset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if not (0.0 < a.frac < 1.0):
        sys.exit(f"FATAL: --frac {a.frac} outside (0,1)")
    df = pd.read_parquet(a.train_parquet)
    if "prompt" not in df.columns:
        sys.exit(f"FATAL: no 'prompt' column in {a.train_parquet} (have {list(df.columns)})")

    def coin(v):
        h = hashlib.sha1((a.salt + json.dumps(v, sort_keys=True, default=str)).encode()).hexdigest()
        return int(h[:8], 16) % 10_000 < a.frac * 10_000

    keep = df["prompt"].map(coin)
    sub = df[keep].reset_index(drop=True)
    realized = len(sub) / len(df)
    # A parity draw over ~14k prompts lands within a couple points of frac;
    # anything far off means the serialization degenerated (all rows hashing
    # equal, or a schema drift) -- refuse loudly rather than train on garbage.
    if not (0.8 * a.frac <= realized <= 1.2 * a.frac):
        sys.exit(f"FATAL: realized fraction {realized:.4f} far from --frac {a.frac} -- "
                 f"hash pathology or schema drift, refusing to write")

    sub.to_parquet(a.out)
    prov = {
        "generator": "scripts/gen_task_subset.py",
        "source": os.path.abspath(a.train_parquet),
        "frac_requested": a.frac,
        "frac_realized": realized,
        "salt": a.salt,
        "n_source": int(len(df)),
        "n_subset": int(len(sub)),
        "columns": list(df.columns),
    }
    with open(a.out + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=1)
    print(f"subset: {len(sub)}/{len(df)} rows ({realized:.4f}) -> {a.out}")
    print(f"provenance sidecar -> {a.out}.provenance.json")


if __name__ == "__main__":
    main()
