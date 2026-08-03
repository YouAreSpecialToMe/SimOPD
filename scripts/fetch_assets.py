"""Fetch the models and evaluation datasets, skipping whatever is already local.

    python scripts/fetch_assets.py --data-dir ~/data/simopd_math
    python scripts/fetch_assets.py --check      # report only, download nothing

`hf download` is already incremental, but it still contacts the hub to compare
every file -- which is precisely where a mirror + Xet setup fails with a 401 CAS
error, on assets that were fully downloaded already. So each asset is resolved
offline first (local_files_only), and the network is touched only for what is
genuinely missing.
"""

import argparse
import os
import sys

MODELS = [
    "Qwen/Qwen3-0.6B-Base",      # screening student
    "Qwen/Qwen3-1.7B",           # sweet-spot teacher
    "Qwen/Qwen3-1.7B-Base",      # final-tier student / anchor
    "Qwen/Qwen3-4B-Instruct-2507",  # mismatch teacher, Demystifying's off-the-shelf slot
    "Qwen/Qwen3-8B",             # D6 teacher ladder
]
# The evaluation suites BENCHMARKS.md pins. AIME24 comes from the H4 mirror:
# math-ai/aime24 ships no answer column, only \boxed inside the solution.
DATASETS = [
    ("HuggingFaceH4/MATH-500", "test"),
    ("math-ai/amc23", "test"),
    ("HuggingFaceH4/aime_2024", "train"),
    ("math-ai/aime25", "test"),
    ("math-ai/minervamath", "test"),
    ("google/IFEval", "train"),
]
IGNORE = ["*.pth", "*.msgpack", "*.h5"]   # torch-only; the flax/tf copies are dead weight


def model_cached(repo):
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo, ignore_patterns=IGNORE, local_files_only=True)
        return True
    except Exception:
        return False


def dataset_cached(name, split):
    import datasets
    try:
        # HF_DATASETS_OFFLINE makes this resolve from cache or fail, never fetch.
        prev = os.environ.get("HF_DATASETS_OFFLINE")
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        try:
            datasets.load_dataset(name, split=split)
            return True
        finally:
            if prev is None:
                os.environ.pop("HF_DATASETS_OFFLINE", None)
            else:
                os.environ["HF_DATASETS_OFFLINE"] = prev
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.expanduser("~/data/simopd_math"))
    p.add_argument("--check", action="store_true", help="report what is missing, download nothing")
    args = p.parse_args()

    missing = 0

    print("models:")
    for repo in MODELS:
        if model_cached(repo):
            print(f"  cached   {repo}")
            continue
        if args.check:
            print(f"  MISSING  {repo}")
            missing += 1
            continue
        print(f"  fetching {repo}")
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(repo, ignore_patterns=IGNORE)
            print(f"  ok       {repo}")
        except Exception as e:
            print(f"  FAILED   {repo}: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
            missing += 1

    print("\neval datasets:")
    for name, split in DATASETS:
        if dataset_cached(name, split):
            print(f"  cached   {name}")
            continue
        if args.check:
            print(f"  MISSING  {name}")
            missing += 1
            continue
        print(f"  fetching {name}")
        try:
            import datasets
            datasets.load_dataset(name, split=split)
            print(f"  ok       {name}")
        except Exception as e:
            print(f"  FAILED   {name}: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
            missing += 1

    print("\ntraining data:")
    train = os.path.join(os.path.expanduser(args.data_dir), "train.parquet")
    val = os.path.join(os.path.expanduser(args.data_dir), "math500.parquet")
    if os.path.exists(train) and os.path.exists(val):
        import pandas as pd
        print(f"  cached   {train} ({len(pd.read_parquet(train))} rows)")
        print(f"  cached   {val} ({len(pd.read_parquet(val))} rows)")
    elif args.check:
        print(f"  MISSING  {args.data_dir}/{{train,math500}}.parquet")
        missing += 1
    else:
        print(f"  building {args.data_dir}")
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        rc = subprocess.run([sys.executable, os.path.join(here, "prep_nemotron_math.py"),
                             "--local_save_dir", args.data_dir]).returncode
        missing += rc != 0

    print()
    if missing:
        print(f"{missing} asset(s) missing or failed")
        return 1
    print("all assets present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
