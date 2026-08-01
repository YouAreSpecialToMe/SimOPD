# Prepare SimOPD math protocol data (Demystifying-aligned):
#   train: nvidia/Nemotron-Cascade-RL-Math (14,476 problems; features: problem/answer/source)
#   val:   HuggingFaceH4/MATH-500 (screening main metric, pass@1)
# Output: verl parquet format under ~/data/simopd_math/
#
# data_source is set to the MATH route so verl's rule-based math scorer
# (boxed extraction + exact match) handles both reward gating (G axis) and val.

import argparse
import os

import datasets

MATH_SCORER_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def to_verl(example, idx, split):
    question = example["problem"].strip() + " " + INSTRUCTION
    return {
        "data_source": MATH_SCORER_SOURCE,
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(example["answer"])},
        "extra_info": {"split": split, "index": idx, "orig_source": example.get("source", "")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/simopd_math")
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)

    train = datasets.load_dataset("nvidia/Nemotron-Cascade-RL-Math", split="train")
    train = train.map(
        lambda ex, i: to_verl(ex, i, "train"),
        with_indices=True,
        remove_columns=train.column_names,
    )
    train.to_parquet(os.path.join(out_dir, "train.parquet"))
    print(f"train: {len(train)} -> {out_dir}/train.parquet")

    # MATH-500: features problem/solution/answer/subject/level/unique_id
    val = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    val = val.map(
        lambda ex, i: to_verl(ex, i, "test"),
        with_indices=True,
        remove_columns=val.column_names,
    )
    val.to_parquet(os.path.join(out_dir, "math500.parquet"))
    print(f"val (MATH500): {len(val)} -> {out_dir}/math500.parquet")
