# Prepare SimOPD instruction-following domain data (anchor-aligned):
#   train: nvidia/Nemotron-Cascade-RL-Instruction-Following (108,938 prompts with
#          IFEval-taxonomy constraint annotations), sampled to the math set's size
#   val:   google/IFEval (541 prompts; strict-prompt-acc is the in-loop metric)
# Output: verl parquet format under ~/data/simopd_if/
#
# Dataset ruling (survey line 208's own precedent): anchor alignment over
# literature mode -- this is the SAME Nemotron-Cascade family as the math set, so
# the two domain cells differ by domain and nothing else. The audited pool offers
# no better anchor: RG-OPD trained a mixed UltraInteract subset, SelecTKD's IF set
# is unnamed, FiRe/Teachability never trained IF at all.
#
# data_source is "simopd/ifeval": verl has no IF entry in its scorer registry, so
# domain runs set custom_reward_function to src/simopd/domain_reward.py, whose
# dispatcher sends this data_source to the vendored Google checker and everything
# else to stock verl. The constraints ride in reward_model.ground_truth as JSON.
#
# Two hygiene steps happen HERE, not at training time:
#   * instruction ids are validated against the vendored registry -- an id the
#     checker cannot instantiate would score 0 forever and read as "hard prompt"
#     rather than "broken row" (dropped, counted, reported);
#   * kwargs dicts lose their None-valued keys -- build_description(**kwargs)
#     treats an explicit None as an argument, not an absence, and IFEval's own
#     evaluation_lib does this exact strip.

import argparse
import json
import os
import sys

import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "third_party"))
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

DATA_SOURCE = "simopd/ifeval"
TRAIN_SET = "nvidia/Nemotron-Cascade-RL-Instruction-Following"
VAL_SET = "google/IFEval"
# The math train set is 14,476 rows; matching it keeps "domain" the only thing the
# two campaigns disagree about (steps-per-epoch, dataloader rhythm all identical).
MATH_SET_SIZE = 14476


def _clean_kwargs(raw):
    """kwargs per instruction: strip None values; tolerate JSON strings."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}
    return {k: v for k, v in dict(raw).items() if v is not None}


def to_verl(prompt, instruction_id_list, kwargs_list, idx, split, orig_source=""):
    gt = {
        "instruction_id_list": list(instruction_id_list),
        "kwargs": [_clean_kwargs(k) for k in kwargs_list],
        "prompt": prompt,
    }
    return {
        "data_source": DATA_SOURCE,
        # No boxed-answer INSTRUCTION suffix here, unlike the math prep: IF prompts
        # are self-contained and appending anything would itself violate constraints
        # like "reply with exactly N words".
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "instruction_following",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
        "extra_info": {"split": split, "index": idx, "orig_source": orig_source},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_save_dir", default="~/data/simopd_if")
    ap.add_argument("--sample", type=int, default=MATH_SET_SIZE,
                    help="training rows to keep (seeded shuffle); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from instruction_following_eval import instructions_registry
    known = set(instructions_registry.INSTRUCTION_DICT)

    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)

    train = datasets.load_dataset(TRAIN_SET, split="train")
    cols = set(train.column_names)
    # The card documents instruction_id_list/prompt/kwargs; fail loudly if the
    # schema moved rather than silently writing empty ground truths.
    for need in ("prompt", "instruction_id_list"):
        assert need in cols, f"{TRAIN_SET} schema drift: no '{need}' in {sorted(cols)}"
    kw_col = "kwargs" if "kwargs" in cols else None
    assert kw_col, f"{TRAIN_SET} schema drift: no kwargs column in {sorted(cols)}"

    dropped_unknown = 0
    rows = []
    for i, ex in enumerate(train):
        ids = list(ex["instruction_id_list"] or [])
        kws = ex[kw_col]
        if isinstance(kws, str):
            kws = json.loads(kws) if kws.strip() else []
        kws = list(kws or [])
        if not ids or len(ids) != len(kws) or any(x not in known for x in ids):
            dropped_unknown += 1
            continue
        rows.append((ex["prompt"], ids, kws, ex.get("dataset", "") or ex.get("source", "")))
    print(f"loaded {len(train)}; dropped {dropped_unknown} rows with unknown/misaligned "
          f"instruction ids (registry has {len(known)})")

    import random
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.sample:
        rows = rows[: args.sample]

    ds = datasets.Dataset.from_list(
        [to_verl(p, ids, kws, i, "train", src) for i, (p, ids, kws, src) in enumerate(rows)]
    )
    ds.to_parquet(os.path.join(out_dir, "train.parquet"))
    print(f"train: {len(ds)} -> {out_dir}/train.parquet")

    val = datasets.load_dataset(VAL_SET, split="train")   # IFEval ships as 'train'
    vrows = []
    for i, ex in enumerate(val):
        ids = list(ex["instruction_id_list"])
        kws = list(ex["kwargs"])
        assert all(x in known for x in ids), f"IFEval id missing from vendored registry: {ids}"
        vrows.append(to_verl(ex["prompt"], ids, kws, i, "test", "google/IFEval"))
    vds = datasets.Dataset.from_list(vrows)
    vds.to_parquet(os.path.join(out_dir, "ifeval.parquet"))
    print(f"val (IFEval): {len(vds)} -> {out_dir}/ifeval.parquet")


if __name__ == "__main__":
    main()
