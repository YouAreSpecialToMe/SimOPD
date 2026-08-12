# Prepare SimOPD code domain data (execution-verified, prime_code-compatible):
#   train: verified competitive-programming problems with stdin/stdout test cases
#   val:   a seeded held-out slice of the SAME distribution (in-loop pass@1);
#          HumanEval+/MBPP+ stay the offline/transfer judges (evalplus, unchanged)
# Output: verl parquet format under ~/data/simopd_code/
#
# Why stdin/stdout format is load-bearing: verl's stock dispatch sends
# data_source="codecontests" to prime_code, whose harness executes
# {"inputs": [...], "outputs": [...]}(+ optional fn_name) test cases with
# timeouts. Every row this script writes runs on that path with ZERO new
# execution code -- which is what keeps the G axis (verified_only and friends)
# and the in-loop val runnable on day one. pytest-style sets (KodCode) do not
# fit this harness and are deliberately not used.
#
# Sources, in order of preference:
#   * agentica-org/DeepCoder-Preview-Dataset (taco/primeintellect/lcb splits,
#     every problem verified against an official solution, >=5 tests each)
#   * likaixin/TACO-verified (TACO rows whose official solution passes its tests)
# Both descend from TACO/codecontests lineage -- the same lineage RG-OPD's
# UltraInteract coding split draws on, which is the audit's only in-pool code
# training precedent.
#
# Test-count cap: prime_code runs every test with a per-test timeout. A training
# reward that fires 128 rollouts x 50 tests is a CPU bill nobody registered, so
# training rows keep at most --max-tests cases (seeded subsample; the cases are
# i.i.d. stdin/stdout pairs). The held-out val keeps up to 2x that. This caps the
# reward's worst case, it does not change what "pass" means for a correct program.
#
# Self-verification (the retention rule KodCode and DeepCoder themselves use, run
# against OUR exact setup rather than trusted from theirs): a row survives only if
# an official solution scores 1.0 through the same prime_code harness, the same
# CAPPED tests, and the same stdin/stdout mode the model will face. The first
# 24-row spot check measured why this is not optional: 17/24 -- three fn_name-style
# rows whose call-convention tests structurally mismatch the stdin/stdout prompt
# (the model can never satisfy them as instructed; dropped outright), and four rows
# whose official solutions fail our harness (timeouts, output-format drift between
# their runner and prime_code). A row whose reward is unreachable by a correct
# program is not a hard problem, it is label noise -- and the G axis would gate on it.

import argparse
import json
import os
import random

import datasets

DATA_SOURCE = "codecontests"   # verl stock dispatch -> prime_code
MATH_SET_SIZE = 14476
INSTRUCTION = ("Write a complete Python program that reads from standard input and "
               "writes the answer to standard output. Put your final solution in a "
               "```python code block.")


def _norm_tests(ex):
    """Return {'inputs': [...], 'outputs': [...] [, 'fn_name']} or None.

    Handles the three layouts seen in the wild: TACO's `input_output` JSON string,
    DeepCoder/verl-style `reward_model.ground_truth`, and a bare `tests` field.
    """
    raw = None
    if ex.get("input_output"):
        raw = ex["input_output"]
    elif isinstance(ex.get("reward_model"), dict) and ex["reward_model"].get("ground_truth"):
        raw = ex["reward_model"]["ground_truth"]
    elif ex.get("tests"):
        raw = ex["tests"]
    elif ex.get("ground_truth"):
        raw = ex["ground_truth"]
    if raw is None:
        return None
    try:
        t = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(t, list):  # some layouts: [{"input":..,"output"/"expected_output":..}]
            ins = [c.get("input", "") for c in t]
            outs = [c.get("output", c.get("expected_output", "")) for c in t]
            t = {"inputs": ins, "outputs": outs}
        if not isinstance(t, dict) or not t.get("inputs") or not t.get("outputs"):
            return None
        if len(t["inputs"]) != len(t["outputs"]):
            return None
        keep = {"inputs": list(t["inputs"]), "outputs": list(t["outputs"])}
        if t.get("fn_name"):
            keep["fn_name"] = t["fn_name"]
        return keep
    except Exception:
        return None


def _strip_fence(sol):
    # primeintellect solutions often arrive already wrapped in ```python fences;
    # prime_code splits on the LAST fence, so nesting is survivable -- but verify
    # with the same text the harness will effectively see.
    s = sol.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[: -3]
    return s


def _first_solutions(ex, k=2):
    sols = ex.get("solutions")
    if sols is None:
        return []
    if isinstance(sols, str):
        try:
            sols = json.loads(sols)
        except Exception:
            sols = [sols]
    if isinstance(sols, dict):  # taco sometimes: {"language": [...], "solution": [...]}
        sols = sols.get("solution", [])
    return [str(x) for x in list(sols)[:k]]


def _verify_one(item):
    """Worker: does ANY of the row's first official solutions pass the capped tests
    through prime_code, exactly as the training reward will run them?"""
    idx, tests_json, sols = item
    from verl.utils.reward_score import prime_code
    for sol in sols:
        try:
            ok, _ = prime_code.compute_score(f"```python\n{_strip_fence(sol)}\n```",
                                             tests_json, continuous=False)
        except Exception:
            ok = False
        if ok is True or ok == 1.0:
            return idx, True
    return idx, False


def _question(ex):
    for k in ("question", "problem", "prompt", "description"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        # verl-style prompt: [{"role","content"}]
        if isinstance(v, list) and v and isinstance(v[0], dict) and v[0].get("content"):
            return v[0]["content"].strip()
    return None


def to_verl(question, tests, idx, split, orig_source=""):
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": question + "\n\n" + INSTRUCTION}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(tests)},
        "extra_info": {"split": split, "index": idx, "orig_source": orig_source},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="agentica-org/DeepCoder-Preview-Dataset")
    ap.add_argument("--configs", default="taco,primeintellect",
                    help="comma-separated HF config names to pool. DeepCoder-Preview ships "
                         "four (codeforces/lcbv5/primeintellect/taco); the default pools the "
                         "two verified competitive-programming lineages and leaves lcbv5 out "
                         "deliberately -- LiveCodeBench is contamination surface for anyone "
                         "comparing against LCB numbers later")
    ap.add_argument("--splits", default="train", help="comma-separated split names to pool")
    ap.add_argument("--local_save_dir", default="~/data/simopd_code")
    ap.add_argument("--sample", type=int, default=MATH_SET_SIZE)
    ap.add_argument("--val-holdout", type=int, default=200,
                    help="held-out problems for the in-loop val (same distribution)")
    ap.add_argument("--max-tests", type=int, default=6)
    ap.add_argument("--self-verify-procs", type=int, default=48,
                    help="0 disables self-verification (NOT for campaign data)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", action="store_true", help="print the schema and 2 rows, then exit")
    args = ap.parse_args()

    pool = []
    pairs = [(c.strip(), sp.strip()) for c in args.configs.split(",") if c.strip()
             for sp in args.splits.split(",") if sp.strip()]
    for config, split in pairs:
        ds = datasets.load_dataset(args.source, config, split=split)
        print(f"{args.source}[{config}/{split}]: {len(ds)} rows, columns={ds.column_names}")
        if args.inspect:
            for r in list(ds.select(range(2))):
                print({k: (str(v)[:160] + "…" if len(str(v)) > 160 else v) for k, v in r.items()})
            continue
        kept = dropped = fn_dropped = 0
        for ex in ds:
            q, t = _question(ex), _norm_tests(ex)
            if not q or not t:
                dropped += 1
                continue
            if "fn_name" in t:
                # call-convention tests: the harness would invoke fn_name(args),
                # but the prompt instructs a stdin/stdout PROGRAM -- a correct
                # response as instructed can never pass. Structural mismatch, out.
                fn_dropped += 1
                continue
            pool.append((q, t, f"{args.source}:{config}/{split}", _first_solutions(ex)))
            kept += 1
        print(f"  kept {kept}, dropped {dropped} unusable + {fn_dropped} fn_name-style")
    if args.inspect:
        return

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    # de-duplicate by the first 400 chars of the question: the pooled sources
    # overlap (taco appears in several lineages), and a duplicated problem would
    # leak between train and the held-out val
    seen, uniq = set(), []
    for q, t, src, sols in pool:
        key = " ".join(q[:400].split())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((q, t, src, sols))
    print(f"pooled {len(pool)} -> unique {len(uniq)}")

    def cap(tests, n):
        if len(tests["inputs"]) <= n:
            return tests
        idx = rng.sample(range(len(tests["inputs"])), n)
        out = {"inputs": [tests["inputs"][i] for i in idx],
               "outputs": [tests["outputs"][i] for i in idx]}
        if "fn_name" in tests:
            out["fn_name"] = tests["fn_name"]
        return out

    # Cap FIRST, verify against the capped tests: the reward sees the capped set,
    # so passing the full set but failing the cap would still be an unreachable row.
    capped = [(q, cap(t, args.max_tests), src, sols) for q, t, src, sols in uniq]

    if args.self_verify_procs:
        from concurrent.futures import ProcessPoolExecutor
        items = [(i, json.dumps(t), sols) for i, (q, t, src, sols) in enumerate(capped) if sols]
        no_sol = len(capped) - len(items)
        verdict = {}
        with ProcessPoolExecutor(max_workers=args.self_verify_procs) as pool_ex:
            done = 0
            for idx, ok in pool_ex.map(_verify_one, items, chunksize=8):
                verdict[idx] = ok
                done += 1
                if done % 2000 == 0:
                    print(f"  self-verify {done}/{len(items)} "
                          f"(pass rate so far {sum(verdict.values())/len(verdict):.1%})", flush=True)
        survivors = [row for i, row in enumerate(capped) if verdict.get(i)]
        print(f"self-verify: {len(survivors)}/{len(items)} rows pass through our exact "
              f"harness+cap ({no_sol} had no official solution and are dropped)")
    else:
        survivors = capped
        print("self-verify DISABLED -- fine for --inspect, not for campaign data")

    val, train = survivors[: args.val_holdout], survivors[args.val_holdout:]
    if args.sample:
        train = train[: args.sample]

    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)
    tds = datasets.Dataset.from_list(
        [to_verl(q, t, i, "train", s) for i, (q, t, s, _) in enumerate(train)])
    tds.to_parquet(os.path.join(out_dir, "train.parquet"))
    print(f"train: {len(tds)} -> {out_dir}/train.parquet")

    vds = datasets.Dataset.from_list(
        [to_verl(q, t, i, "test", s) for i, (q, t, s, _) in enumerate(val)])
    vds.to_parquet(os.path.join(out_dir, "val_holdout.parquet"))
    print(f"val (held-out): {len(vds)} -> {out_dir}/val_holdout.parquet")


if __name__ == "__main__":
    main()
