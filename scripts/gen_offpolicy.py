"""Precompute the teacher's off-policy responses for a1_gkd_mix0.5.

    python scripts/gen_offpolicy.py --dry --limit 8      # CPU: keys + prompt construction
    python scripts/gen_offpolicy.py                      # GPU, once, cached forever

One response per training prompt from the OPD teacher (4B-Instruct-2507), generated
IN THE STUDENT'S TEMPLATE at the protocol's rollout parameters (tau=1.0, top-p=1.0,
seeded) -- these sequences stand in for student rollouts, so they must come from the
distribution the arm claims to mix in: teacher samples under the training regime, not
greedy showpieces. Keys are the same first-16-token prefix hashes gkd_mix looks up,
built by the same verl-native id path as gen_priv_cot (imported, not re-derived).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_priv_cot import _template_ids, prefix_hash  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--student", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--train-parquet", default=os.path.expanduser("~/data/simopd_math/train.parquet"))
    p.add_argument("--out", default=os.path.expanduser("~/data/simopd_math/gkd_offpolicy.parquet"))
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="PROTOCOL 3.8: cache stands in for student rollouts, so it matches the 16k cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--dry", action="store_true")
    # Data-parallel sharding (2026-08-15, for the 8-GPU box): SamplingParams carries a
    # PER-REQUEST seed, so each prompt's sample depends on (seed, prompt) and not on
    # batch composition -- N shards produce token-identical rows to one single-GPU
    # pass, just N times sooner. Shards slice AFTER the dedupe below, so they are
    # disjoint by construction; --merge concatenates and re-checks key uniqueness.
    p.add_argument("--shard", default=None, help="i/n: generate rows i::n into <out>.shard<i>of<n>")
    p.add_argument("--merge", type=int, default=None,
                   help="merge <out>.shard*of<N> into <out> and verify, no GPU")
    a = p.parse_args()

    if a.merge:
        import pandas as pd

        parts = [pd.read_parquet(f"{a.out}.shard{i}of{a.merge}") for i in range(a.merge)]
        out = pd.concat(parts, ignore_index=True)
        assert out["prefix_hash"].is_unique, "merged shards repeat a key; a shard ran twice?"
        caps = set(out["gen_max_tokens"]), set(out["temperature"]), set(out["teacher"])
        assert all(len(c) == 1 for c in caps), f"shards disagree on provenance: {caps}"
        out.to_parquet(a.out)
        print(f"merged {a.merge} shards -> {a.out}   {len(out)} rows   "
              f"resp len p50 {out['response_tokens'].median():.0f}")
        return 0

    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.student)
    tok_t = AutoTokenizer.from_pretrained(a.teacher)
    assert tok.vocab_size == tok_t.vocab_size, (
        "shared vocab is the protocol precondition (S6): teacher generates ids the "
        "student engine must score")
    df = pd.read_parquet(a.train_parquet)
    if a.limit:
        df = df.iloc[: a.limit]
    contents = [(q[0]["content"] if hasattr(q, "__len__") and len(q) else str(q))
                for q in df["prompt"]]
    prefixes = [_template_ids(tok, [{"role": "user", "content": c}], enable_thinking=False)
                for c in contents]
    keys = [prefix_hash(ids) for ids in prefixes]
    # The train parquet carries a few genuinely duplicated prompts (measured
    # 2026-08-15: 9 identical-content pairs in 14,476 rows). Full-prefix keys make
    # key equality mean prompt equality, so a duplicate SHARES its first
    # occurrence's cache row -- the correct GKD semantics (same X -> same fixed
    # (X,Y)). Verify that equality before collapsing: a same-key pair with
    # DIFFERENT ids would be a real 64-bit collision and still dies loudly
    # (audit C1's original point stands for cross-prompt collisions).
    first_row = {}
    keep = []
    for i, k in enumerate(keys):
        j = first_row.setdefault(k, i)
        if j != i:
            assert prefixes[j] == prefixes[i], (
                f"prefix_hash collision between DIFFERENT prompts (rows {j} vs {i}); widen the key")
            continue
        keep.append(i)
    if len(keep) < len(keys):
        print(f"{len(keys) - len(keep)} duplicate prompts share their first occurrence's cache row")
        prefixes = [prefixes[i] for i in keep]
        keys = [keys[i] for i in keep]

    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        assert 0 <= i < n, f"--shard {a.shard}: want i/n with 0 <= i < n"
        prefixes, keys = prefixes[i::n], keys[i::n]
        a.out = f"{a.out}.shard{i}of{n}"
        print(f"shard {i}/{n}: {len(keys)} prompts -> {a.out}")

    if a.dry:
        responses = [[1, 2, 3]] * len(prefixes)
    else:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        llm = LLM(model=a.teacher, gpu_memory_utilization=a.gpu_mem_util,
                  max_model_len=a.max_tokens + 1536, seed=a.seed)
        outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prefixes],
                            SamplingParams(temperature=1.0, top_p=1.0,
                                           max_tokens=a.max_tokens, seed=a.seed))
        responses = [list(o.outputs[0].token_ids) for o in outs]

    out = pd.DataFrame({"prefix_hash": keys, "response_ids": responses,
                        "response_tokens": [len(r) for r in responses],
                        # provenance columns (audit S1): consumers can refuse a
                        # stale-cap or wrong-teacher cache instead of degrading.
                        "gen_max_tokens": a.max_tokens, "temperature": 1.0,
                        "seed": a.seed, "teacher": a.teacher})
    if a.dry:
        # A rehearsal must never clobber the production cache with stub rows (C6):
        # gkd_mix would inject [1,2,3] as "teacher rollouts" with only the mix-rate
        # print to notice.
        a.out = a.out + ".dry"
    out.to_parquet(a.out)
    print(f"{len(out)} rows -> {a.out}   resp len p50 "
          f"{out['response_tokens'].median():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
