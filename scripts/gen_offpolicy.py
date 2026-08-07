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
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.student)
    df = pd.read_parquet(a.train_parquet)
    if a.limit:
        df = df.iloc[: a.limit]
    contents = [(q[0]["content"] if hasattr(q, "__len__") and len(q) else str(q))
                for q in df["prompt"]]
    prefixes = [_template_ids(tok, [{"role": "user", "content": c}], enable_thinking=False)
                for c in contents]
    keys = [prefix_hash(ids) for ids in prefixes]
    assert len(set(keys)) == len(keys), "duplicate prefix hashes; widen the key"

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
                        "response_tokens": [len(r) for r in responses]})
    out.to_parquet(a.out)
    print(f"{len(out)} rows -> {a.out}   resp len p50 "
          f"{out['response_tokens'].median():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
