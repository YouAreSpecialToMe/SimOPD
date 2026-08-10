"""Dump full response TEXT for a chosen handful of MATH500 problems.

The campaign's eval artifacts keep only lengths and correctness, which is enough
to show *that* the late-training runaway happens but not *what* the model is
writing when it happens. This is a text-preserving sibling of
scripts/eval_offline.py: identical model resolution, identical chat template,
identical greedy sampling and identical scorer, restricted to a problem list and
writing the completion string alongside every row.

Read-only with respect to the repo -- it imports scripts/eval_offline.py rather
than copying its logic, so the prompt can never drift from the campaign's.

  python3 eval_offline_textdump.py --model <path|hf-id> --run-id <id> \
      --problems pids.txt --out-dir <dir> [--max-tokens 16384]

The name deliberately contains "eval_offline": the refill daemon's busy-detection
matches on that substring, so a card running this is never double-booked.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, "/mgfs/shared/Group_GY/changhao/SimOPD/scripts")
import eval_offline as EO  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--problems", required=True, help="file with one problem_id per line")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--benchmark", default="math500")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    want = [ln.strip() for ln in open(args.problems) if ln.strip()]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score import default_compute_score

    model = EO.resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model)

    problems, answers, pids = EO.load_benchmark(args.benchmark)
    keep = [i for i, pid in enumerate(pids) if pid in set(want)]
    missing = set(want) - {pids[i] for i in keep}
    if missing:
        print(f"[textdump] WARNING {len(missing)} requested ids not in {args.benchmark}: "
              f"{sorted(missing)[:5]}", file=sys.stderr)
    print(f"[textdump] {len(keep)}/{len(want)} problems, model={model}", flush=True)

    # Byte-identical to the campaign's prompt: same INSTRUCTION, same non-thinking
    # template. A mismatch here would make these texts incomparable to the curves.
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": problems[i].strip() + " " + EO.INSTRUCTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for i in keep
    ]

    llm = LLM(model=model, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=min(args.max_tokens + 1024, EO._derived_max_len(model)),
              seed=args.seed)
    outputs = llm.generate(prompts, SamplingParams(
        n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed))

    rows = []
    for i, out in zip(keep, outputs, strict=True):
        c = out.outputs[0]
        score = default_compute_score(EO.MATH_SCORER_SOURCE, c.text, answers[i])
        if isinstance(score, dict):
            score = score.get("score", 0.0)
        rows.append(dict(run_id=args.run_id, problem_id=pids[i], answer=str(answers[i]),
                         problem=problems[i], response=c.text,
                         resp_len=len(c.token_ids), truncated=int(c.finish_reason == "length"),
                         finish_reason=str(c.finish_reason), correct=int(float(score) > 0.5),
                         temperature=0.0, max_tokens=args.max_tokens))
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(args.out_dir, f"{args.run_id}__textdump__{stamp}.parquet")
    pd.DataFrame(rows).to_parquet(path)
    print(f"[textdump] wrote {path}  rows={len(rows)} "
          f"trunc={sum(r['truncated'] for r in rows)} "
          f"correct={sum(r['correct'] for r in rows)}", flush=True)
    print(json.dumps({"path": path, "n": len(rows)}))


if __name__ == "__main__":
    main()
