"""Offline benchmark evaluation for SimOPD (METRICS.md §6 item 1).

Serves three pre-registered jobs with one code path:
  - decision metrics on a checkpoint (MATH500 pass@1 greedy, AMC23/AIME avg@32)
  - the diversity side-effect panel (pass@k at tau=1.0 on the frozen 100-problem subset)
  - teacher ceilings / D6 ladder (same command, just point --model at a teacher)

Scoring reuses verl's `default_compute_score` under the MATH data_source, i.e. the
exact scorer the in-training validation uses. METRICS.md forbids a second
implementation here: a scorer mismatch would silently shift pass@1 between the
training curve and the final table, and every verdict is a comparison of the two.

Writes the per-problem parquet that `verdict.py` consumes; the McNemar/Wilcoxon
tests are computed from these rows, never from wandb curves.

Examples
--------
  # student checkpoint, screening decision metrics
  python scripts/eval_offline.py --model /scratch/.../global_step_300/actor \
      --benchmarks math500 --run-id vanilla_s0 --step 300

  # teacher ceiling ladder (D6 input + Gap Recovery Rate denominator)
  python scripts/eval_offline.py --model Qwen/Qwen3-8B --benchmarks math500,amc23 \
      --run-id teacher_ceiling_8b --step -1

  # diversity panel
  python scripts/eval_offline.py --model ... --benchmarks math500_sub100 \
      --n 8 --temperature 1.0 --top-p 1.0 --run-id vanilla_s0 --step 300
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

import datasets
import pandas as pd

# data_source that routes to verl's MATH scorer (boxed extraction + equivalence).
# Our training parquets carry this same string, which is what guarantees parity.
MATH_SCORER_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

# name -> (hf id, split, problem field, answer field)
# AIME24 uses the H4 mirror: math-ai/aime24 ships no answer column (only \boxed
# inside solution), verified 2026-07-31. See BENCHMARKS.md.
BENCHMARKS = {
    "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "answer"),
    "amc23": ("math-ai/amc23", "test", "question", "answer"),
    "aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
    "aime25": ("math-ai/aime25", "test", "problem", "answer"),
    "minerva": ("math-ai/minervamath", "test", "question", "answer"),
}
SUBSET_FILE = "data/math500_subset100.json"


def load_benchmark(name):
    """Return (problems, answers, problem_ids). `math500_sub100` is the frozen
    diversity subset, selected by unique_id so a reordered upstream can't shift it."""
    base = "math500" if name == "math500_sub100" else name
    hf_id, split, q_field, a_field = BENCHMARKS[base]
    ds = datasets.load_dataset(hf_id, split=split)

    if name == "math500_sub100":
        with open(SUBSET_FILE) as f:
            keep = set(json.load(f)["unique_ids"])
        ds = ds.filter(lambda ex: ex["unique_id"] in keep)
        assert len(ds) == 100, f"frozen subset resolved to {len(ds)} rows, expected 100"
        ids = ds["unique_id"]
    else:
        ids = ds["id"] if "id" in ds.column_names else [f"{name}/{i}" for i in range(len(ds))]

    return list(ds[q_field]), [str(a) for a in ds[a_field]], [str(i) for i in ids]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or checkpoint dir")
    p.add_argument("--benchmarks", default="math500", help="comma-separated; see BENCHMARKS")
    p.add_argument("--run-id", required=True, help="arm name, matches the wandb run")
    p.add_argument("--step", type=int, default=-1, help="training step; -1 for teachers/ceilings")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1, help="samples per problem (32 for avg@32, 8 for the panel)")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy pass@1; 0.7 for avg@32; 1.0 for the panel")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192, help="match the run's training cap")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--out-dir", default="/scratch/zz865/simopd/evals")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score import default_compute_score

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_tokens + 1024,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.out_dir, exist_ok=True)

    for bench in args.benchmarks.split(","):
        bench = bench.strip()
        problems, answers, pids = load_benchmark(bench)

        # Same non-thinking template as training; a mismatch here would make the
        # offline number incomparable to the in-training val curve.
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": q.strip() + " " + INSTRUCTION}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for q in problems
        ]

        outputs = llm.generate(prompts, sampling)

        rows = []
        for pid, gt, out in zip(pids, answers, outputs, strict=True):
            for sample_idx, comp in enumerate(out.outputs):
                score = default_compute_score(MATH_SCORER_SOURCE, comp.text, gt)
                if isinstance(score, dict):
                    score = score.get("score", 0.0)
                rows.append(
                    {
                        "run_id": args.run_id,
                        "git_sha": git_sha,
                        "seed": args.seed,
                        "step": args.step,
                        "benchmark": bench,
                        "problem_id": pid,
                        "sample_idx": sample_idx,
                        "correct": int(float(score) > 0.5),
                        "resp_len": len(comp.token_ids),
                        "truncated": int(comp.finish_reason == "length"),
                        "finish_reason": str(comp.finish_reason),
                        "temperature": args.temperature,
                        "n": args.n,
                    }
                )

        df = pd.DataFrame(rows)
        path = os.path.join(args.out_dir, f"{args.run_id}__{bench}__step{args.step}__seed{args.seed}__{stamp}.parquet")
        df.to_parquet(path)

        per_problem = df.groupby("problem_id")["correct"].mean()
        metric = "pass@1" if args.n == 1 else f"avg@{args.n}"
        print(
            f"[{bench}] {metric}={per_problem.mean():.4f}  "
            f"pass@{args.n}={df.groupby('problem_id')['correct'].max().mean():.4f}  "
            f"trunc={df['truncated'].mean():.4f}  len_mean={df['resp_len'].mean():.0f}  -> {path}"
        )


if __name__ == "__main__":
    main()
