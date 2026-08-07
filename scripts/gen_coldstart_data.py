"""Build the off-policy cold-start SFT set for arm a2 (Rethinking's recipe).

Stage 1 of two: sample responses from the teacher on a reserved slice of the
training prompts, keep the ones the verifier accepts, and write them in the
`messages` format verl's SFT trainer reads. Stage 2 (SFT) and stage 3 (OPD from
the resulting checkpoint) are driven by slurm/coldstart.sbatch.

Two choices worth stating, because both are confounds a reviewer will look for:

* The cold-start prompts are *reserved* -- removed from the OPD phase for this
  arm -- so warm-up cannot be credited to simply having seen those questions
  twice. The cost is that this arm's OPD phase sees fewer distinct prompts than
  the others (reported below; at 300x128 we are already multi-epoch, so the
  difference is in epochs seen, not in samples trained on).
* Responses are rejection-sampled on the verifier. Confirmed against the
  official repo (audit r5, 2026-08-08): thunlp/OPD's vllm_rollout.py runs with
  --enable-rejection-sampling true, so filtering is their recipe, not our
  reading. It also matches our G-axis discipline that the verifier filters but
  never enters the training input. --keep-all disables it for the ablation.
"""

import argparse
import json
import os

import pandas as pd

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
MATH_SCORER_SOURCE = "DigitalLearningGmbH/MATH-lighteval"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--train-parquet", default=os.path.expanduser("~/data/simopd_math/train.parquet"))
    p.add_argument("--out-dir", default=os.path.expanduser("~/data/simopd_math"))
    p.add_argument("--n-prompts", type=int, default=3000, help="reserved slice size")
    p.add_argument("--n-samples", type=int, default=4, help="teacher samples per prompt before filtering")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--keep-all", action="store_true", help="skip verifier filtering (ablation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from verl.utils.reward_score import default_compute_score
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.train_parquet)
    reserved = df.iloc[: args.n_prompts].reset_index(drop=True)
    remainder = df.iloc[args.n_prompts :].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    prompts = [
        tok.apply_chat_template(r["prompt"].tolist(), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for _, r in reserved.iterrows()
    ]
    truths = [r["reward_model"]["ground_truth"] for _, r in reserved.iterrows()]

    llm = LLM(model=args.teacher, gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_tokens + 1024, seed=args.seed)
    outputs = llm.generate(
        prompts,
        SamplingParams(n=args.n_samples, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed),
    )

    rows, n_kept, n_total = [], 0, 0
    for (_, row), gt, out in zip(reserved.iterrows(), truths, outputs, strict=True):
        user_msg = row["prompt"].tolist()[0]["content"]
        for comp in out.outputs:
            n_total += 1
            if not args.keep_all:
                score = default_compute_score(MATH_SCORER_SOURCE, comp.text, gt)
                if isinstance(score, dict):
                    score = score.get("score", 0.0)
                if float(score) <= 0.5:
                    continue
            n_kept += 1
            rows.append({"messages": [{"role": "user", "content": user_msg},
                                      {"role": "assistant", "content": comp.text}]})

    os.makedirs(args.out_dir, exist_ok=True)
    sft_path = os.path.join(args.out_dir, "coldstart_sft.parquet")
    opd_path = os.path.join(args.out_dir, "train_coldstart_remainder.parquet")
    pd.DataFrame(rows).to_parquet(sft_path)
    remainder.to_parquet(opd_path)

    meta = {
        "teacher": args.teacher,
        "reserved_prompts": len(reserved),
        "opd_prompts_remaining": len(remainder),
        "samples_per_prompt": args.n_samples,
        "generated": n_total,
        "kept": n_kept,
        "accept_rate": round(n_kept / max(n_total, 1), 4),
        "verifier_filtered": not args.keep_all,
    }
    with open(os.path.join(args.out_dir, "coldstart_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"SFT set -> {sft_path}\nOPD prompts for this arm -> {opd_path}")


if __name__ == "__main__":
    main()
