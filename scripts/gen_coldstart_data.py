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
* Responses are filtered the way the official repo filters them (audit r6): a
  VALIDITY check, not a correctness one -- see --filter. The r5 note below read
  their flag as verifier rejection sampling; their code says otherwise.
* (r5, superseded) Responses are rejection-sampled on the verifier. Confirmed against the
  official repo (audit r5, 2026-08-07): thunlp/OPD's vllm_rollout.py runs with
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
    p.add_argument("--student", default="Qwen/Qwen3-1.7B-Base", help="tokenizer for the SFT-side overlong filter (S9)")
    p.add_argument("--train-parquet", default=os.path.expanduser("~/data/simopd_math/train.parquet"))
    p.add_argument("--out-dir", default=os.path.expanduser("~/data/simopd_math"))
    p.add_argument("--n-prompts", type=int, default=3000, help="reserved slice size")
    p.add_argument("--n-samples", type=int, default=4, help="teacher samples per prompt before filtering")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=16384)  # PROTOCOL 3.8: follow the 16k cap
    p.add_argument("--filter", choices=["validity", "verifier", "none"], default="validity",
                   help="validity = the paper's own is_valid_output (boxed + degeneracy, "
                        "audit r6); verifier = correctness (OUR ablation); none = keep all")
    p.add_argument("--keep-all", action="store_true", help="deprecated alias for --filter none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from collections import Counter

    from verl.utils.reward_score import default_compute_score

    # --- thunlp/OPD scripts/infer/vllm_rollout.py, ported verbatim (audit r6
    # 2026-08-09). Their --enable-rejection-sampling, which r5 read as verifier
    # rejection sampling, is a VALIDITY filter: it rejects generations with no
    # \boxed{} or with degenerate repetition and retries the slot (<=3). It never
    # checks whether the answer is right. Their thresholds are imported literally,
    # the convention this audit already follows for d2's k=5 and h1's K=100.
    def _has_boxed(t):
        return "\\boxed" in t

    def _repeated_lines(t, min_len=20, threshold=5):
        lines = [l.strip() for l in t.split("\n") if len(l.strip()) >= min_len]
        return bool(lines) and Counter(lines).most_common(1)[0][1] >= threshold

    def _ngram_repetition(t, n=100, threshold=3):
        if len(t) < n * threshold:
            return False
        seen = {}
        for i in range(0, len(t) - n + 1, 10):
            c = t[i:i + n]
            seen[c] = seen.get(c, 0) + 1
            if seen[c] >= threshold:
                return True
        return False

    def _consecutive_repeat(t, block=50, threshold=3):
        if len(t) < block * threshold:
            return False
        for i in range(len(t) - block * threshold + 1):
            b = t[i:i + block]
            cnt, pos = 1, i + block
            while pos + block <= len(t) and t[pos:pos + block] == b:
                cnt += 1; pos += block
                if cnt >= threshold:
                    return True
        return False

    def is_valid_output(t):
        if not _has_boxed(t):
            return False, "no_boxed"
        if _repeated_lines(t):
            return False, "repeated_lines"
        if _ngram_repetition(t):
            return False, "ngram_repetition"
        if len(t) > 5000 and _consecutive_repeat(t):
            return False, "consecutive_repeat"
        return True, "ok"
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.train_parquet)
    # Sample, don't slice: the parquet is source-ordered, so iloc[:3000] made the
    # SFT set deepscaler/acereason and the OPD remainder numinamath -- near-disjoint
    # source distributions, a confound far beyond the registered "fewer distinct
    # prompts" (audit 2026-08-07 C5). Exact-duplicate prompts are dropped first so
    # no reserved question reappears verbatim in the remainder (C8: 3 pairs
    # straddled the old boundary).
    _content = df["prompt"].map(lambda q: q[0]["content"] if hasattr(q, "__len__") and len(q) else str(q))
    n_dup = int(_content.duplicated(keep="first").sum())
    df = df.loc[~_content.duplicated(keep="first")].reset_index(drop=True)
    reserved = df.sample(n=args.n_prompts, random_state=args.seed)
    remainder = df.drop(reserved.index).reset_index(drop=True)
    reserved = reserved.reset_index(drop=True)

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
    reject_stats = {}
    for (_, row), gt, out in zip(reserved.iterrows(), truths, outputs, strict=True):
        user_msg = row["prompt"].tolist()[0]["content"]
        for comp in out.outputs:
            n_total += 1
            if comp.finish_reason == "length":
                # An unterminated completion is not a solution; even --keep-all
                # (the no-verifier ablation) must not learn from a cut-off (S8).
                continue
            mode = "none" if args.keep_all else args.filter
            if mode == "validity":
                ok, why = is_valid_output(comp.text)
                if not ok:
                    reject_stats[why] = reject_stats.get(why, 0) + 1
                    continue
            elif mode == "verifier":
                score = default_compute_score(MATH_SCORER_SOURCE, comp.text, gt)
                if isinstance(score, dict):
                    score = score.get("score", 0.0)
                if float(score) <= 0.5:
                    reject_stats["wrong_answer"] = reject_stats.get("wrong_answer", 0) + 1
                    continue
            n_kept += 1
            rows.append({"messages": [{"role": "user", "content": user_msg},
                                      {"role": "assistant", "content": comp.text}]})

    # S9: at the 16k cap a row can exceed the SFT engine's max_length and
    # truncation=error then kills training hours in; filter here, count in meta.
    print(f"filter={('none' if args.keep_all else args.filter)}  rejects={reject_stats}")
    stu_tok = AutoTokenizer.from_pretrained(args.student)
    sft_max = int(os.environ.get("SFT_MAX_LEN", "17408"))
    def _row_len(r):
        return len(stu_tok.apply_chat_template(r["messages"], tokenize=True, enable_thinking=False))
    n_overlong = 0
    if rows:
        keep_rows = []
        for r in rows:
            if _row_len(r) <= sft_max:
                keep_rows.append(r)
            else:
                n_overlong += 1
        rows = keep_rows

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
        # Provenance guard (audit 2026-08-07 S1/C3): an artifact must carry the cap
        # and seed it was born with, or a stale-cap regeneration race is invisible.
        "gen_max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "dedup_dropped": n_dup,
        "overlong_dropped_sft": n_overlong,
        "sft_max_len": sft_max,
    }
    with open(os.path.join(args.out_dir, "coldstart_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"SFT set -> {sft_path}\nOPD prompts for this arm -> {opd_path}")


if __name__ == "__main__":
    main()
