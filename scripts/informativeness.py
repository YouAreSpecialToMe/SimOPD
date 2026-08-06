"""Can the teacher's likelihood tell a right answer from a wrong one?

    python scripts/informativeness.py --model <ckpt>/actor --run-id vanilla_s0 --step 250
    python scripts/informativeness.py --model Qwen/Qwen3-1.7B-Base --run-id student_base --step 0

Two METRICS §3 entries that are the same measurement read two ways:

  Informativeness I  (Demystifying Eq.4)  the teacher's probability on tokens of
      CORRECT rollouts minus the same on INCORRECT ones. If it is near zero the
      teacher is not distinguishing them, and every token-level trick downstream is
      reweighting a signal that carries no information about correctness.
  teacher-likelihood AUROC (Rethinking)   the same quantity as a discriminator.
      They report 0.73-0.75, which makes it one of the few numbers in this literature
      we can compare against directly rather than by narrative.

FiRe's s(y) trajectory filter is this quantity used as a gate, so g2_fire_likelihood's
premise is exactly what this measures. A low AUROC would mean that arm is filtering
on noise -- which is a verdict about it, obtained without training it.

Offline on purpose. The natural place is a training-time hook, but that only serves
runs started after it lands, and four are already going. This reads a checkpoint, so
it applies to every run including those.

Two model loads, sequentially on one GPU: the student generates, then the teacher
scores those exact tokens. vLLM does not free a model cleanly enough to trust in one
process, so each phase is its own subprocess and they meet at a parquet.
"""

import argparse
import json
import os
import subprocess
import sys

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney U), so scipy/sklearn are not dependencies.

    Ties get averaged ranks, which matters here: truncated rollouts can share
    identical sequence-mean logprobs and a naive implementation would count each tie
    as half a win in one direction only.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def phase_generate(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score import default_compute_score

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_offline import MATH_SCORER_SOURCE, load_benchmark, resolve_model

    model = resolve_model(a.model)
    tok = AutoTokenizer.from_pretrained(model)
    problems, answers, ids = load_benchmark(a.bench)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q.strip() + " " + INSTRUCTION}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for q in problems
    ]
    llm = LLM(model=model, gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=a.max_tokens + 1024, seed=a.seed)
    # tau=1.0 on purpose: this asks what the teacher thinks of the rollouts the
    # POLICY produces, which is what OPD actually distils. Greedy would sample one
    # mode and tell us nothing about the spread the signal has to separate.
    out = llm.generate(prompts, SamplingParams(n=a.n, temperature=1.0, top_p=1.0,
                                               max_tokens=a.max_tokens, seed=a.seed))
    rows = []
    for pid, gt, o in zip(ids, answers, out, strict=True):
        for k, c in enumerate(o.outputs):
            score = default_compute_score(MATH_SCORER_SOURCE, c.text, gt)
            if isinstance(score, dict):
                score = score.get("score", 0.0)
            rows.append({"problem_id": pid, "sample_idx": k,
                         "correct": int(float(score) > 0.5),
                         "prompt": prompts[ids.index(pid)], "text": c.text,
                         "n_tokens": len(c.token_ids),
                         "truncated": int(c.finish_reason == "length")})
    import pandas as pd
    pd.DataFrame(rows).to_parquet(a.rollouts)
    print(f"[generate] {len(rows)} rollouts, acc={sum(r['correct'] for r in rows)/len(rows):.4f}"
          f" -> {a.rollouts}")


def phase_score(a):
    from vllm import LLM, SamplingParams
    import pandas as pd

    df = pd.read_parquet(a.rollouts)
    llm = LLM(model=a.teacher, gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=a.max_tokens + 1024, seed=a.seed)
    # prompt_logprobs=0 returns the logprob of each ACTUAL token, which is what the
    # teacher assigns to what the student wrote -- the same quantity OPD distils on.
    texts = [p + t for p, t in zip(df["prompt"], df["text"])]
    outs = llm.generate(texts, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0))
    means, sums = [], []
    for row, o in zip(df.itertuples(), outs, strict=True):
        lps = [next(iter(d.values())).logprob for d in (o.prompt_logprobs or []) if d]
        # Only the response half: the prompt is identical across samples of a problem
        # and would dilute the very difference being measured.
        resp = lps[-int(row.n_tokens):] if row.n_tokens and len(lps) >= row.n_tokens else lps
        means.append(sum(resp) / len(resp) if resp else float("nan"))
        sums.append(sum(resp) if resp else float("nan"))
    df["teacher_logprob_mean"] = means
    df["teacher_logprob_sum"] = sums
    df.drop(columns=["prompt", "text"]).to_parquet(a.out)
    print(f"[score] teacher logprobs attached -> {a.out}")


def phase_report(a):
    import pandas as pd

    df = pd.read_parquet(a.out).dropna(subset=["teacher_logprob_mean"])
    ok = df[df.correct == 1]["teacher_logprob_mean"]
    bad = df[df.correct == 0]["teacher_logprob_mean"]
    info = float(ok.mean() - bad.mean()) if len(ok) and len(bad) else float("nan")
    au = auroc(df["teacher_logprob_mean"].tolist(), df["correct"].tolist())
    print(f"\n=== {a.run_id} @ step {a.step}, {a.bench}, teacher {a.teacher} ===")
    print(f"  rollouts            {len(df)}  ({len(ok)} correct, {len(bad)} incorrect, "
          f"{df['truncated'].mean():.1%} truncated)")
    print(f"  Informativeness I   {info:+.4f}   (mean teacher logprob, correct - incorrect)")
    print(f"  teacher AUROC       {au:.4f}     (Rethinking report 0.73-0.75)")
    if len(ok) and len(bad):
        print(f"    correct   {ok.mean():.4f}")
        print(f"    incorrect {bad.mean():.4f}")
    verdict = ("the teacher's likelihood carries real information about correctness"
               if au >= 0.65 else
               "near chance -- a trajectory filter on this signal would be filtering noise, "
               "which is a verdict about g2_fire_likelihood obtained without training it")
    print(f"  -> {verdict}")
    rec = {"run_id": a.run_id, "step": a.step, "bench": a.bench, "teacher": a.teacher,
           "n": len(df), "informativeness": info, "auroc": au}
    with open(a.ledger, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  appended to {a.ledger}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="student checkpoint or hub id")
    p.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, default=-1)
    p.add_argument("--bench", default="math500_sub100",
                   help="the frozen 100-problem subset by default: this needs n samples "
                        "per problem and the full 500 buys precision nobody reads")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--work", default=os.environ.get("SIMOPD_EVAL_ROOT", "/home/zz865/data/simopd_evals"))
    p.add_argument("--phase", choices=["generate", "score", "report", "all"], default="all")
    a = p.parse_args()

    tag = f"{a.run_id}__step{a.step}__{a.bench}"
    a.rollouts = os.path.join(a.work, f"_info_rollouts__{tag}.parquet")
    a.out = os.path.join(a.work, f"informativeness__{tag}.parquet")
    a.ledger = os.path.join(a.work, "informativeness.jsonl")
    os.makedirs(a.work, exist_ok=True)

    if a.phase in ("generate", "score", "report"):
        return globals()[f"phase_{a.phase}"](a)

    # Each model in its own process. vLLM leaves enough behind that loading a second
    # model in the same interpreter is a coin flip, and a coin flip inside a
    # diagnostic is worse than a slower diagnostic.
    for ph in ("generate", "score"):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), *sys.argv[1:],
                            "--phase", ph])
        if r.returncode:
            raise SystemExit(f"phase {ph} failed")
    phase_report(a)


if __name__ == "__main__":
    sys.exit(main())
