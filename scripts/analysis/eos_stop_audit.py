"""Run (cluster venv, no GPU needed):
    N_PER=10 PROBE_THREADS=96 python scripts/analysis/eos_stop_audit.py   # hop pod, CPU, ~2 min; receipt docs/data/eos_stop_audit.txt

Stop-position audit (CPU): on real stopped student responses, what does the teacher
say at the position where the student emitted <|endoftext|>, and how big is the
sampled-column k1 reward there relative to the rest of the response?

For each response: teacher q(eot), q(im_end), rank of eot in the teacher's vocab, and
student p(eot), p(im_end) for base / vanilla@250 / c4@250; the per-token reward
r_t = log q(y_t) - log p_base(y_t) along the response, its terminal value r_T and where
r_T sits in the response's own distribution of r_t.
"""
import glob
import os
import time

D = "/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "96")))
STU = "Qwen/Qwen3-1.7B-Base"
TCH = "Qwen/Qwen3-4B-Instruct-2507"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
IM_END, EOT = 151645, 151643
N_PER = int(os.environ.get("N_PER", "10"))

tok = AutoTokenizer.from_pretrained(STU)
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
prob_by_id = {ex["unique_id"]: ex["problem"] for ex in ds}


def newest_with_text(run_id, bench, step):
    for h in sorted(glob.glob(f"{D}/evals/{run_id}__{bench}__step{step}__seed*.parquet"))[::-1]:
        df = pd.read_parquet(h)
        if "response" in df.columns:
            return df
    return None


rows = []
for run, step in [("vanilla_s0_16k", 125), ("c4_pi_tail_budget_s0_16k", 100), ("vanilla_s0_16k", 225)]:
    df = newest_with_text(run, "math500", step)
    if df is None:
        continue
    stop = df[(df["finish_reason"] == "stop") & (df["resp_len"] > 120) & (df["resp_len"] < 1600)]
    stop = stop.sample(n=min(N_PER, len(stop)), random_state=0)
    for _, r in stop.iterrows():
        rows.append((run, step, r["problem_id"], r["response"], int(r["correct"]), int(r["resp_len"])))
print(f"{len(rows)} stopped responses sampled")

examples = []
for run, step, pid, response, correct, resp_len in rows:
    prompt = tok.apply_chat_template([{"role": "user", "content": prob_by_id[pid].strip() + " " + INSTRUCTION}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok.encode(prompt, add_special_tokens=False)
    r_ids = tok.encode(response, add_special_tokens=False)
    examples.append(dict(run=run, step=step, pid=pid, correct=correct, p_ids=p_ids, r_ids=r_ids + [EOT]))

MODELS = [
    ("teacher", TCH),
    ("base", STU),
    ("vanilla@250", f"{D}/ckpt/simopd/vanilla_s0_16k/global_step_250/actor/huggingface"),
    ("c4@250", f"{D}/ckpt/simopd/c4_pi_tail_budget_s0_16k/global_step_250/actor/huggingface"),
]


@torch.no_grad()
def score(model, e):
    """log-probs of the actual response tokens (incl. terminal eot) + full dist at terminal pos."""
    ids = e["p_ids"] + e["r_ids"]
    x = torch.tensor([ids])
    logits = model(x).logits[0].float()
    P = len(e["p_ids"])
    # position t predicts token t+1; response tokens are ids[P:], predicted by logits[P-1 : -1]
    lp = torch.log_softmax(logits[P - 1:-1], -1)
    tgt = torch.tensor(ids[P:])
    tok_lp = lp.gather(1, tgt[:, None]).squeeze(1)              # [R] incl. terminal eot
    term = lp[-1]                                                # dist at the position predicting eot
    rank_eot = int((term > term[EOT]).sum())                     # 0 = top-1
    return tok_lp.numpy(), float(term[EOT].exp()), float(term[IM_END].exp()), rank_eot, int(torch.argmax(term))


res = {}
for name, path in MODELS:
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    out = []
    for e in examples:
        out.append(score(m, e))
    res[name] = out
    print(f"[{name}] scored {len(out)} responses in {time.time()-t0:.0f}s")
    del m

print("\n=== per-response table: terminal position (where the student emitted <|endoftext|>)")
print("run@step        corr len   q(eot)    q(im_end) rank_eot(T) top1_T      | p_base(eot) p_van(eot) p_c4(eot) | p_base(im_end) p_van(im_end) p_c4(im_end)")
agg = {k: [] for k in ["q_eot", "q_im", "rank", "pb", "pv", "pc", "rT", "pct", "med", "p01", "min_nonterm", "frac_lt10"]}
for i, e in enumerate(examples):
    q_lp, q_eot, q_im, rank, top1 = res["teacher"][i]
    b_lp, pb_eot, pb_im, _, _ = res["base"][i]
    v_lp, pv_eot, pv_im, _, _ = res["vanilla@250"][i]
    c_lp, pc_eot, pc_im, _, _ = res["c4@250"][i]
    r = q_lp - b_lp                       # k1 reward under (teacher, base) at every response token
    rT = float(r[-1]); body = r[:-1]
    pct = float((body < rT).mean())       # fraction of body tokens with an even lower reward
    print(f"{e['run'][:7]}@{e['step']:<4} {e['correct']}   {len(e['r_ids']):<5} {q_eot:.1e}  {q_im:.3f}    {rank:<8} {tok.decode([top1])!r:<10} "
          f"| {pb_eot:.3f}      {pv_eot:.1e}   {pc_eot:.3f}    | {pb_im:.0e}         {pv_im:.0e}         {pc_im:.0e}"
          f"   || r_T={rT:.1f}  body: med={np.median(body):.2f} p1={np.percentile(body,1):.1f} min={body.min():.1f} frac(r<-10)={(body<-10).mean():.4f} pct(body<r_T)={pct:.4f}")
    for k, v in zip(agg.keys(), [q_eot, q_im, rank, pb_eot, pv_eot, pc_eot, rT, pct, np.median(body), np.percentile(body, 1), body.min(), (body < -10).mean()]):
        agg[k].append(v)
print("\n=== aggregate over %d responses" % len(examples))
for k, v in agg.items():
    v = np.array(v, dtype=float)
    print(f"  {k:<12} median={np.median(v):.4g}  mean={v.mean():.4g}  min={v.min():.4g}  max={v.max():.4g}")
print("DONE")
