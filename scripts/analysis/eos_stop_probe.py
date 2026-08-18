"""Run (cluster venv, no GPU needed):
    PROBE_THREADS=64 python scripts/analysis/eos_stop_probe.py   # hop pod, CPU, ~10 min; receipt docs/data/eos_stop_probe.txt

EOS-token probe (CPU only): which token actually ends student rollouts, what the
teacher's stop mass looks like at the end of a student answer, and how top-k arms
end their responses in practice.

Part 1: population stats from eval parquets that carry the response text.
Part 2: HF forward passes on real (prompt, response) pairs for teacher / base /
        trained checkpoints, reporting p(<|im_end|>), p(<|endoftext|>) at the end
        of the answer and along the <|im_end|> -> \\n -> ? path, plus a short greedy
        continuation showing the actual stop path each model takes.
"""
import glob
import json
import os
import re
import sys
import time
from collections import Counter

D = "/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "64")))

STU = "Qwen/Qwen3-1.7B-Base"
TCH = "Qwen/Qwen3-4B-Instruct-2507"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
IM_START, IM_END, EOT = 151644, 151645, 151643

tok = AutoTokenizer.from_pretrained(STU)
NL = tok.encode("\n", add_special_tokens=False)[0]
print("student tokenizer eos:", tok.eos_token, tok.eos_token_id, "| pad:", tok.pad_token, tok.pad_token_id)
print("ids:", {"<|im_start|>": IM_START, "<|im_end|>": IM_END, "<|endoftext|>": EOT, "\\n": NL})
for t in (IM_START, IM_END, EOT):
    print("  decode", t, "->", repr(tok.decode([t])), "| special:", t in tok.all_special_ids)


def newest_with_text(run_id, bench, step):
    hits = sorted(glob.glob(f"{D}/evals/{run_id}__{bench}__step{step}__seed*.parquet"))
    for h in reversed(hits):
        try:
            df = pd.read_parquet(h)
        except Exception:
            continue
        if "response" in df.columns:
            return h, df
    return None, None


# ------------------------------------------------------------------ Part 1
print("\n=================== Part 1: how do rollouts actually end (eval parquets with text)")
CELLS = [
    ("vanilla_s0_16k", "math500", 125),
    ("vanilla_s0_16k", "math500", 150),
    ("vanilla_s0_16k", "math500", 225),
    ("c4_pi_tail_budget_s0_16k", "math500", 100),
    ("c4_pi_tail_budget_s0_16k", "math500", 250),
    ("c2_quantile_budget_s0_16k", "math500", 250),
    ("c1_direct_s0_16k", "math500", 250),
    ("h2_horizon_s0_16k", "math500", 250),
    ("d2_ent_gate_s0_16k", "math500", 250),
    ("vanilla_s0_16k", "aime24", 125),
    ("c4_pi_tail_budget_s0_16k", "aime24", 250),
]
FAKE = re.compile(r"\n(user|assistant|system)\n")
picked = []  # (run, step, problem_id, response) for part 2
for run, bench, step in CELLS:
    f, df = newest_with_text(run, bench, step)
    if df is None:
        print(f"[skip] {run} {bench} step{step}: no artifact with response column")
        continue
    resp = df["response"].fillna("").tolist()
    n_text = [len(tok.encode(r, add_special_tokens=False)) for r in resp]
    df["n_text"] = n_text
    df["delta"] = df["resp_len"] - df["n_text"]
    df["ends_nl"] = df["response"].fillna("").str.endswith("\n")
    df["fake_turn"] = df["response"].fillna("").apply(lambda s: bool(FAKE.search(s)))
    stop = df[df["finish_reason"] == "stop"]
    trunc = df[df["finish_reason"] == "length"]
    print(f"\n--- {run} {bench} step{step}  [{os.path.basename(f)}]  n={len(df)}")
    print("    finish_reason:", dict(df["finish_reason"].value_counts()),
          "| mean resp_len:", round(float(df["resp_len"].mean()), 1),
          "| acc:", round(float(df["correct"].mean()), 3))
    if len(stop):
        hist = Counter(int(min(max(d, -1), 6)) for d in stop["delta"])
        print("    STOPPED: n=%d  delta=resp_len-n_text_tokens histogram (clipped -1..6): %s"
              % (len(stop), dict(sorted(hist.items()))))
        print("             ends_with_\\n: %.3f | fake_turn markers: %.3f | mean len %.0f | acc %.3f"
              % (stop["ends_nl"].mean(), stop["fake_turn"].mean(), stop["resp_len"].mean(), stop["correct"].mean()))
        tails = Counter(repr(r[-12:]) for r in stop["response"])
        print("             most common 12-char tails:", tails.most_common(6))
        # example ending
        ex = stop.iloc[0]["response"]
        print("             example tail:", repr(ex[-160:]))
    if len(trunc):
        hist = Counter(int(min(max(d, -1), 6)) for d in trunc["delta"])
        print("    TRUNCATED: n=%d  delta hist: %s | fake_turn markers: %.3f"
              % (len(trunc), dict(sorted(hist.items())), trunc["fake_turn"].mean()))
        ex = trunc.iloc[0]["response"]
        print("             example tail:", repr(ex[-160:]))
    # collect a few real stopped+correct medium-length examples for part 2
    if bench == "math500" and step in (125, 250, 100) and len(stop):
        cand = stop[(stop["correct"] == 1) & (stop["resp_len"] > 150) & (stop["resp_len"] < 900)]
        for _, r in cand.head(2).iterrows():
            picked.append((run, step, r["problem_id"], r["response"], int(r["resp_len"]), int(r["delta"])))

# ------------------------------------------------------------------ Part 2
print("\n=================== Part 2: model probes on real (prompt, response) pairs (CPU forward)")
from datasets import load_dataset  # noqa: E402

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
prob_by_id = {ex["unique_id"]: ex["problem"] for ex in ds}


def build_ids(problem, response):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": problem.strip() + " " + INSTRUCTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok.encode(prompt, add_special_tokens=False)
    r_ids = tok.encode(response, add_special_tokens=False)
    return prompt, p_ids, r_ids


examples = []
seen = set()
for run, step, pid, response, resp_len, delta in picked:
    if pid in seen or pid not in prob_by_id:
        continue
    seen.add(pid)
    prompt, p_ids, r_ids = build_ids(prob_by_id[pid], response)
    examples.append(dict(run=run, step=step, pid=pid, prompt=prompt, p_ids=p_ids, r_ids=r_ids,
                         response=response, resp_len=resp_len, delta=delta))
    if len(examples) >= 4:
        break
print(f"{len(examples)} examples; prompt tail (as tokenized for training/eval):",
      repr(examples[0]["prompt"][-80:]) if examples else None)
for e in examples:
    print(f"  ex: {e['run']}@{e['step']} {e['pid']} resp_len={e['resp_len']} n_text={len(e['r_ids'])} delta={e['delta']} "
          f"tail={e['response'][-60:]!r}")

MODELS = [
    ("teacher  Qwen3-4B-Instruct-2507", TCH),
    ("student0 Qwen3-1.7B-Base", STU),
    ("vanilla_s0@250", f"{D}/ckpt/simopd/vanilla_s0_16k/global_step_250/actor/huggingface"),
    ("c4_pi_tail_budget_s0@250", f"{D}/ckpt/simopd/c4_pi_tail_budget_s0_16k/global_step_250/actor/huggingface"),
    ("c2_quantile_budget_s0@250", f"{D}/ckpt/simopd/c2_quantile_budget_s0_16k/global_step_250/actor/huggingface"),
]


def show(name, logits_row, k=6):
    lp = torch.log_softmax(logits_row.float(), -1)
    p = lp.exp()
    top = torch.topk(p, k)
    tops = ", ".join(f"{tok.decode([int(i)])!r}:{float(v):.3g}" for v, i in zip(top.values, top.indices))
    print(f"      {name:<28} p(im_end)={float(p[IM_END]):.3g}  p(eot)={float(p[EOT]):.3g}  "
          f"p(\\n)={float(p[NL]):.3g}  p(im_start)={float(p[IM_START]):.3g} | top: {tops}")
    return p


@torch.no_grad()
def probe(model, e):
    ids = e["p_ids"] + e["r_ids"]
    x = torch.tensor([ids])
    out = model(x, use_cache=True)
    last = out.logits[0, -1]
    print("    position: END OF ANSWER (after last text token)")
    p_end = show("", last)
    # path A: append <|im_end|>
    pkv = out.past_key_values
    o2 = model(torch.tensor([[IM_END]]), past_key_values=pkv, use_cache=True)
    print("    position: after <|im_end|>")
    show("", o2.logits[0, -1])
    o3 = model(torch.tensor([[NL]]), past_key_values=o2.past_key_values, use_cache=True)
    print("    position: after <|im_end|>\\n")
    show("", o3.logits[0, -1])
    # greedy continuation from END OF ANSWER, 6 tokens, no eos stopping
    cur = torch.tensor([ids])
    gen = []
    o = model(cur, use_cache=True)
    pk = o.past_key_values
    nxt = int(torch.argmax(o.logits[0, -1]))
    gen.append(nxt)
    for _ in range(5):
        o = model(torch.tensor([[nxt]]), past_key_values=pk, use_cache=True)
        pk = o.past_key_values
        nxt = int(torch.argmax(o.logits[0, -1]))
        gen.append(nxt)
    print("    greedy continuation from END OF ANSWER:", [tok.decode([g]) for g in gen], gen)


for name, path in MODELS:
    if not os.path.isdir(path) and "/" in path and path.startswith("/"):
        print(f"\n[skip] {name}: {path} missing")
        continue
    t0 = time.time()
    try:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    except Exception as ex:  # noqa: BLE001
        print(f"\n[skip] {name}: load failed: {ex}")
        continue
    m.eval()
    gc_path = os.path.join(path, "generation_config.json") if os.path.isdir(path) else None
    gc = getattr(m, "generation_config", None)
    print(f"\n##### {name}  (loaded {time.time()-t0:.0f}s)  generation_config eos={getattr(gc, 'eos_token_id', None)}")
    for e in examples[:3]:
        print(f"  -- example {e['pid']} ({e['run']}@{e['step']}, delta={e['delta']}) tail={e['response'][-40:]!r}")
        probe(m, e)
    del m
print("\nDONE")
