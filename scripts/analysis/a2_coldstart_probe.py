"""Run (cluster venv, CPU): python scripts/analysis/a2_coldstart_probe.py   # receipt docs/data/a2_coldstart_probe.txt

Why a2 (cold-start SFT on teacher CoT, then OPD) is termination-broken from step 1:
what token does the SFT target end with, what does the SFT model emit at the end of an
answer, and what do a2's eval responses contain (CPU, cluster venv)."""
import os, glob, re
from collections import Counter
D="/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import pandas as pd, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(96)
IM_START, IM_END, EOT = 151644, 151645, 151643
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
NL=tok.encode("\n",add_special_tokens=False)[0]

print("=== 1. the SFT data (coldstart_sft.parquet)")
df=pd.read_parquet(f"{D}/simopd_math/coldstart_sft.parquet")
print("rows:", len(df), "cols:", df.columns.tolist())
msgs0=list(df.iloc[0]["messages"]); print("roles:", [m["role"] for m in msgs0])
contents=[list(m)[-1]["content"] for m in df["messages"]]
print("assistant content tails (3):", [repr(c[-60:]) for c in contents[:3]])
print("contents containing literal '<|im_end|>':", sum('<|im_end|>' in c for c in contents), "| '<|endoftext|>':", sum('<|endoftext|>' in c for c in contents), "| '<think>':", sum('<think>' in c for c in contents))
full=tok.apply_chat_template([dict(m) for m in msgs0], tokenize=True, add_generation_prompt=False, enable_thinking=False)
full=list(full["input_ids"]) if not isinstance(full,list) else full
prefix=tok.apply_chat_template([dict(m) for m in msgs0[:-1]], tokenize=True, add_generation_prompt=True, enable_thinking=False)
prefix=list(prefix["input_ids"]) if not isinstance(prefix,list) else prefix
print("full render last 6 ids:", full[-6:], [tok.decode([i]) for i in full[-6:]])
print("prefix (generation prompt) last 6 ids:", prefix[-6:], [tok.decode([i]) for i in prefix[-6:]])
print("supervised span = full[len(prefix):] -> ends with:", [tok.decode([i]) for i in full[len(prefix):][-3:]], "| contains 151643 (<|endoftext|>)?", EOT in full[len(prefix):])
lens=[]
print("teacher-CoT target length (tokens) median over 200 rows:", int(np.median([len(tok.encode(c, add_special_tokens=False)) for c in contents[:200]])))

print("\n=== 2. the SFT model (coldstart_sft/hf) at the end of a teacher-CoT answer")
m=AutoModelForCausalLM.from_pretrained(f"{D}/ckpt/coldstart_sft/hf", dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
import json
print("hf generation_config eos:", json.load(open(f"{D}/ckpt/coldstart_sft/hf/generation_config.json")).get("eos_token_id"))
def show(logits):
    p=torch.softmax(logits.float(),-1); top=torch.topk(p,6)
    return f"p(im_end)={float(p[IM_END]):.3g} p(eot)={float(p[EOT]):.3g} p(\\n)={float(p[NL]):.3g} p(im_start)={float(p[IM_START]):.3g} | top: " + ", ".join(f"{tok.decode([int(i)])!r}:{float(v):.3g}" for v,i in zip(top.values, top.indices))
with torch.no_grad():
    for k in range(3):
        msgs=[dict(x) for x in df.iloc[k]["messages"]]
        pre=tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids=tok.encode(pre, add_special_tokens=False)+tok.encode(msgs[-1]["content"], add_special_tokens=False)
        out=m(torch.tensor([ids]), use_cache=True); print(f"  row {k}: END OF ANSWER   ", show(out.logits[0,-1]))
        o2=m(torch.tensor([[IM_END]]), past_key_values=out.past_key_values, use_cache=True); print(f"         after <|im_end|>  ", show(o2.logits[0,-1]))
        o3=m(torch.tensor([[NL]]), past_key_values=o2.past_key_values, use_cache=True); print(f"         after <|im_end|>\\n", show(o3.logits[0,-1]))
        # greedy 14 tokens from END OF ANSWER
        cur=ids[:]; gen=[]; pk=None; x=torch.tensor([cur])
        o=m(x, use_cache=True); pk=o.past_key_values; nxt=int(torch.argmax(o.logits[0,-1])); gen.append(nxt)
        for _ in range(13):
            o=m(torch.tensor([[nxt]]), past_key_values=pk, use_cache=True); pk=o.past_key_values; nxt=int(torch.argmax(o.logits[0,-1])); gen.append(nxt)
        print("         greedy continuation:", [tok.decode([g]) for g in gen])
    # sampled continuation (tau 0.7 like the protocol) x3 for row 0, 24 tokens, to see the typical path
    msgs=[dict(x) for x in df.iloc[0]["messages"]]
    pre=tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids=tok.encode(pre, add_special_tokens=False)+tok.encode(msgs[-1]["content"], add_special_tokens=False)
    torch.manual_seed(0)
    for s in range(3):
        o=m(torch.tensor([ids]), use_cache=True); pk=o.past_key_values; gen=[]
        nxt=int(torch.multinomial(torch.softmax(o.logits[0,-1].float()/0.7,-1),1)); gen.append(nxt)
        for _ in range(23):
            o=m(torch.tensor([[nxt]]), past_key_values=pk, use_cache=True); pk=o.past_key_values
            nxt=int(torch.multinomial(torch.softmax(o.logits[0,-1].float()/0.7,-1),1)); gen.append(nxt)
            if nxt==EOT: break
        print(f"  sampled path {s} (tau .7):", [tok.decode([g]) for g in gen])
del m

print("\n=== 3. a2's eval responses (text): what is inside the truncated ones")
FAKE=re.compile(r"\n(user|assistant|system)\n")
for f in sorted(glob.glob(f"{D}/evals/a2_coldstart_s0_16k__math500__step*__seed0__*.parquet")):
    d=pd.read_parquet(f)
    if "response" not in d.columns: continue
    d["n_text"]=[len(tok.encode(r or "", add_special_tokens=False)) for r in d["response"]]
    d["delta"]=d["resp_len"]-d["n_text"]
    tr=d[d.finish_reason=="length"]; st=d[d.finish_reason=="stop"]
    print(os.path.basename(f), "n=",len(d), "stop/length:", len(st), len(tr))
    hist=Counter(int(min(x,10)) for x in tr["delta"]); print("   truncated: delta(#special tokens inside) hist:", dict(sorted(hist.items())), "| fake-turn markers:", round(float(tr["response"].fillna("").apply(lambda s: bool(FAKE.search(s))).mean()),3))
    ex=tr.iloc[0]["response"]; i=ex.find("\nuser\n"); print("   example around first fake turn:", repr(ex[max(0,i-150):i+120]) if i>=0 else repr(ex[:300]))
    if len(st):
        hist=Counter(int(min(x,10)) for x in st["delta"]); print("   stopped: delta hist:", dict(sorted(hist.items())), "| mean len", int(st.resp_len.mean()))
    break
print("DONE")
