"""Run (cluster venv, CPU): python scripts/analysis/a2_coldstart_probe_steps.py   # receipt docs/data/a2_coldstart_probe_steps.txt

a2_coldstart_s0@25/50/100: p(<|im_end|>) / p(<|endoftext|>) at the end of a teacher-CoT answer,
plus a tau=0.7 sampled continuation -- does the OPD phase amplify the sampler-refused terminator?
Bears on the R5 appendix P1 prediction (dual-stop re-eval of a2@25 halves truncation)."""
import os, glob, torch, json
D="/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(96)
IM_START, IM_END, EOT = 151644, 151645, 151643
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base"); NL=tok.encode("\n",add_special_tokens=False)[0]
df=pd.read_parquet(f"{D}/simopd_math/coldstart_sft.parquet")
rows=[[dict(x) for x in df.iloc[k]["messages"]] for k in range(6)]
def show(logits):
    p=torch.softmax(logits.float(),-1); top=torch.topk(p,5)
    return f"p(im_end)={float(p[IM_END]):.3g} p(eot)={float(p[EOT]):.3g} p(im_start)={float(p[IM_START]):.2g} | top: "+", ".join(f"{tok.decode([int(i)])!r}:{float(v):.3g}" for v,i in zip(top.values,top.indices))
for name in ["a2_coldstart_s0_16k/global_step_25","a2_coldstart_s0_16k/global_step_50","a2_coldstart_s0_16k/global_step_100"]:
    path=f"{D}/ckpt/simopd/{name}/actor/huggingface"
    if not os.path.isdir(path): print("missing", path); continue
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    print(f"\n##### {name}")
    with torch.no_grad():
        for k,msgs in enumerate(rows):
            pre=tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids=tok.encode(pre, add_special_tokens=False)+tok.encode(msgs[-1]["content"], add_special_tokens=False)
            out=m(torch.tensor([ids]), use_cache=True); print(f"  row {k}: END OF ANSWER ", show(out.logits[0,-1]))
            if k<2:
                o2=m(torch.tensor([[IM_END]]), past_key_values=out.past_key_values, use_cache=True); print("         after <|im_end|>", show(o2.logits[0,-1]))
                torch.manual_seed(k)
                o=m(torch.tensor([ids]), use_cache=True); pk=o.past_key_values; gen=[]
                nxt=int(torch.multinomial(torch.softmax(o.logits[0,-1].float()/0.7,-1),1)); gen.append(nxt)
                for _ in range(29):
                    o=m(torch.tensor([[nxt]]), past_key_values=pk, use_cache=True); pk=o.past_key_values
                    nxt=int(torch.multinomial(torch.softmax(o.logits[0,-1].float()/0.7,-1),1)); gen.append(nxt)
                    if nxt in (EOT,): break
                print("         sampled tau.7:", [tok.decode([g]) for g in gen])
    del m
print("DONE")
