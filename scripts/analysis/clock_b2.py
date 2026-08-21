"""b2 双拼写时钟:固定 15 个答案完成态,逐 bank 测 p(eot) 与 p(im_end)。"""
import glob, os
D = "/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import pandas as pd, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
IM, EOT = 151645, 151643; DEV="cuda:0"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
ds = load_dataset("HuggingFaceH4/MATH-500", split="test"); pb = {e["unique_id"]: e["problem"] for e in ds}
df = pd.read_parquet(sorted(glob.glob(f"{D}/evals_vanilla_sweep/vanilla_s0_16k__math500__step25__*.parquet"))[-1])
sel = df[(df.finish_reason=="stop") & (df.resp_len>150) & (df.resp_len<1500)].sample(n=15, random_state=0)
prefixes = []
for _, r in sel.iterrows():
    p = tok.apply_chat_template([{"role":"user","content":pb[r.problem_id].strip()+" "+INSTRUCTION}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefixes.append(tok.encode(p, add_special_tokens=False) + tok.encode(str(r.response), add_special_tokens=False))
print(f"{'bank':>6} {'p(eot)中位':>12} {'p(im_end)中位':>13} {'p(im)>p(eot)':>12} {'停止=top1':>9}")
for st in [25, 75, 125, 175]:
    d = f"{D}/ckpt/simopd/b2_forward_kl_s0_16k/global_step_{st}/actor/huggingface"
    if not os.path.isdir(d): continue
    m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to(DEV).eval()
    pe, pi, gt, top = [], [], 0, 0
    with torch.no_grad():
        for ids in prefixes:
            lg = m(torch.tensor([ids], device=DEV)).logits[0,-1].float()
            p = torch.softmax(lg, -1)
            pe.append(p[EOT].item()); pi.append(p[IM].item())
            gt += int(p[IM].item() > p[EOT].item())
            top += int(p.argmax().item() in (EOT, IM))
    print(f"{st:>6} {pd.Series(pe).median():>12.2e} {pd.Series(pi).median():>13.2e} {gt:>10}/15 {top:>7}/15")
    del m; torch.cuda.empty_cache()
print("CLOCK_B2_DONE")
