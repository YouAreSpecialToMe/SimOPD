"""答案后探针(GPU):学生后期循环响应里,每次"答案落笔"的位置上——
教师下一个 token 更想要终止(im_end/eot)还是继续?随循环次数如何变化?
并让教师从该位置真实贪心生成 24 token。骨架复用 eos_stop_audit.py。
Run: python post_answer_probe.py  (gpu252, 1 GPU)"""
import glob, os, re
D = "/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pandas as pd, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

STU = "Qwen/Qwen3-1.7B-Base"; TCH = "Qwen/Qwen3-4B-Instruct-2507"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
IM_END, EOT = 151645, 151643
N_PER, DEV = 12, "cuda:0"
ARMS = [("a1_gkd_mix0.5_s0_16k", 125), ("a3_offpolicy_s0_16k", 250), ("h6_gen_sched_s0_16k", 175)]

tok = AutoTokenizer.from_pretrained(STU)
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
prob_by_id = {ex["unique_id"]: ex["problem"] for ex in ds}

def anchors(text, ks=(1, 2, 3, 10)):
    """第 k 次 \\boxed 所在"答案行"的自然停点(闭括号后吞掉 $/$./空格,不含换行)+ 最后一次。"""
    outs, k = [], 0
    for m in re.finditer(r"\\boxed\{", text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0); i += 1
        j = i
        while j < len(text) and text[j] in "$. )": j += 1
        k += 1; outs.append((k, j))
    if not outs: return []
    picked = [o for o in outs if o[0] in ks]
    if outs[-1][0] not in ks: picked.append(outs[-1])
    return picked

# ---- 采样截断响应
cases = []
for run, step in ARMS:
    fs = sorted(glob.glob(f"{D}/evals_v2probe/{run}__math500__step{step}__*.parquet"))
    df = pd.read_parquet(fs[-1])
    tr = df[(df.finish_reason == "length") & df.response.astype(str).str.contains(r"\\boxed", regex=True)]
    tr = tr.sample(n=min(N_PER, len(tr)), random_state=0)
    for _, r in tr.iterrows():
        a = anchors(str(r.response))
        if a: cases.append(dict(run=run, step=step, pid=r.problem_id, text=str(r.response), anchors=a))
print(f"{len(cases)} truncated responses, {sum(len(c['anchors']) for c in cases)} anchor positions")

@torch.no_grad()
def next_dist(model, ids):
    x = torch.tensor([ids], device=DEV)
    lg = model(x).logits[0, -1].float()
    p = torch.softmax(lg, -1)
    top = torch.topk(p, 3)
    return p[IM_END].item(), p[EOT].item(), [(tok.decode([i]), v.item()) for v, i in zip(top.values, top.indices)]

print("== loading teacher (4B, bf16, gpu)"); 
tch = AutoModelForCausalLM.from_pretrained(TCH, torch_dtype=torch.bfloat16).to(DEV).eval()
rows, gens = [], []
for c in cases:
    prompt = tok.apply_chat_template([{"role": "user", "content": prob_by_id[c["pid"]].strip() + " " + INSTRUCTION}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok.encode(prompt, add_special_tokens=False)
    for k, cut in c["anchors"]:
        ids = p_ids + tok.encode(c["text"][:cut], add_special_tokens=False)
        if len(ids) > 16000: continue
        qi, qe, top = next_dist(tch, ids)
        rows.append(dict(run=c["run"], pid=c["pid"], it=k, plen=len(ids), q_im=qi, q_eot=qe,
                         top1=top[0][0], p_top1=top[0][1], top2=top[1][0], top3=top[2][0]))
        if k == 1 and len(gens) < 9 and len(ids) < 6000:
            out = tch.generate(torch.tensor([ids], device=DEV), max_new_tokens=24, do_sample=False,
                               eos_token_id=None, pad_token_id=EOT)
            gens.append((c["run"], c["pid"], tok.decode(out[0, len(ids):], skip_special_tokens=False)))
del tch; torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df["q_stop"] = df.q_im + df.q_eot
df["bucket"] = df.it.map(lambda k: str(k) if k <= 3 else ("10" if k == 10 else "last"))
print("\n=== 教师在答案落笔处的下一 token 意愿(按臂 x 循环轮次)===")
agg = df.groupby(["run", "bucket"]).agg(n=("q_stop", "size"), q_stop_med=("q_stop", "median"),
       q_stop_p25=("q_stop", lambda s: s.quantile(.25)), frac_stop_top1=("p_top1", "size")).reset_index()
for (run, b), g in df.groupby(["run", "bucket"]):
    stop_is_top1 = ((g.top1.str.contains("im_end")) | (g.q_stop > g.p_top1)).mean()
    print(f"{run:<26} 第{b:>4}次答案后: n={len(g):<3} q(im_end)中位={g.q_im.median():.3f} q(eot)中位={g.q_eot.median():.2e} "
          f"教师top1即停止占比={ (g.top1.str.contains('im_end')).mean():.2f} top1样例={g.top1.mode().iloc[0]!r}")
print("\n=== 教师真实接管生成(第 1 次答案后,贪心 24 token)===")
for run, pid, g in gens:
    print(f"[{run} | {pid}] -> {g!r}")
out = f"{D}/tmp_export/post_answer_probe.csv"; df.to_csv(out, index=False); print("\nsaved:", out)
