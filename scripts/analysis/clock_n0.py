"""n0 波 A 轴双拼写时钟 + 末 token rollout 探针:
off-policy 轨迹(教师示范以 im_end 结尾)在 N0 事件级读数下,教给学生的是
"im_end 拼写"还是"用自己的 eot 在教师认可的位置停"?"""
import glob, os
D = "/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

def main():
    import pandas as pd, torch
    from collections import Counter
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    IM, EOT = 151645, 151643; DEV = "cuda:0"
    INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test"); pb = {e["unique_id"]: e["problem"] for e in ds}
    df = pd.read_parquet(sorted(glob.glob(f"{D}/evals_vanilla_sweep/vanilla_s0_16k__math500__step25__*.parquet"))[-1])
    sel = df[(df.finish_reason=="stop") & (df.resp_len>150) & (df.resp_len<1500)].sample(n=15, random_state=0)
    prefixes = []
    for _, r in sel.iterrows():
        p = tok.apply_chat_template([{"role":"user","content":pb[r.problem_id].strip()+" "+INSTRUCTION}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefixes.append(tok.encode(p, add_special_tokens=False) + tok.encode(str(r.response), add_special_tokens=False))
    print(f"{'arm':>20} {'bank':>5} {'p(eot)中位':>12} {'p(im_end)中位':>13} {'im>eot':>7} {'停=top1':>8}")
    for arm in ["a1_gkd_mix0.5_n0", "a4_dagger_anneal_n0", "a5_aggrevate_n0", "a3_offpolicy_n0"]:
        for st in [25, 50, 75]:
            d = f"{D}/ckpt/simopd/{arm}_s0_16k/global_step_{st}/actor/huggingface"
            if not os.path.isdir(d): continue
            m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to(DEV).eval()
            pe, pi, gt, top = [], [], 0, 0
            with torch.no_grad():
                for ids in prefixes:
                    lg = m(torch.tensor([ids], device=DEV)).logits[0,-1].float()
                    p = torch.softmax(lg, -1)
                    pe.append(p[EOT].item()); pi.append(p[IM].item())
                    gt += int(p[IM].item() > p[EOT].item()); top += int(p.argmax().item() in (EOT, IM))
            print(f"{arm:>20} {st:>5} {pd.Series(pe).median():>12.2e} {pd.Series(pi).median():>13.2e} {gt:>5}/15 {top:>6}/15")
            del m; torch.cuda.empty_cache()
    # ---- rollout 末 token:75 步 bank,双停契约,60 题贪心
    from vllm import LLM, SamplingParams
    prompts = [tok.apply_chat_template([{"role":"user","content":ex["problem"].strip()+" "+INSTRUCTION}],
               tokenize=False, add_generation_prompt=True, enable_thinking=False) for ex in ds][:60]
    for arm in ["a1_gkd_mix0.5_n0", "a4_dagger_anneal_n0", "a3_offpolicy_n0"]:
        d = f"{D}/ckpt/simopd/{arm}_s0_16k/global_step_75/actor/huggingface"
        if not os.path.isdir(d): continue
        llm = LLM(model=d, max_model_len=17920, gpu_memory_utilization=0.85, seed=0)
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=16384, stop_token_ids=[IM]))
        fin, last = Counter(), Counter()
        for o in outs:
            c = o.outputs[0]; ids = list(c.token_ids)
            fin[str(c.finish_reason)] += 1
            if c.finish_reason == "stop" and ids:
                last[{IM:"im_end", EOT:"eot"}.get(ids[-1], f"other:{ids[-1]}")] += 1
            elif c.finish_reason == "stop":
                last["im_end(被剥离)"] += 1
        print(f"[rollout {arm}@75 | 双停] finish={dict(fin)} last={dict(last)}")
        del llm
        import gc; gc.collect(); torch.cuda.empty_cache()
    print("CLOCK_N0_DONE")

if __name__ == "__main__":
    main()
