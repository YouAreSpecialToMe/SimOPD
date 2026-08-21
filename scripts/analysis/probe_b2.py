"""b2(teacher-topk forward KL)终止验尸:双契约 rollout。
legacy 契约(只认 eot)下若模型 mid-text 吐 im_end = P-drift 铁证;
dual 契约下看它能否正常停、以哪个拼写停。"""
import os
D = "/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

def main():
    from collections import Counter
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    IM, EOT = 151645, 151643
    INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [tok.apply_chat_template([{"role":"user","content":ex["problem"].strip()+" "+INSTRUCTION}],
               tokenize=False, add_generation_prompt=True, enable_thinking=False) for ex in ds][:120]
    for st in [50, 175]:
        path = f"{D}/ckpt/simopd/b2_forward_kl_s0_16k/global_step_{st}/actor/huggingface"
        llm = LLM(model=path, max_model_len=17920, gpu_memory_utilization=0.9, seed=0)
        for cname, sp in [
            ("legacy(只认eot)", SamplingParams(temperature=0.0, max_tokens=16384)),
            ("dual(双停)",      SamplingParams(temperature=0.0, max_tokens=16384, stop_token_ids=[IM]))]:
            outs = llm.generate(prompts, sp)
            fin, last = Counter(), Counter()
            midim = midim_resp = 0; lens = []
            for o in outs:
                c = o.outputs[0]; ids = list(c.token_ids)
                fin[str(c.finish_reason)] += 1; lens.append(len(ids))
                n_im_mid = sum(1 for t in ids[:-1] if t == IM)
                midim += n_im_mid; midim_resp += (n_im_mid > 0)
                if c.finish_reason == "stop" and ids:
                    last[{IM:"im_end", EOT:"eot"}.get(ids[-1], f"other:{ids[-1]}")] += 1
            lens.sort()
            print(f"[b2@{st} | {cname}] finish={dict(fin)} last={dict(last)} "
                  f"mid_im_end: {midim} 个 / {midim_resp} 条响应含之; len p50/p90={lens[len(lens)//2]}/{lens[int(len(lens)*.9)]}")
        del llm
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
    print("PROBE_B2_DONE")

if __name__ == "__main__":
    main()
