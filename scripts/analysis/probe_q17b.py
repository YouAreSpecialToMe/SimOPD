"""Qwen3-1.7B(instruct)终止 token 实测:120 题 greedy + 120 题 tau=1,
记录每条轨迹 finish_reason / 最后 token id / 长度。nonthink 模板与战役协议一致。"""
import os
D = "/mgfs/shared/Group_GY/changhao/simopd_data"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main():
    M = "Qwen/Qwen3-1.7B"; IM, EOT = 151645, 151643
    INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
    tok = AutoTokenizer.from_pretrained(M)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [tok.apply_chat_template([{"role":"user","content":ex["problem"].strip()+" "+INSTRUCTION}],
               tokenize=False, add_generation_prompt=True, enable_thinking=False) for ex in ds][:120]
    llm = LLM(model=M, max_model_len=17920, gpu_memory_utilization=0.9, seed=0)
    for name, sp in [("greedy", SamplingParams(temperature=0.0, max_tokens=16384)),
                     ("tau1",   SamplingParams(temperature=1.0, top_p=1.0, max_tokens=16384, seed=0))]:
        outs = llm.generate(prompts, sp)
        fin, last, lens = Counter(), Counter(), []
        for o in outs:
            c = o.outputs[0]
            fin[str(c.finish_reason)] += 1
            lens.append(len(c.token_ids))
            if c.finish_reason == "stop":
                t = c.token_ids[-1] if c.token_ids else None
                last[{IM:"im_end", EOT:"eot"}.get(t, f"other:{t}")] += 1
        lens.sort()
        print(f"[{name}] n={len(outs)} finish={dict(fin)} last_token={dict(last)} "
              f"len p50/p90/max={lens[len(lens)//2]}/{lens[int(len(lens)*.9)]}/{lens[-1]}")
    print("PROBE_Q17B_DONE")

if __name__ == "__main__":
    main()
