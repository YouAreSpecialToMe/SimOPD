"""Tokenizer parity + template-shift receipt for the student/teacher pair (CPU, cluster venv).

Run:  python scripts/analysis/tokenizer_parity.py    # receipt docs/data/tokenizer_parity.txt

Part 1: ids / special flags / vocab identity / chat templates for the two tokenizers.
Part 2: merges + tokenizer.json structural diff, encode-equality on 1500 real responses,
        and whether the student-template <think>\n\n</think>\n\n prefix changes the teacher scoring.
"""
import sys
PART = sys.argv[1] if len(sys.argv) > 1 else "all"
if PART in ("all", "1"):
    import os, json, hashlib, glob
    D="/mgfs/shared/Group_GY/changhao/simopd_data"
    os.environ.setdefault("HF_HOME", f"{D}/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
    from transformers import AutoTokenizer, AutoConfig
    STU="Qwen/Qwen3-1.7B-Base"; TCH="Qwen/Qwen3-4B-Instruct-2507"
    toks={"student(1.7B-Base)":AutoTokenizer.from_pretrained(STU),"teacher(4B-Instruct-2507)":AutoTokenizer.from_pretrained(TCH)}
    def snap(name):
        d=glob.glob(f"{D}/hf_cache/hub/models--{name.replace('/','--')}/snapshots/*")[0]
        return d
    for name,t in toks.items():
        print(f"\n### {name}: vocab_size={len(t)} eos={t.eos_token!r}/{t.eos_token_id} pad={t.pad_token!r}/{t.pad_token_id} bos={t.bos_token!r}")
        for s in ["<|endoftext|>","<|im_start|>","<|im_end|>","<think>","</think>","<tool_call>","<|object_ref_start|>","\n<|im_end|>\n","assistant\n<think>\n\n</think>\n\n","<think>\n\n</think>\n\n","<|im_end|>\n<|im_start|>assistant\n"]:
            ids=t.encode(s, add_special_tokens=False)
            print(f"   {s!r:<40} -> {ids}   special:{[i in t.all_special_ids for i in ids]}  decode:{t.decode(ids)!r}")
        atd=t.added_tokens_decoder
        for i in [151643,151644,151645,151667,151668]:
            a=atd.get(i); print(f"   id {i}: {a.content!r} special={a.special}" if a else f"   id {i}: NOT an added token")
        print("   all_special_ids:", sorted(t.all_special_ids)[:12], "... n=", len(t.all_special_ids))
    # vocab identity
    v1={k:v for k,v in toks["student(1.7B-Base)"].get_vocab().items()}; v2=toks["teacher(4B-Instruct-2507)"].get_vocab()
    print("\nvocab identical:", v1==v2, "| sizes", len(v1), len(v2))
    diff=[k for k in set(v1)|set(v2) if v1.get(k)!=v2.get(k)]; print("differing entries:", diff[:20])
    for name in [STU,TCH]:
        d=snap(name)
        for f in ["tokenizer.json","vocab.json","merges.txt","tokenizer_config.json","generation_config.json","config.json"]:
            p=os.path.join(d,f)
            if os.path.exists(p):
                h=hashlib.md5(open(p,"rb").read()).hexdigest()[:10]; print(f"{name:<28} {f:<22} md5={h} size={os.path.getsize(p)}")
        cfg=json.load(open(os.path.join(d,"config.json"))); print(f"{name:<28} vocab_size(model)={cfg.get('vocab_size')} tie={cfg.get('tie_word_embeddings')}")
        tc=json.load(open(os.path.join(d,"tokenizer_config.json")))
        print(f"{name:<28} tokenizer_config: eos={tc.get('eos_token')} pad={tc.get('pad_token')} chat_template_len={len(tc.get('chat_template',''))} add_bos={tc.get('add_bos_token')}")
    # chat template comparison on the training prompt
    msgs=[{"role":"user","content":"What is 1+1? Let's think step by step and output the final answer within \\boxed{}."}]
    for name,t in toks.items():
        for think in [False,True]:
            try:
                s=t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
            except Exception as e:
                s=f"ERR {e!r}"
            print(f"\n[{name}] enable_thinking={think}:\n{s!r}")
    # how the teacher would end / what a full teacher-side conversation looks like
    s=toks["teacher(4B-Instruct-2507)"].apply_chat_template(msgs+[{"role":"assistant","content":"2"}], tokenize=False)
    print("\n[teacher template with assistant turn]:", repr(s))
    s=toks["student(1.7B-Base)"].apply_chat_template(msgs+[{"role":"assistant","content":"2"}], tokenize=False)
    print("[student template with assistant turn]:", repr(s))
if PART in ("all", "2"):
    import os, json, glob, difflib
    D="/mgfs/shared/Group_GY/changhao/simopd_data"
    os.environ.setdefault("HF_HOME", f"{D}/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
    import pandas as pd, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    STU="Qwen/Qwen3-1.7B-Base"; TCH="Qwen/Qwen3-4B-Instruct-2507"
    ts=AutoTokenizer.from_pretrained(STU); tt=AutoTokenizer.from_pretrained(TCH)
    snap=lambda n: glob.glob(f"{D}/hf_cache/hub/models--{n.replace('/','--')}/snapshots/*")[0]
    # 1) merges.txt / tokenizer.json structural diff
    m1=open(os.path.join(snap(STU),"merges.txt")).read().splitlines(); m2=open(os.path.join(snap(TCH),"merges.txt")).read().splitlines()
    print("merges lines:", len(m1), len(m2), "| first lines:", m1[:1], m2[:1])
    d=[l for l in difflib.unified_diff(m1,m2,lineterm="",n=0)]; print("merges diff (first 12 lines):", d[:12])
    j1=json.load(open(os.path.join(snap(STU),"tokenizer.json"))); j2=json.load(open(os.path.join(snap(TCH),"tokenizer.json")))
    for k in ["normalizer","pre_tokenizer","post_processor","decoder"]:
        print(f"tokenizer.json[{k}] equal:", j1.get(k)==j2.get(k))
    print("model.type:", j1["model"]["type"], j2["model"]["type"], "| model keys equal except merges/vocab:", {k:(j1["model"].get(k)==j2["model"].get(k)) for k in j1["model"] if k not in ("vocab","merges")})
    print("merges equal:", j1["model"]["merges"]==j2["model"]["merges"], "| n merges:", len(j1["model"]["merges"]), len(j2["model"]["merges"]), "| merge item type:", type(j1["model"]["merges"][0]).__name__, type(j2["model"]["merges"][0]).__name__)
    print("added_tokens equal:", j1.get("added_tokens")==j2.get("added_tokens"))
    # 2) encode-equality on real texts (student responses + prompts)
    f=sorted(glob.glob(f"{D}/evals/c4_pi_tail_budget_s0_16k__math500__step100__seed0__*.parquet"))[-1]
    df=pd.read_parquet(f); texts=df["response"].fillna("").tolist()[:1500]
    neq=0; first=None
    for s in texts:
        a=ts.encode(s, add_special_tokens=False); b=tt.encode(s, add_special_tokens=False)
        if a!=b:
            neq+=1
            if first is None: first=(s[:80],a[:20],b[:20])
    print(f"encode equality over {len(texts)} real responses: {len(texts)-neq} equal, {neq} differ; first diff: {first}")
    # 3) does the student-template <think>\n\n</think>\n\n prefix change the TEACHER's scoring?
    from datasets import load_dataset
    ds=load_dataset("HuggingFaceH4/MATH-500", split="test"); prob={ex["unique_id"]:ex["problem"] for ex in ds}
    INSTR="Let's think step by step and output the final answer within \\boxed{}."
    stop=df[(df.finish_reason=="stop")&(df.resp_len>150)&(df.resp_len<900)].sample(n=8, random_state=1)
    torch.set_num_threads(96)
    m=AutoModelForCausalLM.from_pretrained(TCH, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    IM_END, EOT = 151645, 151643
    print("\nteacher scoring of the SAME student response under (a) the training prompt = student Base template (has <think>\\n\\n</think>\\n\\n) vs (b) the teacher's own template (no think tags)")
    print("pid                          | mean log q per resp token: (a) student-tpl  (b) teacher-tpl | terminal: q(im_end) a/b   q(eot) a/b   | tokens with |Δ log q|>2: frac")
    import numpy as np
    agg=[]
    with torch.no_grad():
        for _,r in stop.iterrows():
            msgs=[{"role":"user","content":prob[r.problem_id].strip()+" "+INSTR}]
            pa=ts.encode(ts.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False), add_special_tokens=False)
            pb=tt.encode(tt.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True), add_special_tokens=False)
            rid=ts.encode(r.response, add_special_tokens=False)+[EOT]
            out=[]
            for P in (pa,pb):
                ids=P+rid; lg=m(torch.tensor([ids])).logits[0].float()
                lp=torch.log_softmax(lg[len(P)-1:-1],-1); tl=lp.gather(1,torch.tensor(rid)[:,None]).squeeze(1)
                term=lp[-1]; out.append((tl.numpy(), float(term[IM_END].exp()), float(term[EOT].exp())))
            ta,tb=out[0][0],out[1][0]; big=(np.abs(ta-tb)>2).mean()
            print(f"{r.problem_id:<28} | {ta[:-1].mean():8.3f}   {tb[:-1].mean():8.3f}   | {out[0][1]:.3f}/{out[1][1]:.3f}   {out[0][2]:.1e}/{out[1][2]:.1e} | {big:.3f}")
            agg.append((ta[:-1].mean(), tb[:-1].mean(), out[0][1], out[1][1], big))
    a=np.array(agg); print("MEAN:", " ".join(f"{x:.3f}" for x in a.mean(0)))
