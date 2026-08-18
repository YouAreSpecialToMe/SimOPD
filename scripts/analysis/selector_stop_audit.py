"""Run (cluster venv, CPU): python scripts/analysis/selector_stop_audit.py   # receipt docs/data/selector_stop_audit.txt

Selector / gate audit at the student's natural stop position (the -25 convention artifact),
plus the top-k family's own stop-token state -- decides which arms need corrected reruns
(triage 2026-08-19: A = sampled-k1 descendants, B = selectors/gates that may themselves read
the mismatch, D = distributional top-k arms).

Section 1 (B class): on 30 real stopped rollouts, for each selector arm's OWN early checkpoint:
  d2  SelecTKD: is the stop position accepted (student top-1 in teacher top-5)? weight 1 or .01
  d1  TIP: student entropy + teacher-top-k divergence at the stop vs the response's other
      positions -> is the stop inside the top-50% (selected)?
  d3  teachability: disagreement x compatibility, robust-normed within response -> top-5%?
  g5  RG-OPD gate: gap = sum(log q_T - log p_S) with the raw terminal (-25) vs the event-level
      terminal -> sign flips; |gap| < 25 share
  g2  FiRe filter: mean teacher logprob per trajectory raw vs event-fixed -> rank crossings
Section 2 (D class): p(eot) / p(im_end) at the natural stop for the top-k family's own
  step-250 checkpoints (b2/b4/c*/e*/h5) -- confirms they neither erased eot nor learned im_end.
Section 3 (H exposure): from eval parquets, P(stop token inside h1's first-100 window),
  E[100/T] for h3/h4 random windows, per arm at early steps.
"""
import glob
import os
import time
from collections import Counter

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
K = 32                       # teacher top-k as the D arms receive it
SELECTKD_K, TEACH_K, D_RET, D3_RET = 5, 16, 0.5, 0.05
N_PER = int(os.environ.get("N_PER", "10"))
CKPT = f"{D}/ckpt/simopd"

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
        rows.append((run, step, r["problem_id"], r["response"], int(r["correct"])))
examples = []
for run, step, pid, response, correct in rows:
    prompt = tok.apply_chat_template([{"role": "user", "content": prob_by_id[pid].strip() + " " + INSTRUCTION}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok.encode(prompt, add_special_tokens=False)
    r_ids = tok.encode(response, add_special_tokens=False) + [EOT]
    examples.append(dict(run=run, step=step, pid=pid, correct=correct, p_ids=p_ids, r_ids=r_ids))
print(f"{len(examples)} stopped rollouts (terminal token = <|endoftext|>)")


@torch.no_grad()
def teacher_pass(model, e):
    ids = e["p_ids"] + e["r_ids"]
    P = len(e["p_ids"])
    lp = torch.log_softmax(model(torch.tensor([ids])).logits[0, P - 1:-1].float(), -1)   # [R, V]: predicts r_ids
    tgt = torch.tensor(e["r_ids"])
    tok_lp = lp.gather(1, tgt[:, None]).squeeze(1)                                       # log q(y_t)
    top = torch.topk(lp, K, dim=-1)                                                      # teacher top-k (rank order)
    q_stop_T = torch.logsumexp(lp[:, [EOT, IM_END]], -1)                                 # log sum_{E_T} q at every position
    return dict(tok_lp=tok_lp, top_lp=top.values, top_id=top.indices, log_qstop=q_stop_T,
                q_im_end_last=float(lp[-1, IM_END].exp()), q_eot_last=float(lp[-1, EOT].exp()))


@torch.no_grad()
def student_pass(model, e):
    ids = e["p_ids"] + e["r_ids"]
    P = len(e["p_ids"])
    lp = torch.log_softmax(model(torch.tensor([ids])).logits[0, P - 1:-1].float(), -1)   # [R, V]
    tgt = torch.tensor(e["r_ids"])
    tok_lp = lp.gather(1, tgt[:, None]).squeeze(1)
    ent = -(lp.exp() * lp).sum(-1)
    top16 = torch.topk(lp, TEACH_K, dim=-1).indices
    log_pstop = lp[:, EOT]                                                               # E_S = {eot}
    return dict(lp=lp, tok_lp=tok_lp, ent=ent, top16=top16, log_pstop=log_pstop,
                p_eot_last=float(lp[-1, EOT].exp()), p_im_end_last=float(lp[-1, IM_END].exp()))


def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def robust(x):
    lo, hi = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
    return ((x - lo) / (hi - lo + 1e-8)).clamp(0, 1)


t0 = time.time()
teacher = AutoModelForCausalLM.from_pretrained(TCH, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
T = [teacher_pass(teacher, e) for e in examples]
del teacher
print(f"teacher scored in {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ Section 1
print("\n=== Section 1: selectors / gates at the natural stop position (per arm's OWN checkpoint)")
SEL = [("d1_tip_s0_16k", 50), ("d2_selectkd_s0_16k", 25), ("d2_selectkd_s0_16k", 50), ("d2_selectkd_s0_16k", 100),
       ("d3_teachability_s0_16k", 50), ("g2_fire_likelihood_s0_16k", 50), ("g5_rgopd_gate_s0_16k", 50),
       ("g5_rgopd_gate_s0_16k", 125), ("vanilla_s0_16k", 50)]
for run, step in SEL:
    path = f"{CKPT}/{run}/global_step_{step}/actor/huggingface"
    if not os.path.isdir(path):
        print(f"[skip] {run}@{step}: no hf checkpoint"); continue
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    d2_acc, d2_top1_is_eot, tip_sel, tip_rank, d3_sel, gaps_raw, gaps_fix, g2_raw, g2_fix = [], [], [], [], [], [], [], [], []
    p_eot, dl_raw, dl_fix = [], [], []
    for e, t in zip(examples, T):
        s = student_pass(m, e)
        R = len(e["r_ids"]); last = R - 1
        # d2: student top-1 in teacher top-5?
        top1 = int(torch.argmax(s["lp"][last]))
        d2_acc.append(int(top1 in t["top_id"][last, :SELECTKD_K].tolist()))
        d2_top1_is_eot.append(int(top1 == EOT))
        # d1 TIP: soft-OR of minmax(entropy) and minmax(teacher-topk divergence), within-response proxy
        stu_at_t = s["lp"].gather(1, t["top_id"])                                          # log p_S at teacher top-k ids
        delta = (t["top_lp"].exp() * (t["top_lp"] - stu_at_t)).sum(-1)
        h_n, d_n = minmax(s["ent"]), minmax(delta)
        score = h_n + d_n - h_n * d_n
        thr = torch.quantile(score, 1 - D_RET)
        tip_sel.append(int(score[last] >= thr)); tip_rank.append(float((score < score[last]).float().mean()))
        # d3 teachability: disagreement x compatibility (teacher mass on student top-16), robust-normed
        in_top16 = (t["top_id"].unsqueeze(-1) == s["top16"].unsqueeze(-2)).any(-1)
        compat = (t["top_lp"].exp() * in_top16).sum(-1)
        sc3 = robust(delta) * robust(compat)
        d3_sel.append(int(sc3[last] >= torch.quantile(sc3, 1 - D3_RET)))
        # g5 gap raw vs event-fixed terminal
        dl = t["tok_lp"] - s["tok_lp"]                                                     # log q - log p per token
        raw_gap = float(dl.sum())
        fix_term = float(t["log_qstop"][last] - s["log_pstop"][last])
        fix_gap = raw_gap - float(dl[last]) + fix_term
        gaps_raw.append(raw_gap); gaps_fix.append(fix_gap)
        dl_raw.append(float(dl[last])); dl_fix.append(fix_term)
        # g2 filter statistic: mean teacher logprob per trajectory
        g2_raw.append(float(t["tok_lp"].mean()))
        g2_fix.append(float((t["tok_lp"].sum() - t["tok_lp"][last] + t["log_qstop"][last]) / R))
        p_eot.append(s["p_eot_last"])
    del m
    gr, gf = np.array(gaps_raw), np.array(gaps_fix)
    flips = int(((gr > 0) != (gf > 0)).sum())
    g2r, g2f = np.array(g2_raw), np.array(g2_fix)
    n = len(examples); kdrop = max(1, int(round(0.2 * n)))                                   # FiRe drops the bottom fraction (0.2 proxy)
    bottom_raw = set(np.argsort(g2r)[:kdrop]); bottom_fix = set(np.argsort(g2f)[:kdrop])
    print(f"\n--- {run}@{step}   (student p(eot) at the stop: median {np.median(p_eot):.3g})")
    print(f"    d2 SelecTKD at stop : accepted {np.mean(d2_acc):.2f} (weight 1 else .01) | student top-1 == eot {np.mean(d2_top1_is_eot):.2f}")
    print(f"    d1 TIP at stop      : selected (top-50%) {np.mean(tip_sel):.2f} | stop's score percentile median {np.median(tip_rank):.3f}")
    print(f"    d3 teachability     : selected (top-5%)  {np.mean(d3_sel):.2f}")
    print(f"    g5 gate gap         : raw median {np.median(gr):.1f}  fixed median {np.median(gf):.1f} | sign flips {flips}/{n} | |raw gap|<25: {np.mean(np.abs(gr)<25):.2f}")
    print(f"    g2 mean log q       : raw {np.mean(g2r):.3f} fixed {np.mean(g2f):.3f} (shift = 25/T) | bottom-20% set changes: {len(bottom_raw ^ bottom_fix)//2}/{kdrop}")
    print(f"    terminal Delta-ell  : raw median {np.median(dl_raw):.1f} -> event-level median {np.median(dl_fix):.2f}")

# ------------------------------------------------------------------ Section 2
print("\n=== Section 2: top-k family at their own step-250 (or latest) checkpoints -- stop-token state at the natural end")
FAM = ["b2_forward_kl_s0_16k", "b4_jsd_s0_16k", "b4_jsd_b0.1_s0_16k", "b4_jsd_b0.9_s0_16k",
       "c1_lsm_topk32_renorm_s0_16k", "c1_direct_s0_16k", "c1_tailbucket_s0_16k", "c2_qb_fixed8_s0_16k",
       "c3_intersection_s0_16k", "e1_pl_rank_s0_16k", "e2_set_coverage_s0_16k", "e3_zvalue_s0_16k",
       "h5_gen100_s0_16k", "h1_first_segment_s0_16k", "d2_selectkd_s0_16k", "d1_tip_s0_16k", "d3_teachability_s0_16k",
       "g5_rgopd_gate_s0_16k", "g2_fire_likelihood_s0_16k", "b1_skew_kl_s0_16k", "f3_power_s0_16k", "h2_last_segment_s0_16k"]
print("arm@step                          p(eot)@stop median   p(im_end)@stop median   greedy next 3 at end-of-answer (ex 0)")
for run in FAM:
    steps = sorted(int(os.path.basename(p).split("_")[-1]) for p in glob.glob(f"{CKPT}/{run}/global_step_*") if os.path.isdir(f"{p}/actor/huggingface"))
    if not steps:
        print(f"{run:<32} [no hf checkpoint]"); continue
    step = 250 if 250 in steps else steps[-1]
    m = AutoModelForCausalLM.from_pretrained(f"{CKPT}/{run}/global_step_{step}/actor/huggingface", dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    pe, pi = [], []
    with torch.no_grad():
        for e in examples[:12]:
            ids = e["p_ids"] + e["r_ids"][:-1]                                                # up to the last text token
            lg = m(torch.tensor([ids])).logits[0, -1].float()
            p = torch.softmax(lg, -1); pe.append(float(p[EOT])); pi.append(float(p[IM_END]))
        ids = examples[0]["p_ids"] + examples[0]["r_ids"][:-1]
        gen = []
        out = m(torch.tensor([ids]), use_cache=True); pk = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1])); gen.append(nxt)
        for _ in range(2):
            out = m(torch.tensor([[nxt]]), past_key_values=pk, use_cache=True); pk = out.past_key_values; nxt = int(torch.argmax(out.logits[0, -1])); gen.append(nxt)
    del m
    print(f"{run+'@'+str(step):<32} {np.median(pe):>10.3g}          {np.median(pi):>10.3g}          {[tok.decode([g]) for g in gen]}")

# ------------------------------------------------------------------ Section 3
print("\n=== Section 3: H-axis stop exposure from eval length distributions (stopped responses only)")
for run in ["vanilla_s0_16k", "h1_first_segment_s0_16k", "h3_random_segment_s0_16k", "h4_random_scatter_s0_16k"]:
    for step in (25, 50, 100):
        hits = sorted(glob.glob(f"{D}/evals/{run}__math500__step{step}__seed*.parquet"))
        if not hits:
            continue
        df = pd.read_parquet(hits[-1], columns=["resp_len", "finish_reason"])
        st = df[df.finish_reason == "stop"]
        if not len(st):
            print(f"{run}@{step}: no stopped responses"); continue
        L = st.resp_len.values.astype(float)
        p_h1 = float((L <= 100).mean())                       # stop token inside the first-100 window
        p_win = float(np.minimum(1.0, 100.0 / L).mean())      # random 100-window / 100-scatter contains the last token
        print(f"{run}@{step}: stopped {len(st)}/{len(df)}  P(stop in h1 window)={p_h1:.4f}  E[P(stop in random 100-window/scatter)]={p_win:.3f}  len median {np.median(L):.0f}")
print("DONE")
