"""C2/C4 stopping-hazard decomposition probe (CPU, HF forwards).

Why C2/C4 stop at ~8k although the teacher's termination mass on the student's
states is ~0: under renormalized top-k only the teacher-support logits receive
gradient, so the student's stop hazard factorises EXACTLY as

    h(s) = p(eot | s) = [1 - m_S(s)] * r(s),
    m_S(s) = sum_{k in S_T(s)} p(k | s)          student mass on the arm's support
    r(s)   = p(eot | s) / (1 - m_S(s))          eot share of the out-of-support mass

and against the base checkpoint (0)

    log h_theta / h_0 = A_S + A_out,
    A_S   = log (1 - m_theta)/(1 - m_0)          support-mass channel
    A_out = log r_theta / r_0                    outside-shape drift (shared parameters)

Response length is the first-passage time of this hazard along the student's own
trajectory: S(t) = prod_{j<t} (1 - h_j), Lambda(t) = -sum log(1 - h_j).

Along the arm's OWN eval responses (text from the archived eval parquets) this
script forwards three models on the same prefixes -- base student, the arm's
checkpoint, and the teacher (only to obtain the top-K payload the arm trains on)
-- rebuilds the arm's ACTUAL support rule per position (c4: pi-tail prefix of the
teacher ranking reaching student mass 1-eps; c2: margin=max(q,pi) >= tau, tau a
batch quantile pinning the average budget), and records per position:

    h_theta, h_0, m_theta, m_0, r_theta, r_0, A_S, A_out,
    F_stop = m_S * Cov_ptilde(ptilde_k, l_k)     local indirect push of support-RKL on log h
    q_T(im_end), q_T(eot), p_theta(im_end)       reference only

plus the cumulative hazard / survival per response and the correct/wrong split.
Usage (hop pod, CPU):
    ARM=c4_pi_tail_budget STEP=250 N_CORRECT=12 N_WRONG=8 N_TRUNC=4 MAXTOK=4500 \
        PROBE_THREADS=96 python scripts/analysis/c_stop_hazard_probe.py
Writes docs/data/c_stop_hazard_<arm>_<step>.npz (+ a text summary on stdout).
"""
import glob
import os
import sys
import time

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
torch.set_grad_enabled(False)
DEVICE = os.environ.get("PROBE_DEVICE", "cpu")          # "cuda:0" on a GPU box: ~50x faster, same numbers

ARM = os.environ.get("ARM", "c4_pi_tail_budget")
STEP = int(os.environ.get("STEP", "250"))
SEED = int(os.environ.get("SEED", "0"))
N_CORRECT = int(os.environ.get("N_CORRECT", "12"))
N_WRONG = int(os.environ.get("N_WRONG", "8"))
N_TRUNC = int(os.environ.get("N_TRUNC", "4"))
MAXTOK = int(os.environ.get("MAXTOK", "4500"))       # cap per response (CPU cost is quadratic)
TRUNC_KEEP = int(os.environ.get("TRUNC_KEEP", "6000"))
K = int(os.environ.get("PAYLOAD_TOPK", "32"))         # DISTILLATION_TOPK of the banked C arms
PI_TAIL_EPS = float(os.environ.get("SIMOPD_PI_TAIL_EPS", "0.05"))
QB_BUDGET = float(os.environ.get("SIMOPD_QB_TARGET_BUDGET", "8"))
STU = "Qwen/Qwen3-1.7B-Base"
TCH = "Qwen/Qwen3-4B-Instruct-2507"
CKPT_STEP = int(os.environ.get("CKPT_STEP", str(STEP)))     # texts from STEP, weights from CKPT_STEP (default: matched)
CKPT = f"{D}/ckpt/simopd/{ARM}_s{SEED}_16k/global_step_{CKPT_STEP}/actor/huggingface"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
IM_END, EOT = 151645, 151643
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "docs", "data", f"c_stop_hazard_{ARM}_txt{STEP}_ckpt{CKPT_STEP}.npz")

tok = AutoTokenizer.from_pretrained(STU)
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
prob_by_id = {ex["unique_id"]: ex["problem"] for ex in ds}


def newest_with_text(run_id, bench, step):
    roots = [os.environ["SIMOPD_EVAL_ROOT"]] if os.environ.get("SIMOPD_EVAL_ROOT") else []
    roots += [f"{D}/evals"]
    for root in roots:
        for h in sorted(glob.glob(f"{root}/{run_id}__{bench}__step{step}__seed*.parquet"))[::-1]:
            df = pd.read_parquet(h)
            if "response" in df.columns:
                return h, df
    for h in sorted(glob.glob(f"{D}/evals/{run_id}__{bench}__step{step}__seed*.parquet"))[::-1]:
        df = pd.read_parquet(h)
        if "response" in df.columns:
            return h, df
    for h in sorted(glob.glob(f"{D}/archive/*/evals/{run_id}__{bench}__step{step}__seed*.parquet"))[::-1]:
        df = pd.read_parquet(h)
        if "response" in df.columns:
            return h, df
    return None, None


path, df = newest_with_text(f"{ARM}_s{SEED}_16k", "math500", STEP)
assert df is not None, f"no eval parquet with text for {ARM} step {STEP}"
print(f"[probe] {ARM}@{STEP}: {os.path.basename(path)} n={len(df)} "
      f"stop={int((df.finish_reason == 'stop').sum())} trunc={int((df.finish_reason == 'length').sum())}")
rng = np.random.RandomState(0)
stopped = df[(df.finish_reason == "stop") & (df.resp_len <= MAXTOK) & (df.resp_len > 50)]
corr = stopped[stopped.correct == 1].sample(n=min(N_CORRECT, int((stopped.correct == 1).sum())), random_state=0)
wrong = stopped[stopped.correct == 0].sample(n=min(N_WRONG, int((stopped.correct == 0).sum())), random_state=0)
trunc = df[df.finish_reason == "length"].sample(n=min(N_TRUNC, int((df.finish_reason == "length").sum())), random_state=0)
if len(trunc) < N_TRUNC:   # fallback: the longest wrong responses stand in for "deep failure"
    extra = df[(df.correct == 0) & (~df.index.isin(trunc.index)) & (~df.index.isin(wrong.index))].sort_values("resp_len", ascending=False).head(N_TRUNC - len(trunc))
    trunc = pd.concat([trunc, extra])
rows = [("correct", r) for _, r in corr.iterrows()] + [("wrong", r) for _, r in wrong.iterrows()] + \
       [("trunc", r) for _, r in trunc.iterrows()]
print(f"[probe] sampled correct={len(corr)} wrong={len(wrong)} trunc={len(trunc)} (MAXTOK={MAXTOK}, trunc kept to {TRUNC_KEEP})")

examples = []
for grp, r in rows:
    prompt = tok.apply_chat_template([{"role": "user", "content": prob_by_id[r.problem_id].strip() + " " + INSTRUCTION}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok.encode(prompt, add_special_tokens=False)
    r_ids = tok.encode(r.response, add_special_tokens=False)
    if grp == "trunc":
        r_ids = r_ids[:TRUNC_KEEP]
        stopped_flag = False
        if r.finish_reason == "stop" and len(r_ids) < TRUNC_KEEP:   # longest-wrong stand-in: keep its real stop
            r_ids = r_ids + [EOT]; stopped_flag = True
    else:
        r_ids = r_ids + [EOT]
        stopped_flag = True
    examples.append(dict(grp=grp, pid=r.problem_id, correct=int(r.correct), p_ids=p_ids, r_ids=r_ids, stopped=stopped_flag))


def load(path):
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16,
                                             low_cpu_mem_usage=True).to(DEVICE).eval()
    print(f"[probe] loaded {path.split('/')[-1] if '/' in path else path} in {time.time()-t0:.0f}s", flush=True)
    return m


def forward_rows(model, e, want):
    """Run one forward on prompt+response; return per-response-position quantities.
    Position i (0-based over response tokens) is the distribution that PREDICTS
    response token i, i.e. logits at index P-1+i. Returns dict of tensors [R, ...]."""
    ids = e["p_ids"] + e["r_ids"]
    P, R = len(e["p_ids"]), len(e["r_ids"])
    out = {}
    x = torch.tensor([ids], device=DEVICE)
    logits = model(x).logits[0, P - 1:P - 1 + R].float().cpu()    # [R, V] (bf16 weights on GPU; logits to fp32 on CPU)
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    if want == "teacher":
        lp = logits - lse
        top = torch.topk(lp, K, dim=-1)                            # teacher rank order = payload
        out["t_top_lp"] = top.values.double()
        out["t_top_ids"] = top.indices
        out["q_im_end"] = lp[:, IM_END].exp()
        out["q_eot"] = lp[:, EOT].exp()
    else:
        out["logits"] = logits                                     # keep; we gather against the payload later
        out["lse"] = lse.squeeze(-1)
    return out


def student_quantities(logits, lse, t_top_ids, t_top_lp, keep):
    """Given a student's logits [R,V] and the payload + support mask keep [R,K]:
    h, m_S, r, p_im_end, F_stop, KL_S."""
    # float64 throughout: in the body the student's mass on the payload is 1 - 1e-9 and
    # eot is 1e-10, so 1 - m and r are hopeless in float32. The out-of-support mass is
    # computed DIRECTLY as logsumexp over the complement (exact, no cancellation).
    logits = logits.double()
    lp_all = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    h = lp_all[:, EOT].exp()
    p_im = lp_all[:, IM_END].exp()
    stu_at = torch.gather(lp_all, 1, t_top_ids)                    # [R,K] log p on payload
    pi = stu_at.exp()
    m = (pi * keep).sum(-1)
    sup_ids = torch.where(keep, t_top_ids, torch.full_like(t_top_ids, -1))
    masked = lp_all.clone()
    for j in range(K):                                              # mask the support ids out
        idx = sup_ids[:, j]
        ok = idx >= 0
        masked[torch.arange(masked.shape[0])[ok], idx[ok]] = float("-inf")
    log_out = torch.logsumexp(masked, dim=-1)                        # log (1 - m), exact
    out_mass = log_out.exp()
    r = h / out_mass.clamp_min(1e-300)
    t_top_lp = t_top_lp.double()
    neg = torch.finfo(torch.float64).min
    stu_n = torch.where(keep, stu_at, torch.full_like(stu_at, neg))
    tch_n = torch.where(keep, t_top_lp, torch.full_like(t_top_lp, neg))
    stu_n = stu_n - torch.logsumexp(stu_n, -1, keepdim=True)
    tch_n = tch_n - torch.logsumexp(tch_n, -1, keepdim=True)
    pt = torch.where(keep, stu_n.exp(), torch.zeros_like(stu_n))
    ell = torch.where(keep, stu_n - tch_n, torch.zeros_like(stu_n))
    kl = (pt * ell).sum(-1)
    # Cov under ptilde of (ptilde_k, ell_k): E[XY] - E[X]E[Y]
    cov = (pt * pt * ell).sum(-1) - (pt * pt).sum(-1) * kl
    f_stop = m * cov
    q_cov = (t_top_lp.exp() * keep).sum(-1)                         # teacher mass covered by the support
    return dict(h=h, m=m, r=r, p_im=p_im, f_stop=f_stop, kl=kl, pi=pi, log_out=log_out, q_cov=q_cov)


def support_c4(pi_theta):
    cum = pi_theta.cumsum(-1)
    reached = cum >= (1.0 - PI_TAIL_EPS)
    first = torch.where(reached.any(-1), reached.float().argmax(-1), torch.full(reached.shape[:-1], K - 1, dtype=torch.long))
    idx = torch.arange(K)
    return idx.unsqueeze(0) <= first.unsqueeze(-1)


# ------------------------------------------------------------------ forwards
teacher = load(TCH)
T = []
t0 = time.time()
for i, e in enumerate(examples):
    T.append(forward_rows(teacher, e, "teacher"))
    print(f"[probe] teacher {i+1}/{len(examples)} ({e['grp']}, R={len(e['r_ids'])}) {time.time()-t0:.0f}s", flush=True)
del teacher

results = {}
for name, path in (("theta", CKPT), ("base", STU)):
    m = load(path)
    res = []
    t0 = time.time()
    for i, e in enumerate(examples):
        res.append(forward_rows(m, e, "student"))
        print(f"[probe] {name} {i+1}/{len(examples)} {time.time()-t0:.0f}s", flush=True)
    results[name] = res
    del m

# ------------------------------------------------------------------ supports + quantities
# c2's tau is a BATCH quantile over margins: compute it over all probe positions (the
# probe's population plays the micro-batch). c4's support is per position.
keeps = []
if ARM.startswith("c2"):
    margins = []
    for e, t, s in zip(examples, T, results["theta"]):
        lp_all = s["logits"] - s["lse"].unsqueeze(-1)
        pi = torch.gather(lp_all, 1, t["t_top_ids"]).exp()
        margins.append(torch.maximum(t["t_top_lp"].exp(), pi))
    frac = 1.0 - min(QB_BUDGET / K, 1.0)
    tau = torch.quantile(torch.cat([mm.flatten() for mm in margins]), frac)
    for mm in margins:
        kp = mm >= tau
        kp[:, 0] = True
        keeps.append(kp)
    print(f"[probe] c2 tau={float(tau):.4g} (frac {frac}); realised budget mean={np.mean([float(k.float().sum(-1).mean()) for k in keeps]):.2f}")
else:
    for e, t, s in zip(examples, T, results["theta"]):
        lp_all = s["logits"] - s["lse"].unsqueeze(-1)
        pi = torch.gather(lp_all, 1, t["t_top_ids"]).exp()
        keeps.append(support_c4(pi))
    print(f"[probe] c4 realised budget mean={np.mean([float(k.float().sum(-1).mean()) for k in keeps]):.2f}")

per = []
for i, (e, t, kp) in enumerate(zip(examples, T, keeps)):
    q_th = student_quantities(results["theta"][i]["logits"], results["theta"][i]["lse"], t["t_top_ids"], t["t_top_lp"], kp)
    q_0 = student_quantities(results["base"][i]["logits"], results["base"][i]["lse"], t["t_top_ids"], t["t_top_lp"], kp)
    h_t, h_0 = q_th["h"].numpy(), q_0["h"].numpy()
    A_S = (q_th["log_out"] - q_0["log_out"]).numpy()                                # log (1-m_theta)/(1-m_0), exact
    A_out = (torch.log(q_th["r"].clamp_min(1e-300)) - torch.log(q_0["r"].clamp_min(1e-300))).numpy()
    ident = np.log(np.clip(h_t, 1e-300, None) / np.clip(h_0, 1e-300, None)) - (A_S + A_out)
    lam_t = -np.cumsum(np.log1p(-np.clip(h_t, 0, 1 - 1e-12)))
    lam_0 = -np.cumsum(np.log1p(-np.clip(h_0, 0, 1 - 1e-12)))
    per.append(dict(grp=e["grp"], pid=e["pid"], correct=e["correct"], stopped=e["stopped"], R=len(e["r_ids"]),
                    h_theta=h_t, h_base=h_0, m_theta=q_th["m"].numpy(), m_base=q_0["m"].numpy(),
                    r_theta=q_th["r"].numpy(), r_base=q_0["r"].numpy(), A_S=A_S, A_out=A_out, ident_err=ident,
                    f_stop=q_th["f_stop"].numpy(), kl_S=q_th["kl"].numpy(), budget=kp.float().sum(-1).numpy(),
                    q_im_end=t["q_im_end"].numpy(), q_eot=t["q_eot"].numpy(), p_im_theta=q_th["p_im"].numpy(),
                    q_cov=q_th["q_cov"].numpy(), q_ET=(t["q_im_end"] + t["q_eot"]).numpy(), finish=("stop" if e["stopped"] else "length"),
                    lam_theta=lam_t, lam_base=lam_0))

np.savez_compressed(OUT, per=np.array(per, dtype=object), arm=ARM, step=STEP, ckpt_step=CKPT_STEP, K=K)
print(f"[probe] wrote {OUT}")

# ------------------------------------------------------------------ summary
def band(arr, R, which):
    if which == "stop":
        return arr[-1:]                                      # the position that emitted eot (stopped only)
    if which == "close":
        return arr[max(0, R - 201):R - 1]                    # 200 tokens before the stop
    if which == "body":
        return arr[: max(1, R - 201)]
    if which == "deep":
        return arr[2000:] if R > 2000 else arr[:0]           # 2k+ (repetition / deep failure regime)
    return arr


def lam_at(lam, t):
    return float(lam[min(t, len(lam)) - 1]) if len(lam) else float("nan")


print("\n=== VERDICT NUMBERS (" + f"{ARM}: texts Y{STEP}, weights theta{CKPT_STEP})")
stp = [p for p in per if p["stopped"]]
if stp:
    hs = np.array([p["h_theta"][-1] for p in stp]); h0 = np.array([p["h_base"][-1] for p in stp])
    AS = np.array([p["A_S"][-1] for p in stp]); AO = np.array([p["A_out"][-1] for p in stp])
    print(f"  [1] natural-stop hazard: h_theta med={np.median(hs):.3f} (mean {hs.mean():.3f})  h_base med={np.median(h0):.3f}  n={len(stp)}")
    print(f"  [2] stop-hazard attribution |A_S|/(|A_S|+|A_out|) = {np.abs(AS).sum()/max(np.abs(AS).sum()+np.abs(AO).sum(),1e-12):.2f}  "
          f"(A_S med {np.median(AS):+.3f}, A_out med {np.median(AO):+.3f})")
corr_p = [p for p in per if p["grp"] == "correct"]; tr_p = [p for p in per if p["grp"] == "trunc"]
if corr_p and tr_p:
    print(f"  [3] Lambda(6k or end): correct theta med={np.median([lam_at(p['lam_theta'], 6000) for p in corr_p]):.2f} base {np.median([lam_at(p['lam_base'], 6000) for p in corr_p]):.2f} | "
          f"trunc theta med={np.median([lam_at(p['lam_theta'], 6000) for p in tr_p]):.3f} base {np.median([lam_at(p['lam_base'], 6000) for p in tr_p]):.3f} | "
          f"q_T(E_T) med: correct-stop {np.median([p['q_ET'][-1] for p in corr_p]):.2f}, trunc-deep {np.median(np.concatenate([band(p['q_ET'], p['R'], 'deep') for p in tr_p] or [np.array([np.nan])])):.2g}")


print("\n=== identity check  log(h_theta/h_0) - (A_S + A_out): max |err| =",
      f"{max(np.abs(p['ident_err']).max() for p in per):.2e}")
for grp in ("correct", "wrong", "trunc"):
    G = [p for p in per if p["grp"] == grp]
    if not G:
        continue
    print(f"\n=== group {grp}: n={len(G)} mean R={np.mean([p['R'] for p in G]):.0f}")
    for which in (("stop", "close", "body") if grp != "trunc" else ("body", "deep")):
        hs_t = np.concatenate([band(p["h_theta"], p["R"], which) for p in G])
        hs_0 = np.concatenate([band(p["h_base"], p["R"], which) for p in G])
        ms_t = np.concatenate([band(p["m_theta"], p["R"], which) for p in G])
        ms_0 = np.concatenate([band(p["m_base"], p["R"], which) for p in G])
        AS = np.concatenate([band(p["A_S"], p["R"], which) for p in G])
        AO = np.concatenate([band(p["A_out"], p["R"], which) for p in G])
        FS = np.concatenate([band(p["f_stop"], p["R"], which) for p in G])
        qi = np.concatenate([band(p["q_ET"], p["R"], which) for p in G])
        bud = np.concatenate([band(p["budget"], p["R"], which) for p in G])
        qc = np.concatenate([band(p["q_cov"], p["R"], which) for p in G])
        if hs_t.size == 0:
            continue
        share = np.abs(AS).sum() / max(np.abs(AS).sum() + np.abs(AO).sum(), 1e-12)
        print(f"  [{which:<5}] h_theta med={np.median(hs_t):.3g} h_base med={np.median(hs_0):.3g} | m_theta={np.median(ms_t):.3f} m_base={np.median(ms_0):.3f} "
              f"| A_S med={np.median(AS):+.3f} A_out med={np.median(AO):+.3f} |A_S|/(|A_S|+|A_out|)={share:.2f} "
              f"| F_stop>0 frac={float((FS > 0).mean()):.2f} | q_T(E_T) med={np.median(qi):.2g} | |S_T| med={np.median(bud):.1f} q_cov med={np.median(qc):.3f}")
    # cumulative hazard at the observed stop / end, theta vs base
    Lt = [p["lam_theta"][-1] for p in G]
    L0 = [p["lam_base"][-1] for p in G]
    print(f"  Lambda(T_obs): theta med={np.median(Lt):.2f} base med={np.median(L0):.2f}  "
          f"(P(T<=T_obs) theta med={np.median(1 - np.exp(-np.array(Lt))):.2f}, base {np.median(1 - np.exp(-np.array(L0))):.2f})")
    # where does Lambda first cross 1 (median first-passage proxy)?
    def first_cross(lam, thr=1.0):
        idx = np.argmax(lam >= thr)
        return int(idx) if lam[-1] >= thr else -1
    ft = [first_cross(p["lam_theta"]) for p in G]
    f0 = [first_cross(p["lam_base"]) for p in G]
    print(f"  first t with Lambda>=1: theta {sorted(ft)} | base {sorted(f0)}   (-1 = never within the path)")
print("DONE")
