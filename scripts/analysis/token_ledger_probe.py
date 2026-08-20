#!/usr/bin/env python3
"""逐位置台账:一遍前向,回答"哪些 token 贵、次序有没有翻、分层之后呢"。

用户 2026-08-21 提的三件事 —— (1) top1-agreement 与 rank_S(y_T^top1)、(2) 边距
M_t = z_S(y_T^top1) - max_{y != y_T^top1} z_S(y) 及其 M_t<0 -> M_t>0 的翻转、
(3) 不同时期哪些 token 扛着 loss —— 要的前向完全相同,所以不做三个探针,做一张
逐位置记录表,把问题全部变成这张表上的 groupby。

    ARM=c4_pi_tail_budget CKPT_STEP=150 TXT_STEP=50 PROBE_DEVICE=cuda:0 \
        python scripts/analysis/token_ledger_probe.py

为什么位置必须来自一份 FIXED 文本:两个 checkpoint 各自生成的轨迹是不同的,没有
共享位置,翻转就无从谈起。所以取某个参考 step 的 text dump 做 teacher-forcing,
两个 ckpt 都在同一批 token 上前向。代价写在这里:测到的是"参考策略访问的状态上"
的改进,不是"该 ckpt 自己会走的路上"的改进 —— 与 A 轴那个发现同一个口径问题
(状态来源决定一切),不要读成后者。

输出 parquet,一行一个 response 位置:
    pid grp correct pos pos_band token_id token_str tok_class is_rep
    t_top1 t_top1_lp t_ent q_sampled p_sampled dl_sampled kl_topk
    s_top1 rank_t_top1 M_t agree
"""
import glob
import os
import re
import sys
import time

D = os.environ.get("SIMOPD_STORE", "/mgfs/shared/Group_GY/changhao/simopd_data")
os.environ.setdefault("HF_HOME", f"{D}/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

torch.set_grad_enabled(False)
DEVICE = os.environ.get("PROBE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

ARM = os.environ.get("ARM", "c4_pi_tail_budget")
SEED = int(os.environ.get("SEED", "0"))
CKPT_STEP = int(os.environ.get("CKPT_STEP", "150"))       # 权重
TXT_STEP = int(os.environ.get("TXT_STEP", str(CKPT_STEP)))  # 文本(固定位置的来源)
TXT_ARM = os.environ.get("TXT_ARM", ARM)
N_CORRECT = int(os.environ.get("N_CORRECT", "12"))
N_WRONG = int(os.environ.get("N_WRONG", "8"))
N_TRUNC = int(os.environ.get("N_TRUNC", "4"))
MAXTOK = int(os.environ.get("MAXTOK", "4500"))
TRUNC_KEEP = int(os.environ.get("TRUNC_KEEP", "6000"))
K = int(os.environ.get("PAYLOAD_TOPK", "32"))
REP_N = int(os.environ.get("REP_N", "8"))
STU = "Qwen/Qwen3-1.7B-Base"
TCH = "Qwen/Qwen3-4B-Instruct-2507"
CKPT = f"{D}/ckpt/simopd/{ARM}_s{SEED}_16k/global_step_{CKPT_STEP}/actor/huggingface"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
IM_END, EOT = 151645, 151643
OUT = os.environ.get("LEDGER_OUT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs", "data", f"token_ledger_{ARM}_txt{TXT_STEP}_ckpt{CKPT_STEP}.parquet")

tok = AutoTokenizer.from_pretrained(STU)
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
prob_by_id = {ex["unique_id"]: ex["problem"] for ex in ds}


def newest_with_text(run_id, bench, step):
    roots = [os.environ["SIMOPD_EVAL_ROOT"]] if os.environ.get("SIMOPD_EVAL_ROOT") else []
    roots += [f"{D}/evals", f"{D}/archive/*/evals"]
    for root in roots:
        for h in sorted(glob.glob(f"{root}/{run_id}__{bench}__step{step}__seed*.parquet"))[::-1]:
            df = pd.read_parquet(h)
            if "response" in df.columns:
                return h, df
    return None, None


# ---- token 分类。逐 token 字面在 151k 词表上太稀,一半的桶只有个位数样本;类别是
# 稳的那一层,而且直接对应"数学 vs 散文 vs 排版"这个我们真正想问的问题。
_CLASS = [
    ("terminator", lambda s, i: i in (EOT, IM_END)),
    ("digit", lambda s, i: bool(re.fullmatch(r"\s*\d+\s*", s))),
    ("math_op", lambda s, i: bool(re.fullmatch(r"\s*[=+\-*/^<>%|()\[\]{}]+\s*", s))),
    ("latex", lambda s, i: "\\" in s or s.strip() in ("$", "$$")),
    ("newline", lambda s, i: "\n" in s),
    ("punct", lambda s, i: bool(re.fullmatch(r"\s*[.,;:!?'\"`]+\s*", s))),
    ("space", lambda s, i: bool(re.fullmatch(r"\s+", s))),
]


def classify(s, i):
    for name, fn in _CLASS:
        try:
            if fn(s, i):
                return name
        except Exception:
            pass
    return "word"


def pos_band(p):
    return "0-100" if p < 100 else "100-500" if p < 500 else "500-2k" if p < 2000 else "2k+"


def rep_mask(ids, n=REP_N):
    """位置 t 的结尾 n-gram 在同一条回复里更早出现过 -> True。与 kernel 里
    _repetition_mask 同一判据(滚动哈希 + 首次出现),这里是 numpy 版本。"""
    T = len(ids)
    out = np.zeros(T, dtype=bool)
    if T < n:
        return out
    a = np.asarray(ids, dtype=np.int64)
    h = np.zeros(T - n + 1, dtype=np.int64)
    for j in range(n):
        h = h * np.int64(1000003) + a[j:T - n + 1 + j]
    first = {}
    for idx, key in enumerate(h.tolist()):
        if key in first:
            out[idx + n - 1] = True
        else:
            first[key] = idx
    return out


path, df = newest_with_text(f"{TXT_ARM}_s{SEED}_16k", "math500", TXT_STEP)
assert df is not None, f"no eval parquet with text for {TXT_ARM} step {TXT_STEP}"
print(f"[ledger] texts {TXT_ARM}@{TXT_STEP}: {os.path.basename(path)} n={len(df)}; "
      f"weights {ARM}@{CKPT_STEP}; device={DEVICE}", flush=True)

stopped = df[(df.finish_reason == "stop") & (df.resp_len <= MAXTOK) & (df.resp_len > 50)]
corr = stopped[stopped.correct == 1].sample(n=min(N_CORRECT, int((stopped.correct == 1).sum())), random_state=0)
wrong = stopped[stopped.correct == 0].sample(n=min(N_WRONG, int((stopped.correct == 0).sum())), random_state=0)
trunc = df[df.finish_reason == "length"].sample(n=min(N_TRUNC, int((df.finish_reason == "length").sum())), random_state=0)
rows = [("correct", r) for _, r in corr.iterrows()] + [("wrong", r) for _, r in wrong.iterrows()] + \
       [("trunc", r) for _, r in trunc.iterrows()]
print(f"[ledger] sampled correct={len(corr)} wrong={len(wrong)} trunc={len(trunc)}", flush=True)

examples = []
for grp, r in rows:
    prompt = tok.apply_chat_template([{"role": "user", "content": prob_by_id[r.problem_id].strip() + " " + INSTRUCTION}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok.encode(prompt, add_special_tokens=False)
    r_ids = tok.encode(r.response, add_special_tokens=False)
    r_ids = r_ids[:TRUNC_KEEP] if grp == "trunc" else r_ids + [EOT]
    examples.append(dict(grp=grp, pid=r.problem_id, correct=int(r.correct), p_ids=p_ids, r_ids=r_ids))


def load(p):
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        p, dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16, low_cpu_mem_usage=True).to(DEVICE).eval()
    print(f"[ledger] loaded {p.split('/')[-2] if '/' in p else p} in {time.time()-t0:.0f}s", flush=True)
    return m


def logits_for(model, e):
    """位置 i(response 内 0-based)是预测 response token i 的那个分布,即 logits[P-1+i]。"""
    ids = e["p_ids"] + e["r_ids"]
    P, R = len(e["p_ids"]), len(e["r_ids"])
    x = torch.tensor([ids], device=DEVICE)
    return model(x).logits[0, P - 1:P - 1 + R].float()


recs = []
# 教师那一遍只取决于文本,不取决于被测的 checkpoint。整个时间序列(同一批文本 x
# 若干个 ckpt)共享同一份,所以缓存到盘上 —— 4B 的前向是这里最贵的一段,不缓存
# 等于每个 cell 白跑一次。key 里带文本来源和采样参数,换任何一个都会重算。
TCACHE = os.path.join(os.path.dirname(OUT),
                      f".tcache_{TXT_ARM}_s{SEED}_txt{TXT_STEP}_"
                      f"n{N_CORRECT}-{N_WRONG}-{N_TRUNC}_m{MAXTOK}_t{TRUNC_KEEP}_k{K}.npz")
t_cache = {}
if os.path.exists(TCACHE):
    z = np.load(TCACHE, allow_pickle=True)
    for k in range(len(examples)):
        t_cache[k] = dict(
            t_top1=z[f"t_top1_{k}"], t_top1_lp=z[f"t_top1_lp_{k}"],
            t_top_ids=torch.from_numpy(z[f"t_top_ids_{k}"]),
            t_top_lp=torch.from_numpy(z[f"t_top_lp_{k}"]).double(),
            t_ent=z[f"t_ent_{k}"], q_sampled=z[f"q_sampled_{k}"])
    print(f"[ledger] 教师缓存命中 {os.path.basename(TCACHE)}(跳过 4B 前向)", flush=True)
else:
    tch = load(TCH)
    for k, e in enumerate(examples):
        lg = logits_for(tch, e)
        lp = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
        top = torch.topk(lp, K, dim=-1)
        y = torch.tensor(e["r_ids"], device=DEVICE)
        t_cache[k] = dict(
            t_top1=top.indices[:, 0].cpu().numpy(),
            t_top1_lp=top.values[:, 0].cpu().numpy(),
            t_top_ids=top.indices.cpu(),
            t_top_lp=top.values.cpu().double(),
            t_ent=(-(lp.exp() * lp).sum(-1)).cpu().numpy(),
            q_sampled=lp.gather(1, y.unsqueeze(1)).squeeze(1).cpu().numpy(),
        )
        del lg, lp
        if DEVICE != "cpu":
            torch.cuda.empty_cache()
        print(f"[ledger] teacher {k+1}/{len(examples)}", flush=True)
    del tch
    if DEVICE != "cpu":
        torch.cuda.empty_cache()
    np.savez_compressed(TCACHE, **{f"{n}_{k}": (v[n].numpy() if torch.is_tensor(v[n]) else v[n])
                                   for k, v in t_cache.items()
                                   for n in ("t_top1", "t_top1_lp", "t_top_ids", "t_top_lp", "t_ent", "q_sampled")})
    print(f"[ledger] 教师缓存已写 {os.path.basename(TCACHE)}", flush=True)

stu = load(CKPT)
stash = []
for k, e in enumerate(examples):
    lg = logits_for(stu, e)
    lse = torch.logsumexp(lg, dim=-1, keepdim=True)
    lp = lg - lse
    y = torch.tensor(e["r_ids"], device=DEVICE)
    tc = t_cache[k]
    t1 = torch.tensor(tc["t_top1"], device=DEVICE, dtype=torch.long)

    # M_t:学生给教师第一名的 logit,减去它自己最好的"别人"。取 top-2 再按 argmax
    # 是不是 t_top1 来选对手 —— 直接用 max 会在两者相同时得到 0,把"已经对齐"和
    # "差 0 分"混为一谈。
    top2 = torch.topk(lg, 2, dim=-1)
    z_t1 = lg.gather(1, t1.unsqueeze(1)).squeeze(1)
    is_first = top2.indices[:, 0] == t1
    rival = torch.where(is_first, top2.values[:, 1], top2.values[:, 0])
    M_t = (z_t1 - rival)
    rank = (lg > z_t1.unsqueeze(1)).sum(1)                      # 0 = 学生也把它排第一

    p_sampled = lp.gather(1, y.unsqueeze(1)).squeeze(1)
    # 教师 top-K 上重归一化的 KL:各臂各自的支撑规则不同,这里固定 K 只为可比,
    # 不代表任何一个臂真正优化的量(它们的 loss 面板才是)。
    tt_ids = tc["t_top_ids"].to(DEVICE)
    q = tc["t_top_lp"].to(DEVICE).float()
    q = (q - torch.logsumexp(q, dim=-1, keepdim=True))
    p_on_raw = lp.gather(1, tt_ids)                             # 未重归一化:学生在池内各列的真实质量
    p_on = p_on_raw - torch.logsumexp(p_on_raw, dim=-1, keepdim=True)
    kl_topk = (q.exp() * (q - p_on)).sum(-1)

    # 臂自己的支撑上的重归一化 KL。这是关键的一列:c2/c4 根本看不到全词表 k1,
    # 它们只在自己选出来的支撑上算 KL,支撑外的 token 贡献恒为 0 —— 所以"哪些
    # token 扛着 c4 的 loss"这个问题,先要问"哪些 token 在支撑里"。
    # c4 的规则是逐位置的(教师秩前缀 / 学生质量到 1-eps),这里就地算;
    # c2 的 tau 是 BATCH 分位数,要等所有位置都过一遍才能定,所以留到循环外。
    # 必须用未重归一化的质量:c4 的规则是"累计学生质量达到 1-eps",而学生压在池外
    # 的那部分(停止地标处就是 eot)正是让规则够不到目标、退化成全池回落的原因。
    # 用重归一化后的 p_on,累计和恒为 1,那个回落就永远看不到了 —— 而它恰恰是我们
    # 要研究的现象。
    pi_on = p_on_raw.exp()                                      # [R,K] 学生在教师 top-K 上的真实质量
    y_col = (tt_ids == y.unsqueeze(1))
    in_pool = y_col.any(-1)
    y_idx = torch.where(in_pool, y_col.float().argmax(-1), torch.full_like(in_pool, -1, dtype=torch.long))
    stash.append(dict(k=k, q=q.cpu(), p_on=p_on.cpu(), pi_on=pi_on.cpu(),
                      t_top_lp=tc["t_top_lp"], y_idx=y_idx.cpu(), in_pool=in_pool.cpu()))

    rep = rep_mask(e["r_ids"])
    strs = tok.convert_ids_to_tokens(e["r_ids"])
    strs = [tok.convert_tokens_to_string([s]) for s in strs]
    R = len(e["r_ids"])
    dl = tc["q_sampled"] - p_sampled.cpu().numpy()
    recs.append(pd.DataFrame(dict(
        pid=e["pid"], grp=e["grp"], correct=e["correct"],
        pos=np.arange(R), pos_band=[pos_band(p) for p in range(R)],
        token_id=np.asarray(e["r_ids"]), token_str=strs,
        tok_class=[classify(s, i) for s, i in zip(strs, e["r_ids"])],
        is_rep=rep,
        t_top1=tc["t_top1"], t_top1_lp=tc["t_top1_lp"], t_ent=tc["t_ent"],
        # 教师在那个位置想写什么 —— "学生写了 X,教师要的是 Y" 才是可行动的一对,
        # 只知道 X 贵没法改任何东西。
        t_top1_str=[tok.convert_tokens_to_string([t]) for t in
                    tok.convert_ids_to_tokens([int(v) for v in tc["t_top1"]])],
        q_sampled=tc["q_sampled"], p_sampled=p_sampled.cpu().numpy(),
        dl_sampled=dl, kl_topk=kl_topk.cpu().numpy(),
        s_top1=top2.indices[:, 0].cpu().numpy(),
        rank_t_top1=rank.cpu().numpy(), M_t=M_t.cpu().numpy(),
        agree=(rank == 0).cpu().numpy(),
    )))
    del lg, lp, lse
    if DEVICE != "cpu":
        torch.cuda.empty_cache()
    print(f"[ledger] student {k+1}/{len(examples)}", flush=True)

# ---- 臂自己的支撑规则(移植自 c_stop_hazard_probe.py,同一套判据)
PI_TAIL_EPS = float(os.environ.get("SIMOPD_PI_TAIL_EPS", "0.05"))
QB_BUDGET = float(os.environ.get("SIMOPD_QB_TARGET_BUDGET", "8"))
RULE = os.environ.get("LEDGER_RULE", "c2" if ARM.startswith("c2") else
                      "c3" if ARM.startswith("c3") else
                      "c4" if ARM.startswith("c4") else "none")


def support_c4(pi):
    cum = pi.cumsum(-1)
    reached = cum >= (1.0 - PI_TAIL_EPS)
    first = torch.where(reached.any(-1), reached.float().argmax(-1),
                        torch.full(reached.shape[:-1], pi.shape[-1] - 1, dtype=torch.long))
    return torch.arange(pi.shape[-1]).unsqueeze(0) <= first.unsqueeze(-1)


keeps = {}
if RULE == "c2":
    # tau 是 micro-batch 上的分位数;探针的全体位置在这里扮演那个 micro-batch。
    marg = [torch.maximum(st["t_top_lp"].exp().float(), st["pi_on"]) for st in stash]
    frac = 1.0 - min(QB_BUDGET / K, 1.0)
    tau = torch.quantile(torch.cat([m.flatten() for m in marg]), frac)
    for st, m in zip(stash, marg):
        kp = m >= tau
        kp[:, 0] = True                     # 教师第一名恒在支撑里,否则会出现空支撑
        keeps[st["k"]] = kp
    print(f"[ledger] c2 tau={float(tau):.4g} (frac {frac}); 实际预算均值="
          f"{float(torch.cat([keeps[st['k']].float().sum(-1) for st in stash]).mean()):.2f}", flush=True)
elif RULE == "c4":
    for st in stash:
        keeps[st["k"]] = support_c4(st["pi_on"])
    print(f"[ledger] c4 实际预算均值="
          f"{float(torch.cat([keeps[st['k']].float().sum(-1) for st in stash]).mean()):.2f}", flush=True)
elif RULE == "c3":
    # thunlp 交集:学生 top-K 与教师 top-K 的交 —— 学生在教师池内的质量排前 K 的那些
    for st in stash:
        r = st["pi_on"].argsort(dim=-1, descending=True).argsort(dim=-1)
        keeps[st["k"]] = r < K
else:
    for st in stash:
        keeps[st["k"]] = torch.ones_like(st["pi_on"], dtype=torch.bool)

for st, frame in zip(stash, recs):
    kp = keeps[st["k"]]
    q = st["q"].clone()
    p = st["p_on"].clone()
    NEG = torch.finfo(q.dtype).min
    q = torch.where(kp, q, torch.full_like(q, NEG))
    p = torch.where(kp, p, torch.full_like(p, NEG))
    q = q - torch.logsumexp(q, -1, keepdim=True)
    p = p - torch.logsumexp(p, -1, keepdim=True)
    kl = (q.exp() * (q - p)).sum(-1)
    yi = st["y_idx"]
    in_sup = torch.zeros_like(st["in_pool"])
    ok = st["in_pool"]
    if bool(ok.any()):
        in_sup[ok] = kp[ok, yi[ok].clamp(min=0)]
    frame["kl_arm"] = kl.numpy()
    frame["sup_size"] = kp.sum(-1).numpy()
    frame["in_pool"] = st["in_pool"].numpy()
    frame["in_sup"] = in_sup.numpy()

out = pd.concat(recs, ignore_index=True)
out["rule"] = RULE
out["arm"] = ARM
out["ckpt_step"] = CKPT_STEP
out["txt_step"] = TXT_STEP
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_parquet(OUT, index=False)
print(f"[ledger] {len(out)} 行 -> {OUT}")
print(f"[ledger] rule={RULE} 支撑均值={out.sup_size.mean():.2f} 采样 token 在支撑内={out.in_sup.mean():.3f}")
print(f"[ledger] agree={out.agree.mean():.3f}  M_t>0={float((out.M_t > 0).mean()):.3f}  "
      f"rank_t_top1 中位={int(out.rank_t_top1.median())}  |dl| 均值={out.dl_sampled.abs().mean():.3f}")
