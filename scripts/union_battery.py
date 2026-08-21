#!/usr/bin/env python3
"""CPU battery for the union-support kernels (union_rkl / union_fkl, c5 family).

    PYTHONPATH=src SIMOPD_GATHER_EOS=1 SIMOPD_EOS_IDS=7 SIMOPD_EOS_TEACHER_IDS=7,9 \
        SIMOPD_MODEL_EOS_ID=7 python scripts/union_battery.py

Toy vocab (V=40, E_S={7}, E_T={7,9}, payload K=12 -> pool 10 + 2 extras). The payload
builder is c4_state_battery's, verbatim -- same teacher_patch/eos_gather round trip.
The extras-guard failure modes (shuffled ids, -inf, verl's dummy row) are shared code
copied from c4's kernel and battle-tested in that battery's case A/G; here the cases
are the union-specific math:

  A  runs + hand-check: renormalized RKL/FKL against an independent formula on a
     position built by hand
  B  dedup: overlapping student columns are masked; nothing counts twice
  C  terminator columns are EXACT: un_q_imend reads the gathered value, not q̂
  D  the mechanism: at a stop-state row (student mass on eot, outside the pool),
     union RKL pays a large penalty and pushes the eot logit DOWN; teacher-only c1
     on the same row barely sees it -- the blindness the union removes
  E  TERM_EVENT=1 refused loudly
  F  gradients flow, finite everywhere
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("SIMOPD_GATHER_EOS", "1")
os.environ.setdefault("SIMOPD_EOS_IDS", "7")
os.environ.setdefault("SIMOPD_MODEL_EOS_ID", "7")
os.environ.setdefault("SIMOPD_EOS_TEACHER_IDS", "7,9")
os.environ["SIMOPD_SHADOW"] = "0"

import torch  # noqa: E402

from simopd import eos_gather as EG  # noqa: E402
from simopd import teacher_patch  # noqa: E402
from simopd import topk_losses as T  # noqa: E402

PASS = 0


def ok(cond, msg):
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok  {msg}")


torch.manual_seed(0)
V, K = 40, 12
EXTRA = EG.extra_ids()
n = len(EXTRA)
POOL = K - n
TL = 12
EOT, IMEND = 7, 9


class LogprobsTensors(tuple):
    def __new__(cls, ids, lps, ranks, cu=None):
        self = tuple.__new__(cls, (ids, lps, ranks, cu))
        self.logprob_token_ids, self.logprobs, self.selected_token_ranks, self.cu_num_generated_tokens = ids, lps, ranks, cu
        return self


def fake_original(logprobs, num_logprobs, token_ids):
    topk_lp, topk_idx = torch.topk(logprobs, num_logprobs, dim=-1)
    tok = token_ids.unsqueeze(-1)
    tok_lp = logprobs.gather(-1, tok)
    ranks = (logprobs > tok_lp).sum(-1)
    return LogprobsTensors(torch.cat((tok, topk_idx), 1).to(torch.int32), torch.cat((tok_lp, topk_lp), 1), ranks)


def build_payload(teacher_logits, sampled):
    wrapped = EG._wrap(fake_original, LogprobsTensors)
    tw = wrapped(teacher_logits.log_softmax(-1), K, sampled)
    dicts = []
    for r in range(tw.logprob_token_ids.shape[0]):
        rk = [int(tw.selected_token_ranks[r])] + list(range(1, K + 1))
        dicts.append({int(t): types.SimpleNamespace(logprob=float(l), rank=rr)
                      for t, l, rr in zip(tw.logprob_token_ids[r].tolist(), tw.logprobs[r].tolist(), rk)})
    o = types.SimpleNamespace(prompt_token_ids=[0] + sampled.tolist(), prompt_logprobs=[None] + dicts)
    rr = {}
    teacher_patch._extract_with_eos(lambda *a: None)(o, K, rr)
    t_ids = torch.nested.nested_tensor([torch.tensor(rr["prompt_ids"][:-1])], layout=torch.jagged)
    t_lps = torch.nested.nested_tensor([torch.tensor(rr["prompt_logprobs"][:-1])], layout=torch.jagged)
    return t_lps, t_ids


CFG = types.SimpleNamespace(distillation_loss=types.SimpleNamespace(topk=K, log_prob_min_clamp=None))

# scenario: body rows prefer continuation ids 12..30; STOP row: student mass on eot(7),
# teacher wants im_end(9) -- eot is OUTSIDE the teacher pool there (q ~ e^-8 scale).
tch = torch.randn(TL, V) * 2.0
tch[:, [EOT, IMEND]] -= 8.0
STOP = 5
tch[STOP, IMEND] = 10.0                       # q(im_end) ~ 1 at the stop state
stu = torch.randn(1, TL, V) * 2.0
stu[0, :, EOT] -= 6.0
stu[0, STOP, :] = -8.0
stu[0, STOP, EOT] = 2.0                       # p(eot) dominates the student at STOP
sampled = torch.randint(12, 30, (TL,))
data = {"response_mask": torch.ones(1, TL, dtype=torch.bool),
        "responses": sampled.unsqueeze(0)}
t_lps, t_ids = build_payload(tch, sampled)

# ------------------------------------------------------------------ A. math ---
print("== A. runs + independent hand-check")
z = stu.clone().requires_grad_(True)
out_r = T.compute_union_rkl_topk(z, t_lps, t_ids, CFG, None, data=data)
out_f = T.compute_union_fkl_topk(stu.clone().requires_grad_(True), t_lps, t_ids, CFG, None, data=data)
ok(out_r["distillation_losses"].shape[-1] == TL and out_f["distillation_losses"].shape[-1] == TL,
   "both directions run under the carrier payload")

# independent recompute at one body row: rebuild the union support by hand from the
# same payload pieces and compare the losses.
row = 2
lp_t = t_lps.values()[row]
id_t = t_ids.values()[row].long()
pool_lp, pool_id = lp_t[:POOL], id_t[:POOL]           # sampled col is index K (width K+1)
x_lp, x_id = lp_t[POOL:K], id_t[POOL:K]
slp = stu[0].log_softmax(-1)[row]
stu_top = torch.topk(slp, POOL).indices
dup_s = torch.isin(stu_top, pool_id) | torch.isin(stu_top, torch.tensor(EXTRA))
dup_x = torch.isin(x_id, pool_id)
ids = torch.cat([pool_id, stu_top, x_id])
qh = pool_lp.min().expand(POOL)
q = torch.cat([pool_lp, qh, x_lp]).double()
p = slp[ids].double()
keep = torch.cat([torch.ones(POOL, dtype=torch.bool), ~dup_s, ~dup_x])
NEG = torch.finfo(torch.float64).min
qn = (q.masked_fill(~keep, NEG)).log_softmax(-1) if False else None
qm = q.masked_fill(~keep, NEG)
pm = p.masked_fill(~keep, NEG)
qn = qm - qm.logsumexp(-1)
pn = pm - pm.logsumexp(-1)
rkl_hand = (pn.exp() * (pn - qn))[keep].sum()
fkl_hand = (qn.exp() * (qn - pn))[keep].sum()
ok(abs(float(out_r["distillation_losses"][0, row]) - float(rkl_hand)) < 5e-3,
   f"RKL matches hand formula ({float(out_r['distillation_losses'][0, row]):.4f} vs {float(rkl_hand):.4f})")
ok(abs(float(out_f["distillation_losses"][0, row]) - float(fkl_hand)) < 5e-3,
   f"FKL matches hand formula ({float(out_f['distillation_losses'][0, row]):.4f} vs {float(fkl_hand):.4f})")

# ----------------------------------------------------------------- B. dedup ---
print("== B. dedup: nothing counts twice")
bud = out_r["un_budget"][0]
ok(float(bud.max()) <= POOL + POOL + n, "budget never exceeds pool + student-k + extras")
ok(float(bud.min()) >= POOL, "budget never below the pool alone")
exp_row = POOL + int((~dup_s).sum()) + int((~dup_x).sum())
ok(int(bud[row]) == exp_row, f"row budget equals hand count ({int(bud[row])} vs {exp_row})")

# ------------------------------------------------- C. terminator columns exact ---
print("== C. exact terminator columns")
q_imend_true = float(tch.log_softmax(-1)[STOP, IMEND].exp())
ok(abs(float(out_r["un_q_imend"][0, STOP]) - q_imend_true) < 1e-4,
   f"un_q_imend reads the gathered exact value ({float(out_r['un_q_imend'][0, STOP]):.4f} vs {q_imend_true:.4f})")
ok(float(out_r["un_p_eot"][0, STOP]) > 0.5, "un_p_eot sees the student's stop mass")

# ------------------------------------- D. the blindness the union removes ---
print("== D. dense stop artifact: union sees eot, teacher-only support cannot")
body = [i for i in range(TL) if i != STOP]
r_stop = float(out_r["distillation_losses"][0, STOP])
r_body = float(out_r["distillation_losses"][0, body].mean())
ok(r_stop > r_body + 3.0,
   f"union RKL pays heavily at the stop state ({r_stop:.2f} vs body {r_body:.2f})")
z2 = stu.clone().requires_grad_(True)
out2 = T.compute_union_rkl_topk(z2, t_lps, t_ids, CFG, None, data=data)
out2["distillation_losses"][0, STOP].backward()
ok(float(z2.grad[0, STOP, EOT]) > 0,
   "RKL gradient pushes the eot logit DOWN at the stop state (the dense artifact)")
z3 = stu.clone().requires_grad_(True)
out3 = T.compute_union_fkl_topk(z3, t_lps, t_ids, CFG, None, data=data)
out3["distillation_losses"][0, STOP].backward()
ok(float(z3.grad[0, STOP, IMEND]) < 0,
   "FKL gradient pulls the im_end logit UP at the stop state (teaching the convention)")
ok(float(z3.grad[0, STOP, EOT]) > 0,
   "FKL also pushes eot down there (the explicit column, vs b2's 3e-4 tail bucket)")

# --------------------------------------------------------------- E. refusals ---
print("== E. TERM_EVENT refused")
T.TERM_EVENT = True
try:
    T.compute_union_rkl_topk(stu.clone(), t_lps, t_ids, CFG, None, data=data)
    ok(False, "TERM_EVENT=1 must be refused")
except RuntimeError as e:
    ok("TERM_EVENT" in str(e), "TERM_EVENT=1 refused loudly")
finally:
    T.TERM_EVENT = False
bad_id = t_ids.values().clone()
bad_id[:, POOL], bad_id[:, POOL + 1] = bad_id[:, POOL + 1].clone(), bad_id[:, POOL].clone()
bad = torch.nested.nested_tensor([bad_id], layout=torch.jagged)
try:
    T.compute_union_rkl_topk(stu.clone(), t_lps, bad, CFG, None, data=data)
    ok(False, "shuffled extra ids must be refused")
except RuntimeError as e:
    ok("extra-id block" in str(e), "shuffled extra ids refused loudly")

# --------------------------------------------------------------- F. gradients ---
print("== F. gradient health")
z4 = stu.clone().requires_grad_(True)
loss = T.compute_union_rkl_topk(z4, t_lps, t_ids, CFG, None, data=data)["distillation_losses"].mean()
loss.backward()
ok(bool(torch.isfinite(z4.grad).all()), "RKL grads finite")
z5 = stu.clone().requires_grad_(True)
loss = T.compute_union_fkl_topk(z5, t_lps, t_ids, CFG, None, data=data)["distillation_losses"].mean()
loss.backward()
ok(bool(torch.isfinite(z5.grad).all()), "FKL grads finite")
ok(bool(torch.isfinite(out_r["distillation_losses"]).all()) and bool(torch.isfinite(out_f["distillation_losses"]).all()),
   "losses finite everywhere")

print(f"union battery {PASS}/{PASS} pass")
