#!/usr/bin/env python3
"""CPU battery for the termination family -- N2 (n2_termcal) and N0 (n0_termfix): the
exact termination payload contract end to end.

    PYTHONPATH=src SIMOPD_GATHER_EOS=1 SIMOPD_EOS_IDS=7 SIMOPD_EOS_TEACHER_IDS=7,9 \\
        SIMOPD_EOS_DIAG_IDS=11 python scripts/n2_eos_battery.py       # asymmetric sets (the arm)
    PYTHONPATH=src SIMOPD_GATHER_EOS=1 SIMOPD_EOS_IDS=7 SIMOPD_EOS_TEACHER_IDS= \\
        SIMOPD_EOS_DIAG_IDS=9 python scripts/n2_eos_battery.py        # symmetric single token

Sections (torch required; verl only for the kernel section, skipped if absent):
  A  margin/BCE math: stop_margin == log p/(1-p) across [1e-12, 1-1e-7];
     dBCE/dm == p - q; finite at both ends
  B  eos_gather wrapper vs a faithful fake of vLLM's gather_logprobs:
     width K+1, block == exact full-softmax values, no duplicated column, ranks
     and the actual-token column untouched -- stop id inside/outside the top-K,
     actual token == stop id
  C  teacher_patch N2 reader over the wrapper's own output (dict built with the
     last-write-wins rule vLLM uses): rows = [top-(K-n) by value | extras | sampled],
     no -inf, no stop id in the top block, sampled == exact
  D  kernel: distillation_losses == vanilla k1 on the sampled column; eos_term ==
     BCE(m, q_exact) with q over E_T and m over E_S; d term / d z_stop == p - q;
     a teacher-only member of E_T sits on the CONTINUE side of the student's
     log-odds (no loophole); per-member panels exact; refuses -inf and a shuffled block
  E  _repetition_mask flags a planted repeated 8-gram and nothing before it
  F  N0 kernel (k1_termfix): distillation_losses == vanilla k1 everywhere except at
     the sampled-stop position, where it is the EVENT-level log p(E_S) - log q(E_T);
     eos_dl_raw carries the token-level value; no eos_term
"""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.setdefault("SIMOPD_GATHER_EOS", "1")
os.environ.setdefault("SIMOPD_EOS_IDS", "7")
os.environ.setdefault("SIMOPD_MODEL_EOS_ID", "7")     # toy vocab: the 'model eos' is id 7
os.environ.setdefault("SIMOPD_EOS_TEACHER_IDS", "7,9")
os.environ.setdefault("SIMOPD_EOS_DIAG_IDS", "11")
os.environ["SIMOPD_SHADOW"] = "0"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from simopd import eos_gather as EG  # noqa: E402

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
STOP, TCH, DIAG = EG.stop_ids(), EG.teacher_ids(), EG.diag_ids()
UNION, EXTRA = EG.union_ids(), EG.extra_ids()
n = len(EXTRA)
ok(EXTRA == UNION + DIAG and UNION[:len(STOP)] == STOP and set(UNION) == set(STOP) | set(TCH),
   f"env parsed: E_S={STOP} E_T={TCH} diag={DIAG} -> block {EXTRA}")
ok(all(EXTRA[c] in TCH for c in EG.teacher_cols()) and sorted(EXTRA[c] for c in EG.teacher_cols()) == sorted(TCH),
   "teacher_cols index exactly E_T inside the block")
# the contract guard: E_S must equal what ends a rollout (model eos + SIMOPD_STOP_IDS)
_saved = os.environ.get("SIMOPD_STOP_IDS")
os.environ["SIMOPD_STOP_IDS"] = ",".join(map(str, STOP + [13]))        # a contract id E_S does not name
try:
    EG.stop_ids(); ok(False, f"guard must refuse E_S={STOP} under stop contract {STOP + [13]}")
except RuntimeError as e:
    ok("rollout stop set" in str(e), "guard refuses E_S that misses a contract stop id")
os.environ["SIMOPD_STOP_IDS"] = ",".join(map(str, STOP[1:])) or "off"  # exactly E_S beyond the model eos
ok(EG.rollout_stop_set() == STOP and EG.stop_ids() == STOP,
   f"contract {os.environ['SIMOPD_STOP_IDS']!r} -> rollout stop set == E_S == {STOP}, accepted")
if _saved is None:
    del os.environ["SIMOPD_STOP_IDS"]
else:
    os.environ["SIMOPD_STOP_IDS"] = _saved

# ------------------------------------------------------------------ A. math ---
from simopd.topk_losses import stop_margin, _log1mexp  # noqa: E402

print("== A. margin / BCE math")
p = torch.tensor([1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999, 1 - 1e-7], dtype=torch.float64)
m = stop_margin(p.log())
ref = (p / (1 - p)).log()
ok(torch.allclose(m, ref, rtol=1e-6, atol=1e-9), "stop_margin == log p/(1-p) over 12 decades")
ok(torch.isfinite(_log1mexp(torch.tensor([-1e-9, -1e-3, -1.0, -50.0, -700.0], dtype=torch.float64))).all(),
   "log1mexp finite from -1e-9 to -700")
mm = torch.tensor([-30.0, -3.0, 0.0, 2.0, 30.0], dtype=torch.float64, requires_grad=True)
q = torch.tensor([0.0, 0.2, 0.5, 0.9, 1.0], dtype=torch.float64)
F.binary_cross_entropy_with_logits(mm, q, reduction="sum").backward()
ok(torch.allclose(mm.grad, torch.sigmoid(mm.detach()) - q, atol=1e-12), "dBCE/dm == p - q (incl. q=0 and q=1)")
big = stop_margin(torch.tensor([-200.0], dtype=torch.float64))
ok(torch.isfinite(F.binary_cross_entropy_with_logits(big, torch.tensor([0.3], dtype=torch.float64))).all(),
   "term finite at p ~ e^-200")

# ------------------------------------------------- B. the vLLM sampler wrapper ---
print("== B. eos_gather wrapper vs faithful fake of vLLM gather_logprobs")


class LogprobsTensors(tuple):
    """Shape-compatible stand-in for vllm.v1.outputs.LogprobsTensors (a NamedTuple)."""
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


wrapped = EG._wrap(fake_original, LogprobsTensors)
rows = 6
logits = torch.randn(rows, V) * 3
# row 0: stop id 7 forced INTO the top-K; row 1: forced far outside; row 2: actual token IS the stop id
logits[0, 7] = 50.0
logits[1, 7] = -50.0
logits[2, 7] = 0.0
logits[3, 9] = 60.0            # diag id inside top-K
lp_full = logits.log_softmax(-1)
actual = torch.tensor([3, 4, 7, 5, 9, 1])
out = wrapped(lp_full, K, actual)
ids, lps, ranks = out.logprob_token_ids, out.logprobs, out.selected_token_ranks
ok(ids.shape == (rows, K + 1) and lps.shape == (rows, K + 1), f"width K+1 = {K + 1}")
ok((ids[:, 0].long() == actual).all() and torch.allclose(lps[:, 0], lp_full.gather(-1, actual.unsqueeze(-1)).squeeze(-1)),
   "actual-token column untouched")
ok((ranks == (lp_full > lp_full.gather(-1, actual.unsqueeze(-1))).sum(-1)).all(), "ranks == exact rank on the unmasked full softmax")
blk_ids, blk_lps = ids[:, K + 1 - n:], lps[:, K + 1 - n:]
ok((blk_ids.long() == torch.tensor(EXTRA)).all(), "block ids in env order on every row")
ok(torch.allclose(blk_lps, lp_full[:, EXTRA]), "block values == exact full-softmax logprobs (incl. rank-1 and rank-last cases)")
top_ids = ids[:, 1:K + 1 - n].long()
ok(not torch.isin(top_ids, torch.tensor(EXTRA)).any(), "no stop/diag id inside the top block (no duplicated column)")
ok(top_ids.shape[1] == K - n, f"top block width K-n = {K - n}")
for r in range(rows):
    ok(len(set(ids[r].tolist()) | set()) >= K - n + n, f"row {r}: block ∪ top has no repeated id beyond the actual-token dup")
    exp_top = [t for t in torch.topk(lp_full[r], K).indices.tolist() if t not in EXTRA][:K - n]
    ok(top_ids[r].tolist() == exp_top, f"row {r}: top block == top-K minus extras, rank order kept")
untouched = wrapped(lp_full, 0, actual)
ok(untouched.logprobs.shape[1] == 1, "num_logprobs=0 (estimator path) is delegated unchanged")

# ------------------------------------------------- C. the teacher_patch reader ---
print("== C. teacher_patch N2 reader")
from simopd import teacher_patch  # noqa: E402


class Lp:
    def __init__(self, logprob, rank):
        self.logprob, self.rank = logprob, rank


def build_dicts(ids, lps, ranks, K):
    """vLLM's append_logprobs_for_next_position: positional ranks, dict comprehension
    -> last write wins on a duplicated id."""
    out = []
    for r in range(ids.shape[0]):
        rk = [int(ranks[r])] + list(range(1, K + 1))
        out.append({int(t): Lp(float(l), rr) for t, l, rr in zip(ids[r].tolist(), lps[r].tolist(), rk)})
    return out


dicts = build_dicts(ids, lps, ranks, K)
output = types.SimpleNamespace(prompt_token_ids=[0] + actual.tolist(), prompt_logprobs=[None] + dicts)
res = {}
teacher_patch._extract_with_eos(lambda *a: None)(output, K, res)
tid, tlp = torch.tensor(res["prompt_ids"][:-1]), torch.tensor(res["prompt_logprobs"][:-1])
ok(tid.shape == (rows, K + 1), f"reader rows are K+1 = {K + 1} wide (+ dummy last row {len(res['prompt_ids'])})")
ok(torch.isfinite(tlp).all(), "no -inf anywhere")
ok((tid[:, K] == actual).all() and torch.allclose(tlp[:, K], lp_full.gather(-1, actual.unsqueeze(-1)).squeeze(-1)),
   "sampled column == actual token, exact value")
ok((tid[:, K - n:K] == torch.tensor(EXTRA)).all() and torch.allclose(tlp[:, K - n:K], lp_full[:, EXTRA]),
   "extra block == env ids, exact values")
top_r = tid[:, :K - n]
ok(not torch.isin(top_r, torch.tensor(EXTRA)).any(), "top block has no stop/diag id")
ok((tlp[:, :K - n].diff(dim=-1) <= 1e-7).all(), "top block sorted by value desc")
for r in range(rows):
    exp_top = [t for t in torch.topk(lp_full[r], K).indices.tolist() if t not in EXTRA][:K - n]
    ok(sorted(top_r[r].tolist()) == sorted(exp_top), f"row {r}: top block == exact top-(K-n) non-stop set")

# ------------------------------------------------------------- D. the kernel ---
print("== D. kernel (needs verl importable)")
try:
    import verl  # noqa: F401
    HAVE_VERL = True
except Exception as e:  # pragma: no cover
    HAVE_VERL = False
    print(f"  skip  verl not importable here ({e!r}); run this section on the cluster venv")

if HAVE_VERL:
    from simopd import topk_losses as T
    T_len = 9
    z = (torch.randn(1, T_len, V) * 2).requires_grad_(True)
    tlog = (torch.randn(T_len, V) * 2).log_softmax(-1)
    samp = torch.randint(0, V, (T_len,))
    samp[3] = STOP[0]                                  # a sampled-EOS event
    tw = wrapped(tlog, K, samp)
    dd = build_dicts(tw.logprob_token_ids, tw.logprobs, tw.selected_token_ranks, K)
    o = types.SimpleNamespace(prompt_token_ids=[0] + samp.tolist(), prompt_logprobs=[None] + dd)
    rr = {}
    teacher_patch._extract_with_eos(lambda *a: None)(o, K, rr)
    t_ids = torch.nested.nested_tensor([torch.tensor(rr["prompt_ids"][:-1])], layout=torch.jagged)
    t_lps = torch.nested.nested_tensor([torch.tensor(rr["prompt_logprobs"][:-1])], layout=torch.jagged)
    cfg = types.SimpleNamespace(distillation_loss=types.SimpleNamespace(topk=K, log_prob_min_clamp=None))
    out = T.compute_termcal_topk(z, t_lps, t_ids, cfg, None, data=None)
    lse = torch.logsumexp(z.float(), -1)
    stu_samp = z.float()[0, torch.arange(T_len), samp] - lse[0]
    ok(torch.allclose(out["distillation_losses"][0], stu_samp - tlog[torch.arange(T_len), samp], atol=1e-5),
       "distillation_losses == vanilla k1 (log p_S(y) - log p_T(y)) on the sampled column")
    q_exact = tlog[:, TCH].exp().sum(-1)
    ok(torch.allclose(out["eos_q"][0], q_exact, atol=1e-6), "eos_q == exact full-softmax teacher termination mass over E_T")
    for i, tid_ in enumerate(UNION):
        ok(torch.allclose(out[f"eos_qm_{i}"][0], tlog[:, tid_].exp(), atol=1e-6), f"eos_qm_{i} == exact p_T(id {tid_})")
        ok(torch.allclose(out[f"eos_pm_{i}"][0], (z.float()[0][:, tid_] - lse[0]).exp(), atol=1e-5), f"eos_pm_{i} == exact p_S(id {tid_})")
    log_p = torch.logsumexp(z.float()[0][:, STOP], -1) - lse[0]
    term_ref = F.binary_cross_entropy_with_logits(stop_margin(log_p), q_exact, reduction="none")
    ok(torch.allclose(out["eos_term"][0], term_ref, atol=1e-5), "eos_term == BCEWithLogits(m, q_exact)")
    if DIAG:
        ok(torch.allclose(out["eos_diag_q_0"][0], tlog[:, DIAG[0]].exp(), atol=1e-6), "diag column == exact p_T(diag id)")
    ok(out["eos_sampled_is_stop"][0].sum() == 1, "sampled-EOS event counted once")
    g, = torch.autograd.grad(out["eos_term"].sum(), z)
    p_stop = log_p.exp()
    p_full = torch.softmax(z.float()[0], -1)
    for e_ in STOP:   # inside E_S the push is distributed by the student's own conditional p(e|stop)
        want_g = (p_stop - q_exact) * p_full[:, e_] / p_stop
        ok(torch.allclose(g[0][:, e_], want_g, atol=1e-4),
           f"d term / d z_{e_} == (p - q) * p(e|stop) exactly (single-member E_S: == p - q, bounded in [-1,1])")
    for tid_ in [t for t in TCH if t not in STOP]:
        want_g = -(p_stop - q_exact) * p_full[:, tid_] / (1 - p_stop)
        ok(torch.allclose(g[0][:, tid_], want_g, atol=1e-4),
           f"teacher-only terminator {tid_} gets the CONTINUE-side gradient -(p-q)*p_v/(1-p): student mass there never counts as stopped")
    # refusals
    bad = torch.tensor(rr["prompt_logprobs"][:-1]); bad[2, K - n] = float("-inf")
    try:
        T.compute_termcal_topk(z, torch.nested.nested_tensor([bad], layout=torch.jagged), t_ids, cfg, None, data=None)
        ok(False, "kernel must refuse -inf in the stop block")
    except RuntimeError as e:
        ok("fabricated q=0" in str(e), "kernel refuses -inf in the stop block (no silent q=0)")
    sh = torch.tensor(rr["prompt_ids"][:-1]); sh[:, K - n], sh[:, K - n + 1] = sh[:, K - n + 1].clone(), sh[:, K - n].clone()
    try:
        T.compute_termcal_topk(z, t_lps, torch.nested.nested_tensor([sh], layout=torch.jagged), cfg, None, data=None)
        ok(False, "kernel must refuse a block whose ids are not in env order")
    except RuntimeError as e:
        ok("not running eos_gather" in str(e), "kernel refuses a block whose ids are out of order")

    # ------------------------------------------------------------ F. N0 kernel ---
    print("== F. N0 kernel k1_termfix (event-level Delta-ell at the sampled stop only)")
    o0 = T.compute_termfix_topk(z, t_lps, t_ids, cfg, None, data=None)
    van = stu_samp - tlog[torch.arange(T_len), samp]                    # vanilla k1 = log p_S(y) - log p_T(y)
    stop_pos = torch.isin(samp, torch.tensor(STOP))
    ev = log_p - torch.logsumexp(tlog[:, TCH], -1)                       # log p_S(E_S) - log q_T(E_T)
    want = torch.where(stop_pos, ev, van)
    ok(torch.allclose(o0["distillation_losses"][0], want, atol=1e-5),
       "k1_termfix == vanilla k1 off the stop, event-level log p(E_S) - log q(E_T) at the sampled stop")
    ok(torch.allclose(o0["distillation_losses"][0][~stop_pos], van[~stop_pos], atol=1e-5),
       "every non-stop position bit-identical to vanilla")
    ok(torch.allclose(o0["eos_dl_raw"][0], -van, atol=1e-5), "eos_dl_raw == vanilla's token-level Delta-ell (the receipt)")
    ok("eos_term" not in o0, "N0 carries no calibration term")
    ok(torch.allclose(o0["eos_q"][0], q_exact, atol=1e-6) and torch.allclose(o0["eos_p"][0], log_p.exp(), atol=1e-5),
       "N0 panels: eos_q / eos_p identical to N2's")
    ok(o0["eos_sampled_is_stop"][0].sum() == 1, "N0: sampled-stop event counted once")
    # the audit's arithmetic in miniature (needs a teacher-only terminator, i.e. the
    # asymmetric configuration): make the teacher hate the student's stop token but love
    # the teacher-only terminator at the stop position -> vanilla says -big, N0 says ~0
    tonly = [t for t in TCH if t not in STOP]
    if tonly:
        tl2 = tlog.clone(); row = tl2[3].clone()
        row[STOP[0]] = -30.0; row[tonly[0]] = 0.0
        tl2[3] = row.log_softmax(-1)
        tw2 = wrapped(tl2, K, samp)
        dd2 = build_dicts(tw2.logprob_token_ids, tw2.logprobs, tw2.selected_token_ranks, K)
        o2 = types.SimpleNamespace(prompt_token_ids=[0] + samp.tolist(), prompt_logprobs=[None] + dd2)
        rr2 = {}
        teacher_patch._extract_with_eos(lambda *a: None)(o2, K, rr2)
        t_ids2 = torch.nested.nested_tensor([torch.tensor(rr2["prompt_ids"][:-1])], layout=torch.jagged)
        t_lps2 = torch.nested.nested_tensor([torch.tensor(rr2["prompt_logprobs"][:-1])], layout=torch.jagged)
        o0b = T.compute_termfix_topk(z, t_lps2, t_ids2, cfg, None, data=None)
        dl_raw = float(o0b["eos_dl_raw"][0][3].detach())
        dl_fix = float((-o0b["distillation_losses"][0][3]).detach())
        want_fix = float(tl2[3][TCH].exp().sum().log()) - float(log_p[3].detach())
        ok(dl_raw < -20 and abs(dl_fix - want_fix) < 1e-4,
           f"mismatch case: token-level Delta-ell {dl_raw:.1f} (vanilla) vs event-level {dl_fix:.2f} (N0)")
    else:
        print("  skip  mismatch mini-case (no teacher-only terminator in this configuration)")

    # ------------------------------------------------------- E. repetition mask ---
    print("== E. _repetition_mask")
    from simopd import losses as L
    resp = torch.randint(100, 200, (2, 40))
    resp[0, 20:30] = resp[0, 5:15]                      # a planted 10-token repeat
    mask = torch.ones(2, 40, dtype=torch.bool); mask[1, 30:] = False
    data = {"responses": resp}
    rep = L._repetition_mask(data, mask, n=8)
    ok(rep is not None and rep.shape == mask.shape, "mask shape")
    ok(rep[0, :27].sum() == 0 and rep[0, 27:30].all(), "row 0: only the 8-gram-completing positions of the planted repeat flagged (27,28,29)")
    ok(rep[1].sum() == 0, "row 1: random ids, nothing flagged")
    ok(L._repetition_mask({"responses": None}, mask) is None, "no responses -> None (callers zero-fill)")

print(f"\nALL PASS ({PASS} checks)")
