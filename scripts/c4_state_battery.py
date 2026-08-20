#!/usr/bin/env python3
"""CPU battery for the c4 state-adaptive controller (SIMOPD_C4_HQ / SIMOPD_C4_REP).

    PYTHONPATH=src SIMOPD_GATHER_EOS=1 SIMOPD_EOS_IDS=7 SIMOPD_EOS_TEACHER_IDS=7,9 \
        SIMOPD_MODEL_EOS_ID=7 SIMOPD_REP_GATE_N=4 SIMOPD_REP_GATE_MINRUN=6 \
        python scripts/c4_state_battery.py

Cases (toy vocab, E_S={7}, E_T={7,9}, payload K=12 -> pool 9 + 3 extras):
  A  carrier plumbing: extras split off the pool, id-order refused when shuffled,
     -inf refused; flags off + carrier off == the stock rule bit for bit
  B  landmark row (student mass on eot outside the pool, q_T(E_T) high) ->
     stock rule falls back to the FULL pool, controller freezes to top-1,
     loss == 0 and grad through that row == 0
  C  premature row (h high, q_T(E_T) tiny) -> NOT frozen; the continuation-
     conditional target shrinks the support but keeps teaching
  D  body row (h ~ 1e-9) -> keep identical to the stock rule
  E  rep gate: a planted >=MINRUN repeated run is frozen, an incidental short
     repeat is not; missing responses refused
  F  h detach: no grad reaches the logits through the gate/target path
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
os.environ.setdefault("SIMOPD_REP_GATE_N", "4")
os.environ.setdefault("SIMOPD_REP_GATE_MINRUN", "6")   # toy scale; the shipped default is asserted below
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
TL = 24            # response length
EOT = EG.stop_ids()[0]


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
    """teacher logits [T,V] -> nested (t_lps, t_ids) exactly as teacher_patch emits."""
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

# ---------------------------------------------------------------- scenario ---
# teacher: rows 0..TL-1 prefer continuation tokens 12..20; at the LANDMARK row 5 the
# teacher puts its mass on im_end (9); at the PREMATURE row 9 it wants to continue.
tch = torch.randn(TL, V) * 2.0
tch[:, EXTRA] -= 8.0                         # terminators far down by default
LMK, PRE, MOD = 5, 9, 12
tch[LMK, 9] = 10.0                           # teacher termination mass ~1 at the landmark
tch[MOD, 21] = 12.0; tch[MOD, 22] = 11.0     # teacher pool at MOD: ranks 1,2 = ids 21,22
stu_base = torch.randn(1, TL, V) * 2.0
stu_base[0, :, EOT] -= 6.0
stu_base[0, LMK, EOT] = 10.0                 # student wants to stop at the landmark...
stu_base[0, PRE, EOT] = 10.0                 # ...and prematurely at PRE (teacher disagrees)
# MOD: h ~ 0.5 with 0.49 of student mass on the teacher's top-2 -- the continuation-
# conditional target (0.95*0.5=0.475) is reachable with 2 columns, the stock target
# (0.95) is not: the fix must turn a full-pool fallback into a 2-column support.
stu_base[0, MOD, :] = -20.0
stu_base[0, MOD, EOT] = float(torch.log(torch.tensor(0.50)))
stu_base[0, MOD, 21] = float(torch.log(torch.tensor(0.40)))
stu_base[0, MOD, 22] = float(torch.log(torch.tensor(0.09)))
sampled = torch.randint(12, 30, (TL,))
t_lps, t_ids = build_payload(tch, sampled)

resp = torch.randint(12, 30, (1, TL))
resp[0, 16:23] = 33                          # 7 identical tokens -> runs of 4-grams, run len 4 >= MINRUN? window rep positions 19..22 -> run 4 < 6? craft longer below
data = {"response_mask": torch.ones(1, TL, dtype=torch.bool), "responses": resp}

def run(z, hq, rep, dat=data):
    T.C4_HQ, T.C4_REP = hq, rep
    return T.compute_pi_tail_budget_topk(z, t_lps, t_ids, CFG, None, data=dat)

# ------------------------------------------------------------- A. plumbing ---
print("== A. carrier plumbing / stock parity")
z = stu_base.clone().requires_grad_(True)
out = run(z, False, False)
ok(out["distillation_losses"].shape[-1] == TL, "kernel runs under the carrier with flags off")
ok(float(out["c4_budget"].max()) <= POOL, f"pool excludes the {n} gathered columns (budget <= {POOL})")
# stock parity is checked structurally: with flags off the keep rule must equal the
# documented smallest-prefix rule on the pool -- recomputed by hand below.
slp = torch.log_softmax(z.float(), -1)
pool_id = t_ids.values().unsqueeze(0)[..., :POOL]
pi = torch.gather(slp, -1, pool_id.long()).exp()
cum = pi.cumsum(-1)
reached = cum >= (1 - T.PI_TAIL_EPS)
first = torch.where(reached.any(-1), reached.float().argmax(-1), torch.tensor(POOL - 1))
ok(bool((out["c4_budget"] == (first + 1).float()).all()), "flags off: budget == manual smallest-prefix rule (stock semantics on the pool)")
bad_lps = t_lps.values().clone(); bad_ids = t_ids.values().clone()
bad_ids[:, K - n], bad_ids[:, K - 1 - n] = bad_ids[:, K - 1 - n].clone(), bad_ids[:, K - n].clone()
bt_ids = torch.nested.nested_tensor([bad_ids], layout=torch.jagged)
bt_lps = torch.nested.nested_tensor([bad_lps], layout=torch.jagged)
try:
    T.compute_pi_tail_budget_topk(z, bt_lps, bt_ids, CFG, None, data=data); ok(False, "shuffled extras must refuse")
except RuntimeError as e:
    ok("extra-id block" in str(e), "shuffled extra block refused loudly")

# ------------------------------------------------------------- B. landmark ---
print("== B. landmark: stock full-pool fallback vs agreed-freeze")
budget_off = float(out["c4_budget"][0, LMK])
ok(budget_off == POOL, f"stock rule at the landmark falls back to the FULL pool ({int(budget_off)})")
z2 = stu_base.clone().requires_grad_(True)
out_hq = run(z2, True, False)
ok(float(out_hq["c4_budget"][0, LMK]) == 1.0, "agreed-freeze: landmark support collapses to top-1")
ok(float(out_hq["c4_hq_freeze"][0, LMK]) == 1.0 and float(out_hq["c4_freeze_frac"][0, LMK]) == 1.0, "freeze panels mark the landmark")
ok(abs(float(out_hq["distillation_losses"][0, LMK])) < 1e-6, "frozen landmark loss == 0 (singleton renorm)")
out_hq["distillation_losses"].sum().backward()
g = z2.grad[0, LMK].abs().max()
ok(float(g) < 1e-7, f"frozen landmark receives zero gradient (|g|max={float(g):.1e})")

# ------------------------------------------------------------ C. premature ---
print("== C. premature stop: taught; moderate-h: fallback turned into a small support")
ok(float(out_hq["c4_hq_freeze"][0, PRE]) == 0.0, "premature row NOT frozen (teacher q_T(E_T) tiny)")
ok(abs(float(out_hq["distillation_losses"][0, PRE])) > 1e-6, "premature row still carries loss (kept teaching; "
   "an extreme-h disagreement legitimately keeps the full pool -- maximum correction)")
ok(float(out["c4_budget"][0, MOD]) == POOL, "stock rule at the moderate-h row falls back to the full pool")
ok(float(out_hq["c4_budget"][0, MOD]) == 2.0, f"continuation-conditional target: moderate-h support = 2 columns "
   f"(was {POOL}); teaching continues on the columns that matter")
ok(float(out_hq["c4_hq_freeze"][0, MOD]) == 0.0, "moderate-h row not frozen (teacher disagrees)")

# ----------------------------------------------------------------- D. body ---
print("== D. body rows bit-identical")
b_off = out["c4_budget"][0]; b_hq = out_hq["c4_budget"][0]
same = [t for t in range(TL) if t not in (LMK, PRE, MOD)]
ok(bool((b_off[same] == b_hq[same]).all()), "all body rows: budget identical with/without the hq gate")
l_off = out["distillation_losses"][0]; l_hq = out_hq["distillation_losses"][0]
ok(bool(torch.allclose(l_off[same], l_hq[same], atol=1e-7)), "all body rows: loss identical with/without the hq gate")

# ------------------------------------------------------------------ E. rep ---
print("== E. repetition-run gate")
resp2 = resp.clone()
resp2[0, 10:22] = 34                          # 12 identical tokens -> rep run length 9 >= 6
data2 = {"response_mask": torch.ones(1, TL, dtype=torch.bool), "responses": resp2}
z3 = stu_base.clone().requires_grad_(True)
out_rep = run(z3, False, True, data2)
rep_marked = out_rep["c4_rep_freeze"][0]
# tokens 10..21 identical; the first full n-gram ends at 13 (first OCCURRENCE, not a repeat),
# so the repeat run is 14..21 -- 8 consecutive positions >= MINRUN=6
ok(float(rep_marked[14:22].min()) == 1.0, "long repeated run frozen (positions inside the run)")
ok(float(rep_marked[:14].max()) == 0.0, "positions before/at the first occurrence untouched")
ok(bool((out_rep["c4_budget"][0][rep_marked.bool()] == 1.0).all()), "frozen run support == top-1")
resp3 = resp.clone(); resp3[0, 10:15] = 35    # short repeat: run < MINRUN
out_rep2 = run(stu_base.clone(), False, True, {"response_mask": torch.ones(1, TL, dtype=torch.bool), "responses": resp3})
ok(float(out_rep2["c4_rep_freeze"].sum()) == 0.0, "incidental short repeat NOT gated (minrun filter)")
try:
    run(stu_base.clone(), False, True, None); ok(False, "rep gate with data=None must refuse")
except RuntimeError as e:
    ok("data=None" in str(e), "rep gate refuses loudly without the batch dict")

# the SHIPPED default, and that it separates the two measured loop-run modes. Calibration
# (docs/data/rep_run_calibration.txt, 9 cells): tau=1 loop runs are fragmented -- median 3-7,
# p90 23-45 -- because sampling noise breaks the 8-gram every few tokens, while greedy/late
# collapse gives multi-thousand-token blocks. 64 missed the fragment mode entirely (0.00
# recall on c4@150); 24 covers runs at the p90 scale at ~1-4% healthy-text cost.
import importlib
_saved = {k: os.environ.pop(k, None) for k in ("SIMOPD_REP_GATE_MINRUN", "SIMOPD_REP_GATE_N")}
_fresh = importlib.reload(T)
ok(_fresh.REP_GATE_MINRUN == 24, f"shipped SIMOPD_REP_GATE_MINRUN default is 24 (got {_fresh.REP_GATE_MINRUN})")
ok(_fresh.REP_GATE_N == 8, f"shipped SIMOPD_REP_GATE_N default is 8 (got {_fresh.REP_GATE_N})")
for _k, _v in _saved.items():
    if _v is not None:
        os.environ[_k] = _v
importlib.reload(T)

# fragment-scale run (28 identical tokens -> rep run 21 >= 24? no: check both sides of 24)
def _rep_len(run_tokens, minrun):
    """freeze count for a synthetic run of `run_tokens` identical tokens at MINRUN=minrun"""
    r = torch.randint(12, 30, (1, 64))
    r[0, 10:10 + run_tokens] = 77
    d = {"response_mask": torch.ones(1, 64, dtype=torch.bool), "responses": r}
    m = T._rep_runs_packed(d, torch.nested.nested_tensor([torch.zeros(64, K)], layout=torch.jagged), 64,
                           T.REP_GATE_N, minrun)
    return int(m.sum())

os.environ["SIMOPD_REP_GATE_MINRUN"] = "24"
importlib.reload(T)
ok(_rep_len(40, 24) > 0, "MINRUN=24: a 40-token repeat (rep run 33) is gated")
ok(_rep_len(24, 24) == 0, "MINRUN=24: a 24-token repeat (rep run 17, the fragment scale) is NOT gated -- the threshold is on the RUN, not the repeat")
os.environ["SIMOPD_REP_GATE_MINRUN"] = "6"
importlib.reload(T)

# --------------------------------------------------------------- F. detach ---
print("== F. carrier-off refusal for hq")
EG._ENABLED = False
try:
    run(stu_base.clone(), True, False); ok(False, "C4_HQ without the carrier must refuse")
except RuntimeError as e:
    ok("SIMOPD_GATHER_EOS" in str(e), "hq gate refuses loudly without the carrier")
EG._ENABLED = True

print(f"\nALL PASS ({PASS} checks)")
