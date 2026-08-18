"""Force-gather the teacher's EXACT termination-token logprobs alongside its top-k.

N2 (`n2_termcal`, termination-marginal calibration) adds one thing to vanilla: at
every visited state, a direct stop-vs-continue calibration term
BCEWithLogits(m_t, q_t) with

    q_t = sum_{e in E_T} p_T(e | s_t)        the TEACHER's termination mass
    m_t = log p_theta(E_S | s_t) / (1 - p_theta(E_S | s_t))
                                            the STUDENT's stop log-odds

Its gradient on the stop log-odds is p_t - q_t, so everything hinges on q_t being
the teacher's TRUE full-softmax termination mass. It is not available from the
stock payload:

  * vLLM's prompt_logprobs=k returns the top-k plus the ACTUAL token; there is no
    "also give me token X". A termination token outside the teacher's top-k is
    simply absent (verl then drops even the actual token; teacher_patch restores it).
  * Reading a miss as q=0 is not a harmless default: the gradient becomes p_t > 0,
    a manufactured ANTI-stop push -- and misses cluster exactly where the stop's
    rank is sliding out of the top-k, i.e. at collapse onset. Silent q=0 would
    fabricate the starvation the arm exists to test (design review 2026-08-19).
  * A top-k-renormalized q would inflate stop mass systematically and smuggle
    support normalization (nu) into an arm that is supposed to move one knob.

Why TWO sets (audit 2026-08-19, scripts/analysis/eos_stop_audit.py): the student
(Qwen3-1.7B-Base under verl) ends a rollout ONLY on <|endoftext|> 151643 -- the
tokenizer/generation_config eos, vLLM's sole stop -- and puts ~1e-11 on
<|im_end|>; the Instruct teacher ends a response on <|im_end|> 151645 (median
0.95 at the student's stop positions) and puts ~1e-11 on <|endoftext|>. Same
EVENT, disjoint TOKENS. So the event is read on each side with that side's own
terminators: E_S = what actually terminates the student's rollout, E_T = what the
teacher ends a response with. Calibrating the single token 151643 to the teacher
would push p(stop) toward 1e-11 -- an anti-stop channel -- while a symmetric union
would let a student mass on <|im_end|> (which does NOT stop its rollout) count as
"stopped". The aggregate log-odds gradient distributes inside E_S by the student's
own conditional, so with E_S = {151643} it lands on the token that ends rollouts.

So this module makes the sampler emit the extra ids' logprobs as dedicated
columns, taken from the SAME full log_softmax vLLM already computes
(Sampler.compute_logprobs), never renormalized. Layout of one position's row,
width K+1 for a request of num_logprobs=K (what vLLM's dict builder zips):

    [ actual token | top-(K-n) among NON-extra tokens | n extra ids in env order ]

The extra ids are excluded from the top-k so no column ever duplicates another;
teacher_patch's reader then rebuilds the (S, K+1) tensor as

    [ top-(K-n) | n extra ids | sampled ]

which is exactly the (S, K+1) shape `_prepare_streaming(want_sampled=True)`
expects for DISTILLATION_TOPK=K -- the D-axis path, unchanged. The kernel slices
the last n "top-k" columns back out as the exact logprobs.

Env (all SIMOPD_-prefixed so they enter the run fingerprint):
  SIMOPD_GATHER_EOS=1              arm the patch (teacher server + engine processes)
  SIMOPD_EOS_IDS=151643            comma list: E_S, the STUDENT stop set = the
                                   token(s) that actually END a student rollout
                                   (Qwen3-1.7B-Base under verl: generation_config
                                   eos 151643 <|endoftext|>). Required, and checked
                                   against the rollout stop contract: with
                                   SIMOPD_STOP_IDS=151643,151645 (contract v2,
                                   simopd.stop_set) it must be "151643,151645".
  SIMOPD_EOS_TEACHER_IDS=151643,151645
                                   comma list: E_T, the TEACHER termination set =
                                   the token(s) the teacher ends a response with.
                                   Default = E_S (symmetric). Qwen3-4B-Instruct-2507
                                   ends on <|im_end|> 151645, so E_T must name it.
  SIMOPD_EOS_DIAG_IDS=             comma list, optional: gathered and panelled,
                                   in neither set.

Block order: E_S, then E_T \ E_S, then diag \ (E_S u E_T). Per-member panels
(eos_qm_i / eos_pm_i) index the union E_S u E_T in that order.

Cost: torch.isin on [n_tok, K] ints, one gather of n columns, no [T, V] copy --
the compiled rank kernel and the top-k call are vLLM's own, unchanged.
"""

import os
import sys

_ENABLED = os.environ.get("SIMOPD_GATHER_EOS", "0") == "1"
_MOD = "vllm.v1.sample.sampler"


def enabled():
    return _ENABLED


def _parse(raw):
    raw = (raw or "").strip()
    if not raw:
        return []
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


# The protocol student's own eos (Qwen3-1.7B-Base tokenizer/generation_config): the one
# token that ends a rollout when no stop contract is active. The contract (simopd.stop_set,
# SIMOPD_STOP_IDS, A-AXIS R5 appendix) ADDS ids at the rollout seam; E_S must then equal
# the union, or the arm would calibrate/read an event that is not what ends rollouts.
_MODEL_EOS = 151643


def model_eos():
    """151643 for the protocol student; SIMOPD_MODEL_EOS_ID overrides for the CPU battery
    (toy vocab) only -- no arm sets it."""
    raw = os.environ.get("SIMOPD_MODEL_EOS_ID", "")
    return int(raw) if raw.strip() else _MODEL_EOS


def rollout_stop_set():
    """The token ids that actually end a student rollout under the current contract:
    the model eos, plus SIMOPD_STOP_IDS when the dual-terminator contract is armed."""
    ids = [model_eos()]
    raw = os.environ.get("SIMOPD_STOP_IDS", "")
    if raw.strip() and raw.strip().lower() != "off":
        for i in _parse(raw):
            if i not in ids:
                ids.append(i)
    return ids


def stop_ids():
    """E_S, the STUDENT stop set: token ids whose summed student mass is 'stop'.
    Must be the token(s) that actually end a student rollout. Required when enabled,
    and when enabled it must EQUAL rollout_stop_set() -- a loss set that names a token
    the sampler does not honor (or misses one it does) is refused at import."""
    ids = _parse(os.environ.get("SIMOPD_EOS_IDS", ""))
    if _ENABLED and not ids:
        raise RuntimeError("SIMOPD_GATHER_EOS=1 but SIMOPD_EOS_IDS is empty -- the student stop set "
                           "must name the token(s) that actually end a student rollout.")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"SIMOPD_EOS_IDS has duplicates: {ids}")
    if _ENABLED and set(ids) != set(rollout_stop_set()):
        raise RuntimeError(f"SIMOPD_EOS_IDS={ids} != the tokens that end a rollout under the active "
                           f"stop contract {rollout_stop_set()} (SIMOPD_STOP_IDS="
                           f"{os.environ.get('SIMOPD_STOP_IDS', '')!r}). E_S must be exactly the "
                           "rollout stop set: under the legacy contract '151643', under the dual "
                           "contract '151643,151645'.")
    return ids


def teacher_ids():
    """E_T, the TEACHER termination set: token ids whose summed teacher mass is
    'the response ends here'. Defaults to E_S when SIMOPD_EOS_TEACHER_IDS is unset."""
    raw = os.environ.get("SIMOPD_EOS_TEACHER_IDS", "")
    ids = _parse(raw) if raw.strip() else list(stop_ids())
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"SIMOPD_EOS_TEACHER_IDS has duplicates: {ids}")
    if _ENABLED and not ids:
        raise RuntimeError("SIMOPD_EOS_TEACHER_IDS parsed to an empty set")
    return ids


def union_ids():
    """E_S u E_T in block order: E_S first, then the teacher-only ids."""
    s = stop_ids()
    seen = set(s)
    out = list(s)
    for i in teacher_ids():
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def diag_ids():
    """Gathered for the panels only. Anything also in E_S u E_T is dropped here."""
    s = set(union_ids())
    seen, out = set(), []
    for i in _parse(os.environ.get("SIMOPD_EOS_DIAG_IDS", "")):
        if i in s or i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def extra_ids():
    """Column order of the appended block: E_S, then E_T \ E_S, then diagnostic ids."""
    return union_ids() + diag_ids()


def stop_cols():
    """Positions of E_S inside the extra block."""
    return list(range(len(stop_ids())))


def teacher_cols():
    """Positions of E_T inside the extra block (E_S members first if shared)."""
    u = union_ids()
    t = set(teacher_ids())
    return [i for i, tok in enumerate(u) if tok in t]


def n_extra():
    return len(extra_ids())


def _wrap(original, LogprobsTensors):
    import torch

    def gather_logprobs(logprobs, num_logprobs, token_ids):
        ids = extra_ids()
        n = len(ids)
        # Leave stock behaviour alone wherever we cannot honour the layout: the
        # estimator path (num 0 / None), or a request narrower than the block.
        if not n or num_logprobs is None or num_logprobs <= n:
            return original(logprobs, num_logprobs, token_ids)
        K = int(num_logprobs)
        base = original(logprobs, K, token_ids)          # (tok | top-K), exact rank & tok logprob
        idx_all, lp_all, ranks = base.logprob_token_ids, base.logprobs, base.selected_token_ranks
        tok_idx, topk_idx = idx_all[:, :1], idx_all[:, 1:]
        tok_lp, topk_lp = lp_all[:, :1], lp_all[:, 1:]
        ids_t = torch.tensor(ids, device=logprobs.device, dtype=topk_idx.dtype)
        # Drop the extra ids from the top-K and keep the first K-n survivors, in
        # rank order. At most n of the K entries can be extras, so K-n survivors
        # always exist -- no padding, no duplicated column.
        is_extra = torch.isin(topk_idx, ids_t)
        pos = torch.arange(K, device=logprobs.device).unsqueeze(0).expand_as(topk_idx)
        order = torch.argsort(is_extra.to(torch.int64) * (K + 1) + pos, dim=-1, stable=True)[:, : K - n]
        keep_idx = torch.gather(topk_idx, 1, order)
        keep_lp = torch.gather(topk_lp, 1, order)
        # The block itself: exact values from the full log_softmax, one gather.
        rows = logprobs.shape[0]
        gidx = torch.tensor(ids, device=logprobs.device, dtype=torch.long).unsqueeze(0).expand(rows, -1)
        extra_lp = torch.gather(logprobs, 1, gidx).to(lp_all.dtype)
        extra_idx = ids_t.unsqueeze(0).expand(rows, -1)
        indices = torch.cat((tok_idx, keep_idx, extra_idx), dim=1)
        lps = torch.cat((tok_lp, keep_lp, extra_lp), dim=1)
        cu = getattr(base, "cu_num_generated_tokens", None)
        return LogprobsTensors(indices, lps, ranks, cu) if cu is not None else LogprobsTensors(indices, lps, ranks)

    gather_logprobs._simopd_eos_gather = True
    return gather_logprobs


def install():
    """Rebind Sampler.gather_logprobs (a staticmethod) in the module that owns it.

    Both call sites -- the sampled-logprobs path in Sampler.forward and the
    prompt-logprobs path in gpu_model_runner -- reach it as self.gather_logprobs,
    so the class attribute is the single place to patch. Runs in every process
    that imports the sampler (engine core / workers), via the sitecustomize hook.
    """
    if not _ENABLED:
        return
    stop_ids()  # raise early on a bad env, in the process that would misbehave
    mod = sys.modules.get(_MOD)
    if mod is None or not hasattr(mod, "Sampler"):
        raise RuntimeError(f"eos_gather: SIMOPD_GATHER_EOS=1 but {_MOD} has no Sampler to patch "
                           "-- vLLM moved or renamed it; N2 cannot get exact stop-token logprobs.")
    cls = mod.Sampler
    fn = cls.__dict__.get("gather_logprobs")
    orig = fn.__func__ if isinstance(fn, staticmethod) else fn
    if getattr(orig, "_simopd_eos_gather", False):
        return
    from vllm.v1.outputs import LogprobsTensors

    cls.gather_logprobs = staticmethod(_wrap(orig, LogprobsTensors))
    print(f"[simopd] eos_gather armed: teacher rows carry exact logprobs for ids {extra_ids()} "
          f"(loss set {stop_ids()}, diag {diag_ids()})", file=sys.stderr, flush=True)
