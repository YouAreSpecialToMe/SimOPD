"""Force-gather the teacher's EXACT stop-token logprobs alongside its top-k.

N2 (`n2_eos_aux`) adds one thing to vanilla: at every visited state, a direct
EOS-vs-rest calibration term  BCEWithLogits(m_t, q_t)  with  q_t = p_T(stop | s_t)
and  m_t = log p_theta(stop)/(1 - p_theta(stop)).  Its gradient on the stop margin
is p_t - q_t, so everything hinges on q_t being the teacher's TRUE full-softmax
stop mass. It is not available from the stock payload:

  * vLLM's prompt_logprobs=k returns the top-k plus the ACTUAL token; there is no
    "also give me token X". A stop token outside the teacher's top-k is simply
    absent (verl then drops even the actual token; teacher_patch restores it).
  * Reading a miss as q=0 is not a harmless default: the gradient becomes p_t > 0,
    a manufactured ANTI-stop push -- and misses cluster exactly where EOS's rank
    is sliding out of the top-k, i.e. at collapse onset. Silent q=0 would
    fabricate the starvation the arm exists to test (design review 2026-08-19).
  * A top-k-renormalized q would inflate stop mass systematically and smuggle
    support normalization (nu) into an arm that is supposed to move one knob.

So this module makes the sampler emit the stop ids' logprobs as dedicated
columns, taken from the SAME full log_softmax vLLM already computes
(Sampler.compute_logprobs), never renormalized. Layout of one position's row,
width K+1 for a request of num_logprobs=K (what vLLM's dict builder zips):

    [ actual token | top-(K-n) among NON-stop tokens | n stop/diag ids in env order ]

The stop ids are excluded from the top-k so no column ever duplicates another;
teacher_patch's reader then rebuilds the (S, K+1) tensor as

    [ top-(K-n) | n extra ids | sampled ]

which is exactly the (S, K+1) shape `_prepare_streaming(want_sampled=True)`
expects for DISTILLATION_TOPK=K -- the D-axis path, unchanged. The kernel slices
the last n "top-k" columns back out as the exact stop-token logprobs.

Env (all SIMOPD_-prefixed so they enter the run fingerprint):
  SIMOPD_GATHER_EOS=1        arm the patch (teacher server + engine processes)
  SIMOPD_EOS_IDS=151643      comma list: the LOSS set. Must be the token(s) that
                             actually END a student rollout (Qwen3-1.7B-Base under
                             verl: generation_config eos 151643 <|endoftext|>);
                             calibrating a token that does not stop the rollout
                             would "succeed" while nothing terminates.
  SIMOPD_EOS_DIAG_IDS=151645 comma list, optional: gathered and panelled, NOT in
                             the loss (<|im_end|>: the Instruct teacher's own turn
                             end -- where its stop INTENT lives if it does not sit
                             on <|endoftext|>. The panel answers that directly.)

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


def stop_ids():
    """The LOSS set: token ids whose summed mass is 'stop'. Required when enabled."""
    ids = _parse(os.environ.get("SIMOPD_EOS_IDS", ""))
    if _ENABLED and not ids:
        raise RuntimeError("SIMOPD_GATHER_EOS=1 but SIMOPD_EOS_IDS is empty -- the loss set must "
                           "name the token(s) that actually end a student rollout.")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"SIMOPD_EOS_IDS has duplicates: {ids}")
    return ids


def diag_ids():
    """Gathered for the panels only. Anything also in the loss set is dropped here."""
    s = set(stop_ids())
    seen, out = set(), []
    for i in _parse(os.environ.get("SIMOPD_EOS_DIAG_IDS", "")):
        if i in s or i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def extra_ids():
    """Column order of the appended block: loss ids first, then diagnostic ids."""
    return stop_ids() + diag_ids()


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
