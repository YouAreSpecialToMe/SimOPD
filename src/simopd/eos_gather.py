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
# vLLM >= 0.12 ships a second GPU model runner ("V2", vllm/v1/worker/gpu/); it is the
# DEFAULT for dense generate models in 0.26 (VllmConfig.use_v2_model_runner) and it never
# touches Sampler.gather_logprobs: prompt logprobs and sampled logprobs both go through
# vllm.v1.worker.gpu.sample.logprob.compute_topk_scores (a fused Triton top-k log-softmax).
# The first corrected-wave rehearsal (2026-08-19) armed the legacy sampler in every
# teacher process and still shipped rows without the ids -- because the runner in
# use never called it. Both paths are patched; whichever the worker runs, the row
# layout is the same [actual | top-(K-n) | extras].
_MOD_V2 = "vllm.v1.worker.gpu.sample.logprob"          # where compute_topk_scores lives
# The ONLY binding we patch. Both prompt_logprob.py and sampler.py do
# `from ...logprob import compute_topk_scores` at import, so each holds its own reference
# and they can be patched independently. verl reads the TEACHER through prompt logprobs
# (vllm_async_server.py -> extract_prompt_logprobs(output.prompt_logprobs)), which is
# produced by PromptLogprobsWorker -> compute_prompt_logprobs_with_chunking -> this
# binding. The sampler binding drives ordinary generation, including the STUDENT's own
# rollout: patching it would rewrite rollout rows for no reason (measured 2026-08-19:
# every first-call receipt was K=5 -- the student's sampling request, not the teacher's
# K=66 scoring). One knob per arm means the student's payload stays stock.
_MOD_V2_PROMPT = "vllm.v1.worker.gpu.sample.prompt_logprob"


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
        _first_call_receipt("legacy Sampler.gather_logprobs", K, n, rows)
        return LogprobsTensors(indices, lps, ranks, cu) if cu is not None else LogprobsTensors(indices, lps, ranks)

    gather_logprobs._simopd_eos_gather = True
    return gather_logprobs


def _reorder_rows(base_ids, base_lps, K, ids, device):
    """Shared row surgery: (tok | top-K) -> (tok | top-(K-n) non-extra | extras' ids).
    Returns (indices, keep_lp, extra_idx, tok_lp) so the caller supplies the extras'
    VALUES from whatever exact-logprob primitive its runner has."""
    import torch

    n = len(ids)
    tok_idx, topk_idx = base_ids[:, :1], base_ids[:, 1:]
    tok_lp, topk_lp = base_lps[:, :1], base_lps[:, 1:]
    ids_t = torch.tensor(ids, device=device, dtype=topk_idx.dtype)
    # Drop the extra ids from the top-K and keep the first K-n survivors, in rank
    # order. At most n of the K entries can be extras, so K-n survivors always
    # exist -- no padding, no duplicated column.
    is_extra = torch.isin(topk_idx, ids_t)
    pos = torch.arange(K, device=device).unsqueeze(0).expand_as(topk_idx)
    order = torch.argsort(is_extra.to(torch.int64) * (K + 1) + pos, dim=-1, stable=True)[:, : K - n]
    keep_idx = torch.gather(topk_idx, 1, order)
    keep_lp = torch.gather(topk_lp, 1, order)
    # materialized (rows x n), NOT an expand() view: the V2 caller hands it to a Triton
    # kernel that walks raw pointers with row stride n -- a stride-0 view would be read
    # out of bounds. Tiny (rows x 2 ints).
    extra_idx = ids_t.unsqueeze(0).repeat(base_ids.shape[0], 1)
    return tok_idx, tok_lp, keep_idx, keep_lp, extra_idx


def _wrap_v2(original, LogprobsTensors, compute_token_logprobs):
    """V2 runner: wrap vllm.v1.worker.gpu.sample.logprob.compute_topk_scores.

    Fast path only (no per-request logprob_token_ids override, not a *_logits mode):
    the stock result is (tok | top-K) at width K+1; the extras' values come from
    vLLM's own compute_token_logprobs (max + logsumexp per row, emitted at the given
    ids -- the same kernel that produced the top-K values, so the columns are exactly
    comparable and nothing is renormalized). Anything else is returned untouched, and
    the reader then counts the ids as missing -> the kernel refuses (loud, not wrong)."""
    import torch

    def compute_topk_scores(logits, num_logprobs, sampled_token_ids, *args, **kwargs):
        base = original(logits, num_logprobs, sampled_token_ids, *args, **kwargs)
        ids = extra_ids()
        n = len(ids)
        max_per_req = kwargs.get("max_per_req_token_ids", args[3] if len(args) > 3 else 0)
        logits_mode = kwargs.get("logits_mode", args[4] if len(args) > 4 else False)
        if (not n or num_logprobs is None or num_logprobs <= n or max_per_req or logits_mode
                or num_logprobs >= logits.shape[-1]):
            return base
        K = int(num_logprobs)
        if base.logprob_token_ids.shape[-1] != K + 1:
            return base  # unknown layout -- do not guess; the reader will refuse
        tok_idx, tok_lp, keep_idx, keep_lp, extra_idx = _reorder_rows(
            base.logprob_token_ids, base.logprobs, K, ids, logits.device)
        extra_lp = compute_token_logprobs(logits, extra_idx).to(base.logprobs.dtype)
        indices = torch.cat((tok_idx, keep_idx, extra_idx), dim=1)
        lps = torch.cat((tok_lp, keep_lp, extra_lp), dim=1)
        _first_call_receipt("v2 prompt-logprobs compute_topk_scores", K, n, logits.shape[0])
        return LogprobsTensors(indices, lps, base.selected_token_ranks, base.cu_num_generated_tokens)

    compute_topk_scores._simopd_eos_gather = True
    return compute_topk_scores


_RECEIPTED = set()


def _first_call_receipt(path, K, n, rows):
    """Once per process and path: prove the patched code actually RAN (armed != called;
    the 2026-08-19 rehearsal had every process armed on a path the runner never used)."""
    if path in _RECEIPTED:
        return
    _RECEIPTED.add(path)
    msg = f"first gathered rows via {path}: K={K} n={n} rows={rows} ids={extra_ids()}"
    _diag(msg)
    _receipt(f"call_{path.split()[0]}_pid{os.getpid()}.txt", msg)


def _receipt(name, text):
    """filesystem receipt, immune to stdio capture: which processes armed / gathered."""
    try:
        d = os.path.join(os.environ.get("SIMOPD_EVAL_ROOT", "/nonexistent"), "..", "n2", "armed")
        d = os.path.abspath(d)
        if os.path.isdir(os.path.dirname(d)):
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "w") as f:
                f.write(f"{os.uname().nodename} {os.getpid()} {sys.argv[:2]} {text}\n")
    except Exception:
        pass


def _diag(msg):
    print(f"[simopd] eos_gather pid={os.getpid()}: {msg}", file=sys.stderr, flush=True)


def install_v2():
    """Patch the V2 runner's PROMPT-logprobs binding only (see _MOD_V2_PROMPT).

    Idempotent; no-op if the prompt-logprobs module is not loaded (an older vLLM without
    the V2 runner, or a process that never scores) -- ensure_installed() imports it first.
    The home module and the sampler binding are deliberately left stock."""
    if not _ENABLED:
        return False
    stop_ids()
    prompt = sys.modules.get(_MOD_V2_PROMPT)
    home = sys.modules.get(_MOD_V2)
    if prompt is None or not hasattr(prompt, "compute_topk_scores"):
        return False
    if home is None or not hasattr(home, "compute_token_logprobs"):
        raise RuntimeError("eos_gather: the V2 prompt-logprobs module is loaded but "
                           f"{_MOD_V2}.compute_token_logprobs is missing -- vLLM moved the exact "
                           "log-softmax primitive; the teacher cannot get exact terminator logprobs.")
    orig = prompt.compute_topk_scores
    if getattr(orig, "_simopd_eos_gather", False):
        return True
    from vllm.v1.outputs import LogprobsTensors

    prompt.compute_topk_scores = _wrap_v2(orig, LogprobsTensors, home.compute_token_logprobs)
    print(f"[simopd] eos_gather armed pid={os.getpid()} [V2 prompt-logprobs compute_topk_scores]: teacher "
          f"scoring rows carry exact logprobs for ids {extra_ids()} (sampling path left stock)",
          file=sys.stderr, flush=True)
    _receipt(f"v2_pid{os.getpid()}.txt", "armed v2 prompt-logprobs compute_topk_scores")
    return True


def v2_status():
    """(prompt module loaded?, prompt binding patched?, sampler binding patched?) --
    the third must stay False: the student's rollout payload is not ours to rewrite."""
    prompt = sys.modules.get(_MOD_V2_PROMPT)
    sampler = sys.modules.get("vllm.v1.worker.gpu.sample.sampler")
    patched = bool(prompt is not None
                   and getattr(getattr(prompt, "compute_topk_scores", None), "_simopd_eos_gather", False))
    sampler_patched = bool(sampler is not None
                           and getattr(getattr(sampler, "compute_topk_scores", None), "_simopd_eos_gather", False))
    return prompt is not None, patched, sampler_patched


def ensure_installed(where="unknown"):
    """Second install path (2026-08-19): import the sampler module if it is not loaded
    yet, then patch. Called from the sitecustomize hook that provably fires inside
    vLLM's worker processes (the verl worker-extension import), because the first
    rehearsal of the corrected wave showed the sampler hook alone never armed the
    teacher's workers (rows came back without the gathered ids). Idempotent."""
    if not _ENABLED:
        _diag(f"ensure_installed({where}): SIMOPD_GATHER_EOS not set in this process -- nothing to do")
        return
    import importlib
    try:
        importlib.import_module(_MOD)
    except Exception as e:  # pragma: no cover
        _diag(f"ensure_installed({where}): cannot import {_MOD}: {e!r}")
        raise
    install()
    mod = sys.modules.get(_MOD)
    fn = mod.Sampler.__dict__.get("gather_logprobs")
    orig = fn.__func__ if isinstance(fn, staticmethod) else fn
    # V2 runner path (vLLM >= 0.12; the default runner in 0.26). Import-then-patch so a
    # worker that has not loaded it yet still gets the patched name when it does.
    v2_note = "v2 prompt-logprobs module absent (older vLLM)"
    try:
        importlib.import_module(_MOD_V2_PROMPT)
    except Exception as e:  # noqa: BLE001
        v2_note = f"v2 import failed: {e!r}"
    else:
        install_v2()
        loaded, patched, sampler_patched = v2_status()
        v2_note = f"v2 prompt-logprobs patched={patched} (sampler binding patched={sampler_patched}, must be False)"
    _diag(f"ensure_installed({where}): legacy sampler patched={getattr(orig, '_simopd_eos_gather', False)}; {v2_note}")


def install():
    """Rebind Sampler.gather_logprobs (a staticmethod) in the module that owns it.

    Both call sites -- the sampled-logprobs path in Sampler.forward and the
    prompt-logprobs path in gpu_model_runner -- reach it as self.gather_logprobs,
    so the class attribute is the single place to patch. Runs in every process
    that imports the sampler (engine core / workers), via the sitecustomize hook.
    """
    if not _ENABLED:
        _diag("install(): SIMOPD_GATHER_EOS not set in this process")
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

    wrapped = _wrap(orig, LogprobsTensors)
    wrapped._simopd_eos_gather = True
    cls.gather_logprobs = staticmethod(wrapped)
    print(f"[simopd] eos_gather armed pid={os.getpid()} [legacy Sampler.gather_logprobs]: teacher rows carry exact "
          f"logprobs for ids {extra_ids()} (loss set {stop_ids()}, diag {diag_ids()})", file=sys.stderr, flush=True)
    _receipt(f"pid{os.getpid()}.txt", "armed legacy Sampler.gather_logprobs")
    # If the V2 runner module is already loaded in this process, arm it too (the
    # sitecustomize hook for it fires only if it is imported AFTER sitecustomize).
    install_v2()
