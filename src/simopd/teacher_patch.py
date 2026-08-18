"""Keep the sampled token's teacher logprob alongside the teacher's top-k.

D-axis arms change *which tokens* get supervised while keeping vanilla's
objective (sampled-token reverse KL) -- otherwise the arm silently moves the C
axis too and the comparison is confounded. That needs both the sampled token's
teacher logprob (for the loss) and the teacher's top-k (for the selection
criterion), in the same run.

vLLM already returns both: with prompt_logprobs=k it emits the top-k at each
position plus the actual token when the actual token falls outside. verl then
drops it -- vllm_rollout/utils.py:483 does `if rank > num_prompt_logprobs:
continue`. This module re-attaches it as one extra trailing column, so tensors
become (S, K+1) with [..., :K] the top-k support and [..., K] the sampled token.

Opt-in via SIMOPD_KEEP_SAMPLED=1, because widening the tensor would otherwise
change stock forward_kl_topk's objective (it sums over shape[-1]).

N2 (SIMOPD_GATHER_EOS=1, see simopd.eos_gather) changes what the sampler emits:
each row is [actual | top-(K-n) non-stop | n stop/diag ids]. verl's stock reader
cannot consume that -- it fills slots by the dict's rank FIELD, and the extra
block carries positional ranks that collide with the sampled token's slot -- so
under that flag the reader below replaces verl's outright: sampled and stop
columns are looked up BY ID, the top-(K-n) by VALUE order, and no code path
depends on a rank field. Same (S, K+1) tensor width as KEEP_SAMPLED, so
_prepare_streaming is untouched; the kernel slices the block back out.
"""

import os
import sys

_ENABLED = os.environ.get("SIMOPD_KEEP_SAMPLED", "0") == "1"
_SERVER_MODULE = "verl.workers.rollout.vllm_rollout.vllm_async_server"


def _entry(logprobs_dict, token_id):
    """dict entry by id, tolerating str keys (verl's own reader int()s them; the
    teacher is reached over HTTP where JSON keys are strings)."""
    e = logprobs_dict.get(token_id)
    if e is None:
        e = logprobs_dict.get(str(token_id))
    return e


def _extract_with_eos(original):
    """N2 reader: [top-(K-n) by value | extra ids by id | sampled by id], width K+1.

    Replaces verl's reader entirely under SIMOPD_GATHER_EOS=1 (num_prompt_logprobs
    > 0); the estimator path (num == 0) is delegated unchanged. A stop id that the
    sampler patch failed to deliver is written as -inf AND counted -- the kernel
    refuses to train on -inf, so a broken teacher process dies at its first
    micro-batch instead of silently learning from q=0."""
    from simopd import eos_gather

    def extract_prompt_logprobs(output, num_prompt_logprobs, result_dict):
        if not num_prompt_logprobs:
            return original(output, num_prompt_logprobs, result_dict)
        K = int(num_prompt_logprobs)
        extra = eos_gather.extra_ids()
        n = len(extra)
        if K <= n:
            raise RuntimeError(f"teacher_patch: DISTILLATION_TOPK={K} leaves no top-k columns beside "
                               f"the {n} stop/diag ids -- raise the payload width.")
        extra_set = set(extra)
        ids_ls, lps_ls = [], []
        actual_ids = output.prompt_token_ids[1:]
        missing = 0
        for i, logprobs_dict in enumerate(output.prompt_logprobs[1:]):
            actual = int(actual_ids[i])
            rest = []
            for key, e in logprobs_dict.items():
                tid = int(key)
                if tid in extra_set:
                    continue
                rest.append((float(e.logprob), tid))
            rest.sort(key=lambda t: t[0], reverse=True)
            top = rest[: K - n]
            if len(top) < K - n:
                # By construction impossible (K-n non-stop entries always survive);
                # if vLLM ever changes the layout this must not pass silently.
                raise RuntimeError(f"teacher_patch: position {i} has {len(top)} non-stop entries, "
                                   f"need {K - n}; eos_gather layout contract broken")
            row_ids = [t for _, t in top]
            row_lps = [lp for lp, _ in top]
            for eid in extra:
                e = _entry(logprobs_dict, eid)
                if e is None:
                    missing += 1
                    row_ids.append(eid)
                    row_lps.append(float("-inf"))
                else:
                    row_ids.append(eid)
                    row_lps.append(float(e.logprob))
            e = _entry(logprobs_dict, actual)
            row_ids.append(actual)
            row_lps.append(float(e.logprob) if e is not None else float("-inf"))
            ids_ls.append(row_ids)
            lps_ls.append(row_lps)
        # verl's dummy row for the last prompt token, same width as the data rows.
        ids_ls.append([0] * (K + 1))
        lps_ls.append([0.0] * (K + 1))
        result_dict["prompt_ids"] = ids_ls
        result_dict["prompt_logprobs"] = lps_ls
        if missing:
            print(f"[simopd] teacher_patch: {missing} stop/diag entries missing from the sampler "
                  "output (written as -inf; the kernel will refuse them)", file=sys.stderr, flush=True)

    return extract_prompt_logprobs


def _extract_with_sampled(original):
    def extract_prompt_logprobs(output, num_prompt_logprobs, result_dict):
        original(output, num_prompt_logprobs, result_dict)
        if not num_prompt_logprobs:
            return  # estimator path already returns exactly the sampled token

        ids, logprobs = result_dict["prompt_ids"], result_dict["prompt_logprobs"]
        # prompt_logprobs[0] is None, so row i corresponds to prompt_token_ids[i+1];
        # the original appends one dummy row past the end, handled after the loop.
        actual_ids = output.prompt_token_ids[1:]
        for i, logprobs_dict in enumerate(output.prompt_logprobs[1:]):
            token_id = int(actual_ids[i])
            # verl's own reader calls these keys token_id_str and int()s them, and the
            # teacher is reached over HTTP where JSON keys are always strings, so accept
            # either rather than silently missing every lookup.
            entry = logprobs_dict.get(token_id)
            if entry is None:
                entry = logprobs_dict.get(str(token_id))
            ids[i].append(token_id)
            # vLLM guarantees the actual token is present; -inf is a loud fallback
            # rather than a silent wrong number if that ever stops holding.
            logprobs[i].append(entry.logprob if entry is not None else float("-inf"))
        ids[-1].append(0)
        logprobs[-1].append(0.0)

    return extract_prompt_logprobs


def install():
    """Rebind the name inside the server module, where it is actually called.

    Patching vllm_rollout.utils would not work: the server does
    `from ...utils import extract_prompt_logprobs`, so it holds its own reference.
    """
    from simopd import eos_gather

    if eos_gather.enabled():
        module = sys.modules.get(_SERVER_MODULE)
        if module is None or not hasattr(module, "extract_prompt_logprobs"):
            raise RuntimeError(f"teacher_patch: SIMOPD_GATHER_EOS=1 but {_SERVER_MODULE} has no "
                               "extract_prompt_logprobs to patch -- verl moved or renamed it.")
        if getattr(module.extract_prompt_logprobs, "_simopd_patched", False):
            return
        patched = _extract_with_eos(module.extract_prompt_logprobs)
        patched._simopd_patched = True
        module.extract_prompt_logprobs = patched
        print(f"[simopd] teacher scoring: N2 reader armed, rows = [top-(K-{eos_gather.n_extra()}) | "
              f"{eos_gather.extra_ids()} | sampled]", file=sys.stderr)
        return
    if not _ENABLED:
        return
    module = sys.modules.get(_SERVER_MODULE)
    if module is None or not hasattr(module, "extract_prompt_logprobs"):
        # SIMOPD_KEEP_SAMPLED=1 means a D-axis arm is depending on this patch. The
        # kernel does fail loudly on the narrower tensor (_prepare raises on width K
        # instead of K+1), but that error names a shape, not the cause; this one does.
        raise RuntimeError(
            f"teacher_patch: SIMOPD_KEEP_SAMPLED=1 but {_SERVER_MODULE} has no "
            "extract_prompt_logprobs to patch -- verl moved or renamed it. The D-axis "
            "arm cannot keep vanilla's objective without the sampled column.")
    if getattr(module.extract_prompt_logprobs, "_simopd_patched", False):
        return
    patched = _extract_with_sampled(module.extract_prompt_logprobs)
    patched._simopd_patched = True
    module.extract_prompt_logprobs = patched
    print("[simopd] teacher scoring will retain the sampled token (tensors are (S, K+1))", file=sys.stderr)
