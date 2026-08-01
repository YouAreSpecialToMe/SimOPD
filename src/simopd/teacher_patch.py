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
"""

import os
import sys

_ENABLED = os.environ.get("SIMOPD_KEEP_SAMPLED", "0") == "1"
_SERVER_MODULE = "verl.workers.rollout.vllm_rollout.vllm_async_server"


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
            entry = logprobs_dict.get(token_id)
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
    if not _ENABLED:
        return
    module = sys.modules.get(_SERVER_MODULE)
    if module is None or not hasattr(module, "extract_prompt_logprobs"):
        return
    if getattr(module.extract_prompt_logprobs, "_simopd_patched", False):
        return
    patched = _extract_with_sampled(module.extract_prompt_logprobs)
    patched._simopd_patched = True
    module.extract_prompt_logprobs = patched
    print("[simopd] teacher scoring will retain the sampled token (tensors are (S, K+1))")
