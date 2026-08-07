"""a1_gkd_mix0.5: route a coin-chosen fraction of rollout prompts to cached teacher
responses, scored (not generated) by the student engine.

GKD's lambda-mixing (2306.13649), implemented at the student's rollout server instead
of the TransferQueue trainer surgery originally feared. The wrapped entry is
vLLMHttpServer.generate -- the same seam family as teacher_patch/priv_cot. Guards:

  scoring passthrough   requests carrying prompt_logprobs are teacher-side scoring;
                        untouched (the exact complement of priv_cot's guard).
  membership gate       the coin can only fire on prompts present in the cache, which
                        by construction is the training set -- validation prompts miss
                        and generate normally. A stale cache therefore degrades toward
                        lambda=0 rather than crashing validation; the realised mix
                        rate is printed periodically so that degradation is visible.
  per-visit coin        hash(prompt_key, global_steps) -- GKD's mixing is a fresh coin
                        each visit, not a fixed per-prompt split; the server knows the
                        step via set_global_steps, so the coin is deterministic given
                        (prompt, step) and reproducible across relaunches.

On an off-policy hit the student must still report log_probs (verl uses them as the
behavior-policy term). The wrapper reuses the ORIGINAL generate as a scoring call
(prompt = prompt_ids + cached response, prompt_logprobs=0, max_tokens=1) and slices
the response tail -- the same tail-length pattern informativeness.py uses -- so the
returned TokenOutput is shaped exactly like a generation result.
"""

import hashlib
import os
import sys

_MARK = "_simopd_gkd"
_cache = None
_stats = {"hit": 0, "miss": 0, "pass": 0}


# Semantics note (provenance r4): SIMOPD_GKD_LAMBDA is P(off-policy) here, which is
# 1 - lambda_GKD -- the paper's lambda is the STUDENT-data fraction. Identical at 0.5;
# any future ablation away from 0.5 must convert, or it silently inverts the axis.
def _lam():
    return float(os.environ.get("SIMOPD_GKD_LAMBDA", "0.5"))


def _load_cache():
    global _cache
    if _cache is None:
        import pandas as pd

        path = os.environ["SIMOPD_GKD_CACHE"]
        df = pd.read_parquet(path)
        _cache = {r["prefix_hash"]: list(r["response_ids"]) for _, r in df.iterrows()}
        print(f"[simopd] gkd_mix: cache loaded, {len(_cache)} prompts, lambda={_lam()}",
              file=sys.stderr, flush=True)
    return _cache


def prompt_key(prompt_ids):
    return hashlib.sha1(",".join(map(str, list(prompt_ids)[:16])).encode()).hexdigest()[:16]


def coin(key, step, lam):
    """Deterministic per-(prompt, step) Bernoulli(lam)."""
    h = hashlib.sha1(f"{key}:{step}".encode()).hexdigest()
    return (int(h[:8], 16) % 10_000) < lam * 10_000


def tail_logprobs(extra_fields, n):
    """Student logprob of each response token, from a prompt_logprobs=0 scoring call.

    The extractor emits one row per position with the actual token's entry; the
    response is the last n rows minus the extractor's trailing dummy row.
    """
    rows = extra_fields["prompt_logprobs"]
    flat = [r[0] if isinstance(r, (list, tuple)) else r for r in rows]
    return [float(x) for x in flat[-(n + 1):-1]] if len(flat) > n else [float(x) for x in flat[-n:]]


def install():
    if os.environ.get("SIMOPD_GKD_CACHE", "") == "":
        return
    mod = sys.modules.get("verl.workers.rollout.vllm_rollout.vllm_async_server")
    if mod is None:
        return
    cls = getattr(mod, "vLLMHttpServer", None)
    fn = getattr(cls, "generate", None) if cls else None
    if fn is None or getattr(fn, _MARK, False):
        if fn is None:
            raise RuntimeError("gkd_mix: vLLMHttpServer.generate not found -- verl moved it; "
                               "the a1 arm cannot mix and would silently train as vanilla")
        return

    async def generate(self, prompt_ids, sampling_params, request_id, *a, **kw):
        # Teacher-side scoring calls pass through untouched.
        if isinstance(sampling_params, dict) and sampling_params.get("prompt_logprobs") is not None:
            _stats["pass"] += 1
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)
        cache = _load_cache()
        key = prompt_key(prompt_ids)
        cached = cache.get(key)
        step = getattr(self, "global_steps", 0) or 0
        if cached is None or not coin(key, step, _lam()):
            _stats["miss" if cached is None else "hit"] += 0  # counted below either way
            if cached is None:
                _stats["miss"] += 1
                if _stats["miss"] % 500 == 1:
                    tot = _stats["hit"] + _stats["miss"]
                    print(f"[simopd] gkd_mix: {_stats['hit']}/{tot} routed off-policy "
                          f"(misses are val/unseen prompts)", file=sys.stderr, flush=True)
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)

        _stats["hit"] += 1
        max_tokens = sampling_params.get("max_tokens") if isinstance(sampling_params, dict) else None
        resp = cached if max_tokens is None else cached[:max_tokens]
        score_params = dict(sampling_params) if isinstance(sampling_params, dict) else {}
        score_params.update({"max_tokens": 1, "prompt_logprobs": 0, "logprobs": None,
                             "temperature": 0.0, "n": 1})
        scored = await fn(self, list(prompt_ids) + list(resp), score_params, request_id, *a, **kw)
        from verl.workers.rollout.replica import TokenOutput

        extra = dict(getattr(scored, "extra_fields", None) or {})
        lps = tail_logprobs(extra, len(resp))
        extra.pop("prompt_logprobs", None); extra.pop("prompt_ids", None)
        return TokenOutput(token_ids=list(resp), log_probs=lps, routed_experts=None,
                           stop_reason="stop", extra_fields=extra)

    setattr(generate, _MARK, True)
    cls.generate = generate
    print(f"[simopd] gkd_mix armed on vLLMHttpServer.generate (lambda={_lam()})",
          file=sys.stderr, flush=True)
