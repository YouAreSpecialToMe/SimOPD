"""Dual-terminator contract: student rollout stops on the union stop set.

The pairing is a BASE student (eos <|endoftext|>=151643) distilled from an
INSTRUCT teacher (eos <|im_end|>=151645, generation_config lists BOTH ids).
vLLM stops each served model on its OWN generation_config eos, so the student's
sampler never honored the terminator the objective teaches: k1 on student
rollouts suppresses the student's only stop token (teacher puts ~0 mass there),
while teacher-authored data teaches a stop token the sampler refuses. Measured
2026-08-19: base student pre-training stops 100%/257 tok; vanilla@250 stops
8.3%/15k tok, repeating "Final Answer" 980x to the cap; a2_coldstart@25 (pure
teacher text) truncates 98.5% on aime24. The registered fix (A-AXIS R5
appendix) is the teacher's own stop SET {151643, 151645} -- the union of both
conventions, invented by neither.

Env: SIMOPD_STOP_IDS (comma-separated token ids). Unset/empty/"off" -> no-op
(legacy contract). The launcher pins the value per run (stop_contract file in
ckpt_dir) and the var enters the run fingerprint, so a run can never change
contract mid-curve: pre-contract runs resume with the var UNSET (fingerprint
byte-identical to history), new runs carry it from step 0.

Seam: the same vLLMHttpServer.generate wrap as gkd_mix/a5/h_horizon, installed
LAST (outermost). Scoring probes (prompt_logprobs) pass through untouched --
their params stay byte-identical to the recorded protocol. Teacher servers in
mixer arms share the class; injecting the teacher's own set there is a no-op
by construction. STATELESS by design: the closure holds one immutable tuple,
so the cloudpickle by-value law (gkd_mix incident) has nothing to bite.
"""

import os
import sys

_MARK = "_simopd_stop_set"


def parse_ids(raw):
    """None when the contract is off; else a validated, deduped id tuple."""
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "" or raw.lower() == "off":
        return None
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            raise RuntimeError(f"stop_set: SIMOPD_STOP_IDS entry {part!r} is not an "
                               f"integer token id (full value: {raw!r})")
        if not (1 <= v < 1 << 20):
            raise RuntimeError(f"stop_set: token id {v} outside (0, 2^20) -- not a "
                               f"plausible vocab id; refusing a contract that would "
                               f"silently never fire")
        if v not in ids:
            ids.append(v)
    if not ids:
        raise RuntimeError(f"stop_set: SIMOPD_STOP_IDS={raw!r} parsed to zero ids -- "
                           f"set it to 'off' to mean off, never to garbage")
    return tuple(ids)


def install():
    ids = parse_ids(os.environ.get("SIMOPD_STOP_IDS"))
    if ids is None:
        return
    mod = sys.modules.get("verl.workers.rollout.vllm_rollout.vllm_async_server")
    if mod is None:
        return
    cls = getattr(mod, "vLLMHttpServer", None)
    fn = getattr(cls, "generate", None) if cls else None
    if fn is None:
        raise RuntimeError("stop_set: vLLMHttpServer.generate not found -- verl moved "
                           "it; the run would train under a contract fingerprint whose "
                           "contract never applied")
    if getattr(fn, _MARK, False):
        return

    async def generate(self, prompt_ids, sampling_params, request_id, *a, **kw):
        if not isinstance(sampling_params, dict):
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)
        if sampling_params.get("prompt_logprobs") is not None:
            return await fn(self, prompt_ids, sampling_params, request_id, *a, **kw)
        sp = dict(sampling_params)
        have = sp.get("stop_token_ids") or []
        sp["stop_token_ids"] = [t for t in have if t not in ids] + list(ids)
        return await fn(self, prompt_ids, sp, request_id, *a, **kw)

    setattr(generate, _MARK, True)
    cls.generate = generate
    print(f"[simopd] stop_set armed on vLLMHttpServer.generate "
          f"(stop_token_ids {list(ids)})", file=sys.stderr, flush=True)
