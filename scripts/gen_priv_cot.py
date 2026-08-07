"""Precompute the teacher's private scratchpad for every training prompt (i1_priv_cot).

    python scripts/gen_priv_cot.py --limit 8 --dry     # CPU: prefix construction only
    python scripts/gen_priv_cot.py                     # full run (GPU, ~hours, cached forever)

Design pinned in design-thinking-cells.md §2.1.5, restated where it binds:

  greedy, single, UNVERIFIED. Selecting scratchpads by ground truth would let the
      verifier's answer into supervision construction -- the g1-adjacent line. The
      teacher's own post-think answer is graded and STORED (cot_correct) for the
      pre-registered wrong-scratchpad sub-analysis, and never used for filtering.
  think cap 4096, forced close. A scratchpad cut mid-thought still closes with
      </think>; the truncated flag travels with the row.
  the cache row carries BOTH prefixes. The original prefix (empty think block, ids
      exactly as verl's template application produces them) is the match key at
      scoring time; the replacement prefix swaps in the real scratchpad. Injection
      happens by ids, never by re-tokenizing a joint string, so token boundaries
      cannot shift under the response.

Output parquet: one row per training prompt --
  prefix_hash    sha1 of the first 16 original-prefix token ids (fast lookup key)
  orig_ids       student-template prefix ids, taken from apply_chat_template
                 (tokenize=True) itself -- the match key is verl's own ids by definition
  repl_ids       same prefix with the empty think block replaced by the real one
  cot_correct    teacher's post-think boxed answer vs dataset ground truth (metric only)
  truncated      think block hit the cap and was force-closed
  think_tokens   scratchpad length
"""

import argparse
import hashlib
import os
import sys

EMPTY_THINK = "<think>\n\n</think>"


def _template_ids(tok, msgs, **kw):
    """verl's exact path: template's own tokenize + verl's own normalizer."""
    from verl.utils.tokenizer.tokenizer import normalize_token_ids  # verl's, so the key matches by construction

    return list(normalize_token_ids(tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, **kw)))


def build_prefixes(tok, content, think_text):
    """orig = P + B (empty block), repl = P + C (real block), spliced at p_len.

    Facts this rests on, all measured rather than assumed (two wrong versions ago):
    the Base template renders NO think block under enable_thinking=True and the empty
    CLOSED block under False, so P (the no-block render) is the shared prefix and B
    is the block span; and verl produces ids via apply_chat_template(tokenize=True)
    normalized by its own normalize_token_ids -- reproduced verbatim here, because a
    re-tokenized template STRING encodes special tokens differently and the match key
    would be fiction.
    """
    msgs = [{"role": "user", "content": content}]
    orig_ids = _template_ids(tok, msgs, enable_thinking=False)   # P + B
    p_ids = _template_ids(tok, msgs, enable_thinking=True)       # P (no block)
    if orig_ids[: len(p_ids)] != p_ids:
        raise RuntimeError("no-block render is not a prefix of the empty-block render; "
                           "splice point undefined for this tokenizer")
    c_ids = tok(f"<think>\n{think_text}\n</think>\n\n", add_special_tokens=False).input_ids
    return orig_ids, p_ids + list(c_ids), len(p_ids)


def prefix_hash(ids):
    """FULL-prefix hash; MUST stay verbatim-identical to simopd.gkd_mix.prompt_key
    (audit 2026-08-07 C1: the 16-token key collided on 635 real-data groups, 252
    with conflicting ground truths)."""
    return hashlib.sha1(",".join(map(str, list(ids))).encode()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--student", default="Qwen/Qwen3-1.7B-Base",
                   help="the scoring sequence lives in the STUDENT's template (protocol); "
                        "prefixes are built with this tokenizer. The teacher's template is "
                        "used only to GENERATE the scratchpad -- the -Thinking-2507 line "
                        "renders an OPEN think block regardless of enable_thinking, so it "
                        "has no empty-block anchor to swap at all")
    p.add_argument("--train-parquet", default=os.path.expanduser("~/data/simopd_math/train.parquet"))
    p.add_argument("--out", default=os.path.expanduser("~/data/simopd_math/priv_cot.parquet"))
    p.add_argument("--think-cap", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--dry", action="store_true", help="CPU only: build prefixes with a stub scratchpad")
    a = p.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.student)
    tok_t = AutoTokenizer.from_pretrained(a.teacher)
    assert tok.vocab_size == tok_t.vocab_size, "shared vocab is the protocol precondition"
    df = pd.read_parquet(a.train_parquet)
    if a.limit:
        df = df.iloc[: a.limit]
    contents, gts = [], []
    for _, row in df.iterrows():
        q = row["prompt"]
        contents.append(q[0]["content"] if hasattr(q, "__len__") and len(q) else str(q))
        rm = row.get("reward_model")
        gts.append(rm.get("ground_truth") if isinstance(rm, dict) else None)

    if a.dry:
        thinks = ["STUB SCRATCHPAD: consider the structure of the problem."] * len(contents)
        answers = [""] * len(contents)
    else:
        from vllm import LLM, SamplingParams

        open_texts = [tok_t.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                                add_generation_prompt=True)
                      for c in contents]
        llm = LLM(model=a.teacher, gpu_memory_utilization=a.gpu_mem_util,
                  max_model_len=a.think_cap + 1536, seed=0)
        # Greedy on purpose: ONE canonical scratchpad per prompt, no sampling knob to
        # tune, bit-reproducible cache. +512 past the cap lets the post-think answer
        # emerge for cot_correct; only the think segment is ever injected.
        outs = llm.generate(open_texts, SamplingParams(temperature=0.0, max_tokens=a.think_cap + 512))
        thinks, answers = [], []
        for o in outs:
            text = o.outputs[0].text
            if "</think>" in text:
                th, _, ans = text.partition("</think>")
                thinks.append(th.rstrip("\n")[: a.think_cap * 8]); answers.append(ans)
            else:
                thinks.append(text.rstrip("\n")); answers.append("")   # truncated: forced close, no graded answer

    from verl.utils.reward_score import default_compute_score

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_offline import MATH_SCORER_SOURCE

    rows = []
    for i, (content, think, ans, gt) in enumerate(zip(contents, thinks, answers, gts, strict=True)):
        orig_ids, repl_ids, p_len = build_prefixes(tok, content, think)
        correct = False
        if ans and gt is not None:
            sc = default_compute_score(MATH_SCORER_SOURCE, ans, gt)
            correct = float(sc.get("score", 0.0) if isinstance(sc, dict) else sc) > 0.5
        rows.append({"prefix_hash": prefix_hash(orig_ids), "orig_len": len(orig_ids), "p_len": p_len,
                     "orig_ids": list(orig_ids), "repl_ids": list(repl_ids),
                     "cot_correct": correct, "truncated": not bool(ans),
                     "think_tokens": len(repl_ids) - len(orig_ids)})
    out = pd.DataFrame(rows)
    dup = out["prefix_hash"].duplicated().sum()
    assert dup == 0, f"{dup} duplicate prefix hashes -- 16-token key insufficient, widen it"
    out.to_parquet(a.out)
    n_tr = int(out["truncated"].sum())
    print(f"{len(out)} rows -> {a.out}")
    print(f"  cot_correct rate {out['cot_correct'].mean():.3f}   truncated {n_tr} ({n_tr/len(out):.1%})"
          f"   think len p50 {out['think_tokens'].median():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
