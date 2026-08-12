"""13-gram overlap between the training set and every evaluation benchmark.

    python scripts/decontaminate.py                 # all benchmarks
    python scripts/decontaminate.py --n 13 --report contamination.md

BENCHMARKS.md §4's first hygiene item, owed since 2026-07-31. NVIDIA states
Nemotron-Cascade-RL-Math is decontaminated; this checks it rather than repeating it,
because "the provider says so" is not a hygiene table a reviewer will accept, and a
contaminated MATH500 would inflate every arm and vanilla equally -- invisible in the
comparisons this project is built on, and fatal to the absolute numbers next to them.

13-gram is the convention the field settled on (GPT-3, Llama, Qwen technical reports
all use n-gram overlap at this order). Normalisation is deliberately aggressive --
case, whitespace and punctuation folded -- because a leak that survives reformatting
is still a leak, and the false positives that adds are cheap to read.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

# A problem sharing this fraction of its n-grams with training data is a duplicate,
# not a coincidence. Below PARTIAL it is almost always shared phrasing.
DUP = float(os.environ.get("SIMOPD_DECON_DUP", "0.80"))
PARTIAL = float(os.environ.get("SIMOPD_DECON_PARTIAL", "0.40"))

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalise(text):
    """Fold case, punctuation and whitespace. A leak that survives reformatting is
    still a leak, so the comparison should not be defeated by a changed delimiter."""
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def ngrams(text, n):
    toks = normalise(text).split()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.expanduser("~/data/simopd_math"))
    p.add_argument("--n", type=int, default=13)
    p.add_argument("--report", default=None, help="write a markdown table here")
    p.add_argument("--examples", type=int, default=3, help="worst cases to show per benchmark")
    a = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pandas as pd
    from eval_offline import BENCHMARKS, load_benchmark

    train_path = os.path.join(a.data_dir, "train.parquet")
    if not os.path.exists(train_path):
        raise SystemExit(f"no training data at {train_path}; run scripts/fetch_assets.py")
    train = pd.read_parquet(train_path)
    # The prompt column carries the chat-templated question; the raw problem is what
    # a benchmark would share with it, so compare against the whole prompt text.
    col = next((c for c in ("prompt", "question", "problem") if c in train.columns), None)
    if col is None:
        raise SystemExit(f"no prompt-like column in {train_path}: {list(train.columns)}")

    def as_text(v):
        """verl stores prompts as a chat array of {role, content}, not a string.
        Comparing the repr of that array would match on the JSON scaffolding rather
        than on the problem, so pull the user turns out and compare those."""
        if isinstance(v, str):
            return v
        try:
            return " ".join(m["content"] for m in v if isinstance(m, dict) and "content" in m)
        except Exception:
            return str(v)

    texts = [as_text(x) for x in train[col].tolist()]
    print(f"training set: {len(texts)} rows from column {col!r} "
          f"(chat turns joined; mean {sum(len(t) for t in texts)//max(len(texts),1)} chars)")

    # One inverted index over the training set, reused for every benchmark: building it
    # per benchmark would re-tokenise 14k problems six times for nothing.
    index = defaultdict(set)
    for i, t in enumerate(texts):
        for g in ngrams(t, a.n):
            index[g].add(i)
    print(f"{len(index)} distinct {a.n}-grams indexed\n")

    # The math suite is not the only leak surface. A CODE training set's dangerous
    # overlaps are HumanEval+/MBPP+ (the offline judges), an IF set's is IFEval --
    # and the first code-domain run of this script printed five CLEAN rows without
    # ever looking at either. transfer_eval.load returns the same (problems, metas,
    # ids) shape as load_benchmark, so the transfer surfaces join the same loop.
    surfaces = [(name, load_benchmark) for name in BENCHMARKS]
    try:
        import transfer_eval
        surfaces += [(n, transfer_eval.load) for n in ("humanevalplus", "mbppplus", "ifeval")]
    except Exception as e:
        print(f"  transfer surfaces unavailable ({type(e).__name__}: {e}); "
              f"checking the math suite only")

    rows, excluded = [], {}
    for name, loader in surfaces:
        try:
            problems, _, ids = loader(name)
        except Exception as e:
            print(f"  {name}: could not load ({type(e).__name__}); skipped")
            continue
        hits = []
        for pid, text in zip(ids, problems):
            g = ngrams(text, a.n)
            if not g:
                continue
            shared = {x for x in g if x in index}
            if shared:
                overlap = len(shared) / len(g)
                hits.append((overlap, pid, sorted(shared)[:2]))
        hits.sort(reverse=True)
        # A shared 13-gram alone is not contamination. Competition problems are full of
        # boilerplate -- "can be written as m/n where m and n are relatively prime
        # positive integers" is thirteen tokens on its own -- so counting any hit makes
        # AIME look 30% contaminated when nothing leaked. What matters is what FRACTION
        # of a problem is shared: a duplicate reaches ~1.0, boilerplate stays low.
        dup = [h for h in hits if h[0] >= DUP]
        susp = [h for h in hits if PARTIAL <= h[0] < DUP]
        flag = "CLEAN" if not dup else ("DUPLICATES" if dup else "check")
        print(f"  {name:12s} {len(dup):>3} duplicate(s) >={DUP:.0%}, {len(susp)} partial "
              f"{PARTIAL:.0%}-{DUP:.0%}, {len(hits)} share any gram, of {len(problems)}"
              f"   -> {flag}")
        for ov, pid, ex in (dup + susp)[:a.examples]:
            print(f"      {pid}  {ov:.1%} shared   {ex[0][:66]!r}")
        if not dup and hits:
            ov, pid, ex = hits[0]
            print(f"      (worst is only {ov:.1%}: boilerplate, e.g. {ex[0][:56]!r})")
        rows.append((name, len(dup), len(susp), len(hits), len(problems), flag))
        excluded[name] = [pid for _, pid, _ in dup]

    if a.report:
        with open(a.report, "w") as f:
            f.write(f"# Decontamination: {a.n}-gram overlap vs the training set\n\n")
            f.write(f"Training set: `{train_path}`, {len(texts)} rows.\n\n")
            f.write(f"Duplicate = a problem sharing >={DUP:.0%} of its {a.n}-grams with training "
                    f"data. A single shared gram is not contamination: competition boilerplate "
                    f"(\"can be written as m/n where m and n are relatively prime positive "
                    f"integers\") is thirteen tokens by itself.\n\n")
            f.write(f"| benchmark | duplicates (>={DUP:.0%}) | partial ({PARTIAL:.0%}-{DUP:.0%}) "
                    f"| any shared gram | total | verdict |\n|---|---|---|---|---|---|\n")
            for name, d, sp, h, tot, flag in rows:
                f.write(f"| {name} | {d} | {sp} | {h} | {tot} | {flag} |\n")
        print(f"\nwrote {a.report}")
    # Freeze the ids. A contaminated problem inflates every arm AND vanilla equally, so
    # the comparisons this project judges on are unaffected -- but the absolute pass@1
    # printed beside them is not, and a reviewer will ask. Recording the list lets the
    # table carry both numbers instead of an apology.
    # Next to the training set it describes, NOT the repo's tracked data/ file:
    # the first code-domain run silently overwrote the math campaign's committed
    # record on a live worktree -- a tracked-file mutation waiting for the next
    # `git reset --hard` (the same fuse as the 2026-08-12 duplicate-run incident).
    out = os.path.join(a.data_dir, "contaminated_ids.json")
    with open(os.path.normpath(out), "w") as f:
        json.dump({"n": a.n, "threshold": DUP, "excluded": excluded}, f, indent=2)
    print(f"wrote {os.path.normpath(out)}")
    return 1 if any(r[5] == "DUPLICATES" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
