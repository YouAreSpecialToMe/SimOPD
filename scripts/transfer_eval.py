"""The cross-domain transfer column: HumanEval+ / MBPP+ / IFEval.

Loaders, prompt builders and scorers for the three non-math benchmarks, kept apart
from `eval_offline.py`'s math path because each one needs its own OFFICIAL harness:
BENCHMARKS.md §3 pins evalplus for code and Google's `instruction_following_eval`
for IF, and "official" is load-bearing -- the transfer column's whole value is that
it is comparable to the two audited papers that already report one (FiRe, Teachability).
A reimplementation would make our number a different number.

What this measures, and what it does not
----------------------------------------
Train on math, evaluate on code/IF: this is a SIDE-EFFECT measurement (did the trick
damage other domains), reported as Δ vs vanilla. It is not evidence about whether a
trick works when distilling IN those domains -- that needs code-domain distillation,
where the governing quantity is student tail mass pi(S-bar), which does not exist in
an eval with no teacher. See plan §0.

Sandboxing: evalplus runs model-generated code. Its `untrusted_check` applies a
reliability guard (destructive os/shutil calls stubbed out), per-test time limits and
process isolation, which is what makes this safe to run without docker on the cluster.

Load sensitivity (measured 2026-08-03, and the reason for MIN_TIME_LIMIT below)
------------------------------------------------------------------------------
Scoring HumanEval+'s own canonical solutions gave 160/164 on a box at load ~20 and a
stable 163/164 at load ~6 -- identical inputs, three spurious failures. evalplus caps
each test at max(min_time_limit, gt_time_limit_factor * reference_time), and for the
trivial-but-extreme-input tasks that resolves to the 1.0s floor, which CPU contention
alone can blow. Our own campaign is the contention: four training lanes saturate the
node. A trick would then appear to have destroyed code ability when nothing happened,
which is exactly the false verdict this column exists to avoid, so the floor is raised.

HumanEval/32 (`find_zero`) fails the plus tests at any time limit -- Newton's method
at tol 1e-5 on harder polynomials. 163/164 is the harness's reference ceiling, not a
defect of ours. Run `python scripts/transfer_eval.py --selfcheck` on a new machine to
confirm both harnesses reproduce their reference numbers there.
"""

import json
import os
import sys

TRANSFER = ("humanevalplus", "mbppplus", "ifeval")
_EVALPLUS_DATASET = {"humanevalplus": "humaneval", "mbppplus": "mbpp"}

# evalplus's own default is 1.0s; see the load-sensitivity note above.
MIN_TIME_LIMIT = float(os.environ.get("SIMOPD_EVALPLUS_MIN_TIME_LIMIT", 4.0))
# The reference numbers --selfcheck must reproduce: the plus tests scoring each
# dataset's own canonical solutions. Not 1.0 -- a handful of upstream canonical
# solutions genuinely fail the extended tests (HumanEval/32, Mbpp/255, Mbpp/630),
# which is the harness's ceiling and not something to go fix.
SELFCHECK_REFERENCE = {"humanevalplus": 163 / 164, "mbppplus": 376 / 378}

# evalplus's own chat prompt (codegen.py:176-177). Copied rather than imported
# because those live in a CLI-arg branch, not a constant.
_INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)


def _vendored_on_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tp = os.path.join(root, "third_party")
    if tp not in sys.path:
        sys.path.insert(0, tp)


class _NonThinking:
    """Pins `enable_thinking=False` on every chat-template call.

    evalplus builds its official prompt by calling `tokenizer.apply_chat_template`
    itself, with no way to pass the flag through. Our protocol requires the
    non-thinking template on every rollout and every eval (PROTOCOL §1), so evalplus
    gets a proxy that sets it and defers everything else to the real tokenizer.
    """

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return self._tok.apply_chat_template(*args, **kwargs)


# ---------------------------------------------------------------- loading

def load(name):
    """Return (raw_prompts, metas, problem_ids) for one transfer benchmark."""
    if name in _EVALPLUS_DATASET:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus

        probs = get_human_eval_plus() if name == "humanevalplus" else get_mbpp_plus()
        ids = sorted(probs, key=lambda t: int(t.split("/")[1]))
        return (
            [probs[i]["prompt"] for i in ids],
            [{"entry_point": probs[i]["entry_point"]} for i in ids],
            ids,
        )

    if name == "ifeval":
        import datasets

        ds = datasets.load_dataset("google/IFEval", split="train")
        metas = []
        for row in ds:
            # Every kwargs dict carries the union of all argument names with None
            # for the ones this instruction does not take; passing those through
            # would override the checkers' real defaults.
            metas.append(
                {
                    "prompt": row["prompt"],
                    "instruction_id_list": list(row["instruction_id_list"]),
                    "kwargs": [
                        {k: v for k, v in kw.items() if v is not None}
                        for kw in row["kwargs"]
                    ],
                }
            )
        return list(ds["prompt"]), metas, [str(k) for k in ds["key"]]

    raise KeyError(name)


def build_prompt(name, raw, meta, tok):
    """Chat-templated prompt string, non-thinking, one per problem."""
    if name in _EVALPLUS_DATASET:
        from evalplus.provider.utility import make_raw_chat_prompt

        # Ends inside an open ```python fence, so the model starts generating code
        # rather than prose about code. Without this prefill a base-tier student
        # fails mostly on markdown compliance, and the column would measure
        # formatting rather than the side effect we are after.
        return make_raw_chat_prompt(
            task_prompt=raw,
            instruction_prefix=_INSTRUCTION_PREFIX,
            response_prefix=_RESPONSE_PREFIX,
            tokenizer=_NonThinking(tok),
        )

    # IFEval constraints are about the response itself ("no commas", "at least 300
    # words"), so the instruction is passed through verbatim -- any appended
    # boilerplate of ours would be counted against the constraint it violates.
    return tok.apply_chat_template(
        [{"role": "user", "content": raw}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------- scoring

def _score_code(name, ids, metas, completions, workdir, stamp, parallel=None):
    from evalplus.eval import PASS
    from evalplus.evaluate import evaluate
    from evalplus.sanitize import sanitize

    os.makedirs(workdir, exist_ok=True)
    # A fresh path every call. evalplus reuses `<samples>_eval_results.json` when it
    # exists -- silently, or by blocking on input() -- and a batch job that stops to
    # ask a question is a job that burns its wall clock.
    samples = os.path.join(workdir, f"{name}__{stamp}.jsonl")
    with open(samples, "w") as f:
        for pid, meta, per_problem in zip(ids, metas, completions, strict=True):
            for text in per_problem:
                solution = sanitize(text, entrypoint=meta["entry_point"])
                f.write(json.dumps({"task_id": pid, "solution": solution}) + "\n")

    evaluate(
        dataset=_EVALPLUS_DATASET[name],
        samples=samples,
        parallel=parallel,
        min_time_limit=MIN_TIME_LIMIT,
    )

    with open(samples.replace(".jsonl", "_eval_results.json")) as f:
        res = json.load(f)["eval"]

    out = []
    for pid, per_problem in zip(ids, completions, strict=True):
        # evalplus sorts each task's results by completion_id, which increments in
        # the order we wrote them, so index j is our sample j.
        rows = res.get(pid, [])
        out.append(
            [
                {
                    "correct": int(rows[j]["plus_status"] == PASS) if j < len(rows) else 0,
                    "base_correct": int(rows[j]["base_status"] == PASS) if j < len(rows) else 0,
                }
                for j in range(len(per_problem))
            ]
        )
    return out


def _score_ifeval(ids, metas, completions):
    # nltk 3.10 blocks imports whose origin resolves under the CWD when nltk itself
    # triggered them. Our venv lives at <repo>/simopd, so from the repo root EVERY
    # site-package looks like a CWD import and nltk refuses to load `regex`. It is a
    # false positive of our layout, not a shadowed module; this is the documented
    # opt-out, and it must be set before the first nltk import.
    os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")
    _vendored_on_path()
    from instruction_following_eval import instructions_registry

    out, n_raised = [], 0
    for pid, meta, per_problem in zip(ids, metas, completions, strict=True):
        rows = []
        for text in per_problem:
            followed = []
            # Mirrors evaluation_lib.test_instruction_following_strict verbatim,
            # including the rebuild for instructions that take the prompt back
            # (combination:repeat_prompt and friends) -- their description is
            # constructed from it, so skipping the rebuild scores them against an
            # empty target and they pass for free.
            for index, instr_id in enumerate(meta["instruction_id_list"]):
                instruction = instructions_registry.INSTRUCTION_DICT[instr_id](instr_id)
                instruction.build_description(**meta["kwargs"][index])
                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    instruction.build_description(prompt=meta["prompt"])
                try:
                    ok = bool(text.strip()) and instruction.check_following(text)
                except Exception:
                    # A degenerate response can make a checker raise (langdetect on
                    # punctuation-only output, say). That is a failure to follow, not
                    # a reason to lose the run -- but it is counted and reported,
                    # because a systematic checker fault would otherwise read as a
                    # genuine score drop.
                    ok = False
                    n_raised += 1
                followed.append(ok)
            rows.append(
                {
                    # BENCHMARKS.md §3: strict accuracy, prompt-level primary.
                    "correct": int(all(followed)),
                    "n_instructions": len(followed),
                    "n_instructions_followed": int(sum(followed)),
                }
            )
        out.append(rows)
    if n_raised:
        print(f"  [ifeval] {n_raised} checker exception(s) scored as not-followed")
    return out


def score(name, ids, metas, completions, workdir, stamp, parallel=None):
    """completions[i][j] -> per-sample dict with at least `correct`."""
    if name in _EVALPLUS_DATASET:
        return _score_code(name, ids, metas, completions, workdir, stamp, parallel)
    if name == "ifeval":
        return _score_ifeval(ids, metas, completions)
    raise KeyError(name)


# ---------------------------------------------------------------- self-check

def selfcheck(workdir, parallel=None):
    """Score each code dataset's own canonical solutions and compare to reference.

    Answers the question a new cluster actually raises -- does model-generated code
    execute here at all, and does it execute fast enough not to time out -- without
    a GPU or a checkpoint. This is BENCHMARKS.md §4.5's open sandbox item, in a form
    that can be re-run whenever the machine changes.
    """
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    # Each forked test worker pages in a share of MBPP+'s 793MB expected-output
    # table. On a login node with a couple of GB free that does not fail, it stalls
    # -- measured: 28 minutes and still going, versus 1m20s on a compute node.
    try:
        with open("/proc/meminfo") as f:
            avail_gb = int([l for l in f if l.startswith("MemAvailable")][0].split()[1]) / 2**20
        if avail_gb < 16:
            print(f"  WARNING: {avail_gb:.1f}GB available; this wants ~12GB at parallel=8.")
            print("           Run it on a compute node (slurm/transfer_selfcheck.sbatch),")
            print("           or lower --parallel. A login node will stall, not fail.")
    except Exception:
        pass

    ok = True
    for name in _EVALPLUS_DATASET:
        probs = get_human_eval_plus() if name == "humanevalplus" else get_mbpp_plus()
        raws, metas, ids = load(name)
        # Fenced, so sanitize() is exercised the way a real response exercises it.
        comps = [
            [f"```python\n{probs[i]['prompt']}{probs[i]['canonical_solution']}\n```"]
            for i in ids
        ]
        res = _score_code(name, ids, metas, comps, workdir, f"selfcheck_{name}", parallel)
        acc = sum(r[0]["correct"] for r in res) / len(res)
        fails = [ids[k] for k, r in enumerate(res) if not r[0]["correct"]]
        ref = SELFCHECK_REFERENCE.get(name)
        verdict = "ok" if ref is None else ("ok" if abs(acc - ref) < 1e-9 else "MISMATCH")
        ok &= verdict == "ok"
        print(f"  {verdict:9s} {name}: {acc:.4f}" + (f" (reference {ref:.4f})" if ref else ""))
        if fails:
            print(f"            failing: {fails[:10]}{' ...' if len(fails) > 10 else ''}")
    if not ok:
        print("  A mismatch here is usually timeouts under load, not a broken harness:")
        print(f"  retry on an idle node, or raise SIMOPD_EVALPLUS_MIN_TIME_LIMIT (now {MIN_TIME_LIMIT}).")
    return ok


if __name__ == "__main__":
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selfcheck", action="store_true", help="verify the code harnesses on this machine")
    ap.add_argument("--parallel", type=int, default=None)
    a = ap.parse_args()
    if not a.selfcheck:
        ap.error("nothing to do; pass --selfcheck")
    with tempfile.TemporaryDirectory() as d:
        sys.exit(0 if selfcheck(d, a.parallel) else 1)
