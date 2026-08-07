"""Check the wiring before a run spends fourteen GPU-hours proving it was wrong.

    python scripts/preflight.py --student Qwen/Qwen3-1.7B-Base \
        --teacher Qwen/Qwen3-4B-Instruct-2507 --data ~/data/simopd_math/train.parquet \
        --val ~/data/simopd_math/math500.parquet --loss k1_rec

val_before_train used to do this incidentally: a step-0 MATH500 of 0.468 meant the
student and the chat template had loaded correctly, and 0.05 meant they had not. That
number is identical for every arm -- same base model, same greedy decode, same val set
-- so measuring it once per arm costs about 19 GPU-hours to re-learn one constant, and
it is now recorded rather than remeasured.

What it also bought, though, was the gate. Dropping a check because its main product
is redundant leaves the run unguarded against exactly the failures that make a
fourteen-hour run worthless. So the gate stays, without the generation: everything
below is a tokenizer load and a parquet header, and it is the same set of things a
step-0 accuracy could have told us apart.

Not covered: anything that only shows up in the optimiser. This asserts the inputs are
what the protocol says, not that training works.
"""

import argparse
import os
import sys

# The verl-path value. eval_offline.py reports 0.4740 for the same model, and the two
# are different pipelines; the curve's later points come from verl's validation, so its
# step-0 has to as well or the first segment of every arm mixes two measurements.
# 8k-BATCH anchor. The 16k batch (PROTOCOL 3.8, 2026-08-07) mints its own: the first
# 16k vanilla launches with VAL_BEFORE_TRAIN=True once, and that number replaces this
# for 16k-batch curves. Greedy non-thinking rarely nears either cap, so they may
# coincide -- measured, not assumed.
STEP0_MATH500 = 0.468


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--loss", required=True)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    a = p.parse_args()

    fail = []

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.student)

    # The protocol is non-thinking, forced by the student being a Base model. A template
    # that emits a think block silently changes what every arm distils, and the only
    # outward sign is a response-length distribution nobody compares against a run that
    # does not exist yet.
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "2+2=?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # Qwen3 renders non-thinking as an EMPTY, immediately-closed block -- that is the
    # correct output, not a failure. An OPEN block is the failure, and a first version
    # of this check called the correct case a failure by looking only for "<think>".
    # (4B-Instruct-2507 emits no block at all, being non-thinking-native. The two
    # differ legitimately: the teacher scores in the student's template.)
    body = rendered.split("<think>", 1)[1].split("</think>")[0] if "<think>" in rendered else ""
    if "<think>" in rendered and (body.strip() or "</think>" not in rendered):
        fail.append(f"student template opens a non-empty <think> block under "
                    f"enable_thinking=False:\n      {rendered!r}")

    tok_t = AutoTokenizer.from_pretrained(a.teacher)
    # The teacher never generates -- it scores the student's tokens -- so a vocabulary
    # mismatch does not crash, it silently scores the wrong token ids.
    if tok.vocab_size != tok_t.vocab_size:
        fail.append(f"vocab mismatch: student {tok.vocab_size} vs teacher "
                    f"{tok_t.vocab_size}. The teacher scores the student's token ids, "
                    f"so this yields wrong logprobs rather than an error.")

    import pandas as pd

    for label, path in (("train", a.data), ("val", a.val)):
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            fail.append(f"{label} parquet missing: {path}")
            continue
        df = pd.read_parquet(path)
        if not len(df):
            fail.append(f"{label} parquet is empty: {path}")
            continue
        print(f"  {label:5s} {len(df):>6d} rows  {os.path.basename(path)}")
        # Not a failure: filter_overlong_prompts=True drops these and truncation='error'
        # makes anything that slips through loud rather than silent. It is reported
        # because the effective dataset size is otherwise invisible -- the parquet says
        # 14476 and the run trains on about 12420.
        # The extraction is deliberate: these are numpy arrays of dicts, and an
        # isinstance(q, (list, tuple)) guard misses them, falls back to str(q), and
        # measures the repr -- which reported 44% overlong where the true figure is 14%.
        n = min(len(df), 500)
        over = sum(
            len(tok(q[0]["content"] if hasattr(q, "__len__") and len(q) and
                    isinstance(q[0], dict) else str(q)).input_ids) > a.max_prompt_length
            for q in df["prompt"].iloc[:n]) if "prompt" in df else 0
        if over:
            print(f"        {over}/{n} sampled prompts exceed max_prompt_length="
                  f"{a.max_prompt_length}; filter_overlong_prompts drops them, so the "
                  f"effective {label} set is ~{int(len(df) * (1 - over / n)):d} rows")

    # No sys.path fixing on purpose. The run inherits PYTHONPATH, and whether src/ is
    # on it is exactly the condition worth testing: without it k1_rec silently falls
    # back to stock k1, which is numerically identical and drops the Delta-ell panel
    # METRICS.md requires on every run -- a difference nothing downstream would show.
    try:
        import simopd.topk_losses  # noqa: F401  registers the custom modes
        from verl.trainer.distillation.losses import DISTILLATION_LOSS_REGISTRY

        if a.loss not in DISTILLATION_LOSS_REGISTRY:
            fail.append(f"loss mode {a.loss!r} is not registered. Known: "
                        f"{sorted(DISTILLATION_LOSS_REGISTRY)}")
        else:
            print(f"  loss  {a.loss} registered")
    except ImportError as e:
        fail.append(f"cannot import the loss registry ({e}); src/ is probably not on "
                    f"PYTHONPATH, and k1_rec would fall back to stock k1 without the "
                    f"Delta-ell panel METRICS.md requires on every run")

    # Every module sitecustomize's hooks import, checked on the machine that will run.
    # simopd.zmq_lane was absent from all four machines for a day -- a .gitignore
    # pattern meant `git add -A` never staged it -- while sitecustomize imported it,
    # printed one line, and left both ends of the weight-transfer socket unpatched.
    try:
        import sitecustomize

        for m in getattr(sitecustomize, "REQUIRED_MODULES", ()):
            try:
                __import__(m)
            except Exception as e:
                fail.append(f"{m} is not importable here ({type(e).__name__}). "
                            f"sitecustomize hooks import it; without it the patch it "
                            f"installs silently does not apply.")
        else:
            if getattr(sitecustomize, "REQUIRED_MODULES", None):
                print(f"  hooks {len(sitecustomize.REQUIRED_MODULES)} patch modules importable")
    except ImportError:
        fail.append("sitecustomize is not importable; none of the patches apply and "
                    "k1_rec would silently be stock k1")

    # ModelScope's three resurrections all shared one shape: a flag goes truthy
    # somewhere in the chain, and the death certificate is a vLLM stack at engine
    # init, fourteen identical times across lanes. Name it here instead, before a
    # launch is spent: truthy flags must be the deliberate knob AND importable.
    for var in ("VERL_USE_MODELSCOPE", "VLLM_USE_MODELSCOPE"):
        val = os.environ.get(var, "")
        if val.lower() in ("true", "1", "yes"):
            if os.environ.get("SIMOPD_USE_MODELSCOPE", "0") != "1":
                fail.append(f"{var}={val} but SIMOPD_USE_MODELSCOPE is not 1 -- the flag "
                            f"leaked in from somewhere (simopd_env.sh? an inherited shell?) "
                            f"rather than being chosen. run_opd_baseline force-assigns from "
                            f"the knob, so seeing this means the assignment was bypassed.")
            else:
                try:
                    import modelscope  # noqa: F401
                    print(f"  modelscope deliberately ON ({var}) and importable")
                except ImportError:
                    fail.append(f"{var}={val} (deliberate) but modelscope is not installed "
                                f"in this venv -- every vLLM engine will die at init.")

    print(f"  step0 MATH500 anchor {STEP0_MATH500} (recorded; val_before_train is off)")
    if fail:
        print("\npreflight FAILED:", file=sys.stderr)
        for f in fail:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("  preflight ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
