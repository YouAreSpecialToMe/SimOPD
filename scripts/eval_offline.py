"""Offline benchmark evaluation for SimOPD (METRICS.md §6 item 1).

Serves four pre-registered jobs with one code path:
  - decision metrics on a checkpoint (MATH500 pass@1 greedy, AMC23/AIME avg@32)
  - the diversity side-effect panel (pass@k at tau=1.0 on the frozen 100-problem subset)
  - teacher ceilings / D6 ladder (same command, just point --model at a teacher)
  - the cross-domain transfer column (HumanEval+ / MBPP+ / IFEval), per arm

One script rather than two because `verdict.py` reads these parquets and every row
must be stamped with the same run_id / step / seed / git_sha in the same schema; a
separate transfer script would be a second place for that stamping to drift.
Benchmark-specific loading and scoring live in `transfer_eval.py`.

Scoring reuses verl's `default_compute_score` under the MATH data_source, i.e. the
exact scorer the in-training validation uses. METRICS.md forbids a second
implementation here: a scorer mismatch would silently shift pass@1 between the
training curve and the final table, and every verdict is a comparison of the two.

Writes the per-problem parquet that `verdict.py` consumes; the McNemar/Wilcoxon
tests are computed from these rows, never from wandb curves.

Examples
--------
  # student checkpoint, screening decision metrics
  python scripts/eval_offline.py --model /scratch/.../global_step_300/actor \
      --benchmarks math500 --run-id vanilla_s0 --step 300

  # teacher ceiling ladder (D6 input + Gap Recovery Rate denominator)
  python scripts/eval_offline.py --model Qwen/Qwen3-8B --benchmarks math500,amc23 \
      --run-id teacher_ceiling_8b --step -1

  # diversity panel
  python scripts/eval_offline.py --model ... --benchmarks math500_sub100 \
      --n 8 --temperature 1.0 --top-p 1.0 --run-id vanilla_s0 --step 300

  # cross-domain transfer column (greedy; 1083 prompts, ~15 min on one A100)
  python scripts/eval_offline.py --model /scratch/.../global_step_300/actor \
      --benchmarks humanevalplus,mbppplus,ifeval --max-tokens 2048 \
      --run-id vanilla_s0 --step 300
"""

import argparse
import glob
import json
import os
import subprocess
from datetime import datetime, timezone

import datasets
import pandas as pd

import transfer_eval   # sibling module; scripts/ is sys.path[0] when run as a script

# data_source that routes to verl's MATH scorer (boxed extraction + equivalence).
# Our training parquets carry this same string, which is what guarantees parity.
MATH_SCORER_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

# name -> (hf id, split, problem field, answer field)
# AIME24 uses the H4 mirror: math-ai/aime24 ships no answer column (only \boxed
# inside solution), verified 2026-07-31. See BENCHMARKS.md.
BENCHMARKS = {
    "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "answer"),
    "amc23": ("math-ai/amc23", "test", "question", "answer"),
    "aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
    "aime25": ("math-ai/aime25", "test", "problem", "answer"),
    "minerva": ("math-ai/minervamath", "test", "question", "answer"),
}
SUBSET_FILE = "data/math500_subset100.json"

# Local-parquet benchmarks: a domain campaign's own held-out split, read off the
# shared disk rather than the hub. This is deliberately the SAME file the training
# run validates against every 25 steps, so the offline column is comparable to the
# in-loop curve problem-for-problem, and adds the two things in-loop cannot give:
# avg@k and a frozen per-checkpoint artifact. Value is (path, data_source); the
# data_source is the row's own, so scoring lands on the identical verl scorer the
# training reward uses -- METRICS.md's no-second-implementation rule.
LOCAL_BENCHMARKS = {
    "codeval": (os.path.join(os.environ.get("SIMOPD_CODE_DIR",
                             "/mgfs/shared/Group_GY/changhao/simopd_data/simopd_code"),
                             "val_holdout.parquet"), "codecontests"),
}


def load_local_benchmark(name):
    """(prompts, ground_truths, ids, data_source) from a domain campaign's parquet.

    The prompt column already carries that domain's full instruction -- the prep
    script built it and the training loop feeds it verbatim -- so the caller must
    NOT append the math INSTRUCTION here. Doing so would ask a coding model for a
    \\boxed{} answer and the column would measure our prompt, not the model.

    Ids come from extra_info.index, the prep script's own row key, so an artifact
    keeps pairing with the right problem even if the file is ever rewritten in a
    different order (the positional fallback is the audit-2026-08-07 S2 hazard).
    """
    path, data_source = LOCAL_BENCHMARKS[name]
    df = pd.read_parquet(path)
    prompts, golds, ids = [], [], []
    for i, row in enumerate(df.itertuples(index=False)):
        p = row.prompt
        prompts.append(p[0]["content"] if len(p) and isinstance(p[0], dict) else str(p))
        golds.append(row.reward_model["ground_truth"])
        ei = getattr(row, "extra_info", None)
        idx = ei.get("index", i) if isinstance(ei, dict) else i
        ids.append(f"{name}/{idx}")
    return prompts, golds, ids, data_source


def _derived_max_len(model_path):
    """The model's own max context (config.json), so vLLM never gets asked past it."""
    try:
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(model_path)
        return int(getattr(c, "max_position_embeddings", 1 << 30))
    except Exception:
        return 1 << 30


def resolve_stop_contract(model_path):
    """The stop set a checkpoint TRAINED under. run_opd_baseline.sh pins it per run at
    <run_dir>/simopd_stop_contract.txt ('off' or a comma list); the eval --model path
    is <run_dir>/global_step_N/actor[/huggingface], so walk up a few levels. No pin
    (pre-contract runs, hub ids) -> 'off' = the legacy single-eos protocol every
    banked cell was produced under. Returns (raw_value, source)."""
    if isinstance(model_path, str) and os.path.isdir(model_path):
        d = os.path.abspath(model_path)
        for _ in range(4):
            pin = os.path.join(d, "simopd_stop_contract.txt")
            if os.path.isfile(pin):
                v = open(pin).read().strip() or "off"
                return v, "pin"
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return "off", "legacy"


def resolve_model(path):
    """Point at loadable weights, given a verl checkpoint directory.

    verl writes `<step>/actor/` as FSDP shards (model_world_size_*.pt) with the config
    and tokenizer in a `huggingface/` subdirectory beside them. Passing the actor dir
    to vLLM fails on the tokenizer first, which reads as a broken checkpoint rather
    than as the wrong path -- and if `hf_model` was not in save_contents there are no
    HF weights anywhere, which is a different problem with the same symptom. Separate
    the two, because one is a typo and the other means the run must be repeated.
    """
    if not os.path.isdir(path):
        return path                       # a hub id
    hf = os.path.join(path, "huggingface")
    if os.path.isdir(hf):
        weights = glob.glob(os.path.join(hf, "*.safetensors")) + glob.glob(os.path.join(hf, "*.bin"))
        if weights:
            print(f"[eval] verl checkpoint -> {hf}")
            return hf
        shards = glob.glob(os.path.join(path, "model_world_size_*.pt"))
        raise SystemExit(
            f"{path} holds FSDP shards ({len(shards)} found) and a huggingface/ dir with "
            "config and tokenizer but NO weights, so nothing can load it.\n"
            "  That run was trained without 'hf_model' in "
            "actor_rollout_ref.actor.checkpoint.save_contents (fixed in "
            "run_opd_baseline.sh 2026-08-05).\n"
            "  Recover this one with:  python -m verl.model_merger merge "
            f"--backend fsdp --local_dir {path} --target_dir {hf}"
        )
    return path


def load_benchmark(name):
    """Return (problems, answers, problem_ids). `math500_sub100` is the frozen
    diversity subset, selected by unique_id so a reordered upstream can't shift it."""
    base = "math500" if name == "math500_sub100" else name
    hf_id, split, q_field, a_field = BENCHMARKS[base]
    ds = datasets.load_dataset(hf_id, split=split)

    if name == "math500_sub100":
        with open(SUBSET_FILE) as f:
            keep = set(json.load(f)["unique_ids"])
        ds = ds.filter(lambda ex: ex["unique_id"] in keep)
        assert len(ds) == 100, f"frozen subset resolved to {len(ds)} rows, expected 100"
        ids = ds["unique_id"]
    else:
        # Stable ids where the dataset provides them: math500 ships unique_id, and a
        # positional fallback would pair wrong problems across artifacts if the hub
        # copy ever reorders (audit 2026-08-07 S2). The 16k batch regenerates all
        # artifacts, so the id-scheme switch lands at the batch boundary.
        if "unique_id" in ds.column_names:
            ids = ds["unique_id"]
        elif "id" in ds.column_names:
            ids = ds["id"]
        else:
            ids = [f"{name}/{i}" for i in range(len(ds))]

    return list(ds[q_field]), [str(a) for a in ds[a_field]], [str(i) for i in ids]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or checkpoint dir")
    p.add_argument("--benchmarks", default="math500", help="comma-separated; see BENCHMARKS")
    p.add_argument("--run-id", required=True, help="arm name, matches the wandb run")
    p.add_argument("--step", type=int, default=-1, help="training step; -1 for teachers/ceilings")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1, help="samples per problem (32 for avg@32, 8 for the panel)")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy pass@1; 0.7 for avg@32; 1.0 for the panel")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="the training cap (PROTOCOL 3.8: 16384; suite passes 32768, clamped to context)")
    p.add_argument("--think", action="store_true",
                   help="enable_thinking=True in the chat template (thinking-regime ceilings "
                        "and the annex cells; pair with --max-tokens 16384)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--stop-token-ids", default="auto",
                   help="comma-separated extra stop ids; 'off' = model default only; "
                        "'auto' (default) = the contract the checkpoint TRAINED under: the "
                        "run's pin file simopd_stop_contract.txt (written by "
                        "run_opd_baseline.sh) if one is found above --model, else 'off' "
                        "(pre-contract runs and hub ids: the legacy single-eos protocol). "
                        "The dual-terminator contract {151643,151645} (A-AXIS R5 appendix, "
                        "2026-08-19) therefore reaches eval automatically for v2 runs and "
                        "never re-scores a v1 run under a different protocol -- the ongoing "
                        "post-hoc drain reads these defaults from shared disk, so a global "
                        "flip would have split the campaign ledger mid-drain. Pass ids "
                        "explicitly for a cross-contract re-eval (R5 three-way verdict) -- "
                        "every artifact row records the set used and where it came from.")
    # 0.85 left ~10 GiB of an 80 GiB card unused, and KV capacity is the binding
    # constraint of this suite: at 32k per request the cache admits only 18
    # concurrent sequences, so decode runs at ~80 tok/s x 18 while the 1.7B model
    # itself is nowhere near compute-bound. vLLM's own "fully utilize" figure
    # (73.4 GiB KV, util ~0.99) leaves nothing over the measured 5.0 GiB of
    # weights + activation + CUDA graphs; 0.95 keeps a 3.4 GiB physical margin and
    # lifts concurrency to 20 (+11%). The cache is preallocated at engine init, so
    # an over-committed value fails in the first minutes rather than mid-cell --
    # the worker records rc != 0 and moves on, costing minutes, not hours.
    # Resource knob only: sampling, seeds and n are untouched, so cells stay
    # comparable to the ones already on disk.
    p.add_argument("--gpu-mem-util", type=float, default=0.95)
    p.add_argument("--parallel", type=int, default=None,
                   help="evalplus unit-test workers (code benchmarks only); default = cpu count")
    # One resolution chain for BOTH fleets, and it must match verdict.py's read dir
    # or the writer strands artifacts the reader never finds: SIMOPD_EVAL_ROOT is
    # what this fleet's simopd_env.sh has exported since 2026-08-07, SIMOPD_EVALS is
    # cornell's name for the same thing (audit S1: /scratch is node-local there and
    # strands artifacts), and the home path is the shared fallback.
    p.add_argument("--out-dir", default=os.environ.get("SIMOPD_EVAL_ROOT",
                   os.environ.get("SIMOPD_EVALS",
                   os.path.expanduser("~/data/simopd_evals"))),
                   help="default matches verdict.py's read dir")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score import default_compute_score

    args.model = resolve_model(args.model)

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        # Clamp to the model's own context: Qwen3-1.7B-Base derives 32768, and
        # asking for more makes vLLM refuse (or emit NaN logits past the limit
        # under the env-var override). The honest budget is min(requested,
        # context - prompt) -- Rethinking's 31,744 is exactly this convention
        # (audit 2026-08-07 F3; PROTOCOL 3.7 note).
        max_model_len=min(args.max_tokens + 1024, _derived_max_len(args.model)),
        seed=args.seed,
    )
    _stop_raw = (args.stop_token_ids or "").strip()
    _stop_src = "explicit"
    if _stop_raw.lower() == "auto":
        _stop_raw, _stop_src = resolve_stop_contract(args.model)
    _stop_ids = None
    if _stop_raw and _stop_raw.lower() != "off":
        _stop_ids = [int(x) for x in _stop_raw.split(",") if x.strip()]
        if not _stop_ids:
            raise SystemExit(f"--stop-token-ids parsed to zero ids: {_stop_raw!r}")
    print(f"stop contract for this eval: {_stop_ids or 'off'} (source: {_stop_src})", flush=True)
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        **({"stop_token_ids": _stop_ids} if _stop_ids else {}),
    )

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.out_dir, exist_ok=True)

    for bench in args.benchmarks.split(","):
        bench = bench.strip()
        is_transfer = bench in transfer_eval.TRANSFER
        is_local = bench in LOCAL_BENCHMARKS
        local_source = None

        if is_local:
            problems, golds, pids, local_source = load_local_benchmark(bench)
            metas = [{"answer": g} for g in golds]
            # Prompt verbatim from the parquet -- see load_local_benchmark. Chat
            # template and enable_thinking still match training, same as below.
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.think,
                )
                for q in problems
            ]
        elif is_transfer:
            raws, metas, pids = transfer_eval.load(bench)
            prompts = [
                transfer_eval.build_prompt(bench, raw, meta, tok)
                for raw, meta in zip(raws, metas, strict=True)
            ]
        else:
            problems, answers, pids = load_benchmark(bench)
            metas = [{"answer": a} for a in answers]
            # Same non-thinking template as training; a mismatch here would make the
            # offline number incomparable to the in-training val curve.
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": q.strip() + " " + INSTRUCTION}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.think,
                )
                for q in problems
            ]

        outputs = llm.generate(prompts, sampling)
        completions = [[c.text for c in out.outputs] for out in outputs]

        if is_transfer:
            # Batch-level on purpose: evalplus evaluates a whole samples file in one
            # call, under its own process isolation.
            extras = transfer_eval.score(
                bench, pids, metas, completions,
                workdir=os.path.join(args.out_dir, "_transfer"),
                stamp=f"{args.run_id}__step{args.step}__seed{args.seed}__{stamp}",
                parallel=args.parallel,
            )
        elif is_local:
            # verl routes codecontests to prime_code with continuous=True, so the
            # scorer returns the FRACTION of unit tests passed -- which is the right
            # shape for a training reward and the wrong shape for a pass@1 column.
            # `correct` is therefore all-tests-pass, the field convention; the raw
            # fraction rides along as pass_frac so the softer reading stays available
            # without a second scoring run.
            extras = []
            for meta, per_problem in zip(metas, completions, strict=True):
                per = []
                for text in per_problem:
                    score = default_compute_score(local_source, text, meta["answer"])
                    if isinstance(score, dict):
                        score = score.get("score", 0.0)
                    frac = float(score)
                    per.append({"correct": int(frac >= 1.0), "pass_frac": frac})
                extras.append(per)
        else:
            extras = []
            for meta, per_problem in zip(metas, completions, strict=True):
                per = []
                for text in per_problem:
                    score = default_compute_score(MATH_SCORER_SOURCE, text, meta["answer"])
                    if isinstance(score, dict):
                        score = score.get("score", 0.0)
                    per.append({"correct": int(float(score) > 0.5)})
                extras.append(per)

        rows = []
        for pid, out, per_problem in zip(pids, outputs, extras, strict=True):
            for sample_idx, (comp, extra) in enumerate(zip(out.outputs, per_problem, strict=True)):
                rows.append(
                    {
                        "run_id": args.run_id,
                        "git_sha": git_sha,
                        "seed": args.seed,
                        "step": args.step,
                        "benchmark": bench,
                        "problem_id": pid,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        # Measurement-contract marker: cells produced under different
                        # stop sets are different protocols and must never be compared
                        # unstated (mixed-ledger hazard, A-AXIS R5 appendix).
                        "stop_token_ids": ",".join(map(str, _stop_ids)) if _stop_ids else "off",
                        "stop_contract_source": _stop_src,   # pin | legacy | explicit
                        "sample_idx": sample_idx,
                        "resp_len": len(comp.token_ids),
                        "truncated": int(comp.finish_reason == "length"),
                        "finish_reason": str(comp.finish_reason),
                        "temperature": args.temperature,
                        "n": args.n,
                        # The full completion text (2026-08-12, user request). The suite
                        # samples at tau=0.7 and vLLM batch composition is nondeterministic,
                        # so a trajectory NOT saved here is gone -- the greedy textdump
                        # trick that rescued the collapse analysis cannot reproduce sampled
                        # rollouts (late-training-collapse.md section 4 is the receipt for
                        # what discarding text costs). Columnar parquet means readers that
                        # select columns (verdict.py, refill scoring, suite_acc) never pay
                        # for it; terminal-loop text compresses to almost nothing. Scoring
                        # is untouched: the column is written after `correct` is computed.
                        # SIMOPD_EVAL_SAVE_TEXT=0 restores the old artifact shape.
                        **({"response": comp.text}
                           if os.environ.get("SIMOPD_EVAL_SAVE_TEXT", "1") == "1" else {}),
                        **extra,   # always carries `correct`; transfer adds its own columns
                    }
                )

        df = pd.DataFrame(rows)
        path = os.path.join(args.out_dir, f"{args.run_id}__{bench}__step{args.step}__seed{args.seed}__{stamp}.parquet")
        df.to_parquet(path)

        per_problem = df.groupby("problem_id")["correct"].mean()
        if bench == "ifeval":
            metric = "strict_prompt_acc"
        else:
            metric = "pass@1" if args.n == 1 else f"avg@{args.n}"
        line = (
            f"[{bench}] {metric}={per_problem.mean():.4f}  "
            f"pass@{args.n}={df.groupby('problem_id')['correct'].max().mean():.4f}  "
            f"trunc={df['truncated'].mean():.4f}  len_mean={df['resp_len'].mean():.0f}"
        )
        if "n_instructions" in df:   # IFEval's pre-registered secondary metric
            line += f"  strict_instr_acc={df['n_instructions_followed'].sum() / df['n_instructions'].sum():.4f}"
        if "base_correct" in df:     # HumanEval/MBPP without the extra evalplus tests
            line += f"  base={df.groupby('problem_id')['base_correct'].mean().mean():.4f}"
        print(f"{line}  -> {path}")


if __name__ == "__main__":
    main()
