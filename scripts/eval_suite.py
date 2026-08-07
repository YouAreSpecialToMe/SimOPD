"""The campaign's official accuracy: one weighted number per checkpoint.

    python scripts/eval_suite.py run --model <ckpt> --run-id vanilla_s0 --step 50
    python scripts/eval_suite.py acc --run-id vanilla_s0 --step 50
    python scripts/eval_suite.py acc --run-id vanilla_s0            # all steps -> curve

Registered 2026-08-07 (PROTOCOL sec 3.7, user spec): the tracked accuracy is a
weighted combination of AIME24+25 avg@32, AMC23 avg@32, Minerva avg@3 and MATH500
avg@3, all at the protocol sampling triple (tau=0.7, top_p=0.95) and a 32,768-token
budget -- the field's eval convention of 2-4x the training cap (Rethinking 31,744;
TM 64,000), and the reason this suite lives OFFLINE on saved checkpoints rather
than inside verl's val loop: the in-loop path shares the training engine's response
cap and one global n, and a ~5.5k-generation suite would own the lane's wall-clock.
verl's in-loop greedy MATH500 stays as it is -- live health telemetry (Mode-A
detection cannot wait for an offline sweep) -- and the two never mix, same rule as
the 0.468/0.474 anchors.

Weights default to the equal macro average: AIME24+25 pool into one component by
problem (60 problems, one voice), AMC23, Minerva and MATH500 are the other three,
0.25 each. Problem-count pooling would hand MATH500 57% of the metric; equal-macro
is the least arbitrary default and --weights overrides it. The composite exists for
CURVES and checkpoint selection only -- verdicts stay per-benchmark paired tests
(McNemar/Wilcoxon need per-problem binaries, and a weighted scalar has none).

`run` drives eval_offline.py in two invocations (one vLLM load per n-group);
`acc` aggregates the newest artifact per benchmark into the composite.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (component, benchmarks pooled into it, samples per problem)
SUITE = [
    ("aime", ("aime24", "aime25"), 32),
    ("amc23", ("amc23",), 32),
    ("minerva", ("minerva",), 3),
    ("math500", ("math500",), 3),
]
DEFAULT_WEIGHTS = {name: 0.25 for name, _, _ in SUITE}
TEMPERATURE, TOP_P, MAX_TOKENS = 0.7, 0.95, 32768


def newest(out_dir, run_id, bench, step, k=None):
    """Newest artifact OF THE SUITE'S ERA. Filenames carry no sampling params and
    four eras already share this naming (greedy-8k, tau0.7/p1.0, tau0.7/p0.95/8k,
    the suite) -- unfiltered, a touched stale file silently rewrites suite_acc
    (audit 2026-08-07 F1: fixture moved it 0.449 -> 0.283 with zero warning)."""
    hits = sorted(glob.glob(os.path.join(out_dir, f"{run_id}__{bench}__step{step}__seed*.parquet")),
                  key=os.path.getmtime, reverse=True)
    import pandas as pd

    for h in hits:
        try:
            df = pd.read_parquet(h, columns=None)
        except Exception:
            continue
        t = float(df["temperature"].iloc[0]) if "temperature" in df else None
        n_ = df.groupby("problem_id").size().iloc[0] if "problem_id" in df else None
        if t is not None and abs(t - TEMPERATURE) > 1e-6:
            continue
        if k is not None and n_ is not None and int(n_) != int(k):
            continue
        if t is None:
            # era-blind legacy artifact: never silently admissible for the suite
            continue
        return h
    return None


def cmd_run(a):
    by_n = {}
    for _, benches, k in SUITE:
        by_n.setdefault(k, []).extend(benches)
    for k, benches in sorted(by_n.items(), reverse=True):
        cmd = [sys.executable, os.path.join(HERE, "eval_offline.py"),
               "--model", a.model, "--run-id", a.run_id, "--step", str(a.step),
               "--benchmarks", ",".join(benches), "--n", str(k),
               "--temperature", str(TEMPERATURE), "--top-p", str(TOP_P),
               "--max-tokens", str(MAX_TOKENS), "--out-dir", a.out_dir]
        print(f"[suite] n={k}: {' '.join(benches)}", flush=True)
        subprocess.run(cmd, check=True)
    return cmd_acc(a)


def cmd_acc(a):
    import pandas as pd

    weights = json.loads(a.weights) if a.weights else DEFAULT_WEIGHTS
    assert abs(sum(weights.values()) - 1.0) < 1e-6, f"weights must sum to 1: {weights}"

    steps = [a.step] if a.step is not None else sorted({
        int(os.path.basename(f).split("__step")[1].split("__")[0])
        for f in glob.glob(os.path.join(a.out_dir, f"{a.run_id}__*__step*.parquet"))})
    rows = []
    for step in steps:
        comp, missing = {}, []
        for name, benches, _k in SUITE:
            rates = []
            for b in benches:
                f = newest(a.out_dir, a.run_id, b, step, k=_k)
                if f is None:
                    missing.append(b)
                    continue
                # per-problem success rate, then the component pools PROBLEMS
                rates.append(pd.read_parquet(f).groupby("problem_id")["correct"].mean())
            if rates:
                comp[name] = float(pd.concat(rates).mean())
        if missing:
            print(f"step {step}: INCOMPLETE (missing {','.join(missing)}) -- no composite", flush=True)
            continue
        acc = sum(weights[k] * v for k, v in comp.items())
        rows.append({"step": step, "suite_acc": acc, **{f"acc_{k}": v for k, v in comp.items()}})
        detail = "  ".join(f"{k}={v:.4f}" for k, v in comp.items())
        print(f"step {step}: suite_acc={acc:.4f}   {detail}", flush=True)
    if rows and a.write:
        out = os.path.join(a.out_dir, f"{a.run_id}__suite_acc.parquet")
        pd.DataFrame(rows).to_parquet(out)
        print(f"curve -> {out}")
    return 0


def cmd_sweep(a):
    """After a run completes (user decision 2026-08-07: offline, post-training):
    one command evaluates every saved step. Idempotent -- steps whose five
    artifacts already exist are skipped, so it can rerun after late checkpoints."""
    steps = sorted(int(os.path.basename(os.path.dirname(d)).split("global_step_")[1])
                   for d in glob.glob(os.path.join(a.ckpt_dir, "global_step_*", "actor")))
    if not steps:
        sys.exit(f"no global_step_*/actor under {a.ckpt_dir}")
    print(f"[suite] {a.run_id}: {len(steps)} saved steps {steps}")
    for step in steps:
        have = all(newest(a.out_dir, a.run_id, b, step, k=k_)
                   for _, benches, k_ in SUITE for b in benches)
        if have:
            print(f"[suite] step {step}: artifacts complete, skip")
            continue
        ns = argparse.Namespace(model=os.path.join(a.ckpt_dir, f"global_step_{step}", "actor"),
                                run_id=a.run_id, step=step, out_dir=a.out_dir,
                                weights=a.weights, write=False)
        cmd_run(ns)
    return cmd_acc(argparse.Namespace(run_id=a.run_id, step=None, out_dir=a.out_dir,
                                      weights=a.weights, write=True))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "acc", "sweep"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-id", required=True)
        sp.add_argument("--step", type=int, default=None if name == "acc" else -1)
        sp.add_argument("--out-dir", default=os.environ.get("SIMOPD_EVALS",
                        os.path.expanduser("~/data/simopd_evals")))
        sp.add_argument("--weights", help='JSON, e.g. \'{"aime":0.4,"amc23":0.2,...}\'; must sum to 1')
        sp.add_argument("--write", action="store_true", help="acc: persist the curve parquet")
        if name == "run":
            sp.add_argument("--model", required=True)
        if name == "sweep":
            sp.add_argument("--ckpt-dir", required=True,
                            help="the run's checkpoint dir (holds global_step_*/)")
    a = p.parse_args()
    return {"run": cmd_run, "acc": cmd_acc, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
