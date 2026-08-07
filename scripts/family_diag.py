"""Per-arm rollout diagnostics at a fixed step, grouped by loss support family.

    PYTHONPATH=src python scripts/family_diag.py [step]

Answers "what is each family actually DOING to the student", which the accuracy
table cannot: response length says whether the arm walked into Mode A, and entropy
says whether the policy collapsed or spread.

PARSING TRAP, cost three attempts. verl's metric lines reach the log through Ray,
which prefixes them -- but the ANSI colour escape comes BEFORE the paren:

    \\x1b[36m(TaskRunnerV1 pid=3925102)\\x1b[0m step:100 - actor/entropy:0.394 - ...

so a `^\\(...\\)` strip does not match, and anything anchored with match()/startswith()
after it finds nothing at all. Every regex here SEARCHES rather than anchors. The
failure is silent and total -- an empty table reads as "no data yet", not as a bug.
"""
import re
import sys
import glob
import subprocess
import os

RUN = re.compile(r"#+ RUN: ([A-Za-z0-9_.]+)")
STEP = re.compile(r"step:(\d+) - ")
KV = re.compile(r"([a-z_/]+):(-?[0-9.]+)")
STEP_WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = os.environ.get("SEED", "0")


def families():
    """arm -> sampled | topk | topk+samp, from the loss registry and each arm's env."""
    import simopd  # noqa: F401  -- registers our loss modes
    from verl.trainer.distillation.losses import get_distillation_loss_settings
    env = {**os.environ, "PYTHONPATH": "src"}
    out = {}
    for a in subprocess.run([sys.executable, "scripts/arm.py", "list"],
                            capture_output=True, text=True, env=env).stdout.split():
        blk = subprocess.run([sys.executable, "scripts/arm.py", "env", a],
                             capture_output=True, text=True, env=env).stdout
        e = dict(re.findall(r"^export (\w+)=(.*)$", blk, re.M))
        try:
            topk = get_distillation_loss_settings(e.get("DISTILLATION_LOSS_MODE", "k1_rec")).use_topk
        except Exception:
            continue
        out[a] = "sampled" if not topk else ("topk+samp" if e.get("SIMOPD_KEEP_SAMPLED") == "1" else "topk")
    return out


FAM = families()
want = {f"{a}_s{SEED}" for a in FAM}
got = {}
for f in sorted(glob.glob("logs/*/lane*.log")):
    cur = None
    for line in open(f, errors="replace"):
        m = RUN.search(line)
        if m:
            cur = m.group(1)
            continue
        if cur not in want:
            continue
        m = STEP.search(line)
        if m and int(m.group(1)) == STEP_WANT:
            got.setdefault(cur, dict(KV.findall(line[m.start():])))

print(f"step {STEP_WANT}, seed {SEED}\n")
print(f"{'arm':24s} {'family':11s} {'resp_len':>9s} {'entropy':>8s} {'clipfrac':>9s}")
order = {"sampled": 0, "topk": 1, "topk+samp": 2}
for a, fam in sorted(FAM.items(), key=lambda kv: (order.get(kv[1], 9), kv[0] != "vanilla", kv[0])):
    d = got.get(f"{a}_s{SEED}")
    if not d:
        continue
    print(f"{a:24s} {fam:11s} {float(d.get('response_length/mean', 0)):9.0f} "
          f"{float(d.get('actor/entropy', 0)):8.3f} {float(d.get('actor/pg_clipfrac', 0)):9.3f}")
