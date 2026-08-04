"""Live status of every SimOPD run, on either cluster.

    python scripts/watch.py                # one-shot table
    python scripts/watch.py --watch 60     # refresh every 60s
    python scripts/watch.py --run vanilla_s0   # one run, with its val trajectory

Reads the campaign/lane logs rather than wandb: verl's console logger emits every
metric on a `step:N - key:value - ...` line, the format is identical under slurm
and under the DSW lanes, and it keeps working with wandb offline or unreachable.
wandb's own summary file is only written at the end of a run, so it is useless
for watching one in flight.

The health checks are not generic -- each one is a failure this project actually
had, and each cost real GPU hours:

  STALLED    no new step in a while. An 18-minute silent startup once looked like
             a hang; it was a 14.5s-per-process import tax paid by every vLLM worker.
  MODE-A     response length climbing and step time degrading together. This is what
             turned a 300-step run into >24h and got it killed at step 229.
  NO-CKPT    past SAVE_FREQ steps with nothing on disk. That same run left zero
             checkpoints and zero metrics -- 24 GPU-hours for nothing.
  DISK       verl keeps every checkpoint by default; a 17-run campaign is ~850GB
             unless capped, and it fills the volume mid-flight rather than at the start.
  EARLY-STOP-DUE
             the pre-registered stop rule (below) fires. `--enforce` acts on it.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone

STEP_RE = re.compile(r"step:(\d+) - (.*)")
KV_RE = re.compile(r"([\w/@\-.]+):(?:np\.float64\()?(-?[\d.eE+]+)\)?")
RUN_RE = re.compile(r"#+ RUN: (\S+) #+")
DONE_RE = re.compile(r"#+ (\S+) -> (OK|FAIL) #+")
TQDM_RE = re.compile(r"\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)s/it\]")

STALL_MIN = 25          # startup is legitimately ~5 min; 25 means something is wrong
MODE_A_GROWTH = 1.5     # response length this many times its early value

# ---------------------------------------------------------------- early stop
# Pre-registered 2026-08-04 from vanilla_s0 (job 719188), measured on our own stack:
#
#   step  25   val 0.468   len 1.6k   clip 0.05   82 s/it   entropy 1.40
#   step  40                                                entropy 0.15  <- collapse
#   step 100   val 0.392   len 7.8k   clip 0.95  457 s/it   entropy 0.13  <- Mode A
#   step 200   val 0.456   len 8.2k   clip 1.00  482 s/it   entropy 0.10
#
# val pass@1 was flat from step 25 on; steps 100-300 are 26 hours of rollouts that
# every single one hits the length cap. 85% of the wall clock buys nothing.
#
# The rule must be IDENTICAL for every arm and the stop step must be reported, not
# hidden: a per-arm stopping point that nobody records is precisely the "no two
# papers are line-comparable" disease this project exists to diagnose. So two arms
# are compared at their minimum common step, and "survived to H without Mode A" is
# itself a reportable outcome -- which is what the F axis (soft-log, hard clip) is
# for, and why the rule may not fire before MIN_STEPS.
STOP_MIN_STEPS = 50     # never stop before a plateau could even be established
# 0.90, not 0.95: once Mode A is established the ratio oscillates in 0.90-0.97
# (measured, steps 88-105), so a 0.95 cut is broken by noise rather than by
# recovery and delays the stop 39 steps -- 5 hours -- for nothing. At 0.90 the rule
# fires at step 100, which is where the transition actually completes.
STOP_CLIP = 0.90        # "essentially every rollout is hitting the cap"
STOP_CLIP_WINDOW = 10   # consecutive steps of it, so one long batch cannot trip it
STOP_VAL_PATIENCE = 3   # val evaluations (TEST_FREQ=25 -> 75 steps) without gain
# Provisional until W1 delivers the measured noise floor (vanilla x3 seeds); the
# whole point of the noise floor is that "no gain" needs an operational definition.
STOP_VAL_EPS = 0.02


def stop_verdict(steps, val, min_steps=STOP_MIN_STEPS, clip=STOP_CLIP,
                 window=STOP_CLIP_WINDOW, patience=STOP_VAL_PATIENCE, eps=STOP_VAL_EPS):
    """Should this run be stopped now? -> (stop: bool, reason: str, onset: int|None).

    Both conditions are required. Degeneration alone can still be improving, and a
    val plateau alone can be a shelf before a later gain; only together do they mean
    the remaining steps cannot change the verdict.
    """
    if not steps or steps[-1]["step"] < min_steps:
        return False, "", None

    tail = [s for s in steps[-window:] if s.get("clip") is not None]
    degenerate = len(tail) == window and all(s["clip"] >= clip for s in tail)

    onset = None
    if degenerate:                       # first step of the current unbroken run
        for s in reversed(steps):
            if s.get("clip") is None or s["clip"] < clip:
                break
            onset = s["step"]

    accs = [a for _, a in val]
    saturated = len(accs) > patience and max(accs[-patience:]) <= max(accs[:-patience]) + eps

    if degenerate and saturated:
        return True, f"Mode A from step {onset}, val flat for {patience} evals", onset
    return False, "", onset


def parse_log(path):
    """Return {run_name: record} for every run that appears in this log."""
    runs, current = {}, None
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = RUN_RE.search(line)
                if m:
                    current = m.group(1)
                    runs.setdefault(current, {"name": current, "log": path, "steps": [],
                                              "val": [], "status": "running", "tqdm": None})
                    continue
                m = DONE_RE.search(line)
                if m and m.group(1) in runs:
                    runs[m.group(1)]["status"] = m.group(2)
                    continue
                if current is None:
                    continue
                r = runs[current]
                m = TQDM_RE.search(line)
                if m:
                    r["tqdm"] = (int(m.group(1)), int(m.group(2)), m.group(4), float(m.group(5)))
                m = STEP_RE.search(line)
                if not m:
                    continue
                step, kv = int(m.group(1)), dict(KV_RE.findall(m.group(2)))

                def g(key):
                    v = kv.get(key)
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None

                acc = next((float(v) for k, v in kv.items() if "val-core" in k and "acc" in k), None)
                if acc is not None:
                    r["val"].append((step, acc))
                if g("response_length/mean") is not None:
                    r["steps"].append({
                        "step": step,
                        "len": g("response_length/mean"),
                        "clip": g("response_length/clip_ratio"),
                        "t": g("timing_s/step"),
                        "entropy": g("actor/entropy"),
                    })
    except FileNotFoundError:
        return {}
    for r in runs.values():
        r["mtime"] = os.path.getmtime(path)
    return runs


def checkpoints(ckpt_root, run):
    for project in glob.glob(os.path.join(ckpt_root, "*")):
        d = os.path.join(project, run)
        if os.path.isdir(d):
            steps = [int(x.rsplit("_", 1)[1]) for x in os.listdir(d)
                     if x.startswith("global_step_") and x.rsplit("_", 1)[1].isdigit()]
            if steps:
                return max(steps), len(steps)
    return None, 0


def flags(r, ckpt_step, save_freq, live_jobs=None, ckpt_root=None):
    out = []
    idle_min = (time.time() - r["mtime"]) / 60
    if r["status"] == "running" and idle_min > STALL_MIN:
        # A run whose job is gone is abandoned, not stuck: nothing to wait for and
        # its numbers are void. Distinguishing them decides whether you debug or re-queue.
        job = re.search(r"-(\d{5,})\.out$", r["log"])
        gone = live_jobs is not None and job and job.group(1) not in live_jobs
        out.append(f"ABANDONED({idle_min:.0f}m)" if gone else f"STALLED({idle_min:.0f}m)")
    s = r["steps"]
    if len(s) >= 6:
        early = sum(x["len"] for x in s[:3]) / 3
        late = sum(x["len"] for x in s[-3:]) / 3
        t_early = [x["t"] for x in s[:3] if x["t"]]
        t_late = [x["t"] for x in s[-3:] if x["t"]]
        if early > 0 and late / early > MODE_A_GROWTH and t_early and t_late:
            if sum(t_late) / len(t_late) > sum(t_early) / len(t_early) * 1.5:
                out.append(f"MODE-A(len x{late / early:.1f})")
    if s and save_freq > 0 and s[-1]["step"] > save_freq and ckpt_step is None:
        # Only when the checkpoint root is actually reachable. /scratch is mounted on
        # the compute nodes and not on the login node, so from there every run looks
        # checkpoint-less -- and a warning that is usually wrong gets ignored when it
        # is right, which is the one thing this flag exists for.
        if ckpt_root is None or os.path.isdir(ckpt_root):
            out.append("NO-CKPT")
    if r["status"] == "running":
        stop, why, _ = stop_verdict(s, r["val"])
        if stop:
            out.append("EARLY-STOP-DUE")
    return out


def _live_jobs():
    try:
        q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"],
                           capture_output=True, text=True, timeout=10)
        return {x.strip() for x in q.stdout.split()}
    except Exception:
        return None


def collect(log_dir, ckpt_root, save_freq):
    live = _live_jobs()
    runs = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*.out")) +
                       glob.glob(os.path.join(log_dir, "*.log"))):
        for name, r in parse_log(path).items():
            # a run re-run later supersedes the earlier attempt
            if name not in runs or r["mtime"] > runs[name]["mtime"]:
                runs[name] = r
    for name, r in runs.items():
        r["ckpt_step"], r["ckpt_n"] = checkpoints(ckpt_root, name)
        r["flags"] = flags(r, r["ckpt_step"], save_freq, live, ckpt_root)
    return runs


def render(runs, ckpt_root):
    now = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    print(f"\n=== SimOPD runs @ {now} ===")
    if not runs:
        print("  (no runs found -- is LOG_DIR right?)")
        return
    # `age` and `src` matter: a cancelled or superseded job leaves its runs in the
    # logs looking exactly like live ones, and acting on a void run's numbers is
    # worse than having none.
    hdr = (f"{'run':<34}{'state':<9}{'step':>9}{'val@1':>7}{'len':>7}{'trunc':>6}"
           f"{'s/it':>6}{'age':>6}  {'src':<12} flags")
    print(hdr); print("-" * len(hdr))
    for name in sorted(runs):
        r = runs[name]
        s = r["steps"][-1] if r["steps"] else None
        prog = f"{r['tqdm'][0]}/{r['tqdm'][1]}" if r["tqdm"] else (str(s["step"]) if s else "-")
        val = f"{r['val'][-1][1]:.3f}" if r["val"] else "-"
        ln = f"{s['len']:.0f}" if s and s["len"] else "-"
        tr = f"{s['clip']:.2f}" if s and s["clip"] is not None else "-"
        it = f"{s['t']:.0f}" if s and s["t"] else "-"
        ck = f"{r['ckpt_step']}" if r["ckpt_step"] is not None else "-"
        state = {"OK": "done", "FAIL": "FAILED"}.get(r["status"], "running")
        age_m = (time.time() - r["mtime"]) / 60
        age = f"{age_m:.0f}m" if age_m < 90 else f"{age_m / 60:.0f}h"
        src = os.path.basename(r["log"]).replace("simopd-campaign-", "job").replace(".out", "")[:12]
        print(f"{name:<34}{state:<9}{prog:>9}{val:>7}{ln:>7}{tr:>6}{it:>6}{age:>6}  {src:<12} "
              f"{'ckpt@' + ck + ' ' if ck != '-' else ''}{' '.join(r['flags'])}")

    if os.path.isdir(ckpt_root):
        used = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(ckpt_root) for f in fs
                   if os.path.exists(os.path.join(dp, f))) / 2**30
        free = shutil.disk_usage(ckpt_root).free / 2**30
        warn = "  <-- DISK: verl keeps all checkpoints unless MAX_CKPT_KEEP is set" if free < 100 else ""
        print(f"\ncheckpoints {used:.0f}G used, {free:.0f}G free{warn}")

    try:
        q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h",
                            "-o", "%.10i %.18j %.8T %.10M"], capture_output=True, text=True, timeout=10)
        rows = [x for x in q.stdout.splitlines() if "simopd" in x]
        if rows:
            print("\nslurm:"); [print("  " + x) for x in rows]
    except Exception:
        pass


def enforce(runs, ledger="logs/early_stops.tsv"):
    """Cancel runs the stop rule fires on, and record WHY in a ledger.

    The record is the point. An unrecorded early stop makes two arms silently
    incomparable; a recorded one makes the stop step a measured quantity -- how long
    an arm held off Mode A is exactly what the F axis is judged on. verdict.py reads
    this to compare arms at their minimum common step.
    """
    for r in sorted(runs.values(), key=lambda x: x["name"]):
        if r["status"] != "running":
            continue
        stop, why, onset = stop_verdict(r["steps"], r["val"])
        if not stop:
            continue
        job = re.search(r"-(\d{5,})\.out$", r["log"])
        step = r["steps"][-1]["step"]
        line = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{r['name']}\t"
                f"{job.group(1) if job else '?'}\t{step}\t{onset}\t{why}\n")
        os.makedirs(os.path.dirname(ledger) or ".", exist_ok=True)
        if os.path.exists(ledger):
            with open(ledger) as f:
                if any(f"\t{r['name']}\t" in ln for ln in f):
                    continue      # already stopped and recorded; do not cancel twice
        with open(ledger, "a") as f:
            f.write(line)
        print(f"EARLY STOP {r['name']} at step {step}: {why}")
        if job:
            subprocess.run(["scancel", job.group(1)], timeout=20)
            print(f"  scancelled job {job.group(1)}; recorded in {ledger}")
        else:
            print("  could not identify the slurm job from the log name; not cancelled")


def main():
    p = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--log-dir", default=os.environ.get("LOG_DIR", os.path.join(root, "logs")))
    p.add_argument("--ckpt-root", default=os.environ.get("CKPT_ROOT", "/scratch/zz865/simopd/ckpt"))
    p.add_argument("--save-freq", type=int, default=int(os.environ.get("SAVE_FREQ", "50")))
    p.add_argument("--watch", type=int, metavar="SEC", help="refresh every SEC seconds")
    p.add_argument("--run", help="show one run's validation trajectory")
    p.add_argument("--enforce", action="store_true",
                   help="scancel runs the pre-registered stop rule fires on (use with --watch)")
    args = p.parse_args()

    while True:
        runs = collect(args.log_dir, args.ckpt_root, args.save_freq)
        if args.enforce:
            enforce(runs)
        if args.run:
            r = runs.get(args.run)
            if not r:
                raise SystemExit(f"no run named {args.run}; have: {', '.join(sorted(runs)) or '(none)'}")
            print(f"\n=== {args.run} ({r['status']}) — {r['log']} ===")
            print("val trajectory (MATH500 pass@1):")
            for step, acc in r["val"]:
                print(f"  step {step:>4}  {acc:.4f}")
            print("\nstep, response_length, trunc_ratio, s/it, entropy:")
            for s in r["steps"][-15:]:
                print(f"  {s['step']:>4}  {s['len'] or 0:>8.0f}  {s['clip'] or 0:>5.2f}  "
                      f"{s['t'] or 0:>6.1f}  {s['entropy'] or 0:>5.2f}")
        else:
            render(runs, args.ckpt_root)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
