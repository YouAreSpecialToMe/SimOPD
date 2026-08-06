"""Every arm's numbers at one common step, side by side.

    python scripts/crossarm.py            # newest common step
    python scripts/crossarm.py --step 50
    python scripts/crossarm.py --all      # every val step each arm has reached

progress.py answers "what is running" and watch.py answers "how is THIS run doing".
Neither answers "how do the arms compare right now", which is the question the
audit is actually made of -- and reading it off sixteen logs by hand invites
comparing arms at different steps, which is the one comparison the protocol
forbids (plan §4: cross-arm comparison takes the smallest COMMON step).

So this refuses to mix steps: it reports at a step every listed arm has reached,
and says which arms it had to drop to get there.

NOT a verdict tool. It prints numbers; it does not rank them. Ranking needs the
noise floor (vanilla x3 spread) and per-problem McNemar, which is verdict.py's
job, and until that exists any ordering here is a difference of unknown size.
"""

import argparse
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUN_RE = re.compile(r"#+ RUN: (\S+) #+")
STEP_RE = re.compile(r"step:(\d+) - (.*)")
VAL_RE = re.compile(r"val-core/\S+?/acc/mean@1:(?:np\.float64\()?(-?[\d.eE+]+)")


def num(blob, key):
    m = re.search(re.escape(key) + r":(?:np\.float64\()?(-?[\d.eE+]+)", blob)
    return float(m.group(1)) if m else None


def collect(log_dir):
    """{arm: {step: metrics}} from every lane log under log_dir (both layouts)."""
    arms = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "lane*.log"))
                       + glob.glob(os.path.join(log_dir, "*", "lane*.log"))):
        current = None
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    m = RUN_RE.search(line)
                    if m:
                        current = m.group(1)
                        # _lane.sh appends the lane TAG to the run name, so a
                        # rehearsal writes "<arm>_s0_rehearsal" into the same log
                        # tree. Those are 3-step smoke runs; listing them beside
                        # real arms puts a run that will never reach step 25 in the
                        # "not yet there" line forever, which reads as a stalled arm.
                        if current.endswith("_rehearsal"):
                            current = None
                            continue
                        arms.setdefault(current, {})
                        continue
                    if current is None:
                        continue
                    m = STEP_RE.search(line)
                    if not m:
                        continue
                    step, blob = int(m.group(1)), m.group(2)
                    v = VAL_RE.search(blob)
                    arms[current][step] = {
                        "val": float(v.group(1)) if v else None,
                        "entropy": num(blob, "actor/entropy"),
                        "len": num(blob, "response_length/mean"),
                        "clip": num(blob, "response_length/clip_ratio"),
                        "dl_absmean": num(blob, "actor/distillation/delta_ell_absmean"),
                        "dl_p5": num(blob, "actor/distillation/delta_ell_p5"),
                        "loss_max": num(blob, "actor/distillation/loss_max"),
                        "sit": num(blob, "timing_s/step"),
                    }
        except OSError:
            continue
    return arms


def fmt(x, spec):
    return format(x, spec) if x is not None else "-"


def table(arms, step):
    print(f"\n=== all arms @ step {step} ===")
    print(f"{'arm':<26}{'val':>7}{'entropy':>9}{'len':>7}{'clip':>7}"
          f"{'|dl|':>7}{'dl_p5':>8}{'lossmax':>9}{'s/it':>7}")
    for name in sorted(arms):
        r = arms[name].get(step)
        if not r:
            continue
        print(f"{name:<26}{fmt(r['val'], '>7.3f')}{fmt(r['entropy'], '>9.3f')}"
              f"{fmt(r['len'], '>7.0f')}{fmt(r['clip'], '>7.3f')}"
              f"{fmt(r['dl_absmean'], '>7.3f')}{fmt(r['dl_p5'], '>8.2f')}"
              f"{fmt(r['loss_max'], '>9.1f')}{fmt(r['sit'], '>7.1f')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default=os.environ.get("LOG_DIR", os.path.join(ROOT, "logs")))
    p.add_argument("--step", type=int, help="report at this step instead of the newest common one")
    p.add_argument("--all", action="store_true", help="every val step, not just one")
    args = p.parse_args()

    arms = collect(args.log_dir)
    if not arms:
        raise SystemExit(f"no lane logs under {args.log_dir}")

    val_steps = {a: sorted(s for s, r in m.items() if r["val"] is not None) for a, m in arms.items()}
    have_val = {a: s for a, s in val_steps.items() if s}
    if not have_val:
        raise SystemExit("no arm has reached a validation step yet")

    if args.all:
        for step in sorted({s for ss in have_val.values() for s in ss}):
            table(arms, step)
    else:
        step = args.step or min(max(ss) for ss in have_val.values())
        table(arms, step)
        behind = [a for a in arms if step not in arms[a] or arms[a][step]["val"] is None]
        if behind:
            # Named, not silently dropped: an arm missing from the table is a fact
            # about the campaign, and a reader who cannot see which ones are absent
            # will read the table as complete.
            print(f"\nnot yet at step {step} ({len(behind)}): {' '.join(sorted(behind))}")

    print("\nNumbers only -- no ranking. A gap here is not a verdict until the "
          "vanilla x3 noise floor exists and verdict.py has run McNemar on the "
          "per-problem artifacts.")


if __name__ == "__main__":
    main()
