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
VAL_RE = re.compile(r"val-core/\S+?/acc/mean@1:(?:np\.float64\()?" + r"(-?\d*\.?\d+(?:[eE][+-]?\d+)?)")


# A float, INCLUDING a negative exponent. `[\d.eE+]+` -- the obvious spelling, and
# the one watch.py uses -- has no `-` after the first character, so it eats
# "2.5253701702846836e" out of "...e-05" and stops. delta_ell_p50 sits near zero
# and is routinely written that way, so this is not an edge case.
NUM = r"(-?\d*\.?\d+(?:[eE][+-]?\d+)?)"


def num(blob, key):
    m = re.search(re.escape(key) + r":(?:np\.float64\()?" + NUM, blob)
    return float(m.group(1)) if m else None


def signal(blob, stat):
    """The per-token signal panel, whichever name this arm reports it under.

    METRICS.md §3 forbids the two from sharing a key: only the k1 family's loss
    equals -Delta-ell, so C/E-axis arms (a divergence, or a rank loss) publish
    loss_* and never delta_ell_*. Reading only delta_ell_* therefore prints "-"
    for precisely the arms whose signal is most worth seeing -- and an empty cell
    reads as "not measured" rather than "measured under the other name".

    Returned with the family, because the two columns are NOT comparable across
    it: -0.5 of Delta-ell and 0.5 of a KL are different quantities.
    """
    v = num(blob, f"actor/distillation/delta_ell_{stat}")
    if v is not None:
        return v, "dl"
    return num(blob, f"actor/distillation/loss_{stat}"), "loss"


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
                    absmean, fam = signal(blob, "absmean")
                    p5, _ = signal(blob, "p5")
                    arms[current][step] = {
                        "val": float(v.group(1)) if v else None,
                        "entropy": num(blob, "actor/entropy"),
                        "len": num(blob, "response_length/mean"),
                        "clip": num(blob, "response_length/clip_ratio"),
                        "absmean": absmean,
                        "p5": p5,
                        "fam": fam,
                        "loss_max": num(blob, "actor/distillation/loss_max"),
                        "sit": num(blob, "timing_s/step"),
                    }
        except OSError:
            continue
    return arms


def fmt(x, spec):
    # Pad the placeholder to the field width too: an unpadded "-" shifts every
    # column to its right, turning one missing cell into a whole misread row.
    width = int(re.search(r"(\d+)", spec).group(1))
    return format(x, spec) if x is not None else format("-", f">{width}")


def table(arms, step):
    print(f"\n=== all arms @ step {step} ===")
    print(f"{'arm':<26}{'val':>7}{'entropy':>9}{'len':>7}{'clip':>7}"
          f"{'sig|x|':>8}{'sig_p5':>9}{'lossmax':>9}{'s/it':>7}  kind")
    for name in sorted(arms):
        r = arms[name].get(step)
        if not r:
            continue
        print(f"{name:<26}{fmt(r['val'], '>7.3f')}{fmt(r['entropy'], '>9.3f')}"
              f"{fmt(r['len'], '>7.0f')}{fmt(r['clip'], '>7.3f')}"
              f"{fmt(r['absmean'], '>8.3f')}{fmt(r['p5'], '>9.2f')}"
              f"{fmt(r['loss_max'], '>9.1f')}{fmt(r['sit'], '>7.1f')}  {r['fam']}")
    print("  kind: dl = the signal IS -Delta-ell (k1 family); loss = a divergence or "
          "rank loss (C/E axes).\n        Those two columns do not compare across "
          "kinds -- METRICS.md §3 keeps the key names disjoint for this reason.")


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
