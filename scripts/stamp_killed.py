#!/usr/bin/env python3
"""Stamp hard-stop FAIL markers into logs whose runs were killed, not finished.

migrate_stale reads evidence the way campaign.sh writes it: a RUN marker with no
OK/FAIL is a live run, and live runs are finish-don't-kill. stop_pilot.sh created
a third state that convention excludes -- killed mid-run, never to finish -- and
each corpse holds its name hostage: the row renders RUNNING forever, migration
refuses the name, campaign.sh counts the log in-flight for INFLIGHT_HOURS.

This writes the missing end: `## <name> -> FAIL (hard-stopped ...)` for every
unfinished marker in a log silent past --quiet-min. That guard is what makes it
safe next to live lanes: a real validation goes quiet ~75 minutes, so the
120-minute default cannot touch a living run. Idempotent -- rerun until it
reports nothing pending, then run migrate_stale.
"""
import argparse, glob, os, re, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default=os.environ.get("LOG_DIR", os.path.join(ROOT, "logs")))
    p.add_argument("--quiet-min", type=int, default=120,
                   help="only stamp logs silent at least this long (validation silence is ~75min)")
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()

    logs = sorted(glob.glob(os.path.join(a.log_dir, "lane*.log")) +
                  glob.glob(os.path.join(a.log_dir, "*", "lane*.log")))
    stamped, pending = [], []
    now = time.time()
    for f in logs:
        try:
            text = open(f, errors="replace").read()
        except OSError:
            continue
        names = list(dict.fromkeys(re.findall(r"^#+ RUN: ([A-Za-z0-9_.]+)", text, re.M)))
        open_names = [n for n in names
                      if not re.search(rf"^#+ {re.escape(n)} -> (OK|FAIL)", text, re.M)]
        if not open_names:
            continue
        quiet = (now - os.path.getmtime(f)) / 60
        if quiet < a.quiet_min:
            pending.append((f, open_names, quiet))
            continue
        for n in open_names:
            line = f"## {n} -> FAIL (hard-stopped; stamped {time.strftime('%FT%TZ', time.gmtime())} after {int(quiet)}min silence)\n"
            if a.apply:
                with open(f, "a") as fh:
                    fh.write(line)
            stamped.append((f, n, quiet))
            print(f"  {'stamped' if a.apply else 'would stamp'}: {n:32s} {int(quiet):>4}min quiet  {f}")
    print(f"\n{'applied' if a.apply else 'DRY RUN (add --apply)'}: {len(stamped)} marker(s)")
    if pending:
        print(f"not yet quiet enough (<{a.quiet_min}min) -- alive, or rerun later:")
        for f, ns, q in pending:
            print(f"  {','.join(ns):40s} {int(q):>4}min quiet  {f}")
    if stamped and a.apply:
        print("next: python scripts/migrate_stale.py --suffix __pilot8k --names <freed names> --apply")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
