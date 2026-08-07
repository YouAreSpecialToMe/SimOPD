"""Free a run name whose finished evidence belongs to a superseded config.

    python scripts/migrate_stale.py          # dry-run: says what it would do
    python scripts/migrate_stale.py --apply  # do it (idempotent; rerun as lanes drain)

Audit r3/r4 changed what five arms MEAN (divergence arms moved to the direct
optimizer branch; g2 gained its paper's second stage), but campaign.sh's done-list is
name-based: a lane log holding "#### b2_forward_kl_s0 -> OK" blocks the corrected
version of that name forever. The plan's disposition (EXPANSION-PLAN §6.7) is to keep
the old runs as an accidental PG-vs-direct ablation, not to delete them -- so this
renames rather than removes: every marker line, checkpoint dir and pool claim for a
stale name gains the ablation suffix, the name comes free, and the daemon's next pass
relaunches it under the new fingerprint.

What it will not do, in order of how much a mistake would cost:
  running runs are untouched. A stale name with more RUN: markers than results is in
      flight; the plan says finish-don't-kill, and this reports it and moves on.
      Rerun after the lane drains -- everything here is idempotent.
  live files are never rewritten. A lane appends to its log through a held fd, and
      any rewrite swaps the inode out from under it -- the lane's future "-> OK" would
      land in a deleted file and the run would look unfinished forever, which is the
      false-negative twin of the false OK this campaign already refuses to risk. A
      file is live iff any run in it (stale or not) is unfinished; those are skipped
      whole, with a notice.
  valid runs keep their evidence. A finished log that mixes a stale run with valid
      ones is rewritten line-precisely (only the stale name's marker lines), atomically,
      in the same directory. A log whose runs are ALL stale is renamed to .pgab, which
      campaign.sh's lane*.log glob no longer matches.
"""

import argparse
import os
import re
import sys
import time

# The stale set is the audit's, not a guess: arms whose configs/arms.yaml entry changed
# meaning after their runs started (r3: optimizer branch; r4: g2's missing stage).
# cornell's c1_lsm_topk32_renorm_s0 shares the diagnosis but lives on the other island;
# pass --names there.
DEFAULT_STALE = [
    "b2_forward_kl_s0",
    "c2_quantile_budget_s0",
    "e1_pl_rank_s0",
    "b3_eopd_gate_s0",
    "g2_fire_likelihood_s0",
]

# campaign.sh's exact grammar (grep -oE, PREFIX match): the real lines are
# "################ RUN: name ################" with trailing hashes, so anchoring $
# here made every marker invisible -- on the live fleet this script then reported "no
# evidence anywhere" for five finished runs and mistook a RUNNING run's claim for a
# stale one. Prefix-match like the greps do; a decorative "#### DONE" banner still
# counts as nothing because the RUN:/-> shapes don't match it.
START = re.compile(r"^#+ RUN: ([A-Za-z0-9_.]+)")
END = re.compile(r"^#+ ([A-Za-z0-9_.]+) -> (?:OK|FAIL)")


def scan(path):
    """{name: [started, ended]} for one lane log."""
    counts = {}
    with open(path, errors="replace") as f:
        for line in f:
            for i, pat in ((0, START), (1, END)):
                m = pat.match(line)
                if m:
                    counts.setdefault(m.group(1), [0, 0])[i] += 1
    return counts


def rewrite(path, name, suffix):
    """Rename one run's marker lines in place; atomic, same-directory."""
    out = []
    hits = 0
    with open(path, errors="replace") as f:
        for line in f:
            m = START.match(line) or END.match(line)
            if m and m.group(1) == name:
                line = line.replace(name, name + suffix, 1)
                hits += 1
            out.append(line)
    tmp = path + ".migrate.tmp"
    with open(tmp, "w") as f:
        f.writelines(out)
    os.replace(tmp, path)
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="*", default=DEFAULT_STALE)
    p.add_argument("--suffix", default="__pgab",
                   help="the ablation identity; g2's old run is a stage-1 ablation but "
                        "shares the suffix so there is one convention to grep for")
    p.add_argument("--log-dir", default=os.environ.get("LOG_DIR",
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")))
    p.add_argument("--ckpt-root", default=os.environ.get("CKPT_ROOT", "/scratch/zz865/simopd/ckpt"))
    p.add_argument("--claim-dir", default=os.environ.get("CLAIM_DIR",
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".campaign")))
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    act = (lambda msg, fn: (print("  " + msg), fn())) if a.apply else \
          (lambda msg, fn: print("  would: " + msg))

    import glob as g
    logs = sorted(g.glob(os.path.join(a.log_dir, "lane*.log")) +
                  g.glob(os.path.join(a.log_dir, "*", "lane*.log")))
    per_file = {f: scan(f) for f in logs}
    totals = {}
    for counts in per_file.values():
        for name, (s, e) in counts.items():
            t = totals.setdefault(name, [0, 0])
            t[0] += s; t[1] += e

    freed, waiting = [], []
    for name in a.names:
        started, ended = totals.get(name, (0, 0))
        print(f"\n{name}  (started {started}, finished {ended})")
        if started == 0 and ended == 0:
            # Never ran, or already migrated. A stale CLAIM without evidence can still
            # pin a pool row to a machine whose launch died before the first marker.
            claim = os.path.join(a.claim_dir, "claims", name)
            if os.path.isdir(claim) and time.time() - os.path.getmtime(claim) > 600:
                act(f"mv stale claim {claim} -> {name}{a.suffix}",
                    lambda c=claim, n=name: os.rename(c, os.path.join(os.path.dirname(c), n + a.suffix)))
                freed.append(name)
            else:
                print("  no evidence anywhere; nothing to migrate")
            continue
        if started > ended:
            waiting.append(name)
            print("  IN FLIGHT -- finish-don't-kill; rerun this script after it ends")
            continue

        # All-or-nothing per name: a log skipped for hosting a live run while the ckpt
        # moved anyway would leave the name "done" in evidence but pointing at no
        # checkpoint -- a half-state every downstream lookup would trip over.
        hosting = [(f, c) for f, c in per_file.items() if name in c]
        blocked = {f: [n for n, (s, e) in c.items() if s > e] for f, c in hosting}
        if any(blocked.values()):
            waiting.append(name)
            for f, live in blocked.items():
                if live:
                    print(f"  {f} still hosts running run(s) {live}; {name} deferred whole, rerun later")
            continue
        for f, counts in hosting:
            if set(counts) == {name}:
                act(f"mv {f} -> {f}.pgab (sole occupant; drops out of the lane*.log glob)",
                    lambda f=f: os.rename(f, f + ".pgab"))
            else:
                act(f"rewrite {f}: marker lines {name} -> {name}{a.suffix} "
                    f"(other runs' evidence kept verbatim)",
                    lambda f=f, n=name: rewrite(f, n, a.suffix))

        for proj in g.glob(os.path.join(a.ckpt_root, "*")):
            d = os.path.join(proj, name)
            if os.path.isdir(d):
                act(f"mv {d} -> {d}{a.suffix}", lambda d=d: os.rename(d, d + a.suffix))
        claim = os.path.join(a.claim_dir, "claims", name)
        if os.path.isdir(claim):
            act(f"mv claim -> {name}{a.suffix}",
                lambda c=claim, n=name: os.rename(c, os.path.join(os.path.dirname(c), n + a.suffix)))
        if name not in waiting:
            freed.append(name)

    print(f"\n{'applied' if a.apply else 'DRY RUN (add --apply)'}: "
          f"{len(freed)} name(s) freed {freed or ''}")
    if waiting:
        print(f"still running or in live files, rerun later: {sorted(set(waiting))}")
    if freed and a.apply:
        print("next: repin once (REASON=\"...\" bash deploy/campaign.sh --repin), then the "
              "daemon relaunches these rows under the corrected configs on its next pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
