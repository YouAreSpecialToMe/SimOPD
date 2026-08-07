"""Per-run status: latest validation point, best point, and verdict.

    python scripts/run_status.py [substring]

THE BUG THIS EXISTS TO NOT REPEAT. A run name appears in several lane logs: the
launch that died, and every relaunch that resumed from its checkpoint. An earlier
version of this merged all of them, which paired a FAIL verdict from a dead log with
a curve from the live relaunch -- and reported three healthy vanilla runs, sitting at
step 224 of 250, as "all three seeds died at 200-225, that is a pattern". The FAIL was
real; it was the startup HF-429 from the 06:41 launch that HF_HUB_OFFLINE=1 has since
fixed. Nothing had died at all.

So: a run's verdict is taken ONLY from the newest log that mentions it, and the log is
printed so the claim can be checked. The curve is still merged across logs, because a
resume genuinely continues one curve -- but a verdict is about an attempt, not a run.

Even so, this reads logs. A log cannot tell you a process is alive. For that:
    ssh <host> 'for p in $(pgrep -f verl.trainer.main_ppo); do
                  tr "\\0" "\\n" < /proc/$p/environ | grep ^EXPERIMENT_NAME=; done'
which is the only source that cannot claim a run exists when it does not.
"""

import re
import sys
import glob
import os
import time
from collections import defaultdict

RUN = re.compile(r"^#+ RUN: ([A-Za-z0-9_.]+)")
END = re.compile(r"^#+ ([A-Za-z0-9_.]+) -> (OK|FAIL)")
VAL = re.compile(r"step:(\d+) .*?val-core/[^:]*acc/mean@1:np\.float64\(([0-9.]+)\)")
PREFIX = re.compile(r"^\([^)]*\)\s*")          # ray's "(TaskRunnerV1 pid=N) "

curves = defaultdict(dict)                      # name -> {step: acc}
newest = {}                                     # name -> (mtime, log, verdict|None)


def scan(paths):
    for f in paths:
        mt = os.path.getmtime(f)
        cur = None
        with open(f, errors="replace") as fh:
            for line in fh:
                body = PREFIX.sub("", line.lstrip())
                m = RUN.match(body)
                if m:
                    cur = m.group(1)
                    if mt > newest.get(cur, (0,))[0]:
                        newest[cur] = (mt, f, None)
                    continue
                m = END.match(body)
                if m and newest.get(m.group(1), (0, "", None))[1] == f:
                    mt0, f0, _ = newest[m.group(1)]
                    newest[m.group(1)] = (mt0, f0, m.group(2))
                    continue
                if cur:
                    m = VAL.search(body)
                    if m:
                        curves[cur][int(m.group(1))] = float(m.group(2))


scan(sorted(glob.glob("logs/*/lane*.log")) + sorted(glob.glob("logs/lane*.log")))
extra = os.environ.get("EXTRA_LOG_GLOB")
if extra:
    scan(sorted(glob.glob(extra)))

want = sys.argv[1] if len(sys.argv) > 1 else ""
now = time.time()
rows = []
for name, c in curves.items():
    if not c or want not in name:
        continue
    mt, log, verdict = newest.get(name, (0, "?", None))
    steps = sorted(c)
    state = verdict or ("running" if now - mt < 6 * 3600 else "silent>6h")
    rows.append((name, c[steps[-1]], steps[-1], max(c.values()),
                 max(c, key=c.get), state, len(steps), os.path.basename(log)))

rows.sort(key=lambda r: -r[3])
print(f"{'run':30s} {'last':>6s} {'@':>5s} {'best':>6s} {'@':>5s}  {'state':10s} pts  newest log")
for name, lastv, lasts, bestv, bests, st, n, log in rows:
    print(f"{name:30s} {lastv:6.3f} {lasts:5d} {bestv:6.3f} {bests:5d}  {st:10s} {n:3d}  {log}")
