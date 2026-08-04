"""Find the actual failure in a verl run log, without reading 18,000 lines.

    python scripts/triage.py                    # newest log under logs/
    python scripts/triage.py /tmp/envtest.log
    python scripts/triage.py --ray              # also dig through Ray worker logs
    python scripts/triage.py -n 3               # show the first 3 errors, not 1

A failing run buries its cause. A rehearsal log here is ~18k lines of which almost
none matter: `set -x` echoes a 2KB python invocation, every Ray actor prefixes its
output with `(TaskRunnerV1 pid=...)`, tqdm repaints progress bars, vLLM narrates its
startup, and Ray collapses duplicates into `[repeated 7x across cluster]` lines that
are themselves repeated.

Two rules of this stack are built in, because getting them wrong costs an hour each:

  - The FIRST exception is the cause; later ones are re-raises of it as the failure
    propagates back through Ray. Scrolling to the bottom shows you the last one.
  - `ActorUnavailableError` / `rpc_code: 14` means the actor process DIED. That
    traceback names neither the worker nor its error -- the real one is only in the
    worker's own log, which --ray goes and reads.
"""

import argparse
import glob
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RAY_PREFIX = re.compile(r"^\([\w.]+ (?:pid|ip)=[^)]*\)\s?")
REPEATED = re.compile(r"\[repeated \d+x across cluster\]")
TQDM = re.compile(r"\|[\s#█▏▎▍▌▋▊▉]*\|\s*\d+/\d+|\d+%\|")
# Healthy chatter that happens to be voluminous. Each of these drowns a traceback
# that occurred between two of them; none of them has ever been a cause here.
HEARTBEAT = re.compile(
    r"pending: \d+, running: \d+, finished: \d+"
    r"|Per-operation statistics:|Time range: last"
    r"|transfer_queue\.utils\.perf"
    r"|^step:\d+ - "
    r"|^wandb: "
    r"|Using blocking ray\.get inside async actor"
    r"|Using dataset class:"
    r"|^\s*(?:INFO|DEBUG)[: ]"
    r"|UserWarning|FutureWarning|DeprecationWarning"
    r"|torch\.cuda\.memory\._set_allocator_settings"
    r"|<frozen importlib"
)
# Progress bars for dataset mapping look exactly like training progress; only the
# labelled one tells you how far the RUN got.
TRAIN_PROGRESS = re.compile(r"Training Progress:.*?\|\s*(\d+)/(\d+)")
RUN_START = re.compile(r"#+ RUN: (\S+) #+")
RUN_END = re.compile(r"#+ (\S+) -> (OK|FAIL) #+")
# Shutdown races. Real tracebacks, but they fire while a run is tearing down -- often
# a run that SUCCEEDED -- so they must never outrank the failure being looked for.
LOW_PRIORITY = re.compile(r"Exception ignored|atexit|_shutdown_|ResourceWarning")
# Plumbing frames. Ray re-raises a worker's exception through five of its own
# frames and hydra adds four more; none has ever been the bug, and together they
# push the line that IS the bug off the screen.
PLUMBING = re.compile(r'File "[^"]*/(?:site-packages/(?:ray|hydra|omegaconf)|'
                      r'asyncio|concurrent/futures)/')
CARETS = re.compile(r"^\s*[\^~]+\s*$")
TRACEBACK_START = "Traceback (most recent call last):"
# `SomeError: message`, at column 0, ends a traceback.
EXC_LINE = re.compile(r"^(\w[\w.]*(?:Error|Exception|Interrupt|Exit|Warning|Failure))\b(:.*)?$")


def compress(block):
    """Drop plumbing frames from a traceback, noting how many were removed."""
    out, skipped = [], 0
    i = 0
    while i < len(block):
        line = block[i]
        if CARETS.match(line):
            i += 1
            continue
        if PLUMBING.search(line):
            skipped += 1
            i += 2 if i + 1 < len(block) and block[i + 1].startswith("    ") else 1
            continue
        if skipped:
            out.append(f"      ... {skipped} ray/hydra/asyncio frame(s) omitted")
            skipped = 0
        out.append(line)
        i += 1
    if skipped:
        out.append(f"      ... {skipped} ray/hydra/asyncio frame(s) omitted")
    return out

# Failures whose real cause is always somewhere else.
SYMPTOM_HINTS = [
    (re.compile(r"ActorUnavailableError|ActorDiedError|rpc_code: 14"),
     "the actor process died; its own log has the cause. Re-run with --ray, and check\n"
     "        /dev/shm size (64MB in a container kills vLLM workers) and GPU memory."),
    (re.compile(r"has no revision: main"),
     "this is ModelScope, not HuggingFace: VERL_USE_MODELSCOPE is true somewhere.\n"
     "        export VERL_USE_MODELSCOPE=False -- and if a stale Ray cluster is running,\n"
     "        its workers keep the OLD value, so `ray stop --force` first."),
    (re.compile(r"undefined symbol.*c10|undefined symbol.*flash"),
     "flash-attn was built against a different torch than the one installed.\n"
     "        bash deploy/dsw/repair_torch.sh, or run with USE_REMOVE_PADDING=False."),
    (re.compile(r"CUDA out of memory|torch\.OutOfMemoryError"),
     "lower actor_rollout_ref.rollout.gpu_memory_utilization, or\n"
     "        actor_rollout_ref.actor.ppo_max_token_len_per_gpu."),
    (re.compile(r"No such file or directory.*parquet|FileNotFoundError.*parquet"),
     "training data missing: python scripts/fetch_assets.py --check"),
    (re.compile(r"modelscope[/.]"),
     "the model was loaded through ModelScope, not HuggingFace -- VERL_USE_MODELSCOPE\n"
     "        is true in that worker. Note fetch_assets.py checks the HF cache, so it will\n"
     "        report everything present while verl reads a different (possibly corrupt) copy.\n"
     "        ray stop --force; export VERL_USE_MODELSCOPE=False; rm -rf ~/.cache/modelscope"),
    (re.compile(r"UnicodeDecodeError|invalid start byte|codec can't decode"),
     "a model file on disk is corrupt -- a download written half-way. The name and\n"
     "        size look right, which is why it surfaces here and not at fetch time.\n"
     "        python scripts/fetch_assets.py --check     # names the bad repo\n"
     "        python scripts/fetch_assets.py --repair    # re-pulls it"),
]


def clean(line):
    """Strip the decoration. Returns None for lines that are pure noise."""
    line = ANSI.sub("", line.rstrip("\n"))
    line = line.split("\r")[-1]          # tqdm repaints; keep the final state only
    line = RAY_PREFIX.sub("", line)
    line = REPEATED.sub("", line).rstrip()
    if not line.strip():
        return None
    if TQDM.search(line) or HEARTBEAT.search(line):
        return None
    if line.startswith("+ "):            # set -x echo
        return None
    return line


def errors(lines):
    """Yield (index, [traceback lines]) for each traceback, in order."""
    i = 0
    while i < len(lines):
        if TRACEBACK_START in lines[i]:
            start, block = i, [lines[i]]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                # The exception line at column 0 terminates the block; anything
                # after it belongs to the next thing that happened.
                if EXC_LINE.match(lines[i]) and not lines[i].startswith(" "):
                    break
                if len(block) > 200:     # a runaway block is itself the finding
                    break
                i += 1
            yield start, block
        i += 1


def segments(lines):
    """Split into (run_name, status, start, end). One entry per RUN marker.

    A campaign or lane log holds a dozen runs. The first traceback in the FILE is
    routinely a teardown race belonging to a run that succeeded -- measured on a real
    rehearsal log, where 11 arms passed and the failure was the 12th. Only the
    traceback inside the failing run's own section is the one being looked for.
    """
    marks = []
    for i, l in enumerate(lines):
        m = RUN_START.search(l)
        if m:
            marks.append([m.group(1), None, i, len(lines)])
            continue
        m = RUN_END.search(l)
        if m and marks:
            for seg in reversed(marks):
                if seg[0] == m.group(1):
                    seg[1], seg[3] = m.group(2), i
                    break
    return [tuple(x) for x in marks]


def rank(found, segs):
    """Order tracebacks by how likely each is to be the cause.

    Inside a FAILed run beats anywhere else; a teardown race always loses.
    """
    failed = [(s, e) for _, st, s, e in segs if st == "FAIL"]

    def key(item):
        idx, block = item
        text = "\n".join(block)
        in_failed = any(s <= idx <= e for s, e in failed)
        return (0 if in_failed else 1, 1 if LOW_PRIORITY.search(text) else 0, idx)

    return sorted(found, key=key)


def ray_worker_errors(tmpdirs, limit=3):
    """The cause behind an ActorUnavailableError, from the workers' own logs."""
    out = []
    for tmp in tmpdirs:
        pat = os.path.join(tmp, "session_latest", "logs", "*")
        for path in sorted(glob.glob(pat), key=lambda p: -os.path.getmtime(p)):
            if not os.path.isfile(path):
                continue
            try:
                with open(path, errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            if TRACEBACK_START not in text:
                continue
            lines = [c for c in (clean(x) for x in text.splitlines()) if c]
            for _, block in errors(lines):
                out.append((path, block))
                break
            if len(out) >= limit:
                return out
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("log", nargs="?", help="log file; default = newest under logs/")
    p.add_argument("-n", type=int, default=1, help="how many errors to show (default 1)")
    p.add_argument("--ray", action="store_true", help="also search Ray worker logs")
    p.add_argument("--ray-tmp", action="append", default=[],
                   help="Ray temp dir to search (repeatable); default /tmp/ray and /tmp/ray_lane*")
    p.add_argument("--context", type=int, default=6,
                   help="lines of surrounding output to show before the error")
    a = p.parse_args()

    path = a.log
    if not path:
        cands = glob.glob(os.path.join(root, "logs", "*.out")) + \
                glob.glob(os.path.join(root, "logs", "*.log"))
        if not cands:
            raise SystemExit("no logs/ files; pass one explicitly")
        path = max(cands, key=os.path.getmtime)

    with open(path, errors="replace") as f:
        raw = f.readlines()
    lines = [c for c in (clean(x) for x in raw) if c]
    print(f"=== {path}")
    print(f"    {len(raw)} lines -> {len(lines)} after removing decoration "
          f"({100 * (1 - len(lines) / max(len(raw), 1)):.0f}% was noise)")

    segs = segments(lines)
    if segs:
        ok = [n for n, s, _, _ in segs if s == "OK"]
        bad = [n for n, s, _, _ in segs if s == "FAIL"]
        running = [n for n, s, _, _ in segs if s is None]
        print(f"    {len(segs)} run(s): {len(ok)} OK, {len(bad)} FAIL"
              + (f", {len(running)} unfinished" if running else ""))
        if bad:
            print(f"    FAILED: {', '.join(bad)}")
        if running:
            print(f"    never finished: {', '.join(running)}")
    steps = [int(m.group(1)) for m in TRAIN_PROGRESS.finditer("".join(raw))]
    if steps:
        print(f"    reached training step {max(steps)}")

    found = rank(list(errors(lines)), segs)
    if not found:
        print("\nno traceback in this log.")
        tail = [l for l in lines[-15:]]
        print("last 15 meaningful lines:")
        for l in tail:
            print("   ", l[:200])
        return 0

    print(f"\n{len(found)} traceback(s); showing {min(a.n, len(found))}, most-likely-cause "
          f"first (inside a FAILed run beats elsewhere; teardown races rank last)")
    shown = 0
    for idx, block in found:
        if shown >= a.n:
            break
        owner = next((n for n, _, s, e in segs if s <= idx <= e), None)
        if owner:
            print(f"\n=== in run: {owner}")
        if a.context:
            print("\n--- context before it ---")
            for l in lines[max(0, idx - a.context):idx]:
                print("   ", l[:200])
        print("--- traceback ---")
        for l in compress(block):
            print("   ", l[:300])
        text = "\n".join(block)
        for pat, hint in SYMPTOM_HINTS:
            if pat.search(text):
                print(f"\n    >> {hint}")
        shown += 1

    if a.ray:
        tmps = a.ray_tmp or ["/tmp/ray"] + sorted(glob.glob("/tmp/ray_lane*")) + \
               sorted(glob.glob("/tmp/ray_probe*"))
        tmps = [t for t in tmps if os.path.isdir(t)]
        print(f"\n=== Ray worker logs ({', '.join(tmps) or 'none found'})")
        for wpath, block in ray_worker_errors(tmps):
            print(f"\n--- {wpath}")
            for l in compress(block):
                print("   ", l[:300])
    elif any(p.search("\n".join(b)) for _, b in found[:1] for p, _ in SYMPTOM_HINTS[:1]):
        print("\n    (re-run with --ray to read the worker logs)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
