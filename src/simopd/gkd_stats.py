"""Rollout-server -> trainer relay for per-step GKD mixing telemetry.

The mixing decision (gkd_mix / a5_aggrevate) lives in the rollout-server process;
wandb rows are emitted by the trainer's loss function (losses._gkd_relay_metrics).
They share no in-process state, so the server appends one JSON line per finished
step to a sideband file and the trainer tails the newest complete line.

Same-node assumption: the protocol's 2-GPU lane packs actor + teacher on one
host, so a node-local file is a valid channel. A multi-node actor pool must set
SIMOPD_GKD_STATS to a shared-FS path.

Failure posture (review 2026-08-15 #2/#5/#12):
  * path() REFUSES to guess: with neither SIMOPD_GKD_STATS nor EXPERIMENT_NAME
    set, two co-located lanes would silently share one file and the trainer
    would chart another arm's mixing -- raising at first use (writer install /
    first relay call, both at bringup) beats that.
  * append() never propagates: the telemetry writer sits inline in the wrapped
    generate(), and a full /tmp must not fail a rollout request. First failure
    warns once, rows drop.
  * APPEND-ONLY (2026-08-18, empty-sideband incident): sitecustomize arms the
    writer modules in EVERY process importing the server module, and an earlier
    truncate-at-install let late-spawning actors wipe the serving process's
    rows. Nothing here truncates; launchers wanting a fresh file delete it
    before the run (rehearse does), and a resumed same-name lane appends -- the
    reader takes the newest row, gkd_stats_step keeps staleness visible.
  * latest() rewinds when the file shrinks (truncation/recreation), instead of
    holding a stale offset past EOF forever.

Lag contract: the server flushes step N when it first sees a request from step
N+1 (plus atexit for the last step), so the trainer's step-N wandb row usually
carries the server's step N-1. gkd_stats_step is emitted verbatim so the lag is
visible; offline joins should use it, not the trainer step. Rows carry the
writer pid so a second writer on one path is detectable in the JSONL.
"""

import json
import os
import sys

_read = {"offset": 0, "row": None}
_write = {"warned": False}


def path():
    p = os.environ.get("SIMOPD_GKD_STATS", "")
    if p:
        return p
    name = os.environ.get("EXPERIMENT_NAME", "")
    if not name:
        raise RuntimeError(
            "gkd_stats: neither SIMOPD_GKD_STATS nor EXPERIMENT_NAME is set -- "
            "refusing a shared default path that would mix lanes' telemetry; "
            "export EXPERIMENT_NAME (the lane scripts do) or set SIMOPD_GKD_STATS")
    return os.path.join("/tmp", "simopd_gkd_stats_%s.jsonl" % name)


def append(row):
    """Writer side (rollout server): one line per flushed step. Never raises --
    the writer sits inline in the wrapped generate(), and neither a full /tmp
    nor a config error may fail a rollout request (the config case has already
    passed the install-time path() check by the time any row flushes)."""
    try:
        with open(path(), "ab") as f:
            f.write((json.dumps(dict(row, pid=os.getpid())) + "\n").encode())
    except Exception as e:
        if not _write["warned"]:
            _write["warned"] = True
            print(f"[simopd] gkd_stats: telemetry write failed ({e!r}); "
                  f"rows will be dropped, training continues", file=sys.stderr, flush=True)


def latest():
    """Reader side (trainer): newest complete row, incremental tail.

    Reads only bytes appended since the last call; a torn final line stays
    unread (offset holds at the last newline) and is picked up complete on the
    next call. Returns the last successfully parsed row, or None before the
    first flush / when the file does not exist yet.
    """
    try:
        with open(path(), "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size < _read["offset"]:
                # Truncated or recreated underneath us: start over rather than
                # sitting past EOF returning the pre-truncation row forever.
                _read["offset"] = 0
            f.seek(_read["offset"])
            chunk = f.read()
    except OSError:
        return _read["row"]
    if chunk:
        cut = chunk.rfind(b"\n")
        if cut >= 0:
            complete = chunk[: cut + 1]
            _read["offset"] += len(complete)
            for line in reversed(complete.splitlines()):
                if not line.strip():
                    continue
                try:
                    _read["row"] = json.loads(line)
                    break
                except ValueError:
                    # A corrupt line keeps the previous row; the next flush
                    # supersedes it. Monitoring channel: stale beats crash.
                    continue
    return _read["row"]
