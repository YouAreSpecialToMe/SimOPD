"""Rollout-server -> trainer relay for per-step GKD mixing telemetry.

The mixing decision (gkd_mix) lives in the rollout-server process; wandb rows are
emitted by the trainer's loss function (losses._gkd_relay_metrics). They share no
in-process state, so the server appends one JSON line per finished step to a
sideband file and the trainer tails the newest complete line, re-emitting it as
distillation/gkd_* metrics.

Same-node assumption: the protocol's 2-GPU lane packs actor + teacher on one
host, so a node-local file is a valid channel. A multi-node actor pool must set
SIMOPD_GKD_STATS to a shared-FS path or the trainer reads nothing -- which is
visible (the gkd_* panel simply never appears), not silent corruption.

Lag contract: the server flushes step N when it first sees a request from step
N+1 (plus atexit for the last step), so the trainer's step-N wandb row usually
carries the server's step N-1. gkd_stats_step is emitted verbatim so the lag is
visible in wandb rather than papered over; offline joins should use it, not the
trainer step.
"""

import json
import os

_read = {"offset": 0, "row": None}


def path():
    p = os.environ.get("SIMOPD_GKD_STATS", "")
    if p:
        return p
    # Both processes derive the same default from the same launch env.
    name = os.environ.get("EXPERIMENT_NAME", "run")
    return os.path.join("/tmp", "simopd_gkd_stats_%s.jsonl" % name)


def append(row):
    """Writer side (rollout server): one line per flushed step."""
    with open(path(), "ab") as f:
        f.write((json.dumps(row) + "\n").encode())


def latest():
    """Reader side (trainer): newest complete row, incremental tail.

    Reads only bytes appended since the last call; a torn final line stays
    unread (offset rewinds to the last newline) and is picked up complete on
    the next call. Returns the last successfully parsed row, or None before
    the first flush / when the file does not exist yet.
    """
    try:
        with open(path(), "rb") as f:
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
                    # A torn or corrupt line keeps the previous row; the next
                    # flush supersedes it. Monitoring channel: stale beats crash.
                    continue
    return _read["row"]
