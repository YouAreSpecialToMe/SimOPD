"""Trainer -> rollout-server budget relay for h9_prune_adapt (the REVERSE of
gkd_stats: that one carries server telemetry to the trainer; this one carries
the trainer's adapted response budget back to the serving wrapper).

Same channel contract as gkd_stats, pattern-copied on purpose (shared state
would couple the two arms' modules): node-local JSONL, APPEND-ONLY (launchers
delete for a fresh run; sitecustomize arms every process and truncation was
the 2026-08-18 empty-sideband incident), writer never raises, reader tails
incrementally and rewinds on shrink. Rows carry pid.

budget() is the consumer surface: newest adapted budget, or the full protocol
window before the first row lands -- Prune-OPD starts long and adapts DOWN,
so the cold-start default is the cap, never the floor.
"""

import json
import os
import sys

_read = {"offset": 0, "row": None}
_write = {"warned": False}

DEFAULT_BUDGET = 16384


def path():
    p = os.environ.get("SIMOPD_H_BUDGET", "")
    if p:
        return p
    name = os.environ.get("EXPERIMENT_NAME", "")
    if not name:
        raise RuntimeError(
            "h_budget: neither SIMOPD_H_BUDGET nor EXPERIMENT_NAME is set -- "
            "refusing a shared default path that would mix lanes' budgets; "
            "export EXPERIMENT_NAME (the lane scripts do) or set SIMOPD_H_BUDGET")
    return os.path.join("/tmp", "simopd_h_budget_%s.jsonl" % name)


def append(row):
    """Trainer side. Never raises -- a full /tmp must not fail a train step."""
    try:
        with open(path(), "ab") as f:
            f.write((json.dumps(dict(row, pid=os.getpid())) + "\n").encode())
    except Exception as e:
        if not _write["warned"]:
            _write["warned"] = True
            print(f"[simopd] h_budget: relay write failed ({e!r}); "
                  f"budget stays at its last value, training continues",
                  file=sys.stderr, flush=True)


def latest():
    """Server side: newest complete row, incremental tail (gkd_stats pattern)."""
    try:
        with open(path(), "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size < _read["offset"]:
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
                    continue
    return _read["row"]


def budget_row():
    """(budget, row) from ONE read. generate() runs per request, so the caller that
    also needs to know whether the relay has ever been written must not pay a second
    open() for it -- row is falsy exactly when nobody has written the relay yet."""
    row = latest() or {}
    try:
        b = int(row.get("budget", DEFAULT_BUDGET))
    except (TypeError, ValueError):
        b = DEFAULT_BUDGET
    return max(1, min(b, DEFAULT_BUDGET)), row


def budget():
    return budget_row()[0]
