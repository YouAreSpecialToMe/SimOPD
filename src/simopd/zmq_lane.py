"""Make the weight-transfer socket path unique per lane.

verl already anticipated this bug and says so at vllm_async_server.py:127 --

    Forward the Ray job id into the vLLM worker subprocess so the colocated
    weight-transfer IPC socket path is unique per Ray job. Without this, two
    concurrent verl jobs on the same node both bind the same
    /tmp/rl-colocate-zmq-replica-0-rank-0.sock and one fails with EADDRINUSE.

The defence is right; its premise is not true of how we run. Ray job ids are unique
within a cluster, and each lane starts its OWN Ray, so every lane is handed the same
`01000000`. Measured on both clusters, and confirmed on the node itself:

    u_str LISTEN /tmp/rl-colocate-zmq-01000000-replica-0-rank-0.sock  pid=3668011
    u_str LISTEN /tmp/rl-colocate-zmq-01000000-replica-0-rank-0.sock  pid=3668012

Two processes bound to one path. What follows is worse than EADDRINUSE, because
_init_socket does an unconditional `os.remove(ipc_path)` before binding and both ends
re-init EVERY step: each lane deletes the other's live socket, once per step, forever.
The loser stays bound to an unlinked inode that no receiver can ever reach again, and
blocks in send_pyobj with every engine resident, 0% GPU, and nothing in any log.

That is why it survived 34 steps before hanging. It is a die rolled once per step, not
a failure at startup -- which is the shape that made it look like a different bug each
time it appeared.

An earlier version of this file appended a tag to the FILENAME from SIMOPD_LANE_TAG.
It failed, and the reason is worth keeping: environment variables set by the launcher
do not reach Ray workers at all. Verified directly -- inside a ray.remote, RAY_TMPDIR
and SIMOPD_LANE_TAG both read as missing while the Ray session dir matches the driver
exactly. Anything that depends on the launcher's environment reaching the process that
binds the socket cannot work, so the lane identity has to be something each process
derives for itself.

The session directory is that thing: shared by every process in a lane, unique per Ray
instance by construction (it carries a microsecond timestamp and the driver pid).

So this patches the job id rather than the path. Both writers of it are Ray actors and
so can derive the lane themselves; the vLLM worker subprocess needs no patch at all,
because it reads VERL_RAY_JOB_ID from an environment its parent already stamped --
verl's own propagation path, now carrying a value that is actually unique.

Patching one identity instead of two paths also removes the failure mode the previous
version could not rule out: with two independent rewrites, one end could be patched
and the other not, and the two would compute different paths and never meet.
"""

import hashlib
import os
import sys

_MARK = "_simopd_zmq"


def lane_id():
    """A lane identity every process in the lane derives for itself, from Ray.

    Not from SIMOPD_LANE_TAG, even when set: it reaches the driver but not the Ray
    workers, so preferring it would make different processes in one lane disagree --
    exactly the asymmetry this is meant to remove.
    """
    try:
        import ray

        node = getattr(ray._private.worker.global_worker, "node", None)
        session = node.get_session_dir_path() if node is not None else None
    except Exception:
        session = None
    if not session:
        return ""
    return hashlib.sha1(session.encode()).hexdigest()[:8]


def _announce(side, handle):
    """Print the path each end actually computed, with everything that goes into it.

    The failure this guards is silent by construction: the loser blocks in send_pyobj
    with engines resident, 0% GPU, and nothing in any log, so it has twice now been
    diagnosed only by attaching to a live process. Two lines per run turn the next
    occurrence into a grep -- and, more to the point, name WHICH two processes agree
    when they should not. That is the question the socket table cannot answer, since
    it shows pids without saying which lane or which pool they belong to.
    """
    print(f"[simopd] zmq {side} pid={os.getpid()} lane={lane_id() or '-'} "
          f"job={os.environ.get('VERL_RAY_JOB_ID', '-')} "
          f"replica={os.environ.get('VERL_REPLICA_RANK', '-')} "
          f"dev={os.environ.get('CUDA_VISIBLE_DEVICES', '-')} :: {handle}",
          file=sys.stderr, flush=True)


def install_receiver_logging():
    """Report the receiver's path WITHOUT changing it.

    Deliberately read-only. The receiver derives its path from VERL_RAY_JOB_ID, which
    install_server has already lane-stamped, so it needs no rewrite -- and a second
    rewrite here is precisely how the previous version broke, with one end tagged and
    the other not. This exists to make the two ends comparable in the log, so that a
    disagreement is visible immediately instead of after a 30-minute stall.
    """
    mod = sys.modules.get("verl.workers.rollout.vllm_rollout.utils")
    if mod is None:
        return
    for name in dir(mod):
        cls = getattr(mod, name)
        fn = getattr(cls, "_get_zmq_handle", None) if isinstance(cls, type) else None
        if fn is None or getattr(fn, _MARK, False):
            continue

        def make(original):
            def _get_zmq_handle(self):
                h = original(self)
                if not getattr(self, "_simopd_announced", False):
                    self._simopd_announced = True
                    _announce("RECV", h)
                return h

            setattr(_get_zmq_handle, _MARK, True)
            return _get_zmq_handle

        cls._get_zmq_handle = make(fn)


def _wrap_init(cls, mutate):
    init = cls.__dict__.get("__init__")
    if init is None or getattr(init, _MARK, False):
        return False

    def __init__(self, *a, **kw):
        init(self, *a, **kw)
        lane = lane_id()
        if lane:
            mutate(self, lane)

    setattr(__init__, _MARK, True)
    cls.__init__ = __init__
    return True


def install_sender():
    """ServerAdapter.__init__ builds ipc:///tmp/rl-colocate-zmq-<jid>-replica-N-rank-M.sock

    The lane goes in immediately after the job id, so the path matches what the
    receiver derives from the lane-stamped VERL_RAY_JOB_ID character for character.
    """

    def mutate(self, lane):
        h = getattr(self, "zmq_handle", None)
        if isinstance(h, str) and "-replica-" in h and f"-{lane}-replica-" not in h:
            self.zmq_handle = h.replace("-replica-", f"-{lane}-replica-", 1)
            _announce("SEND", self.zmq_handle)

    mod = sys.modules.get("verl.workers.rollout.vllm_rollout.vllm_rollout")
    if mod is None:
        return
    # Every class defined in the module, not a hardcoded name: a first attempt at this
    # patched "vLLMRollout", which does not exist there, and applied to nothing.
    # mutate is a no-op on a class that has no zmq_handle, so over-reaching is free.
    hit = [n for n in dir(mod)
           if isinstance(getattr(mod, n), type)
           and getattr(mod, n).__module__ == mod.__name__
           and _wrap_init(getattr(mod, n), mutate)]
    print(f"[simopd] weight-transfer sender lane-scoped (patched {hit})", file=sys.stderr)
    if not hit:
        print("[simopd] WARNING: no sender class patched. Lanes will share one socket "
              "path and delete each other's, which hangs weight sync with no error.",
              file=sys.stderr)


def install_server():
    """vLLMHttpServer.__init__ stamps VERL_RAY_JOB_ID for the vLLM worker subprocess.

    Appending the lane here is what makes the receiver correct without touching it:
    it reads this variable, so a lane-unique value in it produces a lane-unique path.
    """

    def mutate(self, lane):
        jid = os.environ.get("VERL_RAY_JOB_ID", "")
        if jid and not jid.endswith(f"-{lane}"):
            os.environ["VERL_RAY_JOB_ID"] = f"{jid}-{lane}"

    mod = sys.modules.get("verl.workers.rollout.vllm_rollout.vllm_async_server")
    if mod is None:
        return
    hit = [n for n in dir(mod)
           if isinstance(getattr(mod, n), type)
           and getattr(mod, n).__module__ == mod.__name__
           and n == "vLLMHttpServer"
           and _wrap_init(getattr(mod, n), mutate)]
    print(f"[simopd] weight-transfer job id lane-scoped (patched {hit})", file=sys.stderr)
