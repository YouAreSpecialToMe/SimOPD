"""Named Ray registry handing the teacher server handles to the student's
rollout server (a5_aggrevate's online-continuation route).

verl keeps the teacher pool behind an UNNAMED GlobalRequestLoadBalancer actor
(teacher_loop/teacher_model.py), so a process that was not handed the handle --
the student's vLLMHttpServer actor -- has no way to reach a teacher. This module
adds the missing directory: a small NAMED actor created in the trainer process
right after MultiTeacherModelManager builds the pool (sitecustomize hooks that
module), holding {teacher_key: [server actor handles]}. The student server looks
it up by name from inside the same lane-local Ray cluster.

Lifetime: NOT detached on purpose -- the registry dies with the lane's Ray
cluster, so a relaunched lane can never resolve a stale registry pointing at
dead teacher actors (each lane is its own Ray cluster; no cross-lane collision).

publish() is called by the sitecustomize wrap; resolve() by a5_aggrevate on its
first mixed request, with a short retry because rollout workers and the teacher
pool initialize concurrently. Both fail LOUD: an a5 arm that cannot reach a
teacher must die, not degrade into vanilla (the two-vanillas-under-one-name
failure class, audit r6).
"""

import os
import sys
import time

REGISTRY_NAME = "simopd_teacher_registry"


def _ray():
    import ray

    return ray


def _registry_cls():
    ray = _ray()

    @ray.remote
    class _TeacherRegistry:
        def __init__(self):
            self._handles = {}

        def put(self, handles_by_key):
            self._handles = dict(handles_by_key)
            return sorted(self._handles)

        def get(self, key):
            return self._handles.get(key)

        def keys(self):
            return sorted(self._handles)

    return _TeacherRegistry


def publish(handles_by_key):
    """Trainer side: create the named actor (idempotent) and store the handles.

    get_if_exists closes the create/create race two concurrent publishers would
    hit in the get_actor-then-create window (review 2026-08-15): whichever
    creation lands second attaches to the first's actor instead of raising.
    """
    ray = _ray()
    reg = _registry_cls().options(name=REGISTRY_NAME, get_if_exists=True).remote()
    keys = ray.get(reg.put.remote(handles_by_key))
    print(f"[simopd] teacher_registry: published teacher handles for {keys}",
          file=sys.stderr, flush=True)


async def resolve(key=None, timeout_s=60.0):
    """Student-server side: return the handle list for one teacher key.

    Retries for `timeout_s` (teacher-pool init races the first rollout), then
    raises. Key defaults to SIMOPD_A5_TEACHER_KEY or the launcher's standard
    teacher key.
    """
    import asyncio

    ray = _ray()
    key = key or os.environ.get("SIMOPD_A5_TEACHER_KEY", "teacher_model")
    deadline = time.monotonic() + timeout_s
    last = "registry actor not found"
    while time.monotonic() < deadline:
        try:
            reg = ray.get_actor(REGISTRY_NAME)
            handles = await reg.get.remote(key)
            if handles:
                return handles
            # Single-teacher lane: the dict key is verl's naming (measured
            # 2026-08-18: this verl normalizes to 'default', not the hydra
            # entry name 'teacher_model'), but with exactly ONE teacher there
            # is no ambiguity -- take it and say so. Multiple keys stay a
            # loud failure: guessing among teachers would be silent-wrong.
            ks = await reg.keys.remote()
            if len(ks) == 1:
                print(f"[simopd] teacher_registry: key {key!r} absent; single "
                      f"teacher {ks[0]!r} resolved instead", file=sys.stderr, flush=True)
                return await reg.get.remote(ks[0])
            last = f"registry up but no handles under key {key!r} (have {ks})"
        except ValueError:
            last = "registry actor not found"
        await asyncio.sleep(2.0)
    raise RuntimeError(
        f"teacher_registry: could not resolve teacher handles ({last}); a5 cannot mix "
        f"without a teacher route and refuses to train as vanilla")
