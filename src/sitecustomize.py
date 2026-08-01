"""Auto-imported by every Python process that has this dir on PYTHONPATH.

verl forwards PYTHONPATH to its Ray workers (constants_ppo.py), and both things
we install are process-local: the distillation loss registry is a module-level
dict, and the teacher patch rebinds a name inside the vLLM server module. Doing
it only in the driver would silently leave the workers stock.

Registration is lazy, not a plain import at startup. Importing our losses here
unconditionally drags verl into every interpreter on the node, including the
vLLM engine and each of its worker subprocesses: measured at +14.5s of startup
apiece (0.02s -> 14.52s), which stalled the first all-arm smoke. Instead we hook
the specific verl modules we extend and act right after each one loads, so a
process that never touches them pays nothing.
"""

import sys
from importlib.abc import MetaPathFinder


def _after_verl_losses():
    import simopd.losses  # noqa: F401  (import side effect: registry population)


def _after_vllm_server():
    from simopd import teacher_patch

    teacher_patch.install()


# verl module -> what to run once it has finished executing
_TARGETS = {
    "verl.trainer.distillation.losses": _after_verl_losses,
    "verl.workers.rollout.vllm_rollout.vllm_async_server": _after_vllm_server,
}


class _SimOPDInstallHook(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        callback = _TARGETS.get(fullname)
        if callback is None:
            return None

        # Delegate to the real finders (skipping ourselves) for the true spec, then
        # wrap exec_module so our hook runs right after verl's module body. Wrapping
        # rather than importing the target here avoids re-executing it, which would
        # trip verl's duplicate-registration guard.
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None

        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec

        real_exec_module = spec.loader.exec_module

        def exec_module(module):
            real_exec_module(module)
            _TARGETS.pop(fullname, None)
            if not _TARGETS:
                try:
                    sys.meta_path.remove(_hook)
                except ValueError:
                    pass
            try:
                callback()
            except Exception as e:  # a broken arm must not disappear into a traceback-free hang
                print(f"[simopd] install hook for {fullname} FAILED: {e!r}", file=sys.stderr)

        spec.loader.exec_module = exec_module
        return spec


_hook = _SimOPDInstallHook()
sys.meta_path.insert(0, _hook)
