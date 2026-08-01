"""Auto-imported by every Python process that has this dir on PYTHONPATH.

verl forwards PYTHONPATH to its Ray workers (constants_ppo.py), and the
distillation loss registry is a module-level dict that must be populated in the
*worker* process where the loss is looked up -- registering only in the driver
silently leaves workers with the stock registry.

Registration is lazy, not a plain import at startup. Importing our losses here
unconditionally drags verl into every interpreter on the node, including the
vLLM engine and each of its worker subprocesses: measured at +14.5s of startup
apiece (0.02s -> 14.52s), which stalled the first all-arm smoke. So we watch for
verl's own loss module instead and register right after it finishes loading.
Processes that never touch verl's distillation code pay nothing.
"""

import sys
from importlib.abc import MetaPathFinder

_TARGET = "verl.trainer.distillation.losses"


class _RegisterArmsAfterVerlLosses(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None

        # Delegate to the real finders (skipping ourselves) for the true spec, then
        # wrap exec_module so our registrations run right after verl's. Wrapping
        # rather than importing here avoids re-executing the module, which would
        # trip verl's duplicate-name guard in the registry.
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
            try:
                sys.meta_path.remove(_hook)
            except ValueError:
                pass
            try:
                import simopd.losses  # noqa: F401  (import side effect: registry population)
            except Exception as e:  # a broken arm must not take the trainer down silently
                print(f"[simopd] custom loss registration FAILED: {e!r}", file=sys.stderr)

        spec.loader.exec_module = exec_module
        return spec


_hook = _RegisterArmsAfterVerlLosses()
sys.meta_path.insert(0, _hook)
