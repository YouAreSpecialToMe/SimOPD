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

import os
import sys
from importlib.abc import MetaPathFinder


class _NeuterModelScopePatching(MetaPathFinder):
    """Let ModelScope's `patch_hub` module import, but make it do nothing.

    `modelscope.utils.hf_util.patch_hub()` replaces huggingface_hub and the
    transformers `from_pretrained` classmethods so every model resolves through
    ModelScope instead. verl calls it only when VERL_USE_MODELSCOPE is true -- but
    on a DSW instance it was patched anyway, with the flag set to False, and the
    consequences are not local:

      * ModelScope's default branch is `master`, so asking for HuggingFace's default
        `main` fails with NotExistError on a model that is sitting on disk;
      * models then load from the ModelScope cache, which `fetch_assets.py` does not
        check -- so it reports every asset present while verl reads a different, and
        in that case corrupt, copy (UnicodeDecodeError at byte 22845308 of a
        tokenizer.json).

    An earlier version raised ImportError here instead. That did stop the patching,
    and it also killed the run: something in the worker imports this module
    unconditionally, so refusing turned a silent misroute into a hard crash. Denying
    an import that somebody genuinely makes is not the same as declining the
    behaviour it would have installed.

    So the module is served as a stub whose `patch_hub` is a no-op; every other
    attribute raises with an explanation. The importer succeeds, the patch never
    applies, and models resolve through HuggingFace as the flag asks. It also prints
    the importing frame once -- on a managed image the caller is the thing worth
    knowing, and nothing else in the traceback names it.

    Applies only when NEITHER flag is on. verl reads VERL_USE_MODELSCOPE, and vLLM
    reads its own VLLM_USE_MODELSCOPE in vllm/transformers_utils/__init__.py -- which
    is what was actually routing DSW through ModelScope while verl's was False.
    """

    _TARGET = "modelscope.utils.hf_util"
    _reported = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._TARGET:
            return None
        from importlib.machinery import ModuleSpec

        if not _NeuterModelScopePatching._reported:
            _NeuterModelScopePatching._reported = True
            import traceback

            caller = "unknown"
            for frame in reversed(traceback.extract_stack()[:-1]):
                if "importlib" not in frame.filename and "sitecustomize" not in frame.filename:
                    caller = f"{frame.filename}:{frame.lineno} in {frame.name}"
                    break
            print(
                f"[simopd] neutralised modelscope.utils.hf_util (imported by {caller}). "
                "It would have repointed huggingface_hub and transformers at ModelScope "
                "while VERL_USE_MODELSCOPE is not true, which is how models end up "
                "loading from a cache nothing checks. Set VERL_USE_MODELSCOPE=true to "
                "use ModelScope properly instead.",
                file=sys.stderr,
            )
        return ModuleSpec(fullname, _StubLoader())


class _StubLoader:
    """Produces a module where every attribute is a harmless no-op."""

    def create_module(self, spec):
        import types

        mod = types.ModuleType(spec.name)

        def _noop(*args, **kwargs):
            return None

        mod.patch_hub = _noop
        mod.unpatch_hub = _noop
        mod.__file__ = "<simopd stub for modelscope.utils.hf_util>"

        def _missing(name):
            # Dunders MUST raise. A blanket handler that answers them returns a
            # function for __file__, and then anything walking the stack -- inspect,
            # and through it torch.library's custom-op registration -- does
            # filename.endswith(...) on a function and dies:
            #   AttributeError: 'function' object has no attribute 'endswith'
            # which surfaces as transformers failing to import AutoTokenizer. The
            # stub then breaks far more than it fixes.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            # Everything else raises too, and the message says why. Answering with
            # a no-op looked safer and is worse: vLLM's ModelScope branch imports
            # AutoConfig from this module, so a no-op silently became the config
            # class and the failure surfaced 200 frames away as
            #   AttributeError: 'function' object has no attribute 'from_pretrained'
            # A stub may decline to patch. It must not impersonate a class.
            raise AttributeError(
                f"{name!r} is not provided by simopd's stub for "
                "modelscope.utils.hf_util. The stub exists to stop patch_hub() "
                "repointing this process at ModelScope. If ModelScope is wanted, set "
                "VLLM_USE_MODELSCOPE=true and VERL_USE_MODELSCOPE=true; if it is not, "
                "the caller above should not be reaching for ModelScope at all."
            )

        mod.__getattr__ = _missing
        return mod

    def exec_module(self, module):
        return None


def _modelscope_wanted():
    """True if EITHER package was asked to use ModelScope.

    Two flags, two packages, one effect. verl reads VERL_USE_MODELSCOPE at import and
    calls patch_hub(); vllm.transformers_utils reads VLLM_USE_MODELSCOPE at import and
    calls the same function. Neutering while one of them is on would break that
    package's own ModelScope paths, so the stub only applies when neither asked.
    """
    return any(
        os.environ.get(v, "False").lower() in ("true", "1", "yes")
        for v in ("VERL_USE_MODELSCOPE", "VLLM_USE_MODELSCOPE")
    )


if not _modelscope_wanted():
    sys.meta_path.insert(0, _NeuterModelScopePatching())


def _after_verl_losses():
    import simopd.losses  # noqa: F401  (import side effect: registry population)
    # The additive-term wrapper must arm BEFORE the first distillation_loss call:
    # lazily imported from inside the registry fn, it installed one call too late
    # and b3's first micro-batch silently dropped the EOPD term (audit 2026-08-07).
    import simopd.b3_additive  # noqa: F401  (install() runs at import, idempotent)


def _after_vllm_server():
    """Two independent installs, and neither may take the other down with it.

    They shared a body, with teacher_patch.install() first. Anything it raised meant
    zmq_lane.install_server() never ran -- while the SENDER patch, which lives in a
    different callback, still applied. That is the exact asymmetry zmq_lane's own
    docstring warns about: one end tagged and the other not compute different socket
    paths, never meet, and the run blocks at its first weight sync with engines
    resident, 0% GPU and nothing in any log. A run that dies at the moment it starts
    is what that looks like from outside.

    So each runs in its own try, and a failure is loud rather than silent -- a patch
    that quietly did not apply is worse than one that crashed, because the crash is
    the only thing that would have said so.
    """
    import traceback

    # Attempt every install first (so one failure does not hide the others'
    # status), then RAISE: a swallowed install here is the zmq-incident class --
    # one socket end tagged, the other not, an 8-hour 0%-GPU hang with no error
    # (audit 2026-08-07: the old print-only handler had silently revoked the
    # raise-on-failure discipline for exactly this path).
    errors = []
    for mod, fn in (("teacher_patch", "install"), ("zmq_lane", "install_server"),
                    ("gkd_mix", "install")):
        try:
            m = __import__(f"simopd.{mod}", fromlist=[fn])
            getattr(m, fn)()
        except Exception as e:
            traceback.print_exc()
            errors.append(f"simopd.{mod}.{fn}: {e!r}")
    if errors:
        raise RuntimeError("[simopd] FATAL install failure(s), all were attempted "
                           "first: " + "; ".join(errors))


def _after_vllm_rollout():
    from simopd import zmq_lane

    zmq_lane.install_sender()


def _after_vllm_rollout_utils():
    from simopd import zmq_lane

    zmq_lane.install_receiver_logging()


# What the hooks import. Declared so preflight.py can check they are importable
# BEFORE a run starts, on the machine that will run it -- the failure this guards was
# a module that existed on the author's box and on none of the four that needed it.
REQUIRED_MODULES = ("simopd.losses", "simopd.topk_losses", "simopd.teacher_patch",
                    "simopd.zmq_lane", "simopd.gkd_mix", "simopd.gkd_schedule",
                    "simopd.gkd_stats", "simopd.b3_additive")


# verl module -> what to run once it has finished executing
_TARGETS = {
    "verl.trainer.distillation.losses": _after_verl_losses,
    "verl.workers.rollout.vllm_rollout.vllm_async_server": _after_vllm_server,
    # Both writers of the weight-transfer job id, which collides across lanes because
    # Ray hands every lane's own cluster the same 01000000. The receiver needs no hook:
    # it reads VERL_RAY_JOB_ID, which _after_vllm_server has already lane-stamped.
    # See simopd.zmq_lane.
    "verl.workers.rollout.vllm_rollout.vllm_rollout": _after_vllm_rollout,
    # Read-only: the receiver's path is already correct via the lane-stamped job id.
    # It logs so a disagreement between the two ends is a grep, not a 30-minute stall.
    "verl.workers.rollout.vllm_rollout.utils": _after_vllm_rollout_utils,
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
            # Pop only after the callback SUCCEEDS: popped-before, a raised callback
            # plus any caught-and-retried import would re-execute the module with no
            # hook and no patches (audit 2026-08-07 hardening).
            if not _TARGETS:
                try:
                    sys.meta_path.remove(_hook)
                except ValueError:
                    pass
            # Raise. Every hook here is load-bearing: the loss registry, the teacher
            # patch, and both ends of the weight-transfer socket. This used to print
            # and continue, under a comment about not disappearing into a
            # traceback-free hang -- and that is exactly what it produced. On
            # 2026-08-06 simopd.zmq_lane was missing from four machines (a .gitignore
            # pattern kept it out of git), this printed one line into a hundred
            # thousand, and two runs then hung for eight hours holding four GPUs.
            #
            # Thirty seconds of traceback at import beats eight hours of silence. A
            # patch that quietly did not apply is worse than one that crashed, because
            # the crash is the only thing that reports it.
            callback()

            _TARGETS.pop(fullname, None)

        spec.loader.exec_module = exec_module
        return spec


_hook = _SimOPDInstallHook()
sys.meta_path.insert(0, _hook)
