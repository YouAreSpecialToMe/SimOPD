"""Auto-imported by every Python process that has this dir on PYTHONPATH.

verl forwards PYTHONPATH to its Ray workers (constants_ppo.py), and the
distillation loss registry is a module-level dict that must be populated in the
*worker* process where the loss is looked up -- registering only in the driver
silently leaves workers with the stock registry. Importing here is what makes
our arms visible everywhere, with no edit to verl itself.
"""
try:
    import simopd.losses  # noqa: F401  (import side effect: registry population)
except Exception as e:  # never break the interpreter over this
    import sys
    print(f"[simopd] custom loss registration FAILED: {e!r}", file=sys.stderr)
