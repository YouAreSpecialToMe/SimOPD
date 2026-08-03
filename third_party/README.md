# Vendored third-party code

## `instruction_following_eval/`

Google's official IFEval checker, copied verbatim from
[google-research/google-research](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
on 2026-08-03 (`instructions.py`, `instructions_registry.py`, `instructions_util.py`;
`__init__.py` added because the upstream directory is not a package).

**Why vendored rather than pip-installed:** there is no official PyPI release. The
name `instruction-following-eval` does not exist on PyPI, and `ifeval` (0.0.1) is an
unrelated third-party upload — depending on it would silently substitute someone
else's reimplementation for the checker BENCHMARKS.md §3 pins.

**Why not reimplemented:** the transfer column's value is that it is comparable to
the papers that already report IFEval (Teachability, RG-OPD). A different checker
makes our number a different number.

Not modified. Runtime deps: `absl-py`, `langdetect`, `nltk`, `immutabledict`, plus
the `punkt` corpus (`scripts/fetch_assets.py` downloads it).
