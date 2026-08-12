"""One custom reward entry point for the domain-expansion campaigns (IF / code).

verl dispatches rule rewards by `data_source`, and its registry has no IFEval
entry -- but it does have `custom_reward_function.{path,name}`, which REPLACES
the default for the whole run. So this module is a dispatcher, not a scorer:

    "simopd/ifeval"  -> the vendored Google checker, strict criterion
    anything else    -> verl's own default_compute_score, unchanged

which lets every domain run set the SAME two hydra keys (path=this file,
name="compute_score") while math rows keep the MATH scorer and code rows keep
prime_code (their data_source stays "codecontests", so the fallback branch
routes them exactly where stock verl would have).

The IF criterion is strict-prompt-accuracy: 1.0 iff EVERY constraint on the
prompt is followed. This mirrors `transfer_eval.py`'s loop verbatim -- including
the rebuild for instructions that take the prompt back (combination:repeat_prompt
and friends): their description is constructed from the prompt, and skipping the
rebuild scores them against an empty target, where they pass for free. The two
call sites must never drift apart; if you touch one, touch both.

Ground truth for IF rows is a JSON string:
    {"instruction_id_list": [...], "kwargs": [{...}, ...], "prompt": "..."}
kwargs dicts arrive with their None-valued keys already stripped (the prep
script's job) because build_description(**kwargs) treats an explicit None as an
argument, not an absence.
"""

import json
import os
import sys

_VENDORED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "third_party")


def _vendored_on_path():
    # Same bootstrap as transfer_eval.py: the checker is vendored, not installed.
    if _VENDORED not in sys.path:
        sys.path.insert(0, _VENDORED)
    os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")


def _if_strict_score(solution_str: str, ground_truth) -> float:
    _vendored_on_path()
    from instruction_following_eval import instructions_registry

    meta = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    text = solution_str or ""
    if not text.strip():
        return 0.0
    for index, instr_id in enumerate(meta["instruction_id_list"]):
        instruction = instructions_registry.INSTRUCTION_DICT[instr_id](instr_id)
        kwargs = meta["kwargs"][index] or {}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=meta["prompt"])
        try:
            if not instruction.check_following(text):
                return 0.0
        except Exception:
            # A checker crash is a failure to follow, not a training crash: the
            # transfer harness counts these the same way (n_raised), and one
            # malformed rollout must not kill a 250-step run.
            return 0.0
    return 1.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "simopd/ifeval":
        return _if_strict_score(solution_str, ground_truth)
    # Everything else takes stock verl behavior. Imported lazily so that importing
    # THIS module never drags verl in (the CPU battery runs without it).
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
