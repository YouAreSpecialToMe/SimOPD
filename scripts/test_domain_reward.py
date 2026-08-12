#!/usr/bin/env python3
"""CPU battery for the domain-expansion reward path. No GPU, no downloads.

Three claims, each of which a domain campaign silently depends on:

  IF      the dispatcher's strict criterion agrees with hand-computed verdicts
          on real instruction types, INCLUDING the prompt-rebuild family
          (combination:repeat_prompt) whose skip-bug scores everything 1.0.
  CODE    data_source="codecontests" rows round-trip through verl's own
          prime_code harness -- a correct program scores 1, a wrong one 0 --
          so the G axis and in-loop val have an executable reward on day one.
  ROUTE   unknown data_sources fall through to verl's default dispatch
          (math still scores like math when the custom fn is installed).

Run on any box with the project venv:
    ./simopd/bin/python scripts/test_domain_reward.py
The IF section runs even without verl installed; CODE/ROUTE need the venv.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from simopd import domain_reward  # noqa: E402

FAILS = 0


def check(name, got, want):
    global FAILS
    ok = abs(float(got) - float(want)) < 1e-9
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
    if not ok:
        FAILS += 1


def gt(ids, kwargs, prompt="Write something."):
    return json.dumps({"instruction_id_list": ids, "kwargs": kwargs, "prompt": prompt})


print("--- IF: strict criterion on real instruction types ---")
# 1. keyword inclusion: present vs absent
g = gt(["keywords:existence"], [{"keywords": ["banana", "kiwi"]}])
check("keywords present", domain_reward.compute_score("simopd/ifeval", "I like banana and kiwi a lot.", g), 1.0)
check("keywords absent", domain_reward.compute_score("simopd/ifeval", "I like apples.", g), 0.0)

# 2. multi-constraint AND: both must hold (strict-prompt criterion)
g = gt(["keywords:existence", "detectable_format:number_bullet_lists"],
       [{"keywords": ["banana"]}, {"num_bullets": 2}])
two_bullets_with_kw = "* banana is good\n* banana is yellow"
two_bullets_no_kw = "* apples are good\n* oranges are fine"
check("AND both hold", domain_reward.compute_score("simopd/ifeval", two_bullets_with_kw, g), 1.0)
check("AND one fails", domain_reward.compute_score("simopd/ifeval", two_bullets_no_kw, g), 0.0)

# 3. the prompt-rebuild family: description is built FROM the prompt; skipping the
#    rebuild makes every response pass. Guard the guard.
prompt = "Summarize the meeting notes."
# Real IFEval rows carry prompt_to_repeat in kwargs; the instruction RAISES on a
# bare build (the battery's first run proved it) -- so the realistic-kwargs case
# checks the verdict, and the missing-kwargs case checks the crash-isolation.
g = gt(["combination:repeat_prompt"], [{"prompt_to_repeat": prompt}], prompt=prompt)
check("repeat_prompt followed",
      domain_reward.compute_score("simopd/ifeval", prompt + " The meeting covered budgets.", g), 1.0)
check("repeat_prompt violated",
      domain_reward.compute_score("simopd/ifeval", "The meeting covered budgets.", g), 0.0)

# 4. degenerate rollout: empty text is a fail, not a crash
check("empty response", domain_reward.compute_score("simopd/ifeval", "   ", g), 0.0)

# 4b. malformed row (kwargs missing for an instruction that requires them): the
# reward must survive and score 0, never propagate the exception into the trainer
g_bad = gt(["combination:repeat_prompt"], [{}], prompt=prompt)
check("malformed kwargs -> 0 not crash",
      domain_reward.compute_score("simopd/ifeval", prompt + " extra", g_bad), 0.0)

# 5. length constraint (nltk word counting -- exercises the heavyweight dependency)
g = gt(["length_constraints:number_words"], [{"relation": "at least", "num_words": 5}])
check("word count met", domain_reward.compute_score("simopd/ifeval", "one two three four five six", g), 1.0)
check("word count unmet", domain_reward.compute_score("simopd/ifeval", "too short", g), 0.0)

try:
    import verl  # noqa: F401
    HAVE_VERL = True
except Exception:
    HAVE_VERL = False
    print("--- CODE/ROUTE: skipped (no verl in this interpreter) ---")

if HAVE_VERL:
    print("--- CODE: prime_code round-trip on a stdin/stdout problem ---")
    tests = json.dumps({"inputs": ["3\n1 2 3\n", "2\n10 20\n"], "outputs": ["6\n", "30\n"]})
    good = ("```python\n"
            "n = int(input())\n"
            "print(sum(map(int, input().split())))\n"
            "```")
    bad = ("```python\n"
           "n = int(input())\n"
           "print(max(map(int, input().split())))\n"
           "```")
    check("correct program", domain_reward.compute_score("codecontests", good, tests), 1.0)
    check("wrong program", domain_reward.compute_score("codecontests", bad, tests), 0.0)

    print("--- ROUTE: unknown data_source falls through to stock verl ---")
    math_gt = "42"
    boxed = "The answer is \\boxed{42}."
    got = domain_reward.compute_score("DigitalLearningGmbH/MATH-lighteval", boxed, math_gt)
    got = got.get("score", got) if isinstance(got, dict) else got
    check("math via fallback", float(got) > 0.5, 1.0)

print()
print("RESULT:", "ALL PASS" if FAILS == 0 else f"{FAILS} FAILURE(S)")
sys.exit(1 if FAILS else 0)
