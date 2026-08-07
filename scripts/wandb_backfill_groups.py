"""Give the pre-grouping runs the group/job_type/tags the later ones get for free.

    python scripts/wandb_backfill_groups.py              # dry run, changes nothing
    python scripts/wandb_backfill_groups.py --apply      # writes

DRY RUN BY DEFAULT, and that is not politeness -- this is the only tool here that
writes to someone's wandb account rather than to disk. It prints the exact before
and after for every run and touches nothing until --apply.

WHY ANY RUN LACKS A GROUP. Grouping is set from the environment in
deploy/dsw/_lane.sh, which every lane reads at launch. Runs started before that
edit have no group and never will retroactively -- a wandb run's metadata is
fixed at init unless something updates it. This is that something.

WHAT IT SETS, and why these three:
  group     <student>__from__<teacher>__s<seed>. The collapsible cell: its members
            are the OPD methods being compared at one pair and one seed.
  job_type  the arm. The SECOND grouping axis -- regroup by it in the UI and the
            members become the seeds of one method, whose spread is the run-to-run
            noise. The group above cannot show that, since its members differ by
            method rather than by seed.
  tags      arm, axis, seed, pair -- for filtering without writing a query.

It reads the pair and seed from each run's own config, never from the run name:
verl hands wandb the whole resolved hydra config (ray_trainer.py:1398), so
actor_rollout_ref.model.path, the teacher path and data.seed are all there. A name
can be typed wrong; the config is what the run actually used.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ENTITY = os.environ.get("WANDB_ENTITY", "lichangh2002")
PROJECT = os.environ.get("WANDB_PROJECT", "simopd")

# Mirrors _lane.sh: strip a trailing /hf (the FSDP merger's output dir) so a local
# checkpoint reads as coldstart_sft rather than as "hf".
def short(path: str) -> str:
    p = (path or "").rstrip("/")
    if p.endswith("/hf"):
        p = p[: -len("/hf")]
    return os.path.basename(p) or "?"


def facets(run):
    """(student, teacher, seed) from the run's own config."""
    c = run.config
    student = (c.get("actor_rollout_ref", {}) or {}).get("model", {}).get("path")
    teachers = (c.get("distillation", {}) or {}).get("teacher_models", {}) or {}
    teacher = next((v.get("model_path") for v in teachers.values() if isinstance(v, dict)), None)
    seed = (c.get("data", {}) or {}).get("seed")
    return student, teacher, seed


def axis_of(arm: str) -> str:
    return "baseline" if arm == "vanilla" else f"axis{arm[:1].upper()}"


def arm_of(run) -> str:
    """Arm name = the run name minus its _s<seed> suffix. The name is verl's
    experiment_name, which _lane.sh built as ${ARM}_s${SEED}."""
    return re.sub(r"_s\d+(_.*)?$", "", run.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--only-missing", action="store_true", default=True,
                   help="skip runs that already have a group (default)")
    p.add_argument("--all", dest="only_missing", action="store_false",
                   help="also rewrite runs that already have a group")
    a = p.parse_args()

    import wandb

    api = wandb.Api()
    runs = list(api.runs(f"{a.entity}/{a.project}"))
    print(f"{a.entity}/{a.project}: {len(runs)} runs\n")

    planned, skipped, unresolved = [], 0, []
    for run in runs:
        if a.only_missing and run.group:
            skipped += 1
            continue
        student, teacher, seed = facets(run)
        if not student or not teacher or seed is None:
            # Refuse to guess. A run whose config lacks the facets is usually a
            # crashed init, and inventing a group for it would put a run that
            # computed nothing into a cell used for comparison.
            unresolved.append(run)
            continue
        arm = arm_of(run)
        planned.append((run, {
            "group": f"{short(student)}__from__{short(teacher)}__s{seed}",
            "job_type": arm,
            "tags": sorted({arm, axis_of(arm), f"seed{seed}",
                            f"{short(student)}__from__{short(teacher)}"}),
        }))

    for run, new in planned:
        print(f"  {run.name}")
        print(f"     group    {run.group or '<none>'}  ->  {new['group']}")
        print(f"     job_type {run.job_type or '<none>'}  ->  {new['job_type']}")
        print(f"     tags     {list(run.tags) or '<none>'}  ->  {new['tags']}")

    if unresolved:
        print(f"\n  {len(unresolved)} run(s) have no pair/seed in their config -- SKIPPED, not guessed:")
        for r in unresolved:
            print(f"     {r.name}  (state={r.state})")

    print(f"\n{len(planned)} to update, {skipped} already grouped, {len(unresolved)} unresolved")

    if not a.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to commit.")
        return

    for run, new in planned:
        run.group = new["group"]
        run.job_type = new["job_type"]
        run.tags = new["tags"]
        run.update()
        print(f"  updated {run.name}")
    print(f"\n{len(planned)} runs updated.")


if __name__ == "__main__":
    sys.exit(main())
