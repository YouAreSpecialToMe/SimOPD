"""Walk every arm's full stack and refuse the silent failure classes.

    PYTHONPATH=src python scripts/arm_lint.py

Born from the pre-S/T review (2026-08-07): after a week in which the optimizer
branch, five selector parameters, the sampling triple and the response cap all
moved, the remaining risk is not any single file but the JOINTS -- an env knob
whose name no kernel reads (it silently becomes its default), a loss mode whose
registry settings disagree with what its kernel consumes, a branch flag that
contradicts the provenance verdict, an arm the verdict ledger forgot. Each check
here is one observed or near-miss failure class, not speculation:

  ENV-CONSUMED   every SIMOPD_*/knob an arm sets must be read by src/ or the
                 launch script (the FiRe NameError and the b3 threshold rename
                 both created windows where a typo'd knob would no-op).
  REGISTRY       loss mode registered; use_topk modes present in TOPK_DISPATCH
                 (verl-native forward_kl_topk excepted) and carrying
                 DISTILLATION_TOPK; kernels that unpack the sampled column
                 (want_sampled=True) must get SIMOPD_KEEP_SAMPLED=1 -- _prepare
                 raises at runtime, but a lint catches it before a lane burns.
  BRANCH         USE_POLICY_GRADIENT / USE_TASK_REWARDS against the audit-r5
                 faithful-branch table, hardcoded here on purpose: the lint is
                 the executable form of the provenance verdict.
  TSV            every stock arm exactly once; shelved/needs arms absent.
  VERDICT        every stock arm present in verdict.py's ARMS ledger (PENDING
                 rows are free; a forgotten row is a verdict that never happens).
"""

import os
import re
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Audit r5's faithful optimizer branch, per arm (True = PG). The lint fails if
# arms.yaml drifts from this table without the table being updated -- which is
# exactly the ceremony a branch change should cost.
EXPECT_PG = {
    "vanilla": True, "a1_gkd_mix0.5": True, "a3_offpolicy": True, "a2_coldstart": True,
    "b1_skew_kl": True, "b2_forward_kl": False, "b3_eopd_gate": True,
    "b4_jsd": False, "b4_jsd_b0.1": False, "b4_jsd_b0.9": False, "b5_k2": False,
    "c1_lsm_topk32_renorm": True, "c2_quantile_budget": False,
    "c2_qb_fixed8": False, "c2_qb_perseq": False,
    "c3_intersection": False, "c4_pi_tail_budget": False,
    "d1_tip": True, "d2_selectkd": True, "d3_teachability": True,
    "e1_pl_rank": False, "e1_pl_rank_a0": False, "e2_set_coverage": False,
    "e2_set_coverage_a0": False, "e3_zvalue": False,
    "f1_soft_log": True, "f2_hard_clip": True, "f3_power": True,
    "g1_verified_only": True, "g1_quota": True, "g2_fire_likelihood": True,
    "g4_failure_only": True, "g4_quota": True, "g5_rgopd_gate": True,
    "h1_first_segment": True, "h2_last_segment": True, "h3_random_segment": True,
    "i0_think_scorer": True, "i1_priv_cot": True,
    "vanilla_n8": True, "j1_kdrl": False,
}
# Kernels that split the sampled column off the teacher payload.
WANT_SAMPLED_MODES = {"eopd_entropy_gate", "k1_fire_gate", "tip_select",
                      "selectkd_verify", "teachability_select"}
# verl-native top-k modes with no entry in our dispatch.
VERL_NATIVE_TOPK = {"forward_kl_topk"}


def read_consumers():
    """Every env name src/ or the launch script reads."""
    consumed = set()
    pat_py = re.compile(r'environ(?:\.get)?\(\s*["\']([A-Z0-9_]+)["\']')
    for dirpath, _, files in os.walk(os.path.join(ROOT, "src")):
        for f in files:
            if f.endswith(".py"):
                consumed |= set(pat_py.findall(open(os.path.join(dirpath, f)).read()))
    sh = open(os.path.join(ROOT, "scripts", "run_opd_baseline.sh")).read()
    consumed |= set(re.findall(r"\$\{([A-Z0-9_]+)[:\-}]", sh))
    consumed |= set(pat_py.findall(open(os.path.join(ROOT, "scripts", "eval_offline.py")).read()))
    return consumed


def main():
    arms = yaml.safe_load(open(os.path.join(ROOT, "configs", "arms.yaml")))["arms"]
    tsv = open(os.path.join(ROOT, "configs", "campaign.tsv")).read()
    verdict_src = open(os.path.join(ROOT, "scripts", "verdict.py")).read()
    consumed = read_consumers()

    from verl.trainer.distillation.losses import (
        DISTILLATION_LOSS_REGISTRY, DISTILLATION_SETTINGS_REGISTRY)
    import simopd.topk_losses as T

    tsv_pairs = re.findall(r"^\d+\t\S+\t(\S+)\t(\S+)", tsv, re.M)
    tsv_names = [a for a, _ in tsv_pairs]
    problems, notes = [], []

    for a in arms:
        rid, env, status = a["run_id"], (a.get("env") or {}), a["status"]
        tag = f"[{rid}]"

        # --- env knobs actually consumed somewhere ---
        for k in env:
            if k not in consumed and k not in ("STUDENT_MODEL", "DATA_DIR", "TRAIN_FILE_BASENAME"):
                (problems if status == "stock" else notes).append(
                    f"{tag} sets {k} but nothing reads it"
                    + (" (typo'd knob = silent default)" if status == "stock"
                       else f" ({status}: reader lands with enlistment)"))

        if status != "stock":
            notes.append(f"{tag} {status}; skipped runtime checks")
            continue

        # --- registry and kernel requirements ---
        # Arms list only their deviation; the launch default is the baseline loss.
        mode = env.get("DISTILLATION_LOSS_MODE", "k1_rec")
        if mode not in DISTILLATION_LOSS_REGISTRY:
            problems.append(f"{tag} loss mode {mode!r} unregistered")
            continue
        st = DISTILLATION_SETTINGS_REGISTRY[mode]
        if getattr(st, "use_topk", False):
            if mode not in T.TOPK_DISPATCH and mode not in VERL_NATIVE_TOPK:
                problems.append(f"{tag} use_topk mode {mode!r} missing from TOPK_DISPATCH")
            if "DISTILLATION_TOPK" not in env:
                problems.append(f"{tag} top-k mode without DISTILLATION_TOPK")
        if mode in WANT_SAMPLED_MODES and env.get("SIMOPD_KEEP_SAMPLED") != "1":
            problems.append(f"{tag} kernel unpacks the sampled column but SIMOPD_KEEP_SAMPLED != 1")
        if mode not in WANT_SAMPLED_MODES and env.get("SIMOPD_KEEP_SAMPLED") == "1":
            notes.append(f"{tag} KEEP_SAMPLED set but kernel does not consume the column (harmless width)")

        # --- optimizer branch vs the r5 verdict ---
        pg = env.get("USE_POLICY_GRADIENT", "True") != "False"
        if EXPECT_PG.get(rid) is None:
            problems.append(f"{tag} not in the lint's branch table -- update EXPECT_PG deliberately")
        elif pg != EXPECT_PG[rid]:
            problems.append(f"{tag} branch is {'PG' if pg else 'direct'}, audit r5 says "
                            f"{'PG' if EXPECT_PG[rid] else 'direct'}")
        # A gated arm parks its knobs under env2_pending; flipping status without
        # renaming the key ships an arm with NO env, and gkd_mix installs only when
        # SIMOPD_GKD_CACHE is set -- a1/a3 would train as two more vanillas and
        # nothing downstream would say so (audit r6 2026-08-09).
        if status == "stock" and any(k.startswith("env") and k != "env" for k in a):
            parked = [k for k in a if k.startswith("env") and k != "env"]
            problems.append(f"{tag} is stock but its knobs are still parked under {parked} -- "
                            f"rename to env: or the arm runs with none")
        if rid == "j1_kdrl" and env.get("USE_TASK_REWARDS") != "True":
            problems.append(f"{tag} KDRL without USE_TASK_REWARDS=True is pure KD")

        # --- manifest and ledger wiring ---
        if tsv_names.count(rid) == 0:
            problems.append(f"{tag} has no campaign.tsv row")
        for pair in {p for p in tsv_pairs if p[0] == rid}:
            if tsv_pairs.count(pair) > 1:
                problems.append(f"{tag} seed {pair[1]} appears {tsv_pairs.count(pair)}x in campaign.tsv")
        if rid != "vanilla" and f'"{rid}"' not in verdict_src:
            problems.append(f"{tag} missing from verdict.py ARMS -- its verdict would simply never print")

        # --- arm.py env materializes ---
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "arm.py"), "env", rid],
                           capture_output=True, text=True)
        if r.returncode != 0:
            problems.append(f"{tag} arm.py env failed: {r.stderr.strip().splitlines()[-1]}")

    for a in arms:
        if a["status"] != "stock" and a["run_id"] in tsv_names:
            problems.append(f"[{a['run_id']}] {a['status']} but present in campaign.tsv")

    print(f"arm_lint: {len(arms)} arms, {len(problems)} problem(s), {len(notes)} note(s)\n")
    for p in problems:
        print(f"  PROBLEM  {p}")
    for n in notes:
        print(f"  note     {n}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
