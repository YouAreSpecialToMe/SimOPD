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
# Termination family (vanilla_corr / n2_termcal): the loss-side student stop set E_S must be
# exactly what ends a rollout under the arm's stop contract -- model eos 151643 plus
# SIMOPD_STOP_IDS when that contract is on (simopd.eos_gather.rollout_stop_set mirrors this).
_MODEL_EOS = "151643"


def _term_family_problems(run_id, env):
    out = []
    on_carrier = str(env.get("SIMOPD_GATHER_EOS", "")) == "1" or env.get("DISTILLATION_LOSS_MODE") in ("k1_termfix", "k1_termcal")
    if not on_carrier:
        return out
    if str(env.get("SIMOPD_TERM_EVENT", "")) == "1":
        for k, want in (("SIMOPD_KEEP_SAMPLED", "1"), ("SIMOPD_GATHER_EOS", "1")):
            if str(env.get(k, "")) != want:
                out.append(f"[{run_id}] SIMOPD_TERM_EVENT=1 needs {k}={want} (the N0 carrier); got {env.get(k)!r}")
        try:
            if int(env.get("DISTILLATION_TOPK", 0)) < 3:
                out.append(f"[{run_id}] SIMOPD_TERM_EVENT=1 needs DISTILLATION_TOPK >= 3 (top-k payload carries the terminators)")
        except ValueError:
            out.append(f"[{run_id}] DISTILLATION_TOPK not an int")
    raw = str(env.get("SIMOPD_STOP_IDS", "")).strip()
    want = {_MODEL_EOS}
    if raw and raw.lower() != "off":
        want |= {x.strip() for x in raw.split(",") if x.strip()}
    if raw == "":
        out.append(f"[{run_id}] termination arm without an explicit SIMOPD_STOP_IDS -- the launcher "
                   "defaults NEW runs to the dual contract, which would silently make this a two-knob "
                   "comparison against the legacy vanilla rows; pin 'off' or the dual set")
    have = {x.strip() for x in str(env.get("SIMOPD_EOS_IDS", "")).split(",") if x.strip()}
    if have != want:
        out.append(f"[{run_id}] SIMOPD_EOS_IDS={sorted(have)} != rollout stop set {sorted(want)} under "
                   f"SIMOPD_STOP_IDS={raw!r} (eos_gather would refuse this at import)")
    return out


# 不写 h9 预算中继的 loss mode。2026-08-21 之前这里是一份白名单({"k1_rec"}),
# 因为钩子只挂在 k1_rec 上;现在 _h9_observe 同时挂在 k1_rec 和 top-k 工厂
# (_topk_registry_fn)上,覆盖了除下面这些之外的全部 mode,所以改成黑名单——
# 白名单会随着每加一个 top-k 臂就漏报一次,方向反了。
H9_NO_RELAY_MODES = {"k1"}   # verl 自带的原版 k1:不经过我们的任何 registry 函数


EXPECT_PG = {
    "c4_rep": False, "c4_hq": False, "c4_state": False, "c4_carrier": False,
    "c5_union_rkl": False, "c5_union_fkl": False,
    "a1_gkd_mix0.5_n0": True,
    "a3_offpolicy_n0": True,
    "a4_dagger_anneal_n0": True,
    "a5_aggrevate_n0": True,
    "h6_gen_sched_n0": True,
    "h7_gen512_n0": True,
    "h8_gen2048_n0": True,
    "h9_prune_adapt_n0": True,
    "h10_task_subset_n0": True,
    "vanilla": True, "a1_gkd_mix0.5": True, "a3_offpolicy": True, "a2_coldstart": True,
    "a4_dagger_anneal": True, "a5_aggrevate": True,
    "b1_skew_kl": True, "b2_forward_kl": False, "b3_eopd_gate": True,
    "b4_jsd": False, "b4_jsd_b0.1": False, "b4_jsd_b0.9": False, "b5_k2": False,
    "c1_lsm_topk32_renorm": True, "c1_direct": False, "c1_tailbucket": True,
    "c2_quantile_budget": False,
    "c2_qb_fixed8": False, "c2_qb_perseq": False,
    "c3_intersection": False, "c4_pi_tail_budget": False,
    "c4_pi_tail_budget_corr": False, "c2_quantile_budget_corr": False,
    "d1_tip": True, "d2_selectkd": True, "d3_teachability": True,
    "e1_pl_rank": False, "e1_pl_rank_a0": False, "e2_set_coverage": False,
    "e2_set_coverage_a0": False, "e3_zvalue": False,
    "f1_soft_log": True, "f2_hard_clip": True, "f3_power": True,
    "f2_clip2.3": True, "f4_posclip": True, "f5_tanh": True,
    "vanilla_corr": True,
    "n2_termcal": True,
    "n2_corr": True, "f1_soft_log_corr": True, "f2_hard_clip_corr": True, "f3_power_corr": True,
    "h2_last_segment_corr": True, "b1_skew_kl_corr": True, "g1_verified_only_corr": True, "g4_failure_only_corr": True,
    "g6_seqmean_corr": True,
    "f2_clip2.3_corr": True, "f4_posclip_corr": True, "f5_tanh_corr": True, "b5_k2_corr": False,
    "h1_first_segment_corr": True, "h3_random_segment_corr": True, "h4_random_scatter_corr": True, "g5_rgopd_gate_corr": True,
    "g2_fire_likelihood_corr": True,
    "d1_tip_corr": True, "d2_selectkd_corr": True, "d3_teachability_corr": True,
    "b2_forward_kl_corr": False, "e2_set_coverage_a0_corr": False, "c3_intersection_corr": False,
    "g1_verified_only": True, "g1_quota": True, "g2_fire_likelihood": True,
    "g4_failure_only": True, "g4_quota": True, "g5_rgopd_gate": True,
    "g6_seqmean": True,
    "h1_first_segment": True, "h2_last_segment": True, "h3_random_segment": True,
    "h4_random_scatter": True, "h5_gen100": True,
    "h7_gen512": True, "h8_gen2048": True, "h10_task_subset": True,
    "h6_gen_sched": True, "h9_prune_adapt": True,
    "i0_think_scorer": True, "i1_priv_cot": True,
    "vanilla_n8": True, "j1_kdrl": False,
    # 2026-09-02 登记的四条(ARM-REVIEW §6.3);分支同各自的对照臂:vanilla_te 同 vanilla_corr
    # (k1 族,PG),c2 梯子两条同 c2_quantile_budget_corr(预算族,direct),h5_gen100_n0 同
    # h7_gen512_n0(k1_termfix,PG)。漏在这张表外 = 「not in the lint's branch table」=
    # 舰队发射门直接不起 lane —— 2026-09-04 审查发现时四条都漏着。
    "vanilla_te": True, "c2_qb_fixed8_corr": False, "c2_qb_perseq_corr": False, "h5_gen100_n0": True,
}
# Kernels that split the sampled column off the teacher payload.
WANT_SAMPLED_MODES = {"eopd_entropy_gate", "k1_fire_gate", "tip_select",
                      "selectkd_verify", "teachability_select", "k1_termcal", "k1_termfix"}
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

        # One dose, one knob: gkd_mix.install() refuses SIMOPD_GKD_LAMBDA +
        # SIMOPD_GKD_SCHEDULE together (constant vs schedule claiming the same
        # coin); catch it at registration so that refusal never actually fires.
        for ek in [k for k in a if k.startswith("env")]:
            d = a.get(ek)
            if isinstance(d, dict) and "SIMOPD_GKD_LAMBDA" in d and "SIMOPD_GKD_SCHEDULE" in d:
                problems.append(f"{tag} {ek} sets both SIMOPD_GKD_LAMBDA and "
                                f"SIMOPD_GKD_SCHEDULE -- one dose, one knob")

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
        # under SIMOPD_TERM_EVENT=1 every sampled-k1 family mode rides the N0 carrier and the
        # termfix kernel consumes the sampled column (topk_losses.TERM_EVENT_FAMILY)
        _te = str(env.get("SIMOPD_TERM_EVENT", "")) == "1"
        wants_sampled = mode in WANT_SAMPLED_MODES or (_te and (mode in T.TERM_EVENT_FAMILY or mode == "k1_fire_gate"))
        if wants_sampled and env.get("SIMOPD_KEEP_SAMPLED") != "1":
            problems.append(f"{tag} kernel unpacks the sampled column but SIMOPD_KEEP_SAMPLED != 1")
        if not wants_sampled and env.get("SIMOPD_KEEP_SAMPLED") == "1" and str(env.get("SIMOPD_GATHER_EOS", "")) != "1":
            notes.append(f"{tag} KEEP_SAMPLED set but kernel does not consume the column (harmless width)")

        # h9 的预算是 trainer->server 的中继,而写中继的钩子 _h9_observe 只挂在 k1_rec 上。
        # 换了 loss mode 的 h9 臂,服务端照常 armed、照常读中继,但没有任何东西写它,
        # budget() 于是永远返回冷启动默认值(整个 16384 窗口)—— 那个臂看起来在跑 h9,
        # 实际上是一条没有裁剪的 vanilla。h9_prune_adapt_n0 就这么烧了 66 步才被发现
        # (2026-08-21),从 step 10 起长度就和基线差 4.6 倍。
        if str(env.get("SIMOPD_H9_ADAPT", "")) not in ("", "0") and mode in H9_NO_RELAY_MODES:
            problems.append(f"{tag} SIMOPD_H9_ADAPT=1 但 loss mode {mode!r} 不写预算中继 "
                            f"(不经过 _h9_observe 的挂钩点)—— 服务端会一直用冷启动 "
                            f"默认 16384,这个臂等于没有 h9")

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
        problems.extend(_term_family_problems(rid, env))

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
