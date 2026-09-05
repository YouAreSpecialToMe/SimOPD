#!/usr/bin/env python3
"""把 AGENT.md §2.2 的名册切成舰队要的 lane 图文件(arm:gpus:steps)。

    python scripts/make_lane_map.py --print                 # 先看
    python scripts/make_lane_map.py --out $D/corr_wave --seed 0

为什么要有:名册 35 条要手抄成若干个 slot 文件,每条还得带对步数(30 条 200、5 条 250)。
手抄一次就会错一条,而错的那条不会报错 —— 它会安安静静地多跑 50 步或少跑 50 步。
臂名逐条过 `arm.py env`,解析不了就当场失败,绝不写出一个「看起来对」的 lane 图。
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# AGENT.md §2.2 名册(2026-09-02 定稿 35 条)。250 步的五条是判决基线与两条预注册 c5、
# 以及 N 轴;其余 30 条按名册是 200 步。c4_carrier / c4_rep 是零成本消融(已评满,不重跑)。
STEPS_250 = ["vanilla_corr", "vanilla_te", "c5_union_rkl", "c5_union_fkl", "n2_corr"]
STEPS_200 = [
    "b1_skew_kl_corr", "b2_forward_kl_corr", "c3_intersection_corr",
    "c2_quantile_budget_corr", "c4_pi_tail_budget_corr", "c4_hq", "c4_state",
    "c2_qb_fixed8_corr", "c2_qb_perseq_corr",
    "d2_selectkd_corr", "d3_teachability_corr", "e2_set_coverage_a0_corr",
    "f2_hard_clip_corr", "f3_power_corr", "g6_seqmean_corr", "g1_verified_only_corr",
    "h1_first_segment_corr", "h2_last_segment_corr", "h3_random_segment_corr",
    "h4_random_scatter_corr",
    "h5_gen100_n0", "h7_gen512_n0", "h8_gen2048_n0",
    "h6_gen_sched_n0", "h9_prune_adapt_n0", "h10_task_subset_n0",
    "a1_gkd_mix0.5_n0", "a3_offpolicy_n0", "a4_dagger_anneal_n0", "a5_aggrevate_n0",
]
# 判决基线先跑:载体彩排(Phase R)要 vanilla_corr 通过,别的 lane 才会起。
ROSTER = [(a, 250) for a in STEPS_250] + [(a, 200) for a in STEPS_200]


def check_arms():
    bad = []
    env = dict(os.environ)
    if not env.get("SIMOPD_STORE"):
        # arm.py 会把 arms.yaml 里的 $SIMOPD_STORE 展开、展不开就拒绝。这里只验臂名能不能解析,
        # 不验资产在不在(那是舰队起 lane 前的断言),所以没设时用占位符 —— 并明说。
        env["SIMOPD_STORE"] = "/SIMOPD_STORE-unset"
        print("  (SIMOPD_STORE 未设:用占位符只验臂名;资产路径由舰队起 lane 前断言)", file=sys.stderr)
    for arm, _ in ROSTER:
        r = subprocess.run([sys.executable, os.path.join(HERE, "arm.py"), "env", arm],
                           capture_output=True, text=True, cwd=ROOT, env=env)
        if r.returncode != 0 or not r.stdout.strip():
            bad.append((arm, (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]))
    if bad:
        for arm, msg in bad:
            print(f"  解析不了: {arm}  {msg[0][:90]}", file=sys.stderr)
        sys.exit(f"{len(bad)} 条臂名 arm.py env 解析失败;不写 lane 图")


def slots(per_lane, per_pod):
    lanes_per_pod = per_pod // per_lane
    if lanes_per_pod < 1:
        sys.exit(f"每 pod {per_pod} 卡装不下一条 {per_lane} 卡的 lane")
    out, i = [], 0
    while i < len(ROSTER):
        chunk = ROSTER[i:i + lanes_per_pod]
        specs = []
        for j, (arm, st) in enumerate(chunk):
            gpus = ",".join(str(j * per_lane + k) for k in range(per_lane))
            specs.append(f"{arm}:{gpus}:{st}")
        out.append(" ".join(specs))
        i += lanes_per_pod
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus-per-lane", type=int, default=2)
    ap.add_argument("--gpus-per-pod", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="写 <out>/slot<k>_s<seed>_lanes;不给就只打印")
    ap.add_argument("--print", dest="show", action="store_true")
    ap.add_argument("--skip-check", action="store_true", help="不过 arm.py(离线环境)")
    a = ap.parse_args()

    a.skip_check or check_arms()
    maps = slots(a.gpus_per_lane, a.gpus_per_pod)
    n_lanes = sum(len(m.split()) for m in maps)
    print(f"名册 {len(ROSTER)} 条 -> {len(maps)} 个 slot × 至多 "
          f"{a.gpus_per_pod // a.gpus_per_lane} lane,共 {n_lanes} 条 lane、"
          f"{n_lanes * a.gpus_per_lane} 卡", file=sys.stderr)
    for k, m in enumerate(maps):
        if a.show or not a.out:
            print(f"# slot{k}\n{m}")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            p = os.path.join(a.out, f"slot{k}_s{a.seed}_lanes")
            with open(p, "w") as f:
                f.write(m + "\n")
            print(f"  写了 {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
