#!/usr/bin/env python3
"""导出 corr/n0 波各臂的 in-loop 训练动态(wandb -> 一张 CSV)。

    python scripts/analysis/export_wave_dynamics.py --out docs/data/inloop_wave_dynamics.csv

与 export_corr.py 的区别:那个只取 4 个键、且写死了"corr vs legacy"两波对照;这里是
【结果归档】用途 —— 一次把这一轮所有臂的曲线连同终止塌缩那套仪表(eos_p/q_at_stop、
entropy_student、delta_ell 分位、overlap_*)一起落盘,让分析和作图不必再连 wandb。

同名 EXPERIMENT_NAME 可能对应多个 wandb run(每次开机一个 id,断线续跑就多一个)。
按 step 合并、后创建的 run 胜 —— 与 export_inloop_waves.py 同口径(复盘 4 的教训)。
"""
import argparse
import csv
import re
import sys

import wandb

# 逐 step 落盘的列。左边是 CSV 列名,右边是 wandb 指标名;缺的列留空,不报错 ——
# 不同臂开的仪表不一样(N0 族才有 delta_ell,c5 才有 union 的 overlap 口径)。
KEYS = {
    "val_acc":            "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1",
    "resp_len":           "response_length/mean",
    "resp_len_clip_frac": "response_length/clip_ratio",
    "entropy":            "actor/entropy",
    "grad_norm":          "actor/grad_norm",
    "lr":                 "actor/lr",
    "pg_loss":            "actor/pg_loss",
    "distill_loss":       "actor/distillation/loss",
    # 终止塌缩仪表(docs/MECHANISMS.md §M-I):学生/教师在停止位上的质量
    "eos_p_at_stop":      "actor/distillation/eos_p_at_stop",
    "eos_q_at_stop":      "actor/distillation/eos_q_at_stop",
    "eos_p_mean":         "actor/distillation/eos_p_mean",
    "eos_q_mean":         "actor/distillation/eos_q_mean",
    "eos_sampled_is_stop": "actor/distillation/eos_sampled_is_stop",
    "eos_missing":        "actor/distillation/eos_missing",
    "entropy_student":    "actor/distillation/entropy_student",
    "entropy_teacher_topk": "actor/distillation/entropy_teacher_topk",
    "entropy_gap_abs":    "actor/distillation/entropy_gap_abs",
    "delta_ell_p50":      "actor/distillation/delta_ell_p50",
    "delta_ell_p95":      "actor/distillation/delta_ell_p95",
    "overlap_ratio":      "actor/distillation/overlap_ratio",
    "overlap_student_mass": "actor/distillation/overlap_student_mass",
    "overlap_teacher_mass": "actor/distillation/overlap_teacher_mass",
    # c5 并集格的预注册读数(2026-08-24 补进归档列):学生/教师在两个终止符上的质量。
    # union_fkl 的判据就是「un_p_imend 上升而 un_p_eot 下降」= 学生把教师的终止符
    # 当成 token 学会了。第一版没收这三列,导致集群断线后本地读不出这条判据。
    "un_p_eot":           "actor/distillation/un_p_eot",
    "un_p_imend":         "actor/distillation/un_p_imend",
    "un_q_imend":         "actor/distillation/un_q_imend",
    "reward_mean":        "critic/rewards/mean",
    "time_per_step_s":    "perf/time_per_step",
    "tokens":             "perf/total_num_tokens",
}
STEP = "training/global_step"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="输出 CSV")
    # 别用臂名正则圈这一波 —— 命名习惯不统一,圈不全。第一版写的是 _corr_s0_16k$ 之类,
    # 结果 c2_quantile_budget_corr_s0(没有 _16k 后缀)连同 9 条同窗口的 A/H 轴臂一起漏掉,
    # 判读台的状态表因此少了 10 条。--since 才是这一波的定义,与 export_wave_metrics.py 同规则。
    ap.add_argument("--pattern", default=r".", help="run 名正则(默认全收,靠 --since 圈波次)")
    ap.add_argument("--since", default="2026-08-19", help="只要这天之后创建的 run")
    ap.add_argument("--samples", type=int, default=4000, help="每个 run 取多少行 history")
    a = ap.parse_args()

    api = wandb.Api(timeout=60)
    ent = api.default_entity
    runs = list(api.runs(f"{ent}/simopd", per_page=800))
    pat = re.compile(a.pattern)
    sel = [r for r in runs if pat.search(r.name) and str(r.created_at) >= a.since]
    print(f"{ent}/simopd: {len(runs)} runs -> 命中 {len(sel)}", file=sys.stderr)

    table = {}          # (run, step) -> {列: 值}
    for r in sorted(sel, key=lambda r: str(r.created_at)):     # 后创建的覆盖先创建的
        n = 0
        for row in r.history(samples=a.samples, pandas=False):
            st = row.get(STEP)
            if st is None:
                continue
            dst = table.setdefault((r.name, int(st)), {})
            for col, wk in KEYS.items():
                v = row.get(wk)
                if v is not None:
                    dst[col] = v
            n += 1
        print(f"  {r.name:<34} rows={n:<5} created={str(r.created_at)[:16]} state={r.state}",
              file=sys.stderr)

    arm_of = lambda run: re.sub(r"_s\d+_16k$", "", run)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "run", "step"] + list(KEYS))
        for (run, step) in sorted(table, key=lambda k: (k[0], k[1])):
            d = table[(run, step)]
            w.writerow([arm_of(run), run, step] + [d.get(c, "") for c in KEYS])
    print(f"wrote {a.out}  rows={len(table)}  runs={len({r for r, _ in table})}", file=sys.stderr)


if __name__ == "__main__":
    main()
