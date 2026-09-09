#!/usr/bin/env python3
"""把 2026-09 名册重训跑出来的东西导成表。

只做搬运和汇总,不做判断:每一列都是磁盘上已有的数字,或是由它们按本文件里
写明的公式算出来的。没有阈值、没有"正常/异常"的分类。

    PYTHONPATH= python3 results/20260909/extract.py

读:  $SIMOPD_STORE/ckpt/simopd/<arm>_s0/
       metrics/launch_<ts>.jsonl   每步一行 {"step": int, "data": {指标名: 值}}
       val_gen/<step>.jsonl        每次在环评测的 500 条生成
       global_step_<n>/            存档目录
       traj/                       §2.4 归档层
     configs/retrain_roster.tsv    臂名 / 目标步数 / 优先级 / 备注
     configs/arms.yaml             每条臂的 env 旋钮

写:  results/20260909/*.tsv
"""
import csv
import glob
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.environ.get("SIMOPD_STORE") or os.path.join(os.path.dirname(ROOT), "simopd_data")
CK = os.path.join(STORE, "ckpt", "simopd")
OUT = os.path.dirname(os.path.abspath(__file__))

ACC = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1"
REW = "val-aux/DigitalLearningGmbH/MATH-lighteval/reward/mean@1"

# 汇总表里挑出来的列。挑选标准只是"跨臂都有、且与名册关心的量直接对应",
# 不代表这些比其余 190 多个指标更重要;全部指标见 metrics_full/<arm>.tsv。
SUMMARY_KEYS = [
    ("val_acc", ACC),
    ("val_reward", REW),
    ("train_score_mean", "critic/score/mean"),
    ("resp_len_mean", "response_length/mean"),
    ("resp_len_max", "response_length/max"),
    ("resp_clip_ratio", "response_length/clip_ratio"),
    ("resp_len_mean_non_aborted", "response_length_non_aborted/mean"),
    ("prompt_len_mean", "prompt_length/mean"),
    ("entropy_actor", "actor/entropy"),
    ("entropy_student", "actor/distillation/entropy_student"),
    ("entropy_teacher_topk", "actor/distillation/entropy_teacher_topk"),
    ("entropy_gap_abs", "actor/distillation/entropy_gap_abs"),
    ("ppo_kl", "actor/ppo_kl"),
    ("distill_ppo_kl", "actor/distillation/ppo_kl"),
    ("rollout_kl", "rollout_corr/kl"),
    ("rollout_k3_kl", "rollout_corr/k3_kl"),
    ("eos_dl_at_stop", "actor/distillation/eos_dl_at_stop"),
    ("eos_dl_at_stop_raw", "actor/distillation/eos_dl_at_stop_raw"),
    ("eos_dstop_novel", "actor/distillation/eos_dstop_novel"),
    ("eos_dstop_rep", "actor/distillation/eos_dstop_rep"),
    ("eos_dstop_pos0_100", "actor/distillation/eos_dstop_pos0_100"),
    ("eos_dstop_pos100_500", "actor/distillation/eos_dstop_pos100_500"),
    ("eos_dstop_pos500_2k", "actor/distillation/eos_dstop_pos500_2k"),
    ("eos_dstop_pos2k_up", "actor/distillation/eos_dstop_pos2k_up"),
]


def w(name, header, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t", lineterminator="\n")
        wr.writerow(header)
        wr.writerows(rows)
    print(f"  {name}: {len(rows)} 行")


def roster():
    out = []
    with open(os.path.join(ROOT, "configs", "retrain_roster.tsv")) as fh:
        for r in csv.reader(fh, delimiter="\t"):
            if r and not r[0].startswith("#") and len(r) >= 2:
                out.append((r[0], int(r[1]), r[2] if len(r) > 2 else "", r[3] if len(r) > 3 else ""))
    return out


def launch_series(arm):
    """step -> data dict。同一步被多次写入(续跑重叠)时,后写的文件覆盖先写的。"""
    rows = {}
    files = sorted(glob.glob(f"{CK}/{arm}_s0/metrics/launch_*.jsonl"), key=os.path.getmtime)
    for f in files:
        for line in open(f, errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            s, dat = d.get("step"), d.get("data")
            if isinstance(s, int) and isinstance(dat, dict):
                rows[s] = dat
    return rows, [os.path.basename(f) for f in files]


def ckpt_steps(arm):
    out = []
    for d in glob.glob(f"{CK}/{arm}_s0/global_step_*"):
        t = d.rsplit("_", 1)[1]
        if t.isdigit():
            out.append(int(t))
    return sorted(out)


def main():
    rs = roster()
    arms_yaml = {a["run_id"]: a for a in yaml.safe_load(open(os.path.join(ROOT, "configs", "arms.yaml")))["arms"]}

    def knob_key(a):
        return a if a in arms_yaml else (a[:-3] if a.endswith("_n0") else a[:-5])

    carrier = arms_yaml["vanilla_corr"]["env"]

    def redact(v):
        """env 值里的绝对路径换成变量形式:导出的是实验设定,不是这台机器的布局。"""
        t = str(v)
        for pre, var in ((STORE, "$SIMOPD_STORE"), (ROOT, "$SIMOPD_ROOT")):
            if pre:
                t = t.replace(pre, var)
        return t

    # ---- 1. 臂的设定 -------------------------------------------------------
    rows = []
    for arm, tgt, prio, note in rs:
        d = arms_yaml[knob_key(arm)]
        env = d.get("env", {})
        delta = {k: v for k, v in env.items() if carrier.get(k) != v}
        rows.append([arm, d.get("axis", ""), tgt, prio,
                     ";".join(f"{k}={redact(v)}" for k, v in sorted(delta.items())),
                     " ".join(str(d.get("desc", "")).split()),
                     " ".join(str(note).split())])
    w("arms.tsv", ["arm", "axis", "target_steps", "roster_prio", "env_delta_vs_carrier", "desc", "roster_note"], rows)

    # ---- 2. 进度 -----------------------------------------------------------
    rows = []
    for arm, tgt, _, _ in rs:
        cks = ckpt_steps(arm)
        ser, files = launch_series(arm)
        steps = sorted(ser)
        rows.append([arm, tgt, max(cks) if cks else 0, len(cks),
                     max(steps) if steps else 0, len(steps),
                     int(bool(cks and max(cks) >= tgt)), len(files)])
    w("progress.tsv", ["arm", "target_steps", "ckpt_max_step", "n_checkpoints",
                       "launch_max_step", "n_steps_logged", "reached_target", "n_launch_files"], rows)

    # ---- 3. 存档点清单 ------------------------------------------------------
    rows = [[arm, s] for arm, _, _, _ in rs for s in ckpt_steps(arm)]
    w("checkpoints.tsv", ["arm", "global_step"], rows)

    # ---- 4. 在环评测准确率(宽表) --------------------------------------------
    every = sorted({s for arm, _, _, _ in rs for s in launch_series(arm)[0] if s % 25 == 0})
    rows = []
    for arm, _, _, _ in rs:
        ser, _ = launch_series(arm)
        rows.append([arm] + ["" if not isinstance(ser.get(s, {}).get(ACC), (int, float))
                             else f"{ser[s][ACC]:.4f}" for s in every])
    w("val_accuracy_wide.tsv", ["arm"] + [str(s) for s in every], rows)

    # ---- 5. 关键指标长表 ----------------------------------------------------
    rows = []
    for arm, _, _, _ in rs:
        ser, _ = launch_series(arm)
        for s in sorted(ser):
            d = ser[s]
            rows.append([arm, s] + ["" if not isinstance(d.get(k), (int, float)) else repr(d[k])
                                    for _, k in SUMMARY_KEYS])
    w("metrics_summary.tsv", ["arm", "step"] + [n for n, _ in SUMMARY_KEYS], rows)

    # ---- 6. 每条臂的全量指标 -------------------------------------------------
    n_full = 0
    for arm, _, _, _ in rs:
        ser, _ = launch_series(arm)
        if not ser:
            continue
        keys = sorted({k for d in ser.values() for k in d})
        p = os.path.join(OUT, "metrics_full", f"{arm}.tsv")
        with open(p, "w", newline="") as fh:
            wr = csv.writer(fh, delimiter="\t", lineterminator="\n")
            wr.writerow(["step"] + keys)
            for s in sorted(ser):
                d = ser[s]
                wr.writerow([s] + ["" if d.get(k) is None else repr(d[k]) for k in keys])
        n_full += 1
    print(f"  metrics_full/: {n_full} 个文件")

    # ---- 7. val_gen 逐题汇总 -------------------------------------------------
    rows = []
    for arm, _, _, _ in rs:
        for p in sorted(glob.glob(f"{CK}/{arm}_s0/val_gen/*.jsonl"),
                        key=lambda x: int(os.path.basename(x)[:-6]) if os.path.basename(x)[:-6].isdigit() else -1):
            base = os.path.basename(p)[:-6]
            if not base.isdigit():
                continue
            acc, rew, chars = [], [], []
            for line in open(p, errors="ignore"):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if isinstance(d.get("acc"), (int, float)):
                    acc.append(d["acc"])
                if isinstance(d.get("reward"), (int, float)):
                    rew.append(d["reward"])
                if isinstance(d.get("output"), str):
                    chars.append(len(d["output"]))
            if acc:
                rows.append([arm, int(base), len(acc), f"{sum(acc)/len(acc):.4f}",
                             f"{sum(rew)/len(rew):.4f}" if rew else "",
                             f"{sum(chars)/len(chars):.1f}" if chars else "",
                             max(chars) if chars else ""])
    w("val_gen_summary.tsv", ["arm", "step", "n_items", "acc_mean", "reward_mean",
                              "output_chars_mean", "output_chars_max"], rows)

    # ---- 8. 归档层清点 -------------------------------------------------------
    rows = []
    for arm, _, _, _ in rs:
        d = f"{CK}/{arm}_s0"
        lj = f"{d}/traj/light.jsonl"
        rows.append([arm,
                     sum(1 for _ in open(lj, errors="ignore")) if os.path.exists(lj) else 0,
                     len(glob.glob(f"{d}/traj/ids_*.parquet")),
                     len(glob.glob(f"{d}/traj/summary_*.parquet")),
                     len(glob.glob(f"{d}/traj/step_*.parquet")),
                     len(glob.glob(f"{d}/traj/div/rank*.jsonl")),
                     len(glob.glob(f"{d}/traj/tok_step*_rank*.parquet")),
                     int(os.path.exists(f"{d}/run_manifest.json")),
                     len(glob.glob(f"{d}/val_gen/*.jsonl"))])
    w("archive_inventory.tsv", ["arm", "light_jsonl_lines", "n_ids_parquet", "n_summary_parquet",
                                "n_step_parquet", "n_div_rank_jsonl", "n_tok_parquet",
                                "has_run_manifest", "n_val_gen_files"], rows)

    # ---- 9. 教师列(summary parquet 的数值均值) --------------------------------
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  teacher_columns.tsv: 跳过(无 pyarrow)")
        return
    rows = []
    for arm, _, _, _ in rs:
        for p in sorted(glob.glob(f"{CK}/{arm}_s0/traj/summary_*.parquet"),
                        key=lambda x: int(x.rsplit("_", 1)[1].split(".")[0])):
            s = int(p.rsplit("_", 1)[1].split(".")[0])
            try:
                t = pq.read_table(p).to_pydict()
            except Exception as e:
                rows.append([arm, s, "", "", "", f"read_error:{type(e).__name__}"])
                continue
            n = len(next(iter(t.values()))) if t else 0

            def mean(col):
                v = [x for x in t.get(col, []) if isinstance(x, (int, float))]
                return f"{sum(v)/len(v):.6g}" if v else ""
            rows.append([arm, s, n, mean("tch_lp_nan"), mean("tch_lp_sum"), mean("tch_lp_last")])
    w("teacher_columns.tsv", ["arm", "step", "n_rows", "tch_lp_nan_mean",
                              "tch_lp_sum_mean", "tch_lp_last_mean"], rows)


if __name__ == "__main__":
    sys.exit(main())
