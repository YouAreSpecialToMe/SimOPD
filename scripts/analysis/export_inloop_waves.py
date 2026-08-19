#!/usr/bin/env python3
"""导出 v2(stop2 标签)与 legacy 两波的 in-loop 曲线到一张 CSV。
每个 EXPERIMENT_NAME 可能对应多个 wandb run(每次开机一个 id):
按 wave 分组后逐 step 合并,同 step 后创建的 run 胜(续跑接续,复盘 4 的教训)。
"""
import csv, os, sys
import wandb

NAMES = ["a1_gkd_mix0.5_s0_16k", "a3_offpolicy_s0_16k", "a4_dagger_anneal_s0_16k",
         "a5_aggrevate_s0_16k", "a2_coldstart_s0_16k",
         "h6_gen_sched_s0_16k", "h7_gen512_s0_16k", "h8_gen2048_s0_16k",
         "h9_prune_adapt_s0_16k", "h10_task_subset_s0_16k"]
VAL = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1"
KEYS = {"step": "training/global_step", "val_acc": VAL,
        "resp_len": "response_length/mean", "clip_ratio": "response_length/clip_ratio",
        "tps": "timing_s/step", "tokens": "perf/total_num_tokens",
        "h_target": "actor/distillation/h_target", "h9_budget": "actor/distillation/h9_budget"}

api = wandb.Api(timeout=60)
ent = api.default_entity
runs = list(api.runs(f"{ent}/simopd", per_page=500))
print(f"project simopd @ {ent}: {len(runs)} runs", file=sys.stderr)

table = {}   # (wave, arm) -> {step: rowdict}, later-created run wins
for r in runs:
    if r.name not in NAMES:
        continue
    if str(r.created_at) < "2026-08-17":
        continue
    wave = "v2" if "stop2" in (r.tags or []) else "legacy"
    key = (wave, r.name)
    dst = table.setdefault(key, {})
    n = 0
    for row in r.history(samples=4000, pandas=False):
        st = row.get(KEYS["step"])
        if st is None:
            continue
        st = int(st)
        out = dst.setdefault(st, {})
        for k, wk in KEYS.items():
            v = row.get(wk)
            if v is not None:
                out[k] = v
        n += 1
    print(f"  {wave:<7} {r.name:<28} run={r.id} rows={n} created={str(r.created_at)[:16]}", file=sys.stderr)

os.makedirs("/mgfs/shared/Group_GY/changhao/simopd_data/tmp_export", exist_ok=True)
p = "/mgfs/shared/Group_GY/changhao/simopd_data/tmp_export/inloop_v2_vs_legacy.csv"
with open(p, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["wave", "arm", "step", "val_acc", "resp_len", "clip_ratio", "tps", "tokens", "h_target", "h9_budget"])
    for (wave, name), steps in sorted(table.items()):
        arm = name.replace("_s0_16k", "")
        for st in sorted(steps):
            row = steps[st]
            w.writerow([wave, arm, st] + [row.get(k, "") for k in
                       ["val_acc", "resp_len", "clip_ratio", "tps", "tokens", "h_target", "h9_budget"]])
print(p)
