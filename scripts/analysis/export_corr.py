#!/usr/bin/env python3
"""导出 corr 波(*_corr_s0_16k, created>=08-19)+ 其 legacy 对应臂的 in-loop 曲线。"""
import csv, os, sys, re
import wandb
KEYS = {"step": "training/global_step",
        "val_acc": "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1",
        "resp_len": "response_length/mean", "clip_ratio": "response_length/clip_ratio"}
api = wandb.Api(timeout=60); ent = api.default_entity
runs = list(api.runs(f"{ent}/simopd", per_page=800))
print(f"{len(runs)} runs", file=sys.stderr)
corr = [r for r in runs if re.search(r"_corr_s0_16k$", r.name) and str(r.created_at) >= "2026-08-19"]
legacy_names = sorted({r.name.replace("_corr_s0_16k", "_s0_16k") for r in corr})
legacy = [r for r in runs if r.name in legacy_names]
table = {}
def merge(rlist, wave):
    for r in sorted(rlist, key=lambda r: str(r.created_at)):
        dst = table.setdefault((wave, r.name.replace("_corr_s0_16k","").replace("_s0_16k","")), {})
        n = 0
        for row in r.history(samples=4000, pandas=False):
            st = row.get(KEYS["step"])
            if st is None: continue
            out = dst.setdefault(int(st), {})
            for k, wk in KEYS.items():
                v = row.get(wk)
                if v is not None: out[k] = v
            n += 1
        print(f"  {wave:<7} {r.name:<32} rows={n} created={str(r.created_at)[:16]}", file=sys.stderr)
merge(legacy, "mfleet"); merge(corr, "corr")
p = "/mgfs/shared/Group_GY/changhao/simopd_data/tmp_export/inloop_corr_vs_mfleet.csv"
with open(p, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["wave","arm","step","val_acc","resp_len","clip_ratio"])
    for (wave, arm), steps in sorted(table.items()):
        for st in sorted(steps):
            row = steps[st]
            w.writerow([wave, arm, st] + [row.get(k,"") for k in ["val_acc","resp_len","clip_ratio"]])
print(p)
