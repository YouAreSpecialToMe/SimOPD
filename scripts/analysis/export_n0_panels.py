"""corr 波的终止面板:每步 stop 事件密度 + 事件级/词级 Delta-ell,回答 N0 的修正机会有多少。"""
import csv, os, sys, re
import wandb
KEYS = {"step":"training/global_step",
        "n_stop":"actor/distillation/eos_n_stop",
        "dl":"actor/distillation/eos_dl_at_stop",
        "dl_raw":"actor/distillation/eos_dl_at_stop_raw",
        "is_stop":"actor/distillation/eos_sampled_is_stop",
        "missing":"actor/distillation/eos_missing",
        "pm1":"actor/distillation/eos_pm_1",
        "len":"response_length/mean", "clip":"response_length/clip_ratio"}
api = wandb.Api(timeout=60); ent = api.default_entity
runs = [r for r in api.runs(f"{ent}/simopd", per_page=800)
        if re.search(r"_corr_s0_16k$", r.name) and str(r.created_at) >= "2026-08-19"]
table = {}
for r in sorted(runs, key=lambda r: str(r.created_at)):
    dst = table.setdefault(r.name.replace("_corr_s0_16k",""), {})
    for row in r.history(samples=4000, pandas=False):
        st = row.get(KEYS["step"])
        if st is None: continue
        out = dst.setdefault(int(st), {})
        for k, wk in KEYS.items():
            v = row.get(wk)
            if v is not None: out[k] = v
p = "/mgfs/shared/Group_GY/changhao/simopd_data/tmp_export/n0_term_panels.csv"
with open(p, "w", newline="") as f:
    w = csv.writer(f); cols=[k for k in KEYS if k!="step"]
    w.writerow(["arm","step"]+cols)
    for arm, steps in sorted(table.items()):
        for st in sorted(steps):
            w.writerow([arm, st] + [steps[st].get(k,"") for k in cols])
print(p)
