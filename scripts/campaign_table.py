"""Whole-campaign summary: every arm, both cells, at matched steps.

Cells are kept in separate tables because they are not comparable: different student,
different teacher, different starting accuracy (1.7B-Base 0.468, 8B-Base 0.664). A
number from one says nothing about the other except through its own vanilla.
"""
import re, glob, os, sys
from collections import defaultdict
from statistics import mean

RUN = re.compile(r"^#+ RUN: ([A-Za-z0-9_.]+)")
VAL = re.compile(r"step:(\d+) .*?val-core/[^:]*acc/mean@1:np\.float64\(([0-9.]+)\)")
PRE = re.compile(r"^\([^)]*\)\s*")
BAR = re.compile(r"\| *(\d+)/250 \[")

curves, last_step = defaultdict(dict), {}

def scan(paths):
    for f in paths:
        cur = None
        with open(f, errors="replace") as fh:
            for line in fh:
                b = PRE.sub("", line.lstrip())
                m = RUN.match(b)
                if m:
                    cur = m.group(1); continue
                if not cur:
                    continue
                m = VAL.search(b)
                if m:
                    curves[cur][int(m.group(1))] = float(m.group(2))
                m = BAR.search(b)
                if m:
                    last_step[cur] = max(last_step.get(cur, 0), int(m.group(1)))

scan(sorted(glob.glob("logs/*/lane*.log")) + sorted(glob.glob("logs/lane*.log")))
scan(sorted(glob.glob("/mgfs/shared/Group_GY/changhao/simopd_data/wpair_logs/lane*.log")))

def split(n):
    w = n.endswith("_w")
    core = n[:-2] if w else n
    m = re.match(r"^(.*)_s(\d+)$", core)
    return (m.group(1), int(m.group(2)), w) if m else (core, -1, w)

cells = {False: defaultdict(dict), True: defaultdict(dict)}   # is_w -> arm -> seed -> curve
prog  = {False: defaultdict(dict), True: defaultdict(dict)}
for name, c in curves.items():
    a, s, w = split(name)
    if c:
        cells[w][a][s] = c
        prog[w][a][s] = last_step.get(name, max(c))

STEPS = [25, 50, 100, 150, 200, 250]


def support_family():
    """arm -> which tokens the loss is computed over.

    Resolved from the loss registry (`use_topk`) and the arm's own env, never from
    the arm's name: c1_lsm_topk32_renorm says topk in its name, f2_hard_clip does not
    say sampled in its, and neither is where the truth lives. SIMOPD_KEEP_SAMPLED=1
    makes the teacher tensors carry one extra column holding the sampled token on top
    of the top-k (src/simopd/teacher_patch.py, topk_losses.py:200), which is a third
    family and not a variant of either.
    """
    import subprocess
    try:
        import simopd  # noqa: F401  -- registers our loss modes
        from verl.trainer.distillation.losses import get_distillation_loss_settings
    except Exception:
        return {}
    env = {**os.environ, "PYTHONPATH": "src"}
    out = {}
    names = subprocess.run([sys.executable, "scripts/arm.py", "list"],
                           capture_output=True, text=True, env=env).stdout.split()
    for a in names:
        blk = subprocess.run([sys.executable, "scripts/arm.py", "env", a],
                             capture_output=True, text=True, env=env).stdout
        e = dict(re.findall(r"^export (\w+)=(.*)$", blk, re.M))
        try:
            topk = get_distillation_loss_settings(e.get("DISTILLATION_LOSS_MODE", "k1_rec")).use_topk
        except Exception:
            continue
        if not topk:
            out[a] = "sampled"
        elif e.get("SIMOPD_KEEP_SAMPLED") == "1":
            out[a] = "topk+samp"
        else:
            out[a] = "topk"
    return out


FAM = support_family()

def cell(arm_seeds, step):
    vals = [c[step] for _, c in sorted(arm_seeds.items()) if step in c]
    if not vals:
        return "  –  ", None
    m = mean(vals)
    return (f"{m:.3f}" + (f"·{len(vals)}" if len(vals) > 1 else "    "), m)

for is_w, title, note in [
    (False, "1.7B-Base ← 4B-Instruct-2507", "student starts at 0.468 · `x·N` = mean of N seeds"),
    (True,  "8B-Base ← 32B",                "student starts at 0.664 · one seed each"),
]:
    arms = cells[is_w]
    if not arms:
        continue
    van = arms.get("vanilla", {})
    ranked = sorted(arms, key=lambda a: -(cell(arms[a], 200)[1] or cell(arms[a], 150)[1] or -1))
    print(f"\n### {title}\n\n_{note}_\n")
    print("| arm | support | " + " | ".join(f"s{s}" for s in STEPS) + " | now | seeds |")
    print("|---|---|" + "---|" * (len(STEPS) + 2))
    for a in ranked:
        row = []
        for s in STEPS:
            txt, m = cell(arms[a], s)
            vm = cell(van, s)[1] if van else None
            if m is not None:
                # n ALWAYS shown. It used to be added only on the compare path, so the
                # s250 column -- where vanilla has no point yet, so no compare happens
                # -- printed single-seed finals in the same style as three-seed means.
                n = len([1 for _, c in arms[a].items() if s in c])
                if vm is not None and a != "vanilla":
                    d = m - vm
                    txt = f"**{m:.3f}**" if d > 0.02 else (f"{m:.3f}" if d > -0.02 else f"_{m:.3f}_")
                else:
                    txt = f"{m:.3f}"
                txt += f"·{n}"
            row.append(txt.strip() or "–")
        steps_now = prog[is_w][a]
        now = "/".join(str(steps_now[k]) for k in sorted(steps_now))
        print(f"| {'**' + a + '**' if a == 'vanilla' else a} | {FAM.get(a, '?')} | " +
              " | ".join(row) + f" | {now} | {len(arms[a])} |")
