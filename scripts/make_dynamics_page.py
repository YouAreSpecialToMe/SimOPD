"""Build a standalone training-dynamics comparison browser (one local HTML file).

    python scripts/make_dynamics_page.py            # -> docs/arm-dynamics.html
    python scripts/make_dynamics_page.py --every 2  # smaller file

All 29 arms are drawn together in every panel; hovering or clicking one arm
spotlights it in ALL panels at once, so an arm's val / length / entropy /
truncation are read against the whole field simultaneously. Up to three arms
pin in distinct colours for direct comparison, and when exactly one is pinned
its own instruments (qb_budget, clip_hit_rate, gate_keep_frac, …) appear below.

Every number is embedded, so the file opens by double-click — no server, no
network (file:// cannot fetch the CSV anyway). Regenerate after each metrics
refresh; there is no other dependency.
"""

import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = "actor/distillation/"
VAL = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1"
STUDENT_ANCHOR, CAP = 0.4580, 16384

# Regimes follow the report's own section-1 reading.
REGIME = {
    "c2_quantile_budget": "climb", "c4_pi_tail_budget": "climb", "e1_pl_rank": "climb",
    "c1_lsm_topk32_renorm": "climb", "c3_intersection": "climb", "a2_coldstart": "climb",
    "j1_kdrl": "stable", "h1_first_segment": "stable",
    "g1_verified_only": "deepU", "h2_last_segment": "deepU", "e3_zvalue": "deepU",
    "f3_power": "drift", "b4_jsd": "drift", "h3_random_segment": "drift",
    "b2_forward_kl": "drift", "b3_eopd_gate": "dead",
}
AXIS = {"a2_coldstart": "A", "b1_skew_kl": "B", "b2_forward_kl": "B", "b3_eopd_gate": "B",
        "b4_jsd": "B", "b5_k2": "B", "c1_lsm_topk32_renorm": "C", "c2_quantile_budget": "C",
        "c3_intersection": "C", "c4_pi_tail_budget": "C", "d1_tip": "D", "d2_selectkd": "D",
        "d3_teachability": "D", "e1_pl_rank": "E", "e2_set_coverage": "E", "e3_zvalue": "E",
        "f1_soft_log": "F", "f2_hard_clip": "F", "f3_power": "F", "g1_verified_only": "G",
        "g2_fire_likelihood": "G", "g4_failure_only": "G", "g5_rgopd_gate": "G",
        "h1_first_segment": "H", "h2_last_segment": "H", "h3_random_segment": "H", "j1_kdrl": "J"}
GPUH = {"a2_coldstart": 472, "b1_skew_kl": 208, "b2_forward_kl": 205, "b3_eopd_gate": 1495,
        "b4_jsd": 179, "b5_k2": 352, "c1_lsm_topk32_renorm": 45, "c2_quantile_budget": 303,
        "c3_intersection": 150, "c4_pi_tail_budget": 198, "d1_tip": 521, "d2_selectkd": 389,
        "d3_teachability": 461, "e1_pl_rank": 139, "e2_set_coverage": 410, "e3_zvalue": 259,
        "f1_soft_log": 241, "f2_hard_clip": 255, "f3_power": 172, "g1_verified_only": 240,
        "g2_fire_likelihood": 573, "g4_failure_only": 357, "g5_rgopd_gate": 356,
        "h1_first_segment": 82, "h2_last_segment": 435, "h3_random_segment": 210,
        "j1_kdrl": 101, "vanilla": 356, "vanilla_n8": 694}

# The comparison grid. `frac` keeps a 0..1 quantity off the auto-log path; the two
# signal panels are deliberately SEPARATE — only the k1 family may be published as
# Delta-ell, the top-k arms optimise a divergence, and one cross-arm panel holding
# both would compare different quantities (losses.py's own rule).
METRICS = [
    dict(k=VAL, t="in-loop MATH500", h="greedy mean@1 · 每 25 步", ref=STUDENT_ANCHOR, frac=1, sparse=1),
    dict(k="response_length/mean", t="响应长度", h="训练 rollout 均长 · tokens", ref=CAP),
    dict(k="response_length/clip_ratio", t="截断率", h="撞 16k 帽的比例", ref=1.0, frac=1),
    dict(k="actor/entropy", t="策略熵", h="nats", ref=None),
    dict(k=P + "delta_ell_absmean", t="Δℓ 幅度", h="仅 k1 族(采样点信号)", ref=None),
    dict(k=P + "loss_absmean", t="loss 幅度", h="仅 top-k 族(散度值,与 Δℓ 不同量纲)", ref=None),
    dict(k="actor/grad_norm", t="梯度范数", h="", ref=None),
    dict(k="critic/score/mean", t="verifier 分数", h="训练 rollout · τ=1 · 只记不用", ref=None, frac=1),
    dict(k="timing_s/step", t="步时", h="秒 / 步", ref=None),
]


SUITE_ANCHOR = 0.145   # untrained student's composite (report §3.1 "Δ vs base")


def _num(cell):
    """'0.335±0.003·2' / '**0.354±0.002**' / '0.338·1' / '–' -> float or None."""
    c = cell.strip().strip("*").split("±")[0].split("·")[0].strip()
    try:
        return float(c)
    except ValueError:
        return None


def parse_suite(md_path):
    """Offline 5-bench suite numbers, read from the campaign report's own tables.

    The suite lives in eval parquets on the shared filesystem, not in the
    training CSV -- but the report is regenerated from those parquets and IS in
    the repo, so the page can carry both scoreboards without the fleet. Returns
    (step curve {arm: [[step, composite], ...]}, finals {arm: {...}}).
    """
    if not os.path.exists(md_path):
        return {}, {}
    steps = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    curve, finals, mode = {}, {}, None
    for ln in open(md_path):
        if ln.startswith("| method | s25 |"):
            mode = "curve"; continue
        if ln.startswith("| method | seeds | composite |"):
            mode = "finals"; continue
        if mode and not ln.startswith("|"):
            mode = None; continue
        if not mode or ln.startswith("|---") or ln.startswith("| ↳"):
            continue
        f = [c.strip() for c in ln.strip().strip("|").split("|")]
        arm = f[0].strip().strip("*")
        if not arm or " " in arm:
            continue
        if mode == "curve":
            pts = [[s, v] for s, v in zip(steps, (_num(c) for c in f[1:11])) if v is not None]
            if pts:
                curve[arm] = pts
        else:                                   # | arm | seeds | composite | Δ | AIME | AMC | Minerva | MATH500 |
            vals = [_num(c) for c in f[2:8]]
            if vals[0] is not None:
                finals[arm] = dict(suite=vals[0], aime=vals[2], amc=vals[3],
                                   minerva=vals[4], s500=vals[5])
    return curve, finals


def r4(v):
    if v is None or pd.isna(v):
        return None
    v = float(v)
    if v == 0:
        return 0
    from math import floor, log10
    d = max(0, min(6, 3 - int(floor(log10(abs(v))))))
    out = round(v, d)
    return int(out) if d == 0 else out


def main():
    ap = argparse.ArgumentParser()
    # BOTH metric dumps by default: the 16k batch (29 arms) and the expansion
    # batch (13 more -- c1_direct, qb_fixed8, f5_tanh, ...). Passing one alone
    # silently drops the other half of the campaign from the page, which is how
    # a regeneration can look successful and lose 13 arms.
    ap.add_argument("--csv", nargs="+", default=[
        os.path.join(ROOT, "docs/data/training_metrics_16k_allkeys.csv.gz"),
        os.path.join(ROOT, "docs/data/training_metrics_exp_allkeys.csv.gz")])
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/arm-dynamics.html"))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--suite-md", default=os.path.join(ROOT, "docs/campaign_16k_report.md"),
                    help="offline suite tables are read from the report (regenerated from "
                         "the eval parquets); pass /dev/null to build the training-only page")
    a = ap.parse_args()
    suite_curve, suite_fin = parse_suite(a.suite_md)

    df = pd.concat([pd.read_csv(p) for p in a.csv], ignore_index=True)
    seeds = sorted(int(s) for s in df.seed.unique())
    NMAX = int(df.step.max())
    grid = [s for s in range(1, NMAX + 1) if s % a.every == 0]

    dcols = [c for c in df.columns if c.startswith(P)]
    owners = {c: {arm for arm, g in df.groupby("arm") if g[c].notna().any()} for c in dcols}
    POSSUF = ("pos0_100", "pos100_500", "pos500_2k", "pos2k_up")
    specific_cols = sorted(c for c, arms in owners.items()
                           if 0 < len(arms) <= 6 and not c.endswith(POSSUF))

    arms_meta, metrics, specific = {}, [], {}
    byarm = {arm: g for arm, g in df.groupby("arm")}

    def series_for(g, col, sparse=False):
        out = []
        for s in seeds:
            sg = g[g.seed == s]
            if col not in sg or sg[col].notna().sum() == 0:
                out.append(None)
                continue
            ser = sg.set_index("step")[col]
            if sparse:
                v = ser.dropna()
                out.append([[int(i), r4(x)] for i, x in v.items()])
            else:
                out.append([r4(ser.get(st)) if st in ser.index else None for st in grid])
        return out if any(o for o in out) else None

    for m in METRICS:
        data = {}
        for arm, g in byarm.items():
            if m["k"] not in g or not g[m["k"]].notna().any():
                continue
            s = series_for(g, m["k"], sparse=bool(m.get("sparse")))
            if s:
                data[arm] = s
        metrics.append({**{kk: vv for kk, vv in m.items() if kk != "k"}, "id": m["k"], "d": data})
        # The offline suite goes DIRECTLY under the in-loop panel: the two
        # scoreboards use different decoding (greedy/16k vs tau=0.7/top-p0.95/
        # 32k/avg@3) and disagree by up to +0.21 on collapsed arms, which is a
        # finding, not noise -- side by side is the only honest layout.
        if m["k"] == VAL and suite_curve:
            metrics.append(dict(
                t="离线套件 composite", h="τ=0.7·top-p0.95·avg@3 · 五 bench 复合 · 臂均值(报告表)",
                ref=SUITE_ANCHOR, frac=1, sparse=1, id="suite/composite",
                d={arm: [pts, None, None] for arm, pts in suite_curve.items()}))

    for arm, g in byarm.items():
        clip = g.groupby("step")["response_length/clip_ratio"].mean()
        last_bad = None
        for st, c in clip.items():
            if pd.notna(c) and c < 0.9:
                last_bad = int(st)
        # "first step after which >=90% of rollouts hit the cap for the rest of the run";
        # needs at least two steps of evidence, which is what makes b1's 247 legal.
        lock = None if (last_bad is None or last_bad >= int(g.step.max()) - 1) else last_bad + 1
        v = g.groupby("step")[VAL].mean().dropna()
        last = g.groupby("step").mean(numeric_only=True).iloc[-1]
        arms_meta[arm] = dict(
            axis=AXIS.get(arm, "–"), reg=REGIME.get(arm, "lock"), lock=lock, gpuh=GPUH.get(arm),
            fin=r4(v.iloc[-1]) if len(v) else None, peak=r4(v.max()) if len(v) else None,
            fall=r4(v.max() - v.iloc[-1]) if len(v) else None,
            trunc=r4(last.get("response_length/clip_ratio")), len=r4(last.get("response_length/mean")),
            ent=r4(last.get("actor/entropy")), nsteps=int(g.step.max()),
            **suite_fin.get(arm, {}))
        own = []
        for c in specific_cols:
            if arm in owners[c]:
                s = series_for(g, c)
                if s:
                    own.append({"t": c.split("/")[-1], "s": s})
        # positional signal profile, seed-mean, ordered bins (only where logged)
        for pre in (P + "delta_ell_absmean_", P + "loss_absmean_"):
            if pre + POSSUF[0] in g and g[pre + POSSUF[0]].notna().any():
                mm = g.groupby("step").mean(numeric_only=True)
                bins = [[r4(mm[pre + s].get(st)) if pre + s in mm else None for st in grid]
                        for s in POSSUF if pre + s in mm]
                if bins:
                    own.insert(0, {"t": "信号的位置分布", "s": bins,
                                   "seq": ["0–100", "100–500", "500–2k", "2k+"][:len(bins)]})
                break
        if own:
            specific[arm] = own

    payload = dict(every=a.every, seeds=seeds, nmax=NMAX, anchor=STUDENT_ANCHOR, cap=CAP,
                   arms=arms_meta, metrics=metrics, specific=specific)
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/2**20:.1f} MiB, {len(arms_meta)} arms, "
          f"{len(metrics)} shared panels, every {a.every} step(s))")


TEMPLATE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SimOPD · 训练动力学总览</title>
<style>
:root{color-scheme:light;--bg:#fbfcfd;--panel:#f2f5f8;--raise:#fff;
 --ink:#12161d;--ink2:#4d5663;--ink3:#7f8a99;--faint:#9fadbd;--rule:#dbe2ea;--rule2:#e9eef4;
 --accent:#1f5fa8;--on-accent:#fff;--ok:#1baf7a;--bad:#eb6834;--rec:#4a3aa7;
 --p1:#2a78d6;--p2:#eb6834;--p3:#1baf7a;
 --q1:#c3d7ee;--q2:#84acd9;--q3:#4079b8;--q4:#12457c}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --bg:#15171a;--panel:#1e2126;--raise:#22262c;
 --ink:#e9ecf0;--ink2:#b2bac5;--ink3:#7c8794;--faint:#4d565f;--rule:#2e343c;--rule2:#272c33;
 --accent:#79a9e0;--on-accent:#10141a;--ok:#199e70;--bad:#d95926;--rec:#9085e9;
 --p1:#3987e5;--p2:#d95926;--p3:#199e70;
 --q1:#22384e;--q2:#33608f;--q3:#5b8fc9;--q4:#a3c8ec}}
:root[data-theme=dark]{color-scheme:dark;--bg:#15171a;--panel:#1e2126;--raise:#22262c;
 --ink:#e9ecf0;--ink2:#b2bac5;--ink3:#7c8794;--faint:#4d565f;--rule:#2e343c;--rule2:#272c33;
 --accent:#79a9e0;--on-accent:#10141a;--ok:#199e70;--bad:#d95926;--rec:#9085e9;
 --p1:#3987e5;--p2:#d95926;--p3:#199e70;
 --q1:#22384e;--q2:#33608f;--q3:#5b8fc9;--q4:#a3c8ec}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.5;
 font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Noto Sans CJK SC","PingFang SC",sans-serif}
.mono,.n{font-family:ui-monospace,"SF Mono",Consolas,"Noto Sans Mono CJK SC",monospace;font-variant-numeric:tabular-nums}
.app{display:grid;grid-template-columns:250px 1fr;min-height:100vh}
aside{border-right:1px solid var(--rule);background:var(--panel);padding:13px 11px;position:sticky;top:0;
 height:100vh;overflow-y:auto}
aside h1{font-size:14.5px;margin:0 0 1px;font-weight:650}
aside .sub{font-size:11px;color:var(--ink3);margin-bottom:11px;font-family:ui-monospace,Consolas,monospace}
#q{width:100%;font:inherit;font-size:12.5px;padding:5px 8px;border:1px solid var(--rule);border-radius:5px;
 background:var(--raise);color:var(--ink);margin-bottom:8px}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:9px}
.chip{font-family:ui-monospace,Consolas,monospace;font-size:10.5px;padding:3px 7px;border-radius:999px;
 border:1px solid var(--rule);background:var(--raise);color:var(--ink2);cursor:pointer}
.chip[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.grp{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);margin:11px 0 3px;
 font-family:ui-monospace,Consolas,monospace}
.armbtn{display:flex;align-items:center;gap:6px;width:100%;text-align:left;border:1px solid transparent;
 background:none;color:var(--ink2);font:inherit;font-size:12px;padding:4px 6px;border-radius:5px;cursor:pointer}
.armbtn:hover{background:var(--raise);color:var(--ink)}
.armbtn.hot{background:var(--raise);color:var(--ink);border-color:var(--rule)}
.armbtn.pin{border-color:currentColor;font-weight:650}
.armbtn .nm{font-family:ui-monospace,Consolas,monospace;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.armbtn .v{font-family:ui-monospace,Consolas,monospace;font-size:11px;opacity:.8}
.dot{width:7px;height:7px;border-radius:50%;flex:none}
.dot.climb{background:var(--ok)}.dot.stable{background:transparent;box-shadow:inset 0 0 0 2px var(--ok)}
.dot.deepU{background:var(--rec)}.dot.lock{background:var(--bad)}
.dot.drift{background:transparent;box-shadow:inset 0 0 0 2px var(--bad)}
.dot.dead{background:var(--bad);box-shadow:inset 0 0 0 1.5px var(--panel)}
main{padding:16px 18px 50px;min-width:0}
.top{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:4px}
.top h2{margin:0;font-size:17px;font-weight:650}
.top .hint{color:var(--ink3);font-size:12px;margin-left:auto}
.pins{display:flex;gap:6px;flex-wrap:wrap;min-height:26px;align-items:center;margin:8px 0 2px}
.pinchip{display:inline-flex;align-items:center;gap:6px;font-family:ui-monospace,Consolas,monospace;font-size:11.5px;
 padding:3px 9px;border-radius:999px;border:1.5px solid currentColor;background:var(--raise);cursor:pointer}
.pinchip .x{opacity:.6}
.pinchip .st{color:var(--ink2);font-size:10.5px}
.empty{color:var(--ink3);font-size:12px;font-family:ui-monospace,Consolas,monospace}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:11px;margin-top:8px}
.pan{background:var(--panel);border:1px solid var(--rule2);border-radius:7px;padding:9px 11px 4px}
.pan .ph{display:flex;align-items:baseline;gap:7px;flex-wrap:nowrap;overflow:hidden}
.pan .pt{font-size:12.5px;font-weight:620;white-space:nowrap;flex:none}
/* the hint yields first: title and the value readout must never be squeezed */
.pan .phint{font-size:10px;color:var(--ink3);font-family:ui-monospace,Consolas,monospace;
 flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pan .tagl{font-size:9.5px;color:var(--ink3);border:1px solid var(--rule);border-radius:3px;padding:0 4px;
 font-family:ui-monospace,Consolas,monospace;flex:none}
.pan .pv{margin-left:auto;font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:var(--ink2);
 font-variant-numeric:tabular-nums;white-space:nowrap;flex:none}
.pan svg{display:block;width:100%;height:170px;touch-action:none;cursor:crosshair}
.seqleg{display:flex;gap:8px;flex-wrap:wrap;font-size:10px;color:var(--ink3);font-family:ui-monospace,Consolas,monospace}
.seqleg i{width:11px;height:3px;border-radius:2px;display:inline-block;margin-right:3px;vertical-align:middle}
h3.sec{font-size:13.5px;margin:22px 0 0;font-weight:650}
h3.sec span{font-weight:400;color:var(--ink3);font-size:11.5px;font-family:ui-monospace,Consolas,monospace;margin-left:8px}
footer{padding:14px 18px 0;color:var(--ink3);font-size:11.5px;border-top:1px solid var(--rule);margin-top:26px}
@media(max-width:820px){.app{grid-template-columns:1fr}aside{position:static;height:auto}}
</style></head><body>
<div class="app">
<aside>
  <h1>训练动力学总览</h1>
  <div class="sub" id="meta"></div>
  <input id="q" type="search" placeholder="搜索臂 / 轴…" aria-label="搜索">
  <div class="chips" id="chips"></div>
  <div id="list"></div>
</aside>
<main>
  <div class="top">
    <h2>全部臂同图对比</h2>
    <span class="hint">悬停曲线或左侧臂名 → 全部面板同步高亮 · 点击固定(最多 3)· ← → 移动高亮</span>
  </div>
  <div class="pins" id="pins"></div>
  <div class="grid" id="grid"></div>
  <div id="own"></div>
</main>
<footer id="foot"></footer>
</div>
<script>
const D=/*__DATA__*/;
const NAMES=Object.keys(D.arms).sort();
const ORDER=["climb","stable","deepU","drift","lock","dead"];
const LAB={climb:"上升",stable:"平稳",deepU:"深 U 回升",lock:"锁死塌陷",drift:"不锁而降",dead:"塌陷不返"};
const PINCOL=["--p1","--p2","--p3"];
const $=s=>document.querySelector(s), NS="http://www.w3.org/2000/svg";
const cv=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt=v=>v==null?"–":Math.abs(v)>=1000?Math.round(v).toLocaleString():Math.abs(v)>=10?v.toFixed(1):
  Math.abs(v)>=1?v.toFixed(2):v.toFixed(3);
let hot=null, pins=[], filt="all", qs="", cross=null;

$("#meta").textContent=`${NAMES.length} 臂 · ${D.nmax} 步 · 种子 ${D.seeds.join("/")} · 每 ${D.every} 步`;
$("#foot").innerHTML=`细线 = 该臂三种子均值;固定臂另画三条种子细线。竖虚线 = 该臂 lock 步(此后 ≥90% rollout 撞帽)。`
 +` 两个信号面板<b>刻意分开</b>:k1 族报 Δℓ,top-k 族报散度 loss,量纲不同不得同图。`
 +` <b>两块记分牌不可混读</b>:in-loop 是 greedy·16k 帽的健康遥测,离线套件是 τ=0.7·top-p0.95·32k·avg@3 的正式测量;`
 +` 终止受损的臂在 greedy 下会一路复读到帽而取不出答案,套件分因此可高出 0.2 以上(f3:0.473 → 0.684)。`
 +` 数据 <span class="mono">docs/data/training_metrics_16k_allkeys.csv.gz</span> + 套件表取自 <span class="mono">docs/campaign_16k_report.md</span>(489/861 格已完成),`
 +` 由 <span class="mono">scripts/make_dynamics_page.py</span> 生成。`;

/* ---------- sidebar ---------- */
const chips=[["all","全部"]].concat(ORDER.map(r=>[r,LAB[r]]));
$("#chips").innerHTML=chips.map(([k,l])=>`<button class="chip" data-f="${k}" aria-pressed="${k==="all"}">${l}</button>`).join("");
$("#chips").onclick=e=>{const b=e.target.closest(".chip");if(!b)return;
  filt=b.dataset.f;[...$("#chips").children].forEach(c=>c.setAttribute("aria-pressed",String(c===b)));
  buildList();drawAll();};
$("#q").oninput=e=>{qs=e.target.value.trim().toLowerCase();buildList();drawAll()};
const visible=()=>NAMES.filter(n=>(filt==="all"||D.arms[n].reg===filt||(filt==="lock"&&D.arms[n].reg==="dead"))
  &&(qs===""||n.toLowerCase().includes(qs)||D.arms[n].axis.toLowerCase()===qs));
function buildList(){
  const vis=new Set(visible()), L=$("#list"); L.innerHTML="";
  for(const reg of ORDER){
    const arms=NAMES.filter(n=>D.arms[n].reg===reg&&vis.has(n));
    if(!arms.length)continue;
    const h=document.createElement("div");h.className="grp";h.textContent=`${LAB[reg]} · ${arms.length}`;L.appendChild(h);
    arms.sort((a,b)=>(D.arms[b].fin??-9)-(D.arms[a].fin??-9)).forEach(n=>{
      const pi=pins.indexOf(n), b=document.createElement("button");
      b.className="armbtn"+(n===hot?" hot":"")+(pi>=0?" pin":"");
      if(pi>=0)b.style.color=cv(PINCOL[pi]);
      b.innerHTML=`<i class="dot ${reg}"></i><span class="nm">${n}</span><span class="v">${D.arms[n].fin??"–"}</span>`;
      b.onmouseenter=()=>{hot=n;drawHi();markList()};
      b.onmouseleave=()=>{hot=null;drawHi();markList()};
      b.onclick=()=>togglePin(n);
      b.dataset.arm=n; L.appendChild(b);
    });
  }
}
function markList(){
  document.querySelectorAll(".armbtn").forEach(b=>{
    const pi=pins.indexOf(b.dataset.arm);
    b.classList.toggle("hot",b.dataset.arm===hot);
    b.classList.toggle("pin",pi>=0);
    b.style.color=pi>=0?cv(PINCOL[pi]):"";
  });
}
function togglePin(n){
  const i=pins.indexOf(n);
  if(i>=0)pins.splice(i,1); else {pins.push(n); if(pins.length>3)pins.shift()}
  renderPins();markList();drawHi();renderOwn();
}
function renderPins(){
  const P=$("#pins");
  if(!pins.length){P.innerHTML='<span class="empty">未固定任何臂 — 点击曲线或左侧臂名以固定对比(最多 3)</span>';return}
  P.innerHTML=pins.map((n,i)=>{const A=D.arms[n];
    return `<button class="pinchip" data-a="${n}" style="color:var(${PINCOL[i]})">${n}
      <span class="st">@250 ${A.fin??"–"} · lock ${A.lock??"从不"} · 截断 ${A.trunc} · ${A.gpuh??"–"} GPU·h${
        A.suite!=null?` · 套件 ${A.suite}(AIME ${A.aime??"–"} · AMC ${A.amc??"–"} · Minerva ${A.minerva??"–"} · M500 ${A.s500??"–"})`:""}</span>
      <span class="x">✕</span></button>`}).join("");
  P.querySelectorAll(".pinchip").forEach(b=>b.onclick=()=>togglePin(b.dataset.a));
}

/* ---------- panels ---------- */
const panels=[];
function mkPanel(m,host,seqNames){
  const d=document.createElement("div");d.className="pan";
  d.innerHTML=`<div class="ph"><span class="pt">${m.t}</span><span class="phint">${m.h||""}</span>`
    +`<span class="tagl" style="display:none">log</span><span class="pv"></span></div><svg></svg>`
    +(seqNames?`<div class="seqleg">${seqNames.map((n,i)=>`<span><i style="background:var(--q${i+1})"></i>${n}</span>`).join("")}</div>`:"");
  host.appendChild(d);
  const o={m,el:d,svg:d.querySelector("svg"),base:null,hi:null,seq:seqNames};
  o.svg.appendChild(o.base=document.createElementNS(NS,"g"));
  o.svg.appendChild(o.hi=document.createElementNS(NS,"g"));
  return o;
}
function path(g,pts,color,w,op,dash){
  if(pts.length<2)return null;
  const e=document.createElementNS(NS,"path");
  e.setAttribute("d",pts.map((p,i)=>`${i?"L":"M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(""));
  e.setAttribute("fill","none");e.setAttribute("stroke",color);e.setAttribute("stroke-width",w);
  e.setAttribute("opacity",op);e.setAttribute("stroke-linejoin","round");
  if(dash)e.setAttribute("stroke-dasharray",dash);
  g.appendChild(e);return e;
}
function text(g,x,y,s,fill,size,anchor,weight){
  const e=document.createElementNS(NS,"text");
  e.setAttribute("x",x);e.setAttribute("y",y);e.setAttribute("font-size",size||10);e.setAttribute("fill",fill);
  e.setAttribute("font-family","ui-monospace,Consolas,monospace");
  if(anchor)e.setAttribute("text-anchor",anchor);
  if(weight)e.setAttribute("font-weight",weight);
  e.textContent=s;g.appendChild(e);return e;
}
function vline(g,x,y0,y1,color,w,dash,op){
  const e=document.createElementNS(NS,"line");
  e.setAttribute("x1",x);e.setAttribute("x2",x);e.setAttribute("y1",y0);e.setAttribute("y2",y1);
  e.setAttribute("stroke",color);e.setAttribute("stroke-width",w||1);e.setAttribute("opacity",op??1);
  if(dash)e.setAttribute("stroke-dasharray",dash);g.appendChild(e);
}
function seedMean(ser){
  if(!ser)return null;
  const good=ser.filter(Boolean);
  if(!good.length)return null;
  const n=Math.max(...good.map(s=>s.length));
  const out=new Array(n).fill(null);
  for(let i=0;i<n;i++){let a=0,c=0;good.forEach(s=>{const v=s[i];if(v!=null){a+=v;c++}});if(c)out[i]=a/c}
  return out;
}
const MEAN={}; // metric idx -> arm -> mean array (dense) or [[step,val]] (sparse)
function prep(){
  D.metrics.forEach((m,mi)=>{
    MEAN[mi]={};
    for(const arm in m.d){
      if(m.sparse){
        const per=m.d[arm].filter(Boolean);
        const byx={};per.forEach(s=>s.forEach(([x,v])=>{(byx[x]=byx[x]||[]).push(v)}));
        MEAN[mi][arm]=Object.keys(byx).map(Number).sort((a,b)=>a-b).map(x=>[x,byx[x].reduce((p,c)=>p+c,0)/byx[x].length]);
      }else MEAN[mi][arm]=seedMean(m.d[arm]);
    }
  });
}
function geom(o){
  const W=o.svg.clientWidth||420,H=170,L=48,R=64,T=10,B=18;
  return {W,H,L,R,T,B,iw:W-L-R,ih:H-T-B};
}
function domain(o,arms){
  const mi=o.mi,m=o.m;let lo=Infinity,hi=-Infinity;
  arms.forEach(a=>{const s=MEAN[mi][a];if(!s)return;
    (m.sparse?s.map(p=>p[1]):s).forEach(v=>{if(v!=null){if(v<lo)lo=v;if(v>hi)hi=v}})});
  if(!isFinite(lo)){lo=0;hi=1}
  const log=!m.frac&&lo>0&&hi/Math.max(lo,1e-9)>30;
  const nonneg=lo>=0, dmax=hi;
  if(m.ref!=null&&!log&&m.ref<=hi*1.25&&m.ref>=lo*0.5){lo=Math.min(lo,m.ref);hi=Math.max(hi,m.ref)}
  if(hi===lo)hi=lo+1e-6;
  if(!log){
    const p=(hi-lo)*.07;lo-=p;hi+=p;
    if(nonneg)lo=Math.max(0,lo);              // these quantities cannot go below zero
    if(lo>0&&lo<(hi-lo)*.6)lo=0;
    if(m.frac&&dmax<=1)hi=Math.min(hi,1);     // a fraction's axis stops at 1
  }
  return {lo,hi,log};
}
function scales(o,arms){
  const G=geom(o),dm=domain(o,arms);
  const yv=v=>{if(dm.log){const l=Math.log10(Math.max(v,dm.lo)),a=Math.log10(dm.lo),b=Math.log10(dm.hi);
      return G.T+G.ih-((l-a)/(b-a))*G.ih}
    return G.T+G.ih-((v-dm.lo)/(dm.hi-dm.lo))*G.ih};
  const xs=s=>G.L+((s-1)/(D.nmax-1))*G.iw;
  return {G,dm,yv,xs};
}
function ptsOf(o,arm,S){
  const s=MEAN[o.mi][arm];if(!s)return[];
  return o.m.sparse? s.map(([x,v])=>[S.xs(x),S.yv(v)])
    : s.map((v,i)=>v==null?null:[S.xs((i+1)*D.every),S.yv(v)]).filter(Boolean);
}
function drawBase(){
  const vis=visible();
  panels.forEach(o=>{
    const S=scales(o,vis.filter(a=>MEAN[o.mi][a]));o.S=S;o.vis=vis.filter(a=>MEAN[o.mi][a]);
    o.base.innerHTML="";o.el.querySelector(".tagl").style.display=S.dm.log?"":"none";
    const ink3=cv("--ink3"),rule=cv("--rule"),faint=cv("--faint");
    const ticks=S.dm.log?[S.dm.lo,Math.sqrt(S.dm.lo*S.dm.hi),S.dm.hi]:[S.dm.lo,(S.dm.lo+S.dm.hi)/2,S.dm.hi];
    ticks.forEach(v=>{const y=S.yv(v);
      const l=document.createElementNS(NS,"line");
      l.setAttribute("x1",S.G.L);l.setAttribute("x2",S.G.W-S.G.R);l.setAttribute("y1",y);l.setAttribute("y2",y);
      l.setAttribute("stroke",rule);l.setAttribute("stroke-width",1);l.setAttribute("opacity",.6);o.base.appendChild(l);
      text(o.base,S.G.L-6,y+3.5,fmt(v),ink3,9.5,"end")});
    [1,Math.round(D.nmax/2),D.nmax].forEach(s=>text(o.base,S.xs(s),S.G.H-5,s,ink3,9.5,"middle"));
    if(o.m.ref!=null&&o.m.ref>=S.dm.lo&&o.m.ref<=S.dm.hi){
      const y=S.yv(o.m.ref),l=document.createElementNS(NS,"line");
      l.setAttribute("x1",S.G.L);l.setAttribute("x2",S.G.W-S.G.R);l.setAttribute("y1",y);l.setAttribute("y2",y);
      l.setAttribute("stroke",ink3);l.setAttribute("stroke-width",1.1);l.setAttribute("stroke-dasharray","4 4");
      l.setAttribute("opacity",.75);o.base.appendChild(l);}
    o.vis.forEach(a=>path(o.base,ptsOf(o,a,S),faint,1.1,.62));
  });
  drawHi();
}
function drawHi(){
  panels.forEach(o=>{
    o.hi.innerHTML="";const S=o.S;if(!S)return;
    const ink=cv("--ink"),ink3=cv("--ink3");
    const show=[];
    pins.forEach((p,i)=>{if(o.vis.includes(p))show.push([p,cv(PINCOL[i]),2.2,1])});
    if(hot&&o.vis.includes(hot)&&!pins.includes(hot))show.push([hot,ink,2.2,1]);
    show.forEach(([a,c])=>{
      if(!o.m.sparse){const per=o.m.d[a];if(per)per.filter(Boolean).forEach(s=>
        path(o.hi,s.map((v,i)=>v==null?null:[S.xs((i+1)*D.every),S.yv(v)]).filter(Boolean),c,.8,.35));}
      const L=D.arms[a].lock;
      if(L)vline(o.hi,S.xs(L),S.G.T,S.G.T+S.G.ih,c,1.1,"3 3",.55);
    });
    const labels=[];
    show.forEach(([a,c,w,op])=>{
      const pts=ptsOf(o,a,S);path(o.hi,pts,c,w,op);
      const last=pts[pts.length-1];
      if(last)labels.push({x:Math.min(last[0]+5,S.G.W-4),y:last[1]+3.5,s:a.split("_")[0],c});
    });
    // de-collide the end labels: two arms can finish at the same value (vanilla and
    // h2 both sit on the cap), and overlapping text is unreadable.
    labels.sort((p,q)=>p.y-q.y);
    for(let i=1;i<labels.length;i++)
      if(labels[i].y-labels[i-1].y<11)labels[i].y=labels[i-1].y+11;
    const over=labels.length?labels[labels.length-1].y-(S.G.T+S.G.ih+6):0;
    if(over>0)labels.forEach(l=>l.y-=over);
    labels.forEach(l=>text(o.hi,l.x,l.y,l.s,l.c,10,"start",600));
    if(cross!=null){
      vline(o.hi,S.xs(cross),S.G.T,S.G.T+S.G.ih,ink,1,null,.4);
      const vals=show.map(([a,c])=>{const v=valAt(o,a,cross);return v==null?null:[a,c,v]}).filter(Boolean);
      o.el.querySelector(".pv").innerHTML=vals.length
        ? `<span style="color:var(--ink3)">step ${cross}</span> `+vals.map(([a,c,v])=>swatch(c)+fmt(v)).join(" ")
        : `<span style="color:var(--ink3)">step ${cross}</span>`;
      vals.forEach(([a,c,v])=>{const e=document.createElementNS(NS,"circle");
        e.setAttribute("cx",S.xs(cross));e.setAttribute("cy",S.yv(v));e.setAttribute("r",2.8);
        e.setAttribute("fill",c);o.hi.appendChild(e)});
    }else{
      o.el.querySelector(".pv").innerHTML=show.length
        ? show.map(([a,c])=>{const s=MEAN[o.mi][a];let v="–";
            if(s){if(o.m.sparse)v=fmt(s[s.length-1][1]);
              else for(let i=s.length-1;i>=0;i--)if(s[i]!=null){v=fmt(s[i]);break}}
            return swatch(c)+v}).join(" ")
        : `<span style="color:var(--ink3)">${o.vis.length} 臂</span>`;
    }
  });
}
const swatch=c=>`<i style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${c};margin:0 3px 0 7px"></i>`;
function valAt(o,arm,step){
  const s=MEAN[o.mi][arm];if(!s)return null;
  if(o.m.sparse){let best=null,bd=1e9;s.forEach(([x,v])=>{const d=Math.abs(x-step);if(d<bd){bd=d;best=v}});
    return bd<=13?best:null}
  const i=Math.round(step/D.every)-1;return (i>=0&&i<s.length)?s[i]:null;
}
function nearest(o,step,py){
  let best=null,bd=1e9;
  o.vis.forEach(a=>{const v=valAt(o,a,step);if(v==null)return;
    const d=Math.abs(o.S.yv(v)-py);if(d<bd){bd=d;best=a}});
  return bd<26?best:null;
}
function renderOwn(){
  const host=$("#own");host.innerHTML="";
  if(pins.length!==1||!D.specific[pins[0]])return;
  const arm=pins[0];
  const h=document.createElement("h3");h.className="sec";
  h.innerHTML=`${arm} 的专属仪表 <span>只有这一臂记录的面板</span>`;host.appendChild(h);
  const g=document.createElement("div");g.className="grid";host.appendChild(g);
  D.specific[arm].forEach(p=>{
    const o=mkPanel({t:p.t,h:p.seq?"信号均值 · 按 token 位置分箱":"",frac:0,ref:null},g,p.seq);
    const W=o.svg.clientWidth||420,H=170,L=48,R=20,T=10,B=18,iw=W-L-R,ih=H-T-B;
    const ser=p.seq?p.s:p.s.filter(Boolean);
    let lo=Infinity,hi=-Infinity;ser.forEach(s=>s.forEach(v=>{if(v!=null){if(v<lo)lo=v;if(v>hi)hi=v}}));
    if(!isFinite(lo)){lo=0;hi=1} if(hi===lo)hi=lo+1e-6;
    const nonneg=lo>=0, pad=(hi-lo)*.07;
    lo-=pad;hi+=pad;if(nonneg)lo=Math.max(0,lo);if(lo>0&&lo<(hi-lo)*.6)lo=0;
    const yv=v=>T+ih-((v-lo)/(hi-lo))*ih, xs=s=>L+((s-1)/(D.nmax-1))*iw;
    const ink3=cv("--ink3"),rule=cv("--rule");
    [lo,(lo+hi)/2,hi].forEach(v=>{const y=yv(v);const l=document.createElementNS(NS,"line");
      l.setAttribute("x1",L);l.setAttribute("x2",W-R);l.setAttribute("y1",y);l.setAttribute("y2",y);
      l.setAttribute("stroke",rule);l.setAttribute("stroke-width",1);l.setAttribute("opacity",.6);o.base.appendChild(l);
      text(o.base,L-6,y+3.5,fmt(v),ink3,9.5,"end")});
    [1,Math.round(D.nmax/2),D.nmax].forEach(s=>text(o.base,xs(s),H-5,s,ink3,9.5,"middle"));
    ser.forEach((s,i)=>path(o.base,s.map((v,j)=>v==null?null:[xs((j+1)*D.every),yv(v)]).filter(Boolean),
      p.seq?cv("--q"+(i+1)):cv("--accent"),p.seq?1.8:1.2,p.seq?1:.7));
    const lastv=ser.map(s=>{for(let i=s.length-1;i>=0;i--)if(s[i]!=null)return s[i];return null}).filter(v=>v!=null);
    o.el.querySelector(".pv").textContent=lastv.map(fmt).join(" / ");
  });
}
function build(){
  const G=$("#grid");G.innerHTML="";panels.length=0;
  D.metrics.forEach((m,mi)=>{
    if(!Object.keys(m.d).length)return;
    const o=mkPanel(m,G);o.mi=mi;panels.push(o);
    o.svg.onpointermove=e=>{
      const r=o.svg.getBoundingClientRect(),W=o.svg.clientWidth||420;
      const px=(e.clientX-r.left)/r.width*W, py=(e.clientY-r.top)/r.height*170;
      const s=Math.round(1+(px-o.S.G.L)/o.S.G.iw*(D.nmax-1));
      cross=Math.max(1,Math.min(D.nmax,s));
      const n=nearest(o,cross,py);
      if(n!==hot){hot=n;markList()}
      drawHi();
    };
    o.svg.onpointerleave=()=>{cross=null;hot=null;markList();drawHi()};
    o.svg.onclick=()=>{if(hot)togglePin(hot)};
  });
}
function drawAll(){drawBase();renderOwn()}
prep();build();buildList();renderPins();
requestAnimationFrame(drawAll);
addEventListener("resize",()=>drawAll());
addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT")return;
  const vis=visible();if(!vis.length)return;
  const i=vis.indexOf(hot);
  if(e.key==="ArrowRight"){hot=vis[(i+1)%vis.length];markList();drawHi();e.preventDefault()}
  if(e.key==="ArrowLeft"){hot=vis[(i-1+vis.length)%vis.length];markList();drawHi();e.preventDefault()}
  if(e.key===" "&&hot){togglePin(hot);e.preventDefault()}
});
new MutationObserver(drawAll).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
</script></body></html>
"""

if __name__ == "__main__":
    main()
