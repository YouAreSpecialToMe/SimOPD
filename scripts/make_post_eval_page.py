"""Build a standalone post-eval comparison browser (one local HTML file).

    python scripts/make_post_eval_page.py     # -> docs/post-eval-dynamics.html

Sibling of make_dynamics_page.py, same interaction model (all arms in every
panel; hover/click one arm to spotlight it everywhere; pin up to three), but the
data is the OFFLINE SUITE over saved checkpoints, not the training loop.

Layout is per BENCHMARK, not per metric: one small overview block carrying the
composite, then one block per benchmark holding that benchmark's own setting
line and its acc / length / truncation panels. The composite is a weighted
average of four components measured under two different sample budgets (avg@32
on AIME and AMC, avg@3 on Minerva and MATH500) over problem sets that differ by
17x in size, so a per-benchmark reading is the honest one and the composite is
kept to a single panel.

The two pages must not be read as one series. In-loop is greedy mean@1 on
MATH500 at 16k; the suite is tau=0.7 / top-p 0.95 / 32k. Different decoding,
different budget -- the report says these must never be compared across tables.

What only this page can show: the suite records resp_len / truncated per SAMPLE,
so length splits by whether the answer was right. That split is the direct test
of the "cannot stop" reading of late-training collapse; in-loop logs carry only
rollout means and cannot separate the two populations.

Source: docs/data/post_eval_cells.csv (per arm/seed/step/benchmark) and
docs/data/post_eval_bystep.csv (composite, complete cells only), both written by
simopd_data/extract_post_eval.py from the eval parquets.
"""

import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_COMPOSITE = 0.1453      # untrained 1.7B student, same suite
TEACHER_COMPOSITE = 0.648    # 4B-2507 teacher, same suite
CAP = 32768
BENCHES = ["aime24", "aime25", "amc23", "minerva", "math500"]
# Which composite component each benchmark feeds; AIME24+25 pool into one.
COMPONENT = {"aime24": "aime", "aime25": "aime", "amc23": "amc23",
             "minerva": "minerva", "math500": "math500"}

AXIS = {"a2_coldstart": "A", "b1_skew_kl": "B", "b2_forward_kl": "B", "b3_eopd_gate": "B",
        "b4_jsd": "B", "b5_k2": "B", "c1_lsm_topk32_renorm": "C", "c2_quantile_budget": "C",
        "c3_intersection": "C", "c4_pi_tail_budget": "C", "d1_tip": "D", "d2_selectkd": "D",
        "d3_teachability": "D", "e1_pl_rank": "E", "e2_set_coverage": "E", "e3_zvalue": "E",
        "f1_soft_log": "F", "f2_hard_clip": "F", "f3_power": "F", "g1_verified_only": "G",
        "g2_fire_likelihood": "G", "g4_failure_only": "G", "g5_rgopd_gate": "G",
        "h1_first_segment": "H", "h2_last_segment": "H", "h3_random_segment": "H",
        "j1_kdrl": "J", "vanilla": "–", "vanilla_n8": "–"}

# Per-benchmark panels. `frac` pins a 0..1 quantity to a linear axis anchored at 0.
BENCH_PANELS = [
    dict(c="avg_at_k", t="准确率 avg@k", h="逐题 k 次的正确率,再对题平均", frac=1, ref=None),
    dict(c="pass_at_k", t="pass@k", h="k 次里有一次对就算解出", frac=1, ref=None),
    dict(c="len_mean", t="回复长度", h="全样本均值 · tokens", ref=CAP),
    dict(c="trunc_rate", t="截断率", h="撞 32k 预算的比例", frac=1, ref=1.0),
    dict(c="len_mean_correct", t="对答长度", h="答对的那些样本 · tokens", ref=CAP),
    dict(c="len_mean_wrong", t="错答长度", h="答错的那些样本 · tokens", ref=CAP),
]


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


def series(df, col, steps, seeds, key=None):
    """One arm's curve, both ways.

    {"m": seed-mean per step, "n": how many seeds that mean averaged, "s": per-seed}

    The mean is the default line because a reader comparing 29 arms wants one
    curve each; the per-seed lines stay in the payload because at these sample
    counts the spread is the interesting part (AIME's avg@32 rests on ~42 correct
    samples out of 960). `n` is what makes a mean honest when a step has only
    some of the three seeds evaluated -- the same job the report's `·N` does.
    """
    per = []
    for sd in seeds:
        sg = df[df.seed == sd].set_index("step")[col]
        per.append([sg.get(st) if st in sg.index else None for st in steps]
                   if len(sg) else None)
    if not any(s and any(v is not None and not pd.isna(v) for v in s) for s in per):
        return None
    mean, cnt = [], []
    for i in range(len(steps)):
        vals = [s[i] for s in per if s and s[i] is not None and not pd.isna(s[i])]
        mean.append(r4(sum(vals) / len(vals)) if vals else None)
        cnt.append(len(vals))
    return {"m": mean, "n": cnt,
            "s": [[r4(v) for v in s] if s else None for s in per]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(ROOT, "docs/data/post_eval_cells.csv"))
    ap.add_argument("--bystep", default=os.path.join(ROOT, "docs/data/post_eval_bystep.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/post-eval-dynamics.html"))
    a = ap.parse_args()

    cells = pd.read_csv(a.cells)
    by = pd.read_csv(a.bystep)
    steps = sorted(set(int(s) for s in cells.step.unique()))
    seeds = sorted(set(int(s) for s in cells.seed.unique()))

    blocks = []

    # ---- overview: the composite only ----
    data = {arm: series(g, "composite", steps, seeds, arm) for arm, g in by.groupby("arm")}
    blocks.append(dict(
        id="overview", t="总览 · suite composite",
        note=("四组分等权:AIME24+25(合并池) / AMC23 / Minerva / MATH500。"
              "只在一格的 5 个 benchmark 全部齐全时才计入 —— 因此这里是 %d 格,"
              "少于任一单项 benchmark 的格数。" % len(by)),
        setting="未训练学生 %.4f · 4B 教师 %.3f · τ=0.7 / top-p 0.95 / 32k"
                % (BASE_COMPOSITE, TEACHER_COMPOSITE),
        panels=[dict(id="composite", t="suite composite", h="四组分等权",
                     ref=BASE_COMPOSITE, frac=1,
                     d={k: v for k, v in data.items() if v})]))

    # ---- one block per benchmark ----
    for b in BENCHES:
        gb = cells[cells.bench == b]
        if gb.empty:
            continue
        nprob = int(gb.n_problems.median())
        k = int(round(gb.n_samples.median() / max(nprob, 1)))
        panels = []
        for p in BENCH_PANELS:
            d = {}
            for arm, g in gb.groupby("arm"):
                s = series(g, p["c"], steps, seeds, arm)
                if s:
                    d[arm] = s
            if d:
                panels.append({**{kk: vv for kk, vv in p.items() if kk != "c"},
                               "id": b + "/" + p["c"], "d": d})
        blocks.append(dict(
            id=b, t=b, kind="bench",
            setting="%d 题 × %d 采样 = %d 样本/格 · 指标 avg@%d · 归入 composite 的 %s 组分"
                    % (nprob, k, nprob * k, k, COMPONENT[b]),
            note=("τ=0.7 / top-p 0.95 / max_tokens 32768。已评格数 %d。" % len(gb)
                  + ("AIME24 与 AIME25 先合并成一个题池再取均值,所以两者单看不等于 aime 组分。"
                     if b.startswith("aime") else "")),
            panels=panels))

    # ---- per-arm meta for the sidebar ----
    meta = {}
    for arm in sorted(set(cells.arm.unique())):
        g250 = by[(by.arm == arm) & (by.step == 250)]
        gb = by[by.arm == arm]
        cg = cells[cells.arm == arm]
        last = gb.sort_values("step").iloc[-1] if len(gb) else None
        meta[arm] = dict(
            axis=AXIS.get(arm, "–"),
            done=int(len(gb)), cells=int(len(cg)), maxstep=int(cg.step.max()),
            comp=r4(g250.composite.mean()) if len(g250) else None,
            grr=r4(100 * (g250.composite.mean() - BASE_COMPOSITE)
                   / (TEACHER_COMPOSITE - BASE_COMPOSITE)) if len(g250) else None,
            len=r4(last.len_mean) if last is not None else None,
            trunc=r4(last.trunc_rate) if last is not None else None,
            lc=r4(last.len_mean_correct) if last is not None else None,
            lw=r4(last.len_mean_wrong) if last is not None else None,
            ratio=r4(last.len_mean_wrong / last.len_mean_correct)
            if last is not None and last.len_mean_correct else None,
        )

    payload = dict(steps=steps, seeds=seeds, base=BASE_COMPOSITE,
                   teacher=TEACHER_COMPOSITE, cap=CAP, benches=BENCHES,
                   arms=meta, blocks=blocks,
                   ncells=int(len(by)), nbenchcells=int(len(cells)),
                   narms=int(cells.arm.nunique()))
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))
    with open(a.out, "w") as f:
        f.write(html)
    print("wrote %s  (%.2f MiB, %d arms, %d blocks, %d panels, %d complete cells)"
          % (a.out, os.path.getsize(a.out) / 2 ** 20, len(meta), len(blocks),
             sum(len(b["panels"]) for b in blocks), len(by)))


TEMPLATE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SimOPD · Post-eval 总览</title>
<style>
:root{color-scheme:light;--bg:#fbfcfd;--panel:#f2f5f8;--raise:#fff;
 --ink:#12161d;--ink2:#4d5663;--ink3:#7f8a99;--faint:#9fadbd;--rule:#dbe2ea;--rule2:#e9eef4;
 --accent:#1f5fa8;--on-accent:#fff;--ok:#1baf7a;--bad:#eb6834;
 --p1:#2a78d6;--p2:#eb6834;--p3:#1baf7a}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --bg:#15171a;--panel:#1e2126;--raise:#22262c;
 --ink:#e9ecf0;--ink2:#b2bac5;--ink3:#7c8794;--faint:#4d565f;--rule:#2e343c;--rule2:#272c33;
 --accent:#79a9e0;--on-accent:#10141a;--ok:#199e70;--bad:#d95926;
 --p1:#3987e5;--p2:#d95926;--p3:#199e70}}
:root[data-theme=dark]{color-scheme:dark;--bg:#15171a;--panel:#1e2126;--raise:#22262c;
 --ink:#e9ecf0;--ink2:#b2bac5;--ink3:#7c8794;--faint:#4d565f;--rule:#2e343c;--rule2:#272c33;
 --accent:#79a9e0;--on-accent:#10141a;--ok:#199e70;--bad:#d95926;
 --p1:#3987e5;--p2:#d95926;--p3:#199e70}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.5;
 font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Noto Sans CJK SC","PingFang SC",sans-serif}
.mono,.n{font-family:ui-monospace,"SF Mono",Consolas,"Noto Sans Mono CJK SC",monospace;font-variant-numeric:tabular-nums}
.app{display:grid;grid-template-columns:262px 1fr;min-height:100vh}
aside{border-right:1px solid var(--rule);background:var(--panel);padding:13px 11px;position:sticky;top:0;
 height:100vh;overflow-y:auto}
main{padding:14px 16px 40px;min-width:0}
h1{font-size:15px;margin:0 0 2px}
.sub{color:var(--ink3);font-size:12px;margin-bottom:10px}
.note{background:var(--raise);border:1px solid var(--rule2);border-left:3px solid var(--accent);
 border-radius:5px;padding:8px 10px;font-size:12px;color:var(--ink2);margin:0 0 12px;line-height:1.55}
.arm{display:flex;align-items:center;gap:6px;padding:3px 6px;border-radius:5px;cursor:pointer;
 font-size:12px;user-select:none}
.arm:hover{background:var(--raise)}
.arm.pin{background:var(--raise);box-shadow:inset 0 0 0 1px var(--rule)}
.arm .sw{width:9px;height:9px;border-radius:2px;background:var(--faint);flex:none}
.arm .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.arm .v{color:var(--ink3);font-size:11px}
.arm.dim{opacity:.42}
.ax{color:var(--faint);font-size:10px;width:11px;flex:none;text-align:center}
.blk{margin-bottom:16px;border:1px solid var(--rule);border-radius:9px;overflow:hidden}
.blk>header{background:var(--panel);border-bottom:1px solid var(--rule2);padding:8px 12px}
.blk>header h2{margin:0;font-size:13.5px;display:flex;align-items:baseline;gap:9px}
.blk>header .set{color:var(--accent);font-size:11.5px}
.blk>header .nt{color:var(--ink3);font-size:11px;margin-top:2px;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;padding:10px 12px}
.pan{background:var(--panel);border:1px solid var(--rule2);border-radius:7px;padding:8px 10px 3px}
.pan h3{margin:0;font-size:12px;font-weight:600;display:flex;align-items:baseline;gap:6px}
.pan .hh{color:var(--ink3);font-weight:400;font-size:10.5px}
.pan .pv{margin-left:auto;font-size:11px;color:var(--accent)}
.pan svg{width:100%;height:124px;display:block;overflow:visible;margin-top:3px}
.hd{display:flex;align-items:center;gap:10px;margin-bottom:9px;flex-wrap:wrap}
button{font:inherit;font-size:12px;padding:3px 9px;border-radius:5px;border:1px solid var(--rule);
 background:var(--raise);color:var(--ink2);cursor:pointer}
button:hover{border-color:var(--accent);color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:11.5px;margin-top:4px}
th,td{text-align:right;padding:2px 6px;border-bottom:1px solid var(--rule2)}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--ink3);font-weight:500}
</style></head><body>
<div class="app">
<aside>
  <div style="font-size:11px;color:var(--ink3);margin-bottom:7px">悬停=聚光 · 点击=钉住(≤3)</div>
  <div id="list"></div>
</aside>
<main>
  <h1>SimOPD · Post-eval 总览</h1>
  <div class="sub mono" id="cap"></div>
  <div class="note">
    离线套件跑在<b>保存的 checkpoint</b> 上,τ=0.7 / top-p 0.95 / 32k 预算。
    <b>与 in-loop 页面不可跨表比较</b> —— 那边是 16k 预算下 MATH500 的贪心 mean@1。
    本页独有的是<b>按对错拆分的长度</b>:逐样本产物才能把答对与答错两群分开,
    这是「不会停」读法的直接检验,in-loop 日志只有 rollout 均值,做不到。
    <br>对答/错答长度切的是<b>样本</b>而不是题目 —— 一道题的 k 次采样会同时落进两边,
    两者按样本数加权合起来才等于「回复长度」。因此两群的题目构成并不相同
    (答对的样本偏向易题),这个倍数里混着「失败时收不住」和「难题本来就写得长」两个来源,
    要分离需按题内配对比较。
  </div>
  <div class="hd">
    <button id="mode">显示:seed 均值</button>
    <button id="clr">清除钉选</button>
    <span class="mono" style="font-size:11.5px;color:var(--ink3)" id="pins">未钉选</span>
  </div>
  <div id="blocks"></div>
</main></div>
<script>
const D=/*__DATA__*/;
const NS="http://www.w3.org/2000/svg";
const ARMS=Object.keys(D.arms).sort((a,b)=>(D.arms[b].comp??-1)-(D.arms[a].comp??-1)||a.localeCompare(b));
const PC=["var(--p1)","var(--p2)","var(--p3)"];
let hover=null, pins=[], meanMode=true;

const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e;};
const sv=(t,at)=>{const e=document.createElementNS(NS,t);for(const k in at)e.setAttribute(k,at[k]);return e;};
const fmt=v=>v==null?"–":(Math.abs(v)>=1000?Math.round(v).toLocaleString():(+v).toFixed(3).replace(/0+$/,"").replace(/\.$/,""));

/* ---------- sidebar ---------- */
const list=document.getElementById("list");
for(const a of ARMS){
  const m=D.arms[a], r=el("div","arm"); r.dataset.arm=a;
  r.appendChild(el("span","ax",m.axis));
  r.appendChild(el("span","sw"));
  r.appendChild(el("span","nm",a));
  r.appendChild(el("span","v",m.comp==null?(m.done+"/30"):fmt(m.comp)));
  r.title=a+"  完整格 "+m.done+"/30 · 单项格 "+m.cells
    +(m.comp!=null?"\nstep250 composite="+fmt(m.comp)+"  GRR="+fmt(m.grr)+"%":"\n(无 step250 完整格)")
    +"\n末步长度 对答 "+fmt(m.lc)+" / 错答 "+fmt(m.lw)+(m.ratio?" (×"+fmt(m.ratio)+")":"");
  r.onmouseenter=()=>{hover=a;paint();};
  r.onmouseleave=()=>{hover=null;paint();};
  r.onclick=()=>{const i=pins.indexOf(a); if(i>=0)pins.splice(i,1); else if(pins.length<3)pins.push(a); paint();};
  list.appendChild(r);
}
document.getElementById("cap").textContent=
  D.narms+" arms · "+D.ncells+" 完整格 / "+D.nbenchcells+" 单项格 · steps "
  +D.steps[0]+"–"+D.steps[D.steps.length-1];

/* ---------- blocks + panels ---------- */
const host=document.getElementById("blocks"), P=[];
for(const blk of D.blocks){
  const sec=el("section","blk"), hd=el("header");
  const h2=el("h2"); h2.appendChild(el("span",null,blk.t));
  if(blk.setting) h2.appendChild(el("span","set mono",blk.setting));
  hd.appendChild(h2);
  if(blk.note) hd.appendChild(el("div","nt",blk.note));
  sec.appendChild(hd);
  const g=el("div","grid"); sec.appendChild(g); host.appendChild(sec);

  for(const p of blk.panels){
    const d=el("div","pan");
    const h=el("h3"); h.appendChild(el("span",null,p.t));
    h.appendChild(el("span","hh",p.h)); const pv=el("span","pv mono"); h.appendChild(pv);
    d.appendChild(h);
    const s=sv("svg",{}); d.appendChild(s); g.appendChild(d);
    const base=sv("g",{}), hi=sv("g",{}); s.appendChild(base); s.appendChild(hi);
    // pooled range over every arm+seed of THIS panel, so arms stay comparable
    // (uses per-seed values, so switching to the mean never overflows the axis)
    let lo=Infinity, hi2=-Infinity;
    for(const a in p.d) for(const ser of p.d[a].s) if(ser) for(const v of ser)
      if(v!=null){lo=Math.min(lo,v);hi2=Math.max(hi2,v);}
    if(!isFinite(lo)){lo=0;hi2=1;}
    if(p.frac) lo=Math.min(lo,0);
    if(lo===hi2) hi2=lo+1;
    const pad=(hi2-lo)*0.08; lo-=pad; hi2+=pad;
    P.push({p,svg:s,base,hi,pv,lo,hi2});
  }
}
const H=110, ML=40, MR=8, MT=6;
const wOf=o=>o.svg.clientWidth||360;
const xs=(o,i)=>ML+(wOf(o)-ML-MR)*i/Math.max(1,D.steps.length-1);

function path(o,ser){
  let d="",up=false;
  ser.forEach((v,i)=>{ if(v==null){up=false;return;}
    const y=MT+H-(v-o.lo)/(o.hi2-o.lo)*H;
    d+=(up?"L":"M")+xs(o,i).toFixed(1)+" "+y.toFixed(1)+" "; up=true; });
  return d;
}
function paint(){
  const spot=hover||(pins.length===1?pins[0]:null);
  for(const r of list.children){
    const a=r.dataset.arm, pi=pins.indexOf(a);
    r.classList.toggle("pin",pi>=0);
    r.classList.toggle("dim",!!spot&&a!==spot&&pi<0);
    r.querySelector(".sw").style.background=pi>=0?PC[pi]:(a===spot?"var(--accent)":"var(--faint)");
  }
  for(const o of P){
    o.base.textContent=""; o.hi.textContent="";
    o.base.appendChild(sv("line",{x1:ML,y1:MT+H,x2:wOf(o)-MR,y2:MT+H,stroke:"var(--rule)"}));
    for(const [fy,lab] of [[0,fmt(o.lo)],[1,fmt(o.hi2)]]){
      const y=MT+H-fy*H;
      const t=sv("text",{x:ML-5,y:y+3.5,"text-anchor":"end","font-size":9.5,fill:"var(--ink3)"});
      t.textContent=lab; o.base.appendChild(t);
    }
    if(o.p.ref!=null&&o.p.ref>=o.lo&&o.p.ref<=o.hi2){
      const y=MT+H-(o.p.ref-o.lo)/(o.hi2-o.lo)*H;
      o.base.appendChild(sv("line",{x1:ML,y1:y,x2:wOf(o)-MR,y2:y,stroke:"var(--faint)",
        "stroke-dasharray":"3 3","stroke-width":1}));
    }
    for(const a in o.p.d){
      const e=o.p.d[a], pi=pins.indexOf(a), on=(a===spot)||pi>=0;
      const col=pi>=0?PC[pi]:(a===spot?"var(--accent)":"var(--faint)");
      const tgt=on?o.hi:o.base;
      const dim=on?1:(spot||pins.length?0.16:0.42);
      // seed lines: always in per-seed mode; in mean mode only for the arm in focus,
      // so the spread stays visible exactly where someone is reading it
      if(!meanMode||on){
        for(const ser of e.s){ if(!ser)continue;
          const d=path(o,ser); if(!d)continue;
          tgt.appendChild(sv("path",{d,fill:"none",stroke:col,
            "stroke-width":meanMode?0.7:(on?1.9:0.8),
            opacity:meanMode?(on?0.34:dim):dim}));
        }
      }
      if(meanMode){
        const d=path(o,e.m);
        if(d) tgt.appendChild(sv("path",{d,fill:"none",stroke:col,
          "stroke-width":on?2.1:0.85,opacity:dim}));
      }
    }
    const who=spot||pins[0];
    o.pv.textContent = who&&o.p.d[who] ? (()=>{
        const e=o.p.d[who];
        for(let i=e.m.length-1;i>=0;i--) if(e.m[i]!=null)
          return fmt(e.m[i])+(e.n[i]<D.seeds.length?("·"+e.n[i]):"");
        return "";})() : "";
  }
  document.getElementById("pins").textContent=pins.length?("已钉:"+pins.join(" · ")):"未钉选";
}
document.getElementById("clr").onclick=()=>{pins=[];paint();};
document.getElementById("mode").onclick=e=>{
  meanMode=!meanMode;
  e.target.textContent="显示:"+(meanMode?"seed 均值":"逐 seed");
  paint();
};
addEventListener("resize",paint);
paint();
</script></body></html>
"""


if __name__ == "__main__":
    main()
