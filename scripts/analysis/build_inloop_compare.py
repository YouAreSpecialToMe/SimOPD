#!/usr/bin/env python3
"""生成 docs/inloop-stop2-vs-legacy.html:v2(双停)vs legacy 的 in-loop 对比。
设计系统沿用 docs/arm-dynamics.html 的 token(明暗三态、等宽数字、面板栅格)。
数据内联(自包含,file:// 与 Artifact CSP 双友好)。"""
import csv, json, collections

SRC = "/private/tmp/claude-501/-Users-lichanghao/0bcd641a-d76b-42bd-8469-dc49bb26e78c/scratchpad/inloop_v2_vs_legacy.csv"
DST = "/Users/lichanghao/Desktop/Changhao/GT_Sem4/project/SimOPD/docs/inloop-stop2-vs-legacy.html"

rows = list(csv.DictReader(open(SRC)))
data = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    arm, wave, st = r["arm"], r["wave"], int(r["step"])
    def f(k):
        return float(r[k]) if r[k] not in ("", None) else None
    data[arm][wave].append({"s": st, "v": f("val_acc"), "l": f("resp_len"),
                            "ht": f("h_target"), "hb": f("h9_budget")})
for arm in data:
    for wave in data[arm]:
        data[arm][wave].sort(key=lambda d: d["s"])

ARMS = [("a1_gkd_mix0.5", "GKD 在线混合 λ=.5"), ("a3_offpolicy", "纯离线缓存 λ=0 端点"),
        ("a4_dagger_anneal", "DAgger 退火"), ("a5_aggrevate", "AggreVaTe 教师续写"),
        ("a2_coldstart", "SFT 冷启动 → OPD(P-untaught 谱系)"),
        ("h6_gen_sched", "递进帽 128→16k"), ("h7_gen512", "固定帽 512"),
        ("h8_gen2048", "固定帽 2048"), ("h9_prune_adapt", "教师失线自适应预算"),
        ("h10_task_subset", "50% 任务子集 · 全深")]

# 摘要表:val @ 25/50/.../250
STEPS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
def val_at(arm, wave, st):
    for d in data[arm][wave]:
        if d["s"] == st and d["v"] is not None:
            return d["v"]
    return None
sumrows = []
for arm, _ in ARMS:
    cells = []
    for st in STEPS:
        lv, vv = val_at(arm, "legacy", st), val_at(arm, "v2", st)
        if lv is None and vv is None:
            cells.append("<td class=n>·</td>")
        else:
            l = f"{lv:.3f}"[1:] if lv is not None else "·"
            v = f"{vv:.3f}"[1:] if vv is not None else "·"
            cls = ""
            if lv is not None and vv is not None:
                cls = " up" if vv > lv + 1e-9 else (" dn" if vv < lv - 1e-9 else "")
            cells.append(f'<td class="n{cls}"><span class=lg>{l}</span> <span class=v2>{v}</span></td>')
    sumrows.append(f'<tr><th class=mono>{arm}</th>{"".join(cells)}</tr>')

payload = json.dumps({a: data[a] for a, _ in ARMS}, separators=(",", ":"))
arms_js = json.dumps(ARMS, ensure_ascii=False)

html = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SimOPD · 双停对比台</title>
<style>
:root{color-scheme:light;--bg:#fbfcfd;--panel:#f2f5f8;--raise:#fff;
 --ink:#12161d;--ink2:#4d5663;--ink3:#7f8a99;--rule:#dbe2ea;--rule2:#e9eef4;
 --accent:#1f5fa8;--v2:#2a78d6;--lg:#eb6834;--ok:#1baf7a;--cap:#9f8ade}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --bg:#15171a;--panel:#1e2126;--raise:#22262c;--ink:#e9ecf0;--ink2:#b2bac5;--ink3:#7c8794;
 --rule:#2e343c;--rule2:#272c33;--accent:#79a9e0;--v2:#3987e5;--lg:#d95926;--ok:#199e70;--cap:#8f7fd9}}
:root[data-theme=dark]{color-scheme:dark;
 --bg:#15171a;--panel:#1e2126;--raise:#22262c;--ink:#e9ecf0;--ink2:#b2bac5;--ink3:#7c8794;
 --rule:#2e343c;--rule2:#272c33;--accent:#79a9e0;--v2:#3987e5;--lg:#d95926;--ok:#199e70;--cap:#8f7fd9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.55;
 font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Noto Sans CJK SC","PingFang SC",sans-serif}
.mono,.n{font-family:ui-monospace,"SF Mono",Consolas,"Noto Sans Mono CJK SC",monospace;font-variant-numeric:tabular-nums}
header{padding:26px 28px 14px;border-bottom:1px solid var(--rule);background:var(--panel)}
header h1{margin:0 0 4px;font-size:19px;font-weight:680;text-wrap:balance}
header .sub{color:var(--ink2);font-size:12.5px;max-width:72ch}
.key{display:flex;gap:16px;margin-top:10px;font-size:11.5px;color:var(--ink2)}
.key .sw{display:inline-block;width:22px;height:0;border-top:2.5px solid var(--v2);vertical-align:middle;margin-right:5px}
.key .sw.lg{border-top:2.5px dashed var(--lg)}
.key .sw.cap{border-top:2px dotted var(--cap)}
main{padding:20px 28px 60px;max-width:1180px;margin:0 auto}
h2{font-size:14px;margin:26px 0 10px;letter-spacing:.04em}
.tblwrap{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--raise)}
table{border-collapse:collapse;font-size:11.5px;min-width:860px}
th,td{padding:5px 9px;border-bottom:1px solid var(--rule2);text-align:right;white-space:nowrap}
thead th{color:var(--ink3);font-weight:600;border-bottom:1px solid var(--rule)}
tbody th{text-align:left;color:var(--ink2);font-weight:560;font-size:11px}
td .lg{color:var(--lg)}td .v2{color:var(--v2);font-weight:640}
td.up .v2::after{content:"▲";font-size:8.5px;color:var(--ok);margin-left:2px}
td.dn .v2::after{content:"▽";font-size:8.5px;color:var(--ink3);margin-left:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(500px,1fr));gap:14px;margin-top:12px}
.card{background:var(--raise);border:1px solid var(--rule);border-radius:9px;padding:12px 14px 8px}
.card h3{margin:0;font-size:12.5px}.card h3 .mono{color:var(--accent)}
.card .why{color:var(--ink3);font-size:11px;margin:1px 0 6px}
.duo{display:grid;grid-template-columns:1fr 1fr;gap:10px}
svg{display:block;width:100%;height:auto}
.axis{stroke:var(--rule);stroke-width:1}
.tick{fill:var(--ink3);font-size:8.5px;font-family:ui-monospace,Consolas,monospace}
.pt{fill:var(--v2)}.ptl{fill:var(--lg)}
.note{color:var(--ink2);font-size:12px;max-width:78ch}
footer{color:var(--ink3);font-size:11px;padding:0 28px 30px;max-width:1180px;margin:0 auto}
@media(prefers-reduced-motion:no-preference){.card{transition:border-color .15s}.card:hover{border-color:var(--accent)}}
</style>
<header>
 <h1>双停契约(v2)与 legacy 的 in-loop 对比 · A/H 轴 seed 0</h1>
 <div class="sub">同臂同协议,唯一差异是终止符契约:legacy 采样器只认学生
 <span class=mono>&lt;|endoftext|&gt;</span>;v2 采纳教师停止集
 <span class=mono>{151643,151645}</span>(R5 附录)。val 为 in-house MATH-lighteval
 greedy@1,每 25 步;长度为训练 rollout 均值。legacy 波在 08-19 团灭/退役处截止。</div>
 <div class="key"><span><span class="sw"></span>v2(stop2)</span>
 <span><span class="sw lg"></span>legacy</span>
 <span><span class="sw cap"></span>帽/预算(h6 ramp · h9 budget)</span></div>
</header>
<main>
<h2>VAL 摘要(每格:<span class="lg mono">legacy</span> → <span class="v2 mono">v2</span>,▲=v2 高)</h2>
<div class="tblwrap"><table>
<thead><tr><th style="text-align:left">arm</th>__STEPHEAD__</tr></thead>
<tbody>__SUMROWS__</tbody></table></div>
<h2>逐臂曲线(左 val,右 rollout 长度)</h2>
<div class="grid" id="grid"></div>
<p class="note" style="margin-top:18px">读法:a1/a3/a4/a5(教师混合)v2 早段 val 全面高于
legacy;a2 与 h10 的长度曲线是 P-untaught / P-suppress 病理的现场测量——v2 只修采样契约、
不动损失,长度是否回落取决于终止符能否被(重新)学会。h6 长度贴 ramp 帽、h9 贴自适应预算,
属设计而非病理。两波均为 seed 0 单种子,监控性读数;裁决以离线套件为准。</p>
</main>
<footer>数据:docs/data/inloop_v2_vs_legacy.csv(wandb 导出,多次开机按步合并,后开机胜)·
生成 2026-08-20 · git c022d57 后</footer>
<script>
const DATA=__PAYLOAD__;
const ARMS=__ARMS__;
function line(pts,xk,yk,xmax,ymax,W,H,PX,PY){
  const xs=s=>PX+ (s/xmax)*(W-PX-6), ys=v=>H-PY-(v/ymax)*(H-PY-8);
  let d="";
  for(const p of pts){ if(p[yk]==null)continue; d+=(d?"L":"M")+xs(p[xk]).toFixed(1)+","+ys(p[yk]).toFixed(1);}
  return d;
}
function panel(arm,yk,ymax,fmt,extra){
  const W=250,H=120,PX=30,PY=16,xmax=250;
  const v2=(DATA[arm]||{}).v2||[], lg=(DATA[arm]||{}).legacy||[];
  let s='<svg viewBox="0 0 '+W+' '+H+'" role="img">';
  s+='<line class="axis" x1="'+PX+'" y1="'+(H-PY)+'" x2="'+(W-4)+'" y2="'+(H-PY)+'"/>';
  s+='<line class="axis" x1="'+PX+'" y1="6" x2="'+PX+'" y2="'+(H-PY)+'"/>';
  for(const t of [0,.5,1]){const y=H-PY-t*(H-PY-8);
    s+='<text class="tick" x="'+(PX-4)+'" y="'+(y+3)+'" text-anchor="end">'+fmt(t*ymax)+'</text>';}
  for(const t of [0,125,250]){const x=PX+(t/xmax)*(W-PX-6);
    s+='<text class="tick" x="'+x+'" y="'+(H-4)+'" text-anchor="middle">'+t+'</text>';}
  if(extra){const d=line(v2,"s",extra,xmax,ymax,W,H,PX,PY);
    if(d)s+='<path d="'+d+'" fill="none" stroke="var(--cap)" stroke-width="1.4" stroke-dasharray="2 3"/>';}
  const dl=line(lg,"s",yk,xmax,ymax,W,H,PX,PY);
  if(dl)s+='<path d="'+dl+'" fill="none" stroke="var(--lg)" stroke-width="1.8" stroke-dasharray="5 4" opacity=".9"/>';
  const dv=line(v2,"s",yk,xmax,ymax,W,H,PX,PY);
  if(dv)s+='<path d="'+dv+'" fill="none" stroke="var(--v2)" stroke-width="2"/>';
  if(yk==="v"){
    for(const [set,cls] of [[lg,"ptl"],[v2,"pt"]])
      for(const p of set){ if(p.v==null)continue;
        const x=PX+(p.s/xmax)*(W-PX-6), y=H-PY-(p.v/ymax)*(H-PY-8);
        s+='<circle class="'+cls+'" cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="2.4"><title>'+p.s+": "+p.v.toFixed(3)+'</title></circle>';}
  }
  return s+"</svg>";
}
const g=document.getElementById("grid");
for(const [arm,why] of ARMS){
  const extra = arm==="h6_gen_sched" ? "ht" : (arm==="h9_prune_adapt" ? "hb" : null);
  const el=document.createElement("div");el.className="card";
  el.innerHTML='<h3><span class="mono">'+arm+'</span> · '+why+'</h3><div class="why">左:in-house val(greedy@1)  右:rollout 长度(token,帽 16384)</div>'
   +'<div class="duo"><div>'+panel(arm,"v",0.7,v=>v.toFixed(1))+'</div>'
   +'<div>'+panel(arm,"l",16800,v=>(v/1000|0)+"k",extra)+'</div></div>';
  g.appendChild(el);
}
</script>
"""
stephead = "".join(f"<th>@{s}</th>" for s in STEPS)
html = html.replace("__STEPHEAD__", stephead).replace("__SUMROWS__", "\n".join(sumrows))
html = html.replace("__PAYLOAD__", payload).replace("__ARMS__", arms_js)
open(DST, "w").write(html)
print(DST, len(html), "bytes")
