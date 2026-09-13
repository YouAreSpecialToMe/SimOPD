#!/usr/bin/env python3
"""Build the eval-evolution page: one MATH500 problem, its answer at every checkpoint."""
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "eval_data.json"
out = sys.argv[2] if len(sys.argv) > 2 else "traj_eval.html"
data = open(src, encoding="utf-8").read()

HTML = r"""<!doctype html>
<html lang="zh"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Same Problem, Every Checkpoint</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#f4f5f7; --panel:#ffffff; --panel2:#eceef2; --ink:#1b2028; --dim:#5d6673;
  --rule:#d9dde4; --right:#2b7a86; --wrong:#b4562a; --accent:#2b7a86; --warn:#a8791f;
  --sans:"IBM Plex Sans",system-ui,sans-serif;
  --cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#11151b; --panel:#171c24; --panel2:#1e242e; --ink:#dee3ea; --dim:#8d97a6;
  --rule:#252c37; --right:#4fb3c0; --wrong:#e08050; --accent:#4fb3c0; --warn:#d0a44a;
}}
:root[data-theme="dark"]{
  --ground:#11151b; --panel:#171c24; --panel2:#1e242e; --ink:#dee3ea; --dim:#8d97a6;
  --rule:#252c37; --right:#4fb3c0; --wrong:#e08050; --accent:#4fb3c0; --warn:#d0a44a;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.55;margin:0}
.wrap{display:grid;grid-template-columns:288px minmax(0,1fr);min-height:100vh}
.rail{border-right:1px solid var(--rule);background:var(--panel);padding:15px 12px;display:flex;flex-direction:column;gap:13px;max-height:100vh;position:sticky;top:0}
.main{padding:16px 22px 60px;display:flex;flex-direction:column;gap:15px;min-width:0}
h1{font-size:16px;font-weight:600;margin:0}
.sub{color:var(--dim);font-size:11.5px;margin-top:3px}
.sub a{color:var(--accent)}
.lab{font-family:var(--cond);font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);margin-bottom:6px}
button{font:inherit;color:inherit;background:none;border:1px solid transparent;cursor:pointer;border-radius:3px}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
input[type=search]{width:100%;padding:5px 8px;font:inherit;font-size:12px;color:inherit;background:var(--panel2);border:1px solid var(--rule);border-radius:3px}
.filters{display:flex;gap:4px;flex-wrap:wrap}
.f{font-family:var(--cond);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;padding:4px 8px;border-color:var(--rule);background:var(--panel2);color:var(--dim)}
.f[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.plist{overflow-y:auto;flex:1;display:flex;flex-direction:column;gap:1px;margin:0 -4px;padding:0 4px}
.p{display:grid;grid-template-columns:38px 1fr;gap:7px;align-items:center;padding:5px 6px;text-align:left;border-color:transparent;width:100%}
.p:hover{background:var(--panel2)}
.p[aria-pressed="true"]{background:var(--accent);color:#fff}
.p[aria-pressed="true"] .pq,.p[aria-pressed="true"] .pn{color:#fff}
.pn{font-family:var(--mono);font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums}
.pq{font-size:11.5px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.strip{display:flex;gap:1.5px;margin-top:2px}
.dot{width:7px;height:7px;border-radius:1px;background:var(--rule)}
.dot.r{background:var(--right)} .dot.w{background:var(--wrong)}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:14px 16px}
.q{font-size:14.5px;line-height:1.65;white-space:pre-wrap}
.meta{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--mono);font-size:12px;color:var(--dim);margin-top:9px;font-variant-numeric:tabular-nums}
.meta b{font-family:var(--cond);font-size:10px;letter-spacing:.07em;text-transform:uppercase;font-weight:500;margin-right:5px}
#chart{width:100%;height:104px;display:block}
.ans{border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.ans+.ans{margin-top:8px}
.ah{display:flex;gap:12px;align-items:center;width:100%;padding:7px 11px;background:var(--panel2);text-align:left;border-radius:0}
.ah:hover{background:var(--rule)}
.ah .st{font-family:var(--mono);font-size:12.5px;font-weight:500;font-variant-numeric:tabular-nums;min-width:62px}
.ah .vb{font-family:var(--cond);font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:1px 6px;border-radius:2px;color:#fff}
.ah .ln{font-family:var(--mono);font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums}
.ah .pv{font-size:11.5px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.ab{padding:11px 13px;font-family:var(--mono);font-size:12.5px;line-height:1.75;white-space:pre-wrap;word-break:break-word;max-height:62vh;overflow-y:auto}
.note{font-size:11.5px;color:var(--dim);max-width:72ch}
code{font-family:var(--mono);font-size:11.5px;background:var(--panel2);padding:1px 4px;border-radius:2px}
@media (max-width:860px){.wrap{grid-template-columns:1fr}.rail{position:static;max-height:none;border-right:none;border-bottom:1px solid var(--rule)}.plist{max-height:40vh}}
</style>
</head>
<body>

<div class="wrap">
<aside class="rail">
  <div>
    <h1>同题演变</h1>
    <div class="sub">在环评测 · 500 道 MATH500 每 25 步各答一次<br>
      <a href="traj-viewer.html">← 逐 token 训练信号</a></div>
  </div>
  <div>
    <div class="lab">筛选 <span id="cnt" style="letter-spacing:0;text-transform:none"></span></div>
    <div class="filters" id="filters"></div>
    <input type="search" id="q" placeholder="搜题干" autocomplete="off" style="margin-top:6px">
  </div>
  <div class="plist" id="plist"></div>
</aside>

<main class="main">
  <div class="card">
    <div class="lab">题目</div>
    <div class="q" id="qtext">从左边选一道题。</div>
    <div class="meta" id="qmeta"></div>
  </div>
  <div class="card">
    <div class="lab">逐步:对错与输出长度</div>
    <canvas id="chart"></canvas>
    <div class="note" style="margin-top:8px">
      柱=输出字符数,青=答对、锈=答错。贪心解码(τ=0),与训练 rollout(τ=1)不是同一个分布。
    </div>
  </div>
  <div class="card">
    <div class="lab" id="anslab">各步的答案</div>
    <div id="answers"></div>
  </div>
</main>
</div>

<script type="application/json" id="DATA">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('DATA').textContent);
const ARM=Object.keys(D.arms)[0];
const A=D.arms[ARM], STEPS=A.steps, PROBS=A.probs;
const PBASE='eval-data/';
const cache=new Map();
let mode='flip', query='', cur=null, tok=0;

const el=i=>document.getElementById(i);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

function kind(p){
  const v=p.acc.filter(a=>a!==null&&a!==undefined);
  if(!v.length)return 'none';
  const s=new Set(v);
  if(s.size>1)return 'flip';
  return v[0]===1?'right':'wrong';
}
const FILTERS=[['flip','翻转过'],['right','全程对'],['wrong','全程错'],['all','全部']];

function shown(){
  const q=query.trim().toLowerCase();
  return PROBS.filter(p=>(mode==='all'||kind(p)===mode) &&
    (!q||p.q.toLowerCase().indexOf(q)>=0||String(p.i)===q));
}

function renderFilters(){
  el('filters').innerHTML=FILTERS.map(f=>
    '<button class="f" data-m="'+f[0]+'" aria-pressed="'+(f[0]===mode)+'">'+f[1]+'</button>').join('');
  el('filters').querySelectorAll('.f').forEach(b=>b.onclick=()=>{
    mode=b.dataset.m;renderFilters();renderList();});
}

function renderList(){
  const list=shown();
  el('cnt').textContent='('+list.length+'/'+PROBS.length+')';
  el('plist').innerHTML=list.slice(0,400).map(p=>{
    const dots=p.acc.map(a=>'<span class="dot '+(a===1?'r':(a===0?'w':''))+'"></span>').join('');
    return '<button class="p" data-i="'+p.i+'" aria-pressed="'+(cur===p.i)+'">'+
      '<span class="pn">#'+p.i+'</span>'+
      '<span><span class="pq">'+esc(p.q.slice(0,64))+'</span><span class="strip">'+dots+'</span></span>'+
      '</button>';}).join('');
  el('plist').querySelectorAll('.p').forEach(b=>b.onclick=()=>select(+b.dataset.i));
}

function drawChart(p){
  const cv=el('chart'),w=cv.clientWidth||600,h=104;
  cv.width=w*devicePixelRatio;cv.height=h*devicePixelRatio;
  const x=cv.getContext('2d');x.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  x.clearRect(0,0,w,h);
  const hi=Math.max.apply(null,p.len.concat([1])),bw=(w-10)/p.len.length,base=h-17;
  p.len.forEach((L,i)=>{
    const px=5+i*bw,ht=(L/hi)*(base-14);
    x.fillStyle=p.acc[i]===1?css('--right'):css('--wrong');x.globalAlpha=.85;
    x.fillRect(px+3,base-ht,bw-6,Math.max(1,ht));x.globalAlpha=1;
    x.fillStyle=css('--ink');x.font='10px '+css('--mono');x.textAlign='center';
    x.fillText(L>=1000?(L/1000).toFixed(1)+'k':String(L),px+bw/2,base-ht-4);
    x.fillStyle=css('--dim');x.font='10px '+css('--cond');
    x.fillText(String(STEPS[i]),px+bw/2,h-4);});
  x.textAlign='left';
}

async function select(i){
  cur=i;renderList();
  const p=PROBS.find(z=>z.i===i);if(!p)return;
  el('qtext').textContent=p.q;
  el('qmeta').innerHTML='<span><b>题号</b>#'+p.i+'</span>'+
    '<span><b>标准答案</b>'+esc(p.gt)+'</span>'+
    '<span><b>答对</b>'+p.acc.filter(a=>a===1).length+'/'+p.acc.length+' 步</span>'+
    '<span><b>长度</b>'+p.len[0]+' → '+p.len[p.len.length-1]+' 字符</span>';
  drawChart(p);

  const mine=++tok;
  el('answers').innerHTML='<span class="note">载入各步答案…</span>';
  let full=cache.get(i);
  if(!full){
    try{
      const res=await fetch(PBASE+ARM+'__'+i+'.json');
      if(!res.ok)throw new Error('HTTP '+res.status);
      full=await res.json();cache.set(i,full);
    }catch(e){
      if(mine!==tok)return;
      el('answers').innerHTML='<span class="note">取不到答案文件 ('+esc(e.message)+
        ')。本页需要从本地服务器打开(<code>python3 -m http.server</code>),file:// 会被同源策略挡住。</span>';
      return;}
  }
  if(mine!==tok)return;
  el('anslab').textContent='各步的答案 · '+STEPS.length+' 个检查点';
  el('answers').innerHTML=STEPS.map((s,k)=>{
    const a=full.answers[String(s)]||'';
    const ok=full.acc[k]===1;
    const prev=a.replace(/\s+/g,' ').slice(0,110);
    return '<div class="ans"><button class="ah" data-k="'+k+'">'+
      '<span class="st">step '+s+'</span>'+
      '<span class="vb" style="background:'+(ok?'var(--right)':'var(--wrong)')+'">'+(ok?'对':'错')+'</span>'+
      '<span class="ln">'+a.length+' 字符</span>'+
      '<span class="pv">'+esc(prev)+'</span></button>'+
      '<div class="ab" id="ab'+k+'" hidden>'+esc(a)+'</div></div>';}).join('');
  el('answers').querySelectorAll('.ah').forEach(b=>b.onclick=()=>{
    const d=el('ab'+b.dataset.k);d.hidden=!d.hidden;});
}

el('q').oninput=e=>{query=e.target.value;renderList();};
addEventListener('resize',()=>{if(cur!==null){const p=PROBS.find(z=>z.i===cur);if(p)drawChart(p);}});

renderFilters();renderList();
const first=shown()[0];if(first)select(first.i);
</script>
</body></html>
"""

open(out, "w", encoding="utf-8").write(HTML.replace("__DATA__", data))
print("wrote", out, len(HTML) + len(data), "bytes")
