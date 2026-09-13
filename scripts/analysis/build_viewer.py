#!/usr/bin/env python3
"""Build the trajectory viewer HTML from viewer_data.json (standalone document)."""
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "viewer_data.json"
out = sys.argv[2] if len(sys.argv) > 2 else "traj_viewer.html"
data = open(src, encoding="utf-8").read()

HTML = r"""<!doctype html>
<html lang="zh"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token Signal Scope</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#f4f5f7; --panel:#ffffff; --panel2:#eceef2; --ink:#1b2028; --dim:#5d6673;
  --rule:#d9dde4; --hot:#b4562a; --cold:#2b7a86; --mark:#6b5bc4; --accent:#2b7a86;
  --ref:#9aa3b0;
  --sans:"IBM Plex Sans",system-ui,sans-serif;
  --cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#11151b; --panel:#171c24; --panel2:#1e242e; --ink:#dee3ea; --dim:#8d97a6;
  --rule:#252c37; --hot:#e08050; --cold:#4fb3c0; --mark:#9b8ce8; --accent:#4fb3c0;
  --ref:#59626f;
}}
:root[data-theme="dark"]{
  --ground:#11151b; --panel:#171c24; --panel2:#1e242e; --ink:#dee3ea; --dim:#8d97a6;
  --rule:#252c37; --hot:#e08050; --cold:#4fb3c0; --mark:#9b8ce8; --accent:#4fb3c0;
  --ref:#59626f;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5;margin:0}
.wrap{display:grid;grid-template-columns:232px minmax(0,1fr);min-height:100vh}
.rail{border-right:1px solid var(--rule);background:var(--panel);padding:16px 13px;display:flex;flex-direction:column;gap:16px;max-height:100vh;overflow-y:auto;position:sticky;top:0}
.main{padding:16px 20px 48px;display:flex;flex-direction:column;gap:16px;min-width:0}
h1{font-size:16px;font-weight:600;margin:0;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:11.5px;margin-top:3px}
.lab{font-family:var(--cond);font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);margin-bottom:7px}
.btns{display:flex;flex-direction:column;gap:2px}
.chips{display:flex;flex-wrap:wrap;gap:4px}
button{font:inherit;color:inherit;background:none;border:1px solid transparent;cursor:pointer;border-radius:3px}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
input[type=search]{width:100%;padding:5px 8px;font:inherit;font-size:12px;color:inherit;background:var(--panel2);border:1px solid var(--rule);border-radius:3px}
.axgrp{margin-bottom:7px}
.axname{font-family:var(--cond);font-size:10px;letter-spacing:.1em;color:var(--dim);margin:0 0 3px 2px}
.arm{text-align:left;padding:4px 7px;font-size:12px;border-color:transparent;font-family:var(--mono);width:100%}
.arm:hover{background:var(--panel2)}
.arm[aria-pressed="true"]{background:var(--accent);color:#fff}
.step{font-family:var(--mono);font-size:11.5px;padding:3px 6px;border-color:var(--rule);background:var(--panel2);font-variant-numeric:tabular-nums}
.step:hover{border-color:var(--accent)}
.step[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:13px 15px}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
.series{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:13px;flex:1 1 420px}
.spark{display:flex;flex-direction:column;gap:2px}
.spark canvas{width:100%;height:32px;display:block}
.sv{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums}
.prof{flex:0 0 210px}
#profc{width:100%;height:108px;display:block}
.seqrow{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.seq{padding:4px 8px;border-color:var(--rule);background:var(--panel2);font-size:12px;display:flex;gap:6px;align-items:baseline}
.seq:hover{border-color:var(--accent)}
.seq[aria-pressed="true"]{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
.seq b{font-family:var(--mono);font-weight:500;font-variant-numeric:tabular-nums}
.seq i{font-style:normal;color:var(--dim);font-size:11px}
.badge{font-family:var(--cond);font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:1px 5px;border-radius:2px;background:var(--hot);color:#fff}
.jump{font-family:var(--cond);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;padding:4px 8px;border-color:var(--rule);background:var(--panel2);color:var(--dim)}
.jump:hover{border-color:var(--accent);color:var(--ink)}
#trace{width:100%;height:92px;display:block}
.stream{font-family:var(--mono);font-size:13px;line-height:2.05;white-space:pre-wrap;word-break:break-word;overflow-x:auto}
.tok{border-radius:2px;padding:1px 0}
.tok.hotstop{box-shadow:0 -2px 0 var(--mark) inset,0 2px 0 var(--mark) inset}
.tok.dis{border-bottom:1.5px dotted var(--hot)}
.tok.flash{outline:2px solid var(--accent);outline-offset:1px}
.alt{color:var(--mark);font-size:10.5px;opacity:.85}
.gap{display:block;margin:13px 0;border-top:1px dashed var(--rule);color:var(--dim);font-family:var(--cond);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;padding-top:5px}
.readout{position:sticky;bottom:0;background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:8px 12px;display:flex;gap:18px;flex-wrap:wrap;font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums}
.readout span b{font-weight:500;color:var(--dim);font-family:var(--cond);font-size:10px;letter-spacing:.07em;text-transform:uppercase;margin-right:5px}
.legend{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:11.5px;color:var(--dim)}
.ramp{width:120px;height:9px;border-radius:2px;background:linear-gradient(90deg,var(--cold),var(--panel2) 50%,var(--hot))}
.note{font-size:11.5px;color:var(--dim);max-width:70ch}
code{font-family:var(--mono);font-size:11.5px;background:var(--panel2);padding:1px 4px;border-radius:2px}
label.tgl{display:flex;gap:5px;align-items:center;font-size:11.5px;color:var(--dim);cursor:pointer}
@media (max-width:860px){.wrap{grid-template-columns:1fr}.rail{position:static;max-height:none;border-right:none;border-bottom:1px solid var(--rule)}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style>
</head>
<body>

<div class="wrap">
<aside class="rail">
  <div>
    <h1>Token Signal Scope</h1>
    <div class="sub">SimOPD 2026-09 · 逐 token 训练信号</div>
  </div>
  <div>
    <div class="lab">臂 <span id="armcount" style="letter-spacing:0;text-transform:none"></span></div>
    <input type="search" id="filter" placeholder="筛选,如 c2 / union / seg" autocomplete="off">
    <div class="btns" id="arms" style="margin-top:6px"></div>
  </div>
  <div><div class="lab">训练步</div><div class="chips" id="steps"></div></div>
  <div class="note">
    颜色 = <code>r</code> = log π<sub>θ</sub> − log π<sub>T</sub>,取采样列的原始值。<br>
    损失是全批 <code>r</code> 的均值,每个 token 被压制的力度正比于它的 <code>r</code>。<br>
    键盘:← → 换步,↑ ↓ 换臂。
  </div>
</aside>

<main class="main">
  <div class="card">
    <div class="row">
      <div style="flex:1 1 420px;min-width:0">
        <div class="lab">该臂逐步 <span id="refnote" style="letter-spacing:0;text-transform:none"></span></div>
        <div class="series" id="series"></div>
      </div>
      <div class="prof">
        <div class="lab">r 随位置(本步)</div>
        <canvas id="profc"></canvas>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="lab">序列</div>
    <div class="seqrow" id="seqs"></div>
    <div class="seqrow" id="jumps" style="margin-top:9px"></div>
    <div style="margin-top:12px"><div class="lab">整序列信号 · 横轴=位置</div><canvas id="trace"></canvas></div>
    <div class="legend" style="margin-top:8px">
      <span style="display:flex;gap:6px;align-items:center"><span class="ramp"></span>
        <span style="font-family:var(--mono)">r −3 … 0 … +8</span></span>
      <span>青=被强化 · 锈=被压制</span>
      <span style="color:var(--mark)">▮ 教师想在此结束</span>
      <span>▯ 字节片段</span>
    </div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:baseline;gap:14px;flex-wrap:wrap">
      <div class="lab" id="streamlab" style="margin:0">Token 流</div>
      <label class="tgl"><input type="checkbox" id="showalt"> 显示教师想要的 token(标记处)</label>
    </div>
    <div class="stream" id="stream" style="margin-top:9px"></div>
  </div>

  <div class="readout" id="readout">
    <span><b>token</b><span id="r-tok">—</span></span>
    <span><b>位置</b><span id="r-pos">—</span></span>
    <span><b>r</b><span id="r-r">—</span></span>
    <span><b>熵</b><span id="r-e">—</span></span>
    <span><b>q(im_end)</b><span id="r-q">—</span></span>
    <span><b>教师想要</b><span id="r-a">—</span></span>
  </div>
</main>
</div>

<script type="application/json" id="DATA">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('DATA').textContent);
const SEQBASE=(D.meta&&D.meta.seqdir)||'traj-data/';
const seqCache=new Map();
let streamToken=0;                       // guards against out-of-order fetches
const armNames=Object.keys(D.arms);
const REF='vanilla_corr';
let curArm=armNames.includes(REF)?REF:armNames[0], curStep=null, curSeq=0, filter='';

const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el=id=>document.getElementById(id);
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const attr=s=>esc(s).replace(/"/g,'&quot;');

function hex2rgb(h){h=h.replace('#','');if(h.length===3)h=h.split('').map(c=>c+c).join('');
  return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}
function mix(a,b,t){return a.map((v,i)=>Math.round(v+(b[i]-v)*t));}

function sigColor(r){
  if(r===null||r===undefined) return 'transparent';
  const base=hex2rgb(css('--panel2'));
  if(r>=0){const t=Math.min(1,Math.tanh(r/3.2));const m=mix(base,hex2rgb(css('--hot')),t);
    return `rgba(${m[0]},${m[1]},${m[2]},${0.16+0.84*t})`;}
  const t=Math.min(1,Math.tanh(-r/1.6));const m=mix(base,hex2rgb(css('--cold')),t);
  return `rgba(${m[0]},${m[1]},${m[2]},${0.16+0.84*t})`;
}

function dpi(cv,h){const w=cv.clientWidth||300;cv.width=w*devicePixelRatio;cv.height=h*devicePixelRatio;
  const x=cv.getContext('2d');x.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);return [x,w,h];}

function axisOf(n){const m=/^([a-z])\d/.exec(n);return m?m[1].toUpperCase():(n.indexOf('vanilla')===0?'对照':'其他');}

function sparkline(cv,vals,mark,ref){
  const [x,w,h]=dpi(cv,32);
  const pool=vals.concat(ref||[]).filter(z=>z!==null&&z!==undefined);
  if(!pool.length)return;
  const lo=Math.min.apply(null,pool),hi=Math.max.apply(null,pool),sp=(hi-lo)||1;
  const py=z=>h-3-((z-lo)/sp)*(h-8);
  x.clearRect(0,0,w,h);
  x.strokeStyle=css('--rule');x.lineWidth=1;x.beginPath();x.moveTo(0,h-.5);x.lineTo(w,h-.5);x.stroke();
  const line=(arr,col,wd)=>{x.strokeStyle=col;x.lineWidth=wd;x.beginPath();
    arr.forEach((z,i)=>{if(z===null||z===undefined)return;
      const px=arr.length<2?0:i/(arr.length-1)*(w-2)+1;
      i===0?x.moveTo(px,py(z)):x.lineTo(px,py(z));});x.stroke();};
  if(ref&&ref.length)line(ref,css('--ref'),1);
  line(vals,css('--accent'),1.6);
  if(mark>=0&&vals.length>1){const px=mark/(vals.length-1)*(w-2)+1;
    x.strokeStyle=css('--ink');x.globalAlpha=.3;x.beginPath();x.moveTo(px,0);x.lineTo(px,h);x.stroke();
    x.globalAlpha=1;const z=vals[mark];
    if(z!==null&&z!==undefined){x.fillStyle=css('--accent');x.beginPath();x.arc(px,py(z),2.6,0,7);x.fill();}}
}

function drawProfile(){
  const cv=el('profc');const [x,w,h]=dpi(cv,108);
  x.clearRect(0,0,w,h);
  const p=(D.arms[curArm].prof||{})[String(curStep)];
  if(!p){x.fillStyle=css('--dim');x.font='11px '+css('--sans');x.fillText('本步无逐 token 样本',4,20);return;}
  const labs=['0–100','100–500','.5–2k','2–8k','8k+'],vals=p.bins,mu=p.mean;
  const pool=vals.filter(v=>v!==null).concat(mu===null?[]:[mu]);
  if(!pool.length)return;
  const hi=Math.max.apply(null,pool.concat([0.01])),bw=(w-8)/vals.length,base=h-16;
  vals.forEach((v,i)=>{
    const px=4+i*bw;
    if(v!==null){const ht=Math.max(0,(v/hi)*(base-12));
      x.fillStyle=css('--hot');x.globalAlpha=.78;x.fillRect(px+3,base-ht,bw-6,ht);x.globalAlpha=1;
      x.fillStyle=css('--ink');x.font='10px '+css('--mono');x.textAlign='center';
      x.fillText(v.toFixed(2),px+bw/2,base-ht-3);}
    x.fillStyle=css('--dim');x.font='9px '+css('--cond');x.textAlign='center';
    x.fillText(labs[i],px+bw/2,h-4);});
  x.textAlign='left';
  if(mu!==null){const y=base-(mu/hi)*(base-12);
    x.strokeStyle=css('--accent');x.setLineDash([3,3]);x.lineWidth=1;
    x.beginPath();x.moveTo(0,y);x.lineTo(w,y);x.stroke();x.setLineDash([]);
    x.fillStyle=css('--accent');x.font='9px '+css('--mono');x.fillText('批均值 '+mu.toFixed(2),4,y-3);}
}

function drawTrace(s){
  const cv=el('trace');const [x,w,h]=dpi(cv,92);
  x.clearRect(0,0,w,h);
  const r=s.trace.r,q=s.trace.q,n=r.length,zero=h*0.62;
  for(let i=0;i<n;i++){const v=r[i];if(v===null)continue;
    const px=i/n*w,bw=Math.max(1,w/n);
    const t=v>=0?Math.min(1,Math.tanh(v/3.2)):-Math.min(1,Math.tanh(-v/1.6));
    const ht=t*(v>=0?zero-4:h-zero-4);
    x.fillStyle=v>=0?css('--hot'):css('--cold');x.globalAlpha=.72;
    x.fillRect(px,zero-ht,bw,Math.abs(ht)||0.6);}
  x.globalAlpha=1;
  x.strokeStyle=css('--rule');x.lineWidth=1;x.beginPath();x.moveTo(0,zero+.5);x.lineTo(w,zero+.5);x.stroke();
  x.strokeStyle=css('--mark');x.lineWidth=1.4;
  for(let i=0;i<n;i++){const v=q[i];if(v===null||v<0.05)continue;
    const px=i/n*w,ht=Math.min(1,v)*(zero-6);
    x.globalAlpha=.85;x.beginPath();x.moveTo(px,zero);x.lineTo(px,zero-ht);x.stroke();}
  x.globalAlpha=1;
  // mark the positions the jump buttons target
  const flag=(p,col)=>{if(p===null||p===undefined||!s.len)return;
    const px=p/s.len*w;
    x.strokeStyle=col;x.lineWidth=1;x.globalAlpha=.9;x.setLineDash([2,2]);
    x.beginPath();x.moveTo(px,0);x.lineTo(px,h-12);x.stroke();
    x.setLineDash([]);x.globalAlpha=1;};
  flag(s.rmax,css('--hot')); flag(s.rmin,css('--cold'));
  if(s.hotq>0.5)flag(s.hot,css('--mark'));
  x.fillStyle=css('--dim');x.font='10px '+css('--mono');
  x.fillText('0',2,zero-3);x.fillText('位置 0',2,h-2);
  x.textAlign='right';x.fillText(String(s.len),w-2,h-2);x.textAlign='left';
}

function renderSeries(){
  const a=D.arms[curArm],idx=a.steps.indexOf(curStep);
  const ref=(curArm!==REF&&D.arms[REF])?D.arms[REF]:null;
  el('refnote').textContent=ref?'(灰线 = vanilla_corr)':'';
  const defs=[['len','平均长度',v=>v.toFixed(0)],['trunc','截断率',v=>v.toFixed(2)],
              ['stop','停止率',v=>v.toFixed(2)],['dl_last','停止位 r',v=>v.toFixed(1)],
              ['rep4','rep4',v=>v.toFixed(2)],['q_ime','教师 q(结束)',v=>v.toFixed(3)]];
  el('series').innerHTML=defs.map(d=>
    `<div class="spark"><div class="lab" style="margin:0">${d[1]}</div>
     <canvas data-k="${d[0]}"></canvas><div class="sv" id="sv-${d[0]}"></div></div>`).join('');
  defs.forEach(d=>{
    sparkline(el('series').querySelector('canvas[data-k="'+d[0]+'"]'),a.series[d[0]],idx,ref?ref.series[d[0]]:null);
    const v=a.series[d[0]][idx];
    el('sv-'+d[0]).textContent=(v===null||v===undefined)?'—':d[2](v);});
  drawProfile();
}

async function renderStream(){
  const s=D.arms[curArm].samples[String(curStep)][curSeq];
  drawTrace(s);
  const mine=++streamToken;
  el('streamlab').textContent='Token 流 · 序列 #'+s.seq+' · '+s.len+' token';
  el('jumps').innerHTML='';

  let full=seqCache.get(s.key);
  if(!full){
    el('stream').innerHTML='<span class="note">载入整条序列…</span>';
    try{
      const res=await fetch(SEQBASE+encodeURIComponent(s.key)+'.json');
      if(!res.ok) throw new Error('HTTP '+res.status);
      full=await res.json();
      seqCache.set(s.key,full);
    }catch(err){
      if(mine!==streamToken)return;
      el('stream').innerHTML='<span class="note">取不到整条序列 ('+esc(err.message)+
        ')。这份页面需要从本地服务器打开(<code>python3 -m http.server</code>),'+
        '直接用 file:// 双击会被浏览器的同源策略挡住。</span>';
      return;
    }
  }
  if(mine!==streamToken)return;

  const showAlt=el('showalt').checked, toks=full.toks, extra=full.extra||{};
  const out=new Array(toks.length);
  for(let j=0;j<toks.length;j++){
    const txt=toks[j][0], r=toks[j][1], ex=extra[j];
    const alt=ex?ex[2]:null, q=ex?ex[1]:null;
    const dis=!!alt;
    const cls='tok'+((q!==null&&q!==undefined&&q>0.5)?' hotstop':'')+(dis?' dis':'');
    const safe=esc(txt);
    const body=(showAlt&&dis)?safe+'<span class="alt">\u2192'+esc(alt)+'</span>':safe;
    // a 16k-token response makes 16k spans, so only emit attributes that carry
    // something; the empty ones cost ~40 chars each across the whole stream
    out[j]='<span class="'+cls+'" style="background:'+sigColor(r)+'" id="p'+j+
      '" data-p="'+j+'" data-r="'+r+
      (ex&&ex[0]!==null?'" data-e="'+ex[0]:'')+
      (q!==null&&q!==undefined?'" data-q="'+q:'')+
      (alt?'" data-a="'+attr(alt):'')+
      '" data-t="'+attr(txt)+'">'+body+'</span>';
  }
  el('stream').innerHTML=out.join('');

  const j=[];
  const mk=(p,lab)=>{if(p!==null&&p!==undefined&&p>=0&&p<toks.length)
    j.push('<button class="jump" data-j="'+p+'">'+lab+'</button>');};
  mk(0,'开头');
  mk(s.rmax,'最受压制 r='+s.rmaxv);
  mk(s.rmin,'最受强化 r='+s.rminv);
  if(s.hotq>0.5) mk(s.hot,'教师想结束 @'+s.hot);
  mk(toks.length-1,'结尾');
  el('jumps').innerHTML=j.join('');
  el('jumps').querySelectorAll('.jump').forEach(b=>b.onclick=()=>{
    const t=el('p'+b.dataset.j);if(!t)return;
    t.scrollIntoView({block:'center',behavior:'smooth'});
    t.classList.add('flash');setTimeout(()=>t.classList.remove('flash'),1600);});
}

function renderSeqs(){
  const arr=D.arms[curArm].samples[String(curStep)]||[];
  if(curSeq>=arr.length)curSeq=0;
  el('seqs').innerHTML=arr.length?arr.map((s,i)=>
    '<button class="seq" data-i="'+i+'" aria-pressed="'+(i===curSeq)+'">'+
    '<b>#'+s.seq+'</b><i>'+s.len+' tok</i>'+
    (s.trunc?'<span class="badge">截断</span>':'')+
    (s.score===1?'<i style="color:var(--cold)">✓</i>':(s.score===0?'<i>✗</i>':''))+
    '</button>').join(''):'<span class="note">该步无整序列样本</span>';
  el('seqs').querySelectorAll('.seq').forEach(b=>b.onclick=()=>{
    curSeq=+b.dataset.i;renderSeqs();renderStream();});
  if(arr.length){renderStream();}else{el('stream').innerHTML='';el('jumps').innerHTML='';}
}

function renderSteps(){
  const a=D.arms[curArm];
  if(a.steps.indexOf(curStep)<0)curStep=a.steps[Math.min(1,a.steps.length-1)];
  el('steps').innerHTML=a.steps.map(s=>
    '<button class="step" data-s="'+s+'" aria-pressed="'+(s===curStep)+'">'+s+'</button>').join('');
  el('steps').querySelectorAll('.step').forEach(b=>b.onclick=()=>{
    curStep=+b.dataset.s;curSeq=0;renderSteps();renderSeries();renderSeqs();});
}

function visibleArms(){
  const f=filter.trim().toLowerCase();
  return armNames.filter(n=>!f||n.toLowerCase().indexOf(f)>=0);
}

function renderArms(){
  const vis=visibleArms();
  el('armcount').textContent='('+vis.length+'/'+armNames.length+')';
  const groups={};
  vis.forEach(n=>{const a=axisOf(n);(groups[a]=groups[a]||[]).push(n);});
  el('arms').innerHTML=Object.keys(groups).sort().map(ax=>
    '<div class="axgrp"><div class="axname">'+ax+'</div>'+
    groups[ax].map(n=>'<button class="arm" data-a="'+n+'" aria-pressed="'+(n===curArm)+'">'+n+'</button>').join('')+
    '</div>').join('');
  el('arms').querySelectorAll('.arm').forEach(b=>b.onclick=()=>{
    curArm=b.dataset.a;curStep=null;curSeq=0;
    renderArms();renderSteps();renderSeries();renderSeqs();});
}

el('stream').addEventListener('mousemove',ev=>{
  const t=ev.target.closest('.tok');if(!t)return;
  el('r-tok').textContent=JSON.stringify(t.dataset.t);
  el('r-pos').textContent=t.dataset.p;
  el('r-r').textContent=t.dataset.r;
  el('r-e').textContent=t.dataset.e||'—';
  el('r-q').textContent=t.dataset.q||'—';
  el('r-a').textContent=t.dataset.a?JSON.stringify(t.dataset.a):'(与学生相同)';
});
el('showalt').onchange=()=>{const arr=D.arms[curArm].samples[String(curStep)];if(arr&&arr.length)renderStream();};
el('filter').oninput=e=>{filter=e.target.value;renderArms();};

addEventListener('keydown',ev=>{
  if(/^(INPUT|TEXTAREA)$/.test(ev.target.tagName))return;
  const a=D.arms[curArm],si=a.steps.indexOf(curStep),vis=visibleArms(),ai=vis.indexOf(curArm);
  let hit=true;
  if(ev.key==='ArrowRight'&&si<a.steps.length-1)curStep=a.steps[si+1];
  else if(ev.key==='ArrowLeft'&&si>0)curStep=a.steps[si-1];
  else if(ev.key==='ArrowDown'&&ai>=0&&ai<vis.length-1){curArm=vis[ai+1];curStep=null;}
  else if(ev.key==='ArrowUp'&&ai>0){curArm=vis[ai-1];curStep=null;}
  else hit=false;
  if(hit){ev.preventDefault();curSeq=0;renderArms();renderSteps();renderSeries();renderSeqs();}
});

addEventListener('resize',()=>{renderSeries();
  const arr=D.arms[curArm].samples[String(curStep)];if(arr&&arr[curSeq])drawTrace(arr[curSeq]);});

renderArms();renderSteps();renderSeries();renderSeqs();
</script>
</body></html>
"""

open(out, "w", encoding="utf-8").write(HTML.replace("__DATA__", data))
print("wrote", out, len(HTML) + len(data), "bytes")
