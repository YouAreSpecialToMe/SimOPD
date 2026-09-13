#!/usr/bin/env python3
"""Build the trajectory viewer HTML from viewer_data.json."""
import json
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "viewer_data.json"
out = sys.argv[2] if len(sys.argv) > 2 else "traj_viewer.html"
data = open(src, encoding="utf-8").read()

HTML = r"""<title>Token Signal Scope</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#f4f5f7; --panel:#ffffff; --panel2:#eceef2; --ink:#1b2028; --dim:#5d6673;
  --rule:#d9dde4; --hot:#b4562a; --cold:#2b7a86; --mark:#6b5bc4; --accent:#2b7a86;
  --sans:"IBM Plex Sans",system-ui,sans-serif;
  --cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#11151b; --panel:#171c24; --panel2:#1e242e; --ink:#dee3ea; --dim:#8d97a6;
  --rule:#252c37; --hot:#e08050; --cold:#4fb3c0; --mark:#9b8ce8; --accent:#4fb3c0;
}}
:root[data-theme="dark"]{
  --ground:#11151b; --panel:#171c24; --panel2:#1e242e; --ink:#dee3ea; --dim:#8d97a6;
  --rule:#252c37; --hot:#e08050; --cold:#4fb3c0; --mark:#9b8ce8; --accent:#4fb3c0;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5}
.wrap{display:grid;grid-template-columns:210px minmax(0,1fr);gap:0;min-height:100vh}
.rail{border-right:1px solid var(--rule);background:var(--panel);padding:18px 14px;display:flex;flex-direction:column;gap:20px}
.main{padding:18px 22px 60px;display:flex;flex-direction:column;gap:18px;min-width:0}
h1{font-size:17px;font-weight:600;margin:0;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:12px;margin-top:3px}
.lab{font-family:var(--cond);font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);margin-bottom:7px}
.btns{display:flex;flex-direction:column;gap:3px}
.chips{display:flex;flex-wrap:wrap;gap:4px}
button{font:inherit;color:inherit;background:none;border:1px solid transparent;cursor:pointer;border-radius:3px}
button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.arm{text-align:left;padding:6px 8px;font-size:12.5px;border-color:var(--rule);background:var(--panel2)}
.arm:hover{border-color:var(--accent)}
.arm[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.step{font-family:var(--mono);font-size:12px;padding:4px 7px;border-color:var(--rule);background:var(--panel2);font-variant-numeric:tabular-nums}
.step:hover{border-color:var(--accent)}
.step[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:14px 16px}
.series{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:14px}
.spark{display:flex;flex-direction:column;gap:3px}
.spark canvas{width:100%;height:34px;display:block}
.sv{font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums}
.seqrow{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.seq{padding:5px 9px;border-color:var(--rule);background:var(--panel2);font-size:12px;display:flex;gap:7px;align-items:baseline}
.seq:hover{border-color:var(--accent)}
.seq[aria-pressed="true"]{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
.seq b{font-family:var(--mono);font-weight:500;font-variant-numeric:tabular-nums}
.seq i{font-style:normal;color:var(--dim);font-size:11px}
.badge{font-family:var(--cond);font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:1px 5px;border-radius:2px;background:var(--hot);color:#fff}
#trace{width:100%;height:96px;display:block}
.stream{font-family:var(--mono);font-size:13px;line-height:1.95;white-space:pre-wrap;word-break:break-word;overflow-x:auto}
.tok{border-radius:2px;padding:1px 0}
.tok.hotstop{box-shadow:0 -2px 0 var(--mark) inset,0 2px 0 var(--mark) inset}
.gap{display:block;margin:14px 0;border-top:1px dashed var(--rule);color:var(--dim);font-family:var(--cond);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;padding-top:5px}
.readout{position:sticky;bottom:0;background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:9px 13px;display:flex;gap:20px;flex-wrap:wrap;font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums}
.readout span b{font-weight:500;color:var(--dim);font-family:var(--cond);font-size:10px;letter-spacing:.07em;text-transform:uppercase;margin-right:5px}
.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:11.5px;color:var(--dim)}
.ramp{width:132px;height:9px;border-radius:2px;background:linear-gradient(90deg,var(--cold),var(--panel2) 50%,var(--hot))}
.note{font-size:12px;color:var(--dim);max-width:68ch}
code{font-family:var(--mono);font-size:12px;background:var(--panel2);padding:1px 4px;border-radius:2px}
@media (max-width:760px){.wrap{grid-template-columns:1fr}.rail{border-right:none;border-bottom:1px solid var(--rule)}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
<aside class="rail">
  <div>
    <h1>Token Signal Scope</h1>
    <div class="sub">SimOPD 2026-09 · 逐 token 训练信号</div>
  </div>
  <div><div class="lab">臂</div><div class="btns" id="arms"></div></div>
  <div><div class="lab">训练步</div><div class="chips" id="steps"></div></div>
  <div class="note" style="font-size:11.5px">
    颜色 = <code>r</code> = log π<sub>θ</sub> − log π<sub>T</sub>。<br>
    损失是全批 <code>r</code> 的均值,每个 token 被压制的力度正比于它的 <code>r</code>。
  </div>
</aside>

<main class="main">
  <div class="card">
    <div class="lab">该臂逐步</div>
    <div class="series" id="series"></div>
  </div>

  <div class="card">
    <div class="lab">序列</div>
    <div class="seqrow" id="seqs"></div>
    <div style="margin-top:14px"><div class="lab">整序列信号 · 横轴=位置</div><canvas id="trace"></canvas></div>
    <div class="legend" style="margin-top:9px">
      <span style="display:flex;gap:7px;align-items:center"><span class="ramp"></span>
        <span style="font-family:var(--mono)">r&nbsp;−3 … 0 … +8</span></span>
      <span>青 = 被强化 · 锈 = 被压制</span>
      <span style="color:var(--mark)">▮ 教师认为此处该结束 (q&gt;0.5)</span>
      <span>▯ = 多字节字符被切开的字节片段</span>
    </div>
  </div>

  <div class="card">
    <div class="lab" id="streamlab">Token 流</div>
    <div class="stream" id="stream"></div>
  </div>

  <div class="readout" id="readout">
    <span><b>token</b><span id="r-tok">—</span></span>
    <span><b>位置</b><span id="r-pos">—</span></span>
    <span><b>r</b><span id="r-r">—</span></span>
    <span><b>熵</b><span id="r-e">—</span></span>
    <span><b>q(im_end)</b><span id="r-q">—</span></span>
  </div>
</main>
</div>

<script type="application/json" id="DATA">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('DATA').textContent);
const armNames=Object.keys(D.arms);
let curArm=armNames[0], curStep=null, curSeq=0;

const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el=id=>document.getElementById(id);

function mix(a,b,t){return a.map((v,i)=>Math.round(v+(b[i]-v)*t));}
function hex2rgb(h){h=h.replace('#','');if(h.length===3)h=h.split('').map(c=>c+c).join('');
  return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}

// diverging scale: 0 -> panel2, positive -> hot (clamped near +8), negative -> cold (near -3)
function sigColor(r,alphaBoost){
  if(r===null||r===undefined) return 'transparent';
  const base=hex2rgb(css('--panel2'));
  if(r>=0){const t=Math.min(1,Math.tanh(r/3.2));const c=hex2rgb(css('--hot'));
    const m=mix(base,c,t);return `rgba(${m[0]},${m[1]},${m[2]},${(0.16+0.84*t)*(alphaBoost||1)})`;}
  const t=Math.min(1,Math.tanh(-r/1.6));const c=hex2rgb(css('--cold'));
  const m=mix(base,c,t);return `rgba(${m[0]},${m[1]},${m[2]},${(0.16+0.84*t)*(alphaBoost||1)})`;
}

function dpi(cv,h){const w=cv.clientWidth||300;cv.width=w*devicePixelRatio;cv.height=h*devicePixelRatio;
  const x=cv.getContext('2d');x.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);return [x,w,h];}

function sparkline(cv,vals,mark){
  const [x,w,h]=dpi(cv,34);const v=vals.filter(z=>z!==null);
  if(!v.length)return;const lo=Math.min(...v),hi=Math.max(...v),sp=(hi-lo)||1;
  x.clearRect(0,0,w,h);
  x.strokeStyle=css('--rule');x.lineWidth=1;x.beginPath();x.moveTo(0,h-.5);x.lineTo(w,h-.5);x.stroke();
  x.strokeStyle=css('--accent');x.lineWidth=1.6;x.beginPath();
  vals.forEach((z,i)=>{if(z===null)return;const px=vals.length<2?0:i/(vals.length-1)*(w-2)+1;
    const py=h-3-((z-lo)/sp)*(h-8);i===0?x.moveTo(px,py):x.lineTo(px,py);});
  x.stroke();
  if(mark>=0&&vals.length>1){const px=mark/(vals.length-1)*(w-2)+1;
    x.strokeStyle=css('--ink');x.globalAlpha=.35;x.lineWidth=1;
    x.beginPath();x.moveTo(px,0);x.lineTo(px,h);x.stroke();x.globalAlpha=1;
    const z=vals[mark];if(z!==null){const py=h-3-((z-lo)/sp)*(h-8);
      x.fillStyle=css('--accent');x.beginPath();x.arc(px,py,2.6,0,7);x.fill();}}
}

function drawTrace(s){
  const cv=el('trace');const [x,w,h]=dpi(cv,96);
  x.clearRect(0,0,w,h);
  const r=s.trace.r, q=s.trace.q, n=r.length;
  const zero=h*0.62;
  // r as filled bars around zero
  for(let i=0;i<n;i++){
    const v=r[i];if(v===null)continue;
    const px=i/n*w, bw=Math.max(1,w/n);
    const t=v>=0?Math.min(1,Math.tanh(v/3.2)):-Math.min(1,Math.tanh(-v/1.6));
    const ht=t*(v>=0?zero-4:h-zero-4);
    x.fillStyle=v>=0?css('--hot'):css('--cold');x.globalAlpha=.72;
    x.fillRect(px,zero-ht,bw,Math.abs(ht)||0.6);
  }
  x.globalAlpha=1;
  x.strokeStyle=css('--rule');x.lineWidth=1;
  x.beginPath();x.moveTo(0,zero+.5);x.lineTo(w,zero+.5);x.stroke();
  // teacher terminator mass
  x.strokeStyle=css('--mark');x.lineWidth=1.4;
  for(let i=0;i<n;i++){const v=q[i];if(v===null||v<0.05)continue;
    const px=i/n*w;const ht=Math.min(1,v)*(zero-6);
    x.globalAlpha=.85;x.beginPath();x.moveTo(px,zero);x.lineTo(px,zero-ht);x.stroke();}
  x.globalAlpha=1;
  // window shading
  x.fillStyle=css('--ink');x.globalAlpha=.06;
  s.wins.forEach(win=>{const a=win.o/s.len*w, b=(win.o+win.toks.length)/s.len*w;
    x.fillRect(a,0,Math.max(2,b-a),h);});
  x.globalAlpha=1;
  x.fillStyle=css('--dim');x.font='10px '+css('--mono');
  x.fillText('0',2,zero-3);x.fillText('位置 0',2,h-2);
  x.textAlign='right';x.fillText(String(s.len),w-2,h-2);x.textAlign='left';
}

function renderSeries(){
  const a=D.arms[curArm];const idx=a.steps.indexOf(curStep);
  const defs=[['len','平均长度',v=>v.toFixed(0)],['trunc','截断率',v=>v.toFixed(2)],
              ['stop','停止率',v=>v.toFixed(2)],['dl_last','停止位 r',v=>v.toFixed(1)],
              ['rep4','rep4',v=>v.toFixed(2)],['q_ime','教师 q(结束)',v=>v.toFixed(3)]];
  el('series').innerHTML=defs.map(([k,lab])=>
    `<div class="spark"><div class="lab" style="margin:0">${lab}</div>
     <canvas data-k="${k}"></canvas>
     <div class="sv" id="sv-${k}"></div></div>`).join('');
  defs.forEach(([k,lab,fmt])=>{
    const cv=el('series').querySelector(`canvas[data-k="${k}"]`);
    sparkline(cv,a.series[k],idx);
    const v=a.series[k][idx];
    el('sv-'+k).textContent=(v===null||v===undefined)?'—':fmt(v);
  });
}

function renderStream(){
  const s=D.arms[curArm].samples[String(curStep)][curSeq];
  drawTrace(s);
  el('streamlab').textContent=`Token 流 · 序列 #${s.seq} · ${s.len} token`+
    (s.hot!==null&&s.hotq>0.5?` · 教师在第 ${s.hot} 位就想结束`:'');
  const out=[];let prevEnd=null;
  s.wins.forEach(win=>{
    if(prevEnd!==null&&win.o>prevEnd)
      out.push(`<span class="gap">略去 ${win.o-prevEnd} 个 token</span>`);
    win.toks.forEach((t,i)=>{
      const pos=win.o+i;const [txt,r,e,q]=t;
      const hs=(q!==null&&q>0.5)?' hotstop':'';
      const safe=txt.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      out.push(`<span class="tok${hs}" style="background:${sigColor(r)}" data-p="${pos}" `+
        `data-r="${r}" data-e="${e}" data-q="${q}" data-t="${safe.replace(/"/g,'&quot;')}">${safe}</span>`);
    });
    prevEnd=win.o+win.toks.length;
  });
  if(prevEnd!==null&&prevEnd<s.len)
    out.push(`<span class="gap">略去 ${s.len-prevEnd} 个 token</span>`);
  el('stream').innerHTML=out.join('');
}

function renderSeqs(){
  const arr=D.arms[curArm].samples[String(curStep)]||[];
  if(curSeq>=arr.length)curSeq=0;
  el('seqs').innerHTML=arr.map((s,i)=>
    `<button class="seq" data-i="${i}" aria-pressed="${i===curSeq}">
       <b>#${s.seq}</b><i>${s.len} tok</i>
       ${s.trunc?'<span class="badge">截断</span>':''}
       ${s.score===1?'<i style="color:var(--cold)">✓ 答对</i>':(s.score===0?'<i>✗</i>':'')}
     </button>`).join('') || '<span class="note">该步无整序列样本</span>';
  el('seqs').querySelectorAll('.seq').forEach(b=>b.onclick=()=>{
    curSeq=+b.dataset.i;renderSeqs();renderStream();});
  if(arr.length)renderStream();else{el('stream').innerHTML='';}
}

function renderSteps(){
  const a=D.arms[curArm];
  if(!a.steps.includes(curStep))curStep=a.steps[Math.min(1,a.steps.length-1)];
  el('steps').innerHTML=a.steps.map(s=>
    `<button class="step" data-s="${s}" aria-pressed="${s===curStep}">${s}</button>`).join('');
  el('steps').querySelectorAll('.step').forEach(b=>b.onclick=()=>{
    curStep=+b.dataset.s;curSeq=0;renderSteps();renderSeries();renderSeqs();});
}

function renderArms(){
  el('arms').innerHTML=armNames.map(n=>
    `<button class="arm" data-a="${n}" aria-pressed="${n===curArm}">${n}</button>`).join('');
  el('arms').querySelectorAll('.arm').forEach(b=>b.onclick=()=>{
    curArm=b.dataset.a;curStep=null;curSeq=0;renderArms();renderSteps();renderSeries();renderSeqs();});
}

el('stream').addEventListener('mousemove',ev=>{
  const t=ev.target.closest('.tok');if(!t)return;
  el('r-tok').textContent=JSON.stringify(t.dataset.t);
  el('r-pos').textContent=t.dataset.p;
  el('r-r').textContent=t.dataset.r;
  el('r-e').textContent=t.dataset.e;
  el('r-q').textContent=t.dataset.q;
});

addEventListener('resize',()=>{renderSeries();
  const arr=D.arms[curArm].samples[String(curStep)];if(arr&&arr[curSeq])drawTrace(arr[curSeq]);});

renderArms();renderSteps();renderSeries();renderSeqs();
</script>
"""

open(out, "w", encoding="utf-8").write(HTML.replace("__DATA__", data))
print("wrote", out, len(HTML) + len(data), "bytes")
