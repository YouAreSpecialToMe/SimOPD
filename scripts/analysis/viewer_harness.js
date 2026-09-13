// Verify the MERGED page: both views live in one document, so the checks that
// matter are (a) the two scripts coexist without clobbering each other, (b) each
// view still renders every one of its items from its own data file, and (c) the
// switch shows one view and hides the other.
const fs = require('fs');
const path = require('path');

const SEQDIR = path.join(__dirname, 'seqdata_all');
const PDIR = path.join(__dirname, 'evaldata');
const HTML = fs.readFileSync(path.join(__dirname, 'traj_all.html'), 'utf8');

const TOKENS = {
  '--ground': '#f4f5f7', '--panel': '#ffffff', '--panel2': '#eceef2',
  '--ink': '#1b2028', '--dim': '#5d6673', '--rule': '#d9dde4',
  '--hot': '#b4562a', '--cold': '#2b7a86', '--mark': '#6b5bc4',
  '--accent': '#2b7a86', '--ref': '#9aa3b0', '--right': '#2b7a86',
  '--wrong': '#b4562a', '--mono': 'monospace', '--sans': 'sans-serif',
  '--cond': 'sans-serif',
};

const calls = { fillRect: 0, stroke: 0, arc: 0, fillText: 0 };
const ctx2d = () => new Proxy({}, {
  get: (_, k) => (k in calls ? () => { calls[k]++; } : () => {}),
  set: () => true,
});

function mkEl(id) {
  return {
    id, _html: '', dataset: {}, style: {}, clientWidth: 700, checked: false,
    hidden: false, classList: { add() {}, remove() {} }, scrollIntoView() {},
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = v; },
    textContent: '', value: '',
    getContext: () => ctx2d(),
    querySelector: () => mkEl('c'),
    querySelectorAll: () => [],
    addEventListener: () => {}, closest: () => null,
    setAttribute(k, v) { this.dataset['attr_' + k] = v; },
  };
}

const els = {};
let seqFetch = 0, probFetch = 0, missing = 0;
global.window = global;
global.document = { getElementById: (i) => (els[i] = els[i] || mkEl(i)), documentElement: {} };
global.getComputedStyle = () => ({ getPropertyValue: (n) => TOKENS[n] || '#888' });
global.devicePixelRatio = 2;
global.addEventListener = () => {};
global.setTimeout = (f) => f;
global.location = { hash: '' };
global.fetch = async (url) => {
  let p;
  if (url.startsWith('traj-data/')) { p = path.join(SEQDIR, decodeURIComponent(url.slice(10))); seqFetch++; }
  else if (url.startsWith('eval-data/')) { p = path.join(PDIR, url.slice(10)); probFetch++; }
  else return { ok: false, status: 404 };
  if (!fs.existsSync(p)) { missing++; return { ok: false, status: 404 }; }
  return { ok: true, status: 200, json: async () => JSON.parse(fs.readFileSync(p, 'utf8')) };
};

// feed both inline data blobs, then run the page's three script blocks in order
const blobs = [...HTML.matchAll(/<script type="application\/json" id="(\w+)">([\s\S]*?)<\/script>/g)];
blobs.forEach(([, id, body]) => { els[id] = mkEl(id); els[id].textContent = body; });
const scripts = [...HTML.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);

let fail = 0;
const ok = (c, m) => { console.log((c ? '  ok   ' : '  FAIL ') + m); if (!c) fail++; };

console.log('inline blobs:', blobs.map((b) => b[1]).join(', '));
console.log('script blocks:', scripts.length);

// block 0 = viewer, block 1 = eval (IIFE), block 2 = the switch
try {
  eval(scripts[0] + '\n;globalThis.__setV=(a,s,q)=>{curArm=a;curStep=s;curSeq=q;};' +
       '\n;globalThis.__renderV=renderStream;globalThis.__seriesV=renderSeries;' +
       '\n;globalThis.__cacheV=seqCache;');
  ok(true, 'viewer script ran');
} catch (e) { ok(false, 'viewer script threw: ' + e.message); }

try { eval(scripts[1]); ok(true, 'eval script ran (no redeclaration clash)'); }
catch (e) { ok(false, 'eval script threw: ' + e.message); }

try { eval(scripts[2]); ok(true, 'switch script ran'); }
catch (e) { ok(false, 'switch script threw: ' + e.message); }

(async () => {
  const V = JSON.parse(els.DATA.textContent);
  const E = JSON.parse(els.EDATA.textContent);

  // the shim defaults hidden=false, so the initial state is checked in the markup
  const vTag = /<div class="wrap" id="wrapV"([^>]*)>/.exec(HTML)[1];
  const eTag = /<div class="wrap" id="wrapE"([^>]*)>/.exec(HTML)[1];
  ok(!/\bhidden\b/.test(vTag) && /\bhidden\b/.test(eTag),
    'markup ships signal view visible and eval view hidden');
  // and the switch must actually toggle them
  els.swE.dataset.attr_ariapressed = '';
  globalThis.__show ? globalThis.__show('E') : els.swE.onclick();
  ok(els.wrapV.hidden === true && els.wrapE.hidden === false, 'switch reveals the eval view');
  els.swV.onclick();
  ok(els.wrapV.hidden === false && els.wrapE.hidden === true, 'switch returns to the signal view');
  ok(typeof global.evalBoot === 'function', 'evalBoot exposed for deferred start');

  // --- signal view: every arm / step / sequence ---
  let n = 0, bad = 0;
  for (const a of Object.keys(V.arms)) {
    for (const s of V.arms[a].steps) {
      const arr = V.arms[a].samples[String(s)] || [];
      for (let i = 0; i < arr.length; i++) {
        globalThis.__setV(a, s, i);
        globalThis.__seriesV();
        await globalThis.__renderV();
        const h = els.stream._html;
        n++;
        if (!h.includes('class="tok') || /rgba\(NaN/.test(h) ||
            /(data-(p|r|e|q)="undefined"|background:undefined)/.test(h) ||
            h.includes('取不到整条序列')) { if (bad < 3) console.log('  FAIL signal ' + a + ' ' + s + ' #' + i); bad++; fail++; }
      }
    }
  }
  ok(bad === 0, `signal view rendered ${n} (arm, step, seq) combinations`);
  ok(seqFetch === n && globalThis.__cacheV.size === n,
    `signal view: ${seqFetch} fetches, ${globalThis.__cacheV.size} cached, for ${n} renders`);

  // --- eval view: boot it, then every problem ---
  global.evalBoot();
  ok(els.plist._html.includes('class="p"'), 'eval view rendered its problem list after boot');
  const probs = E.arms[Object.keys(E.arms)[0]].probs;
  let ebad = 0;
  for (const p of probs) {
    await globalThis.__selectE(p.i);
    const h = els.answers._html;
    if (!h.includes('class="ans"') || h.includes('取不到答案文件')) {
      if (ebad < 3) console.log('  FAIL eval #' + p.i); ebad++; fail++;
    }
  }
  ok(ebad === 0, `eval view rendered all ${probs.length} problems`);
  ok(probFetch === probs.length, `eval view: ${probFetch} fetches for ${probs.length} problems`);
  ok(missing === 0, `no missing data files (missing ${missing})`);
  console.log('  canvas ops:', JSON.stringify(calls));

  console.log(fail ? `\n${fail} FAILURES` : '\nall checks passed');
  process.exit(fail ? 1 : 0);
})();
