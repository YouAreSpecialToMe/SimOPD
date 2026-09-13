// Run the viewer's real render logic against a minimal DOM shim, so every code
// path that builds markup or draws is actually executed and its output checked.
// fetch() is shimmed onto the local sequence-file directory, which is what the
// page loads from in the browser.
const fs = require('fs');
const path = require('path');

const SEQDIR = path.join(__dirname, 'seqdata_all');

const TOKENS = {
  '--ground': '#f4f5f7', '--panel': '#ffffff', '--panel2': '#eceef2',
  '--ink': '#1b2028', '--dim': '#5d6673', '--rule': '#d9dde4',
  '--hot': '#b4562a', '--cold': '#2b7a86', '--mark': '#6b5bc4',
  '--accent': '#2b7a86', '--ref': '#9aa3b0', '--mono': 'monospace',
  '--sans': 'sans-serif', '--cond': 'sans-serif',
};

const calls = { fillRect: 0, stroke: 0, arc: 0, fillText: 0 };
function ctx2d() {
  const noop = () => {};
  return new Proxy({}, {
    get(_, k) {
      if (k in calls) return () => { calls[k]++; };
      return noop;
    },
    set() { return true; },
  });
}

function mkEl(id) {
  return {
    id, _html: '', dataset: {}, style: {}, clientWidth: 640, checked: false,
    classList: { add() {}, remove() {} },
    scrollIntoView() {},
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = v; },
    textContent: '', value: '',
    getContext: () => ctx2d(),
    querySelector: (sel) => {
      const m = /data-k="([^"]+)"/.exec(sel);
      return mkEl('canvas:' + (m ? m[1] : sel));
    },
    querySelectorAll: () => [],
    addEventListener: () => {},
    closest: () => null,
  };
}

const els = {};
let fetched = 0, fetchMiss = 0;
global.document = { getElementById: (id) => (els[id] = els[id] || mkEl(id)), documentElement: {} };
global.getComputedStyle = () => ({ getPropertyValue: (n) => TOKENS[n] || '#888888' });
global.devicePixelRatio = 2;
global.addEventListener = () => {};
global.setTimeout = (f) => f;
global.fetch = async (url) => {
  const name = decodeURIComponent(url.replace(/^traj-data\//, ''));
  const p = path.join(SEQDIR, name);
  if (!fs.existsSync(p)) { fetchMiss++; return { ok: false, status: 404 }; }
  fetched++;
  const body = fs.readFileSync(p, 'utf8');
  return { ok: true, status: 200, json: async () => JSON.parse(body) };
};

const data = fs.readFileSync(__dirname + '/data.json', 'utf8');
els.DATA = mkEl('DATA');
els.DATA.textContent = data;

eval(fs.readFileSync(__dirname + '/app.js', 'utf8') +
  '\n;globalThis.__set=(a,s,q)=>{curArm=a;curStep=s;curSeq=q;};' +
  '\n;globalThis.__cache=seqCache;' +
  '\n;globalThis.__render=renderStream;globalThis.__series=renderSeries;');
const setState = globalThis.__set;

let fail = 0;
const ok = (c, m) => { console.log((c ? '  ok   ' : '  FAIL ') + m); if (!c) fail++; };

(async () => {
  const D2 = JSON.parse(data);
  const arms = Object.keys(D2.arms);
  console.log('arms:', arms.length);

  ok(els.arms._html.includes('vanilla_corr'), 'arm list rendered');
  ok(els.steps._html.includes('data-s='), 'step chips rendered');
  ok(els.series._html.includes('canvas'), 'sparkline canvases created');
  ok(els.seqs._html.includes('class="seq"'), 'sequence buttons rendered');

  // exercise every arm / step / sequence, awaiting the async stream render
  let n = 0, widest = 0, longest = 0, bad = 0;
  for (const a of arms) {
    for (const s of D2.arms[a].steps) {
      const arr = D2.arms[a].samples[String(s)] || [];
      for (let i = 0; i < arr.length; i++) {
        setState(a, s, i);
        renderSeries();
        await renderStream();
        const h = els.stream._html;
        n++;
        widest = Math.max(widest, h.length);
        longest = Math.max(longest, arr[i].len);
        // data-t / data-a hold arbitrary model text and a token really can BE the
        // word "undefined" (models write it in maths answers), so only the numeric
        // attributes and the style value are checked for it.
        if (!h.includes('class="tok') || /rgba\(NaN/.test(h) ||
            /(data-(p|r|e|q)="undefined"|background:undefined|style="[^"]*undefined)/.test(h) ||
            h.includes('取不到整条序列')) {
          if (bad < 5) console.log(`  FAIL ${a} step ${s} seq ${i}`);
          bad++; fail++;
        }
      }
    }
  }
  ok(bad === 0, `every (arm, step, seq) rendered from its own file: ${n} combinations`);
  ok(fetchMiss === 0, `no missing sequence files (fetched ${fetched}, missing ${fetchMiss})`);
  ok(fetched === n, `each combination loaded its OWN file: ${fetched} fetches for ${n} renders`);
  ok(globalThis.__cache.size === n, `cache holds ${globalThis.__cache.size} distinct sequences`);
  console.log('  longest sequence:', longest, 'tokens ->', widest, 'chars of markup');
  console.log('  canvas ops:', JSON.stringify(calls));

  // the whole sequence must be present now, not a window
  setState('vanilla_corr', 50, 0);
  renderSeries(); await renderStream();
  const s0 = D2.arms.vanilla_corr.samples['50'][0];
  const spans = (els.stream._html.match(/class="tok/g) || []).length;
  ok(spans === s0.len, `full sequence rendered: ${spans} spans for a ${s0.len}-token response`);
  ok(!els.stream._html.includes('略去'), 'no elision markers left');

  console.log(fail ? `\n${fail} FAILURES` : '\nall checks passed');
  process.exit(fail ? 1 : 0);
})();
