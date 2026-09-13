#!/usr/bin/env python3
"""Build ONE page carrying both views.

  训练信号  — per-token r over a training rollout (index inline, sequences fetched)
  同题演变  — one MATH500 problem's answer at every checkpoint (ditto)

Both indices are inlined (~2.1 MB together); the detail files stay on disk and are
fetched on demand, because the two detail sets are 96 MB and inlining them would
mean parsing 96 MB of JSON before the page could show anything.

    python3 build_all.py viewer_full.json eval_data.json traj_viewer.html
"""
import re
import sys

vsrc = sys.argv[1] if len(sys.argv) > 1 else "viewer_full.json"
esrc = sys.argv[2] if len(sys.argv) > 2 else "eval_data.json"
out = sys.argv[3] if len(sys.argv) > 3 else "traj_viewer.html"

vdata = open(vsrc, encoding="utf-8").read()
edata = open(esrc, encoding="utf-8").read()


def piece(builder, kind):
    h = open(builder, encoding="utf-8").read().split('HTML = r"""', 1)[1].rsplit('"""', 1)[0]
    if kind == "style":
        return re.search(r"<style>(.*?)</style>", h, re.S).group(1)
    if kind == "body":
        return re.search(r'<body>(.*?)<script type="application/json"', h, re.S).group(1)
    return re.findall(r"<script>(.*?)</script>", h, re.S)[-1]


vstyle, vbody, vjs = (piece("build_viewer.py", k) for k in ("style", "body", "js"))
estyle, ebody, ejs = (piece("build_eval.py", k) for k in ("style", "body", "js"))

# Both scripts declare el / css / esc at top level, and top-level const in classic
# scripts shares one global lexical scope, so a second declaration is a SyntaxError.
# The eval script therefore gets wrapped in an IIFE; it also has to read its own
# data blob, and its initial render is deferred until the view is first shown.
ejs = ejs.replace("document.getElementById('DATA')", "document.getElementById('EDATA')")
ejs = ejs.replace("""renderFilters();renderList();
const first=shown()[0];if(first)select(first.i);""",
"""window.evalBoot=function(){renderFilters();renderList();
  const first=shown()[0];if(first)select(first.i);};""")
assert "window.evalBoot" in ejs, "eval boot anchor not found"
ejs = "(function(){\n" + ejs + "\nwindow.__selectE=select;\n})();"

# the eval page's rail/main live inside its own .wrap; strip the outer grid so the
# two views can share one shell, and namespace its colliding ids
ebody = ebody.replace('<div class="wrap">', '<div class="wrap" id="wrapE" hidden>')
vbody = vbody.replace('<div class="wrap">', '<div class="wrap" id="wrapV">')
# the eval view drops its own cross-link, the switch replaces it
ebody = re.sub(r'<br>\s*<a href="traj-viewer\.html"[^>]*>.*?</a>', "", ebody, flags=re.S)
vbody = re.sub(r'<br>\s*<a href="traj-eval\.html"[^>]*>.*?</a>', "", vbody, flags=re.S)

# the eval builder's CSS redefines the shared tokens; keep only its component rules
estyle = re.sub(r":root\s*\{.*?\}", "", estyle, count=1, flags=re.S)
estyle = re.sub(r"@media \(prefers-color-scheme:dark\)\{:root:not\(\[data-theme=\"light\"\]\)\{.*?\}\}",
                "", estyle, count=1, flags=re.S)
estyle = re.sub(r":root\[data-theme=\"dark\"\]\{.*?\}", "", estyle, count=1, flags=re.S)
estyle = re.sub(r"\*\{box-sizing:border-box\}", "", estyle)
estyle = re.sub(r"body\{[^}]*\}", "", estyle, count=1)
# the two views style .wrap/.rail/.main/.card/.lab the same way already; the eval
# rules that differ are scoped under #wrapE so they cannot leak into the other view
estyle = "\n".join(
    ("#wrapE " + line) if line.strip().startswith((".", "#")) and "{" in line else line
    for line in estyle.splitlines())

SHARED_TOKENS = """
:root{ --right:#2b7a86; --wrong:#b4562a; --warn:#a8791f; }
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --right:#4fb3c0; --wrong:#e08050; --warn:#d0a44a; }}
:root[data-theme="dark"]{ --right:#4fb3c0; --wrong:#e08050; --warn:#d0a44a; }
.switch{position:sticky;top:0;z-index:5;display:flex;gap:6px;align-items:center;
  padding:9px 20px;background:var(--panel);border-bottom:1px solid var(--rule)}
.switch b{font-family:var(--cond);font-size:11px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--dim);margin-right:8px}
.sw{font-size:12.5px;padding:5px 13px;border:1px solid var(--rule);
  background:var(--panel2);border-radius:3px;cursor:pointer;color:inherit;font-family:inherit}
.sw:hover{border-color:var(--accent)}
.sw[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.sw:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
"""

HTML = f"""<!doctype html>
<html lang="zh"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token Signal Scope</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
{vstyle}
{SHARED_TOKENS}
{estyle}
</style>
</head>
<body>

<nav class="switch">
  <b>视图</b>
  <button class="sw" id="swV" aria-pressed="true">训练信号 · 逐 token</button>
  <button class="sw" id="swE" aria-pressed="false">同题演变 · 评测</button>
</nav>

{vbody}
{ebody}

<script type="application/json" id="DATA">{vdata}</script>
<script type="application/json" id="EDATA">{edata}</script>
<script>
{vjs}
</script>
<script>
{ejs}
</script>
<script>
(function(){{
  const V=document.getElementById('wrapV'), E=document.getElementById('wrapE');
  const bV=document.getElementById('swV'), bE=document.getElementById('swE');
  let started=false;
  function show(which){{
    const ev = which==='E';
    V.hidden=ev; E.hidden=!ev;
    bV.setAttribute('aria-pressed', String(!ev));
    bE.setAttribute('aria-pressed', String(ev));
    location.hash = ev ? '#eval' : '#signal';
    if(ev && !started){{ started=true; if(typeof evalBoot==='function') evalBoot(); }}
  }}
  bV.onclick=()=>show('V'); bE.onclick=()=>show('E');
  if(location.hash==='#eval') show('E');
}})();
</script>
</body></html>
"""

open(out, "w", encoding="utf-8").write(HTML)
print("wrote", out, len(HTML), "bytes")
