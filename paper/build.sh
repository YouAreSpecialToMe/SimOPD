#!/usr/bin/env bash
# Typeset a memo: markdown -> HTML (pandoc) -> paginated PDF (MathJax + paged.js
# in headless Chrome). No LaTeX needed, which is the point: this box has none,
# and BasicTeX needs an admin password.
#
#   bash paper/build.sh                        # builds opd-framework-zh.md
#   bash paper/build.sh some-other-memo.md
#
# One-time: npm install   (mathjax, pagedjs, playwright-core -- all local)
set -euo pipefail
cd "$(dirname "$0")"

SRC=${1:-opd-framework-zh.md}
BASE=$(basename "${SRC%.md}")
mkdir -p out

[ -d node_modules/mathjax ] || { echo "FATAL: run 'npm install' in paper/ first" >&2; exit 1; }

# --standalone + our template: title block, abstract, TOC, then body.
# --mathjax leaves $...$ untouched for the in-page MathJax to typeset.
pandoc "$SRC" \
    --from markdown+yaml_metadata_block+footnotes+pipe_tables \
    --to html5 \
    --standalone \
    --template assets/template.html \
    --toc --toc-depth=2 \
    --mathjax \
    --output "out/${BASE}.html"

node assets/render.mjs "$PWD/out/${BASE}.html" "$PWD/out/${BASE}.pdf"

echo "PDF: paper/out/${BASE}.pdf  ($(du -h "out/${BASE}.pdf" | cut -f1))"
