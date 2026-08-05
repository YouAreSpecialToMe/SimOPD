#!/usr/bin/env bash
# Find out WHAT is routing this environment through ModelScope. Read-only.
#
#   bash deploy/dsw/why_modelscope.sh
#
# Three mechanisms have already been found and fixed on this project, each of which
# looked like the last one: verl's VERL_USE_MODELSCOPE, vLLM's separate
# VLLM_USE_MODELSCOPE, and a sitecustomize guard that arrived too late. Rather than
# guess a fourth, this reports which of them is actually live -- in particular
# whether transformers is ALREADY patched before anything of ours is imported, which
# is the signature of a .pth file (site.py runs those before sitecustomize, so our
# guard cannot see the import at all).

set -uo pipefail
SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SIMOPD_ROOT"
[ -d simopd ] && [ -z "${VIRTUAL_ENV:-}" ] && source simopd/bin/activate

echo "=== 1. every MODELSCOPE variable in this environment ==="
env | grep -i modelscope || echo "  (none set)"

echo
echo "=== 2. is transformers patched at interpreter startup? ==="
# -S skips site.py entirely, so nothing user-level has run: if the two disagree, the
# patch is being applied by site processing (a .pth, sitecustomize or usercustomize)
# rather than by verl or vLLM.
echo "  with site processing:"
python - <<'PY' 2>&1 | sed 's/^/    /'
try:
    from transformers import AutoTokenizer
    m = getattr(AutoTokenizer.from_pretrained, "__module__", "?")
    print(("PATCHED by " + m) if "modelscope" in str(m) else "clean (" + str(m) + ")")
except Exception as e:
    print(f"could not import transformers: {type(e).__name__}: {e}")
PY
echo "  without site processing (python -S):"
PYTHONPATH="${VIRTUAL_ENV:-}/lib/python3.12/site-packages" python -S - <<'PY' 2>&1 | tail -1 | sed 's/^/    /'
try:
    from transformers import AutoTokenizer
    m = getattr(AutoTokenizer.from_pretrained, "__module__", "?")
    print(("PATCHED by " + m) if "modelscope" in str(m) else "clean (" + str(m) + ")")
except Exception as e:
    print(f"could not import transformers: {type(e).__name__}")
PY

echo
echo "=== 3. .pth files (these run BEFORE sitecustomize) ==="
for d in $(python -c 'import site,sys; print(" ".join(site.getsitepackages()+[site.getusersitepackages()]))' 2>/dev/null); do
    for f in "$d"/*.pth; do
        [ -e "$f" ] || continue
        if grep -qil "modelscope\|patch_hub" "$f" 2>/dev/null; then
            echo "  SUSPECT $f"; sed 's/^/          /' "$f" | head -5
        fi
    done
done
echo "  (nothing above = no .pth mentions ModelScope)"

echo
echo "=== 4. sitecustomize / usercustomize actually in use ==="
python -c "
import sys
for name in ('sitecustomize','usercustomize'):
    m = sys.modules.get(name)
    print(f'  {name}: {getattr(m, \"__file__\", \"not loaded\")}')
" 2>/dev/null

echo
echo "=== 5. is modelscope even installed? ==="
python -c "
import importlib.util as u, os
s = u.find_spec('modelscope')
print('  installed at', os.path.dirname(s.origin) if s and s.origin else s)
" 2>/dev/null || echo "  not installed"

echo
echo "=== what to do ==="
cat <<'EOF'
  If (2) says PATCHED even with `python -S`, or (3) names a .pth, the patch is
  applied before any code of ours runs and no flag can stop it. The fix that works
  regardless of mechanism is to remove the package -- nothing can patch what is not
  importable, and we fetch every model from HuggingFace anyway:

      python -m pip uninstall -y modelscope
      bash deploy/dsw/why_modelscope.sh      # (2) should now be clean

  If (1) shows a flag still ON, that is the simpler answer: regenerate the env file
  (VERL_USE_MODELSCOPE=False VLLM_USE_MODELSCOPE=False bash deploy/dsw/setup.sh),
  because `${VAR:-False}` does not override a value the image already exported.
EOF
