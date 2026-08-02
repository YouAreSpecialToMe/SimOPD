#!/usr/bin/env bash
# Rebuild a broken torch/vllm/flash-attn install from scratch, aggressively.
#
#   bash deploy/dsw/repair_torch.sh
#
# For the case where torch imports as a namespace package (spec.origin is None,
# no __version__, no torch.Tensor) while its nvidia-* dependencies are all present
# -- an install that placed the dependencies and then died before the package body.
#
# Everything here is deliberately blunt: delete the trees, ignore the cache, copy
# instead of link. A gentler reinstall is what already failed.

set -uo pipefail

SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SIMOPD_ROOT"
[ -d .venv ] || { echo "no .venv here; run deploy/dsw/setup.sh first" >&2; exit 1; }
source .venv/bin/activate

# uv hardlinks from its cache by default. On DSW the cache is on local disk and the
# venv is on the /mnt/workspace network volume, and hardlinking across filesystems
# is exactly the kind of thing that half-succeeds. Copying is slower and reliable.
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
export UV_INDEX=${UV_INDEX:-$UV_DEFAULT_INDEX}

UV=$(command -v uv || echo "$HOME/.local/bin/uv")
[ -x "$UV" ] || { echo "uv not found" >&2; exit 1; }
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "venv site-packages: $SITE"
echo "uv link mode:       $UV_LINK_MODE"

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -ge 580 ]; then FLAVOR=cu130; else FLAVOR=cu129; fi
echo "driver ${DRV:-unknown} -> $FLAVOR"

echo
echo "=== 1. removing the broken trees ==="
"$UV" pip uninstall torch torchvision torchaudio vllm flash-attn >/dev/null 2>&1 || true
for d in torch torchvision torchaudio vllm flash_attn; do
    # A namespace stub leaves a directory with no dist-info, so uninstall does not
    # touch it. Delete by hand or the reinstall lands on top of the wreckage.
    [ -e "$SITE/$d" ] && { rm -rf "${SITE:?}/$d"; echo "  removed $SITE/$d"; }
done
rm -rf "$SITE"/torch-*.dist-info "$SITE"/vllm-*.dist-info 2>/dev/null || true

echo
echo "=== 2. free space (a full volume is how an install dies partway) ==="
df -h "$SITE" | tail -1

echo
echo "=== 3. reinstalling ($FLAVOR, no cache, copy mode) ==="
if [ "$FLAVOR" = "cu130" ]; then
    "$UV" pip install --no-cache --force-reinstall \
        "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" || exit 1
else
    "$UV" pip install --no-cache --force-reinstall \
        --find-links "${TORCH_FIND_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu129/}" \
        "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129" || exit 1
fi

echo
echo "=== 4. verifying torch BEFORE anything is built against it ==="
python - <<'PY' || { echo "torch still broken -- stop here, do not install anything else" >&2; exit 1; }
import torch, importlib.util
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}")
print(f"  origin {importlib.util.find_spec('torch').origin}")
assert torch.Tensor is not None
PY

echo
echo "=== 5. vllm ==="
if [ "$FLAVOR" = "cu130" ]; then
    "$UV" pip install --no-cache "vllm==0.26.0" || exit 1
else
    VLLM_WHEEL=${VLLM_WHEEL:-${GITHUB_PROXY:-}https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl}
    "$UV" pip install --no-cache "vllm @ $VLLM_WHEEL" || exit 1
fi
python -c "import vllm; print('  vllm', vllm.__version__)" || exit 1

echo
echo "=== done. flash-attn is NOT rebuilt here -- it must come last, after verl ==="
echo "next:  bash deploy/dsw/setup.sh    # picks up from verl + flash-attn"
echo "then:  bash deploy/dsw/doctor.sh"
