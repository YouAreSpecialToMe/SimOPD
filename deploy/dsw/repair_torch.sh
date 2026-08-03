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
# The venv lives at ./simopd. An existing .venv is still honoured so the Cornell
# box, which has one and is mid-campaign, keeps working; new installs get simopd.
# SIMOPD_VENV overrides either way.
VENV=${SIMOPD_VENV:-}
if [ -z "$VENV" ]; then
    for _c in simopd .venv; do [ -d "$_c" ] && { VENV=$_c; break; }; done
fi
VENV=${VENV:-simopd}
[ -d "$VENV" ] || { echo "no venv at ./$VENV; run deploy/dsw/setup.sh first" >&2; exit 1; }
source "$VENV/bin/activate"

# uv hardlinks from its cache by default. On DSW the cache is on local disk and the
# venv is on the /mnt/workspace network volume, and hardlinking across filesystems
# is exactly the kind of thing that half-succeeds. Copying is slower and reliable.
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
export UV_INDEX=${UV_INDEX:-$UV_DEFAULT_INDEX}

# uv validates TLS against its own bundled roots, not the system store. Behind a
# TLS-intercepting proxy that yields "invalid peer certificate: UnknownIssuer"
# even though curl and git are fine, because those use the system CA bundle.
# UV_NATIVE_TLS makes uv use it too; curl is the fallback when it still refuses.
export UV_NATIVE_TLS=${UV_NATIVE_TLS:-1}

fetch_and_install() {   # fetch_and_install <url> <label>
    local url=$1 label=$2 tmp
    pipi --no-cache "$label @ $url" && return 0
    echo "  uv could not fetch it; retrying via curl (system CA store)" >&2
    tmp=$(mktemp -d)/$(basename "${url%%\?*}")
    curl -fL --retry 3 --retry-delay 5 -o "$tmp" "$url" || {
        echo "  curl failed too. Download it on a machine that can reach GitHub and pass" >&2
        echo "    VLLM_WHEEL=/path/to/the.whl" >&2
        return 1
    }
    pipi --no-cache "$tmp"
}

# pip by default: the TLS-roots and hardlink problems that broke this box are both
# uv-specific, and pip needs nothing installed first. SIMOPD_USE_UV=1 to force uv.
UV=""
if [ "${SIMOPD_USE_UV:-0}" = "1" ]; then
    UV=$(command -v uv 2>/dev/null || true)
    [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
fi
pipi() {
    if [ -n "$UV" ]; then pipi "$@"; else
        local args=(); for a in "$@"; do case "$a" in --no-cache) args+=(--no-cache-dir);; *) args+=("$a");; esac; done
        python -m pip install -i "${UV_DEFAULT_INDEX:-https://pypi.org/simple/}" "${args[@]}"
    fi
}
pipu() { if [ -n "$UV" ]; then pipu "$@"; else python -m pip uninstall -y "$@"; fi; }
echo "installer: ${UV:-python -m pip}"
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "venv site-packages: $SITE"
echo "uv link mode:       $UV_LINK_MODE"

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
# nvcc decides the family, not the driver: flash-attn is compiled from source and
# torch refuses to build an extension across a CUDA major mismatch.
NVCC_VER=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
if [ "${NVCC_VER%%.*}" = "13" ] && [ "${DRV:-0}" -ge 580 ]; then FLAVOR=cu130; else FLAVOR=cu129; fi
echo "driver ${DRV:-unknown}, nvcc ${NVCC_VER:-none} -> $FLAVOR"

echo
echo "=== 1. removing the broken trees ==="
pipu torch torchvision torchaudio vllm flash-attn >/dev/null 2>&1 || true
for d in torch torchvision torchaudio vllm flash_attn; do
    # A namespace stub leaves a directory with no dist-info, so uninstall does not
    # touch it. Delete by hand or the reinstall lands on top of the wreckage.
    [ -e "$SITE/$d" ] && { rm -rf "${SITE:?}/$d"; echo "  removed $SITE/$d"; }
done
    # flash-attn installs three things: the flash_attn/ package, its dist-info, and a
    # TOP-LEVEL flash_attn_2_cuda*.so sitting beside them. Removing only the package
    # directory leaves that .so behind, and a reinstall that uv considers satisfied
    # will not replace it -- so the stale extension keeps being imported and keeps
    # raising "undefined symbol: ...c10::impl::cow::materialize_cow_storage...".
for f in "$SITE"/flash_attn_2_cuda*.so "$SITE"/flash_attn-*.dist-info \
         "$SITE"/torch-*.dist-info "$SITE"/vllm-*.dist-info; do
    [ -e "$f" ] && { rm -rf "$f"; echo "  removed $f"; }
done

echo
echo "=== 2. free space (a full volume is how an install dies partway) ==="
df -h "$SITE" | tail -1

echo
echo "=== 3. reinstalling ($FLAVOR, no cache, copy mode) ==="
if [ "$FLAVOR" = "cu130" ]; then
    pipi --no-cache --force-reinstall \
        "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" || exit 1
else
    pipi --no-cache --force-reinstall \
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
    pipi --no-cache "vllm==0.26.0" || exit 1
else
    VLLM_WHEEL=${VLLM_WHEEL:-${GITHUB_PROXY:-}https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl}
    case "$VLLM_WHEEL" in
        /*|file://*) pipi --no-cache "$VLLM_WHEEL" || exit 1 ;;
        *)           fetch_and_install "$VLLM_WHEEL" vllm || exit 1 ;;
    esac
fi
python -c "import vllm; print('  vllm', vllm.__version__)" || exit 1

echo
echo "=== done. flash-attn is NOT rebuilt here -- it must come last, after verl ==="
echo "next:  bash deploy/dsw/setup.sh    # picks up from verl + flash-attn"
echo "then:  bash deploy/dsw/doctor.sh"
