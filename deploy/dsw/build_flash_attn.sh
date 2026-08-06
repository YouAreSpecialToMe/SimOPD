#!/usr/bin/env bash
# Build a flash-attn wheel for THIS box's GPU, torch and interpreter.
#
#   bash deploy/dsw/build_flash_attn.sh              # arch auto-detected
#   FA_ARCHS=90 bash deploy/dsw/build_flash_attn.sh  # force one
#
# Why this exists alongside slurm/build_flash_attn.sbatch: that script pins
# FLASH_ATTN_CUDA_ARCHS="80;86", i.e. A100 and A6000, and produced the cp312 wheel
# vendored under deploy/dsw/. Neither half of that fits an H100 DSW box, and the
# Python half is the less interesting one -- an sm80/sm86 build has no kernels for
# sm90 whatever interpreter loads it, so "install a cp312 Python to match the old
# environment" does not reach the actual problem. The GPU decides this file's
# contents, not the interpreter.
#
# There is also no prebuilt wheel to fall back on: Dao-AILab's latest release
# (v2.8.3.post1) ships cp311 wheels up to torch 2.8, and this stack is torch 2.11.
#
# Everything lands under $BUILD_ROOT. Nothing is installed into the shared venv --
# other machines run from it, and a campaign in flight must not have site-packages
# rewritten underneath it. Install the wheel yourself when the lanes are idle:
#
#   source simopd_env.sh
#   pip install --no-deps <the wheel this prints>
set -euo pipefail

SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SIMOPD_ROOT"
[ -f simopd_env.sh ] && source simopd_env.sh
[ -d simopd ] && [ -z "${VIRTUAL_ENV:-}" ] && source simopd/bin/activate

BUILD_ROOT=${BUILD_ROOT:-$SIMOPD_ROOT/../simopd_data/fa_build}
FA_VERSION=${FA_VERSION:-v2.8.3.post1}
CUDA_PKG_VER=${CUDA_PKG_VER:-12.9.86-1}
CUDART_PKG_VER=${CUDART_PKG_VER:-12.9.79-1}
CCCL_PKG_VER=${CCCL_PKG_VER:-12.9.27-1}
NV_REPO=${NV_REPO:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64}
mkdir -p "$BUILD_ROOT"/{deb,cuda,wheels,tools}

# Compute capability of the card actually in this box, as nvcc wants it (90, not 9.0).
# --id=0 rather than `| head -1`: under `set -e` with pipefail, head exits first, the
# query takes SIGPIPE and returns 141, and the script dies with no error text.
if [ -z "${FA_ARCHS:-}" ]; then
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0)
    FA_ARCHS=${_cc/./}
fi
echo "=== target: sm${FA_ARCHS}  torch $(python -c 'import torch;print(torch.__version__)')  $(python -V 2>&1) ==="

# ---------------------------------------------------------------------------
# A CUDA_HOME assembled from .debs, unpacked rather than installed.
#
# pip cannot supply this. nvidia-cuda-nvcc-cu12 -- from PyPI and from
# pypi.nvidia.com alike -- ships bin/ptxas and nvvm/ but NOT bin/nvcc: it exists so
# torch and triton can JIT, not so anything can compile. cuda-toolkit[nvcc]
# resolves to the same wheel. The .deb of the same version does carry the driver,
# and dpkg-deb -x unpacks without root and without touching the system.
# ---------------------------------------------------------------------------
echo "=== [1/4] CUDA toolchain ==="
if [ ! -x "$BUILD_ROOT/cuda/bin/nvcc" ]; then
    # cuda-cccl is not optional decoration: cuda_fp16.h includes <nv/target>, which
    # lives in CCCL (libcu++), so without it every single .cu fails at
    #   cuda_fp16.h:4492:10: fatal error: nv/target: No such file or directory
    # after the toolchain has otherwise reported itself healthy.
    for p in cuda-nvcc-12-9_${CUDA_PKG_VER} cuda-crt-12-9_${CUDA_PKG_VER} \
             cuda-nvvm-12-9_${CUDA_PKG_VER} cuda-cudart-dev-12-9_${CUDART_PKG_VER} \
             cuda-cccl-12-9_${CCCL_PKG_VER}; do
        f="$BUILD_ROOT/deb/${p}_amd64.deb"
        [ -f "$f" ] || { echo "  fetching ${p}"; curl -fsSL -o "$f" "$NV_REPO/${p}_amd64.deb"; }
        dpkg-deb -x "$f" "$BUILD_ROOT/deb/unpacked"
    done
    # The debs lay everything out under /usr/local/cuda-12.9; lift that to be CUDA_HOME.
    cp -a "$BUILD_ROOT/deb/unpacked/usr/local/cuda-12.9/." "$BUILD_ROOT/cuda/"
fi
export CUDA_HOME="$BUILD_ROOT/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvcc --version | sed -n 's/^.*release/  nvcc release/p'
[ "$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')" = "$(python -c 'import torch;print(torch.version.cuda.split(".")[0])')" ] \
    || echo "  WARNING: nvcc major != torch CUDA major; the extension build will refuse" >&2

# ---------------------------------------------------------------------------
# Build deps, into a private prefix rather than the shared venv.
# ---------------------------------------------------------------------------
echo "=== [2/4] build deps ==="
# --no-build-isolation means setup.py's imports must already be importable: pip does
# not create an environment for them. flash-attn's setup.py imports `wheel` at line
# 20 and `packaging` right after, and this venv has neither -- the build then dies in
# 1.4s at "Preparing metadata", with ModuleNotFoundError: No module named 'wheel'
# buried 18 lines inside a pyproject_hooks traceback that names pip, setuptools and
# pyproject_hooks but not the missing package.
#
# Into a private --target, never the shared venv: other boxes run from that venv, and
# rewriting its site-packages under a live campaign is the accident this whole script
# is arranged to avoid.
for _m in ninja wheel packaging; do
    python -c "import $_m" 2>/dev/null || _missing="${_missing:-} $_m"
done
if [ -n "${_missing:-}" ]; then
    echo "  installing into $BUILD_ROOT/tools:${_missing}"
    # pypi.org explicitly: a pip.conf or inherited PIP_INDEX_URL can point at a
    # mirror that serves the index fast and package bodies at <1MB/s, measured here.
    python -m pip install -q --target "$BUILD_ROOT/tools" \
        -i "${PIP_INDEX_URL:-https://pypi.org/simple/}" ${_missing}
fi
export PYTHONPATH="$BUILD_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$BUILD_ROOT/tools/bin:$PATH"
for _m in ninja wheel packaging; do
    python -c "import $_m" 2>/dev/null || { echo "FATAL: $_m still not importable" >&2; exit 1; }
done
echo "  ninja $(ninja --version 2>/dev/null || echo '(no binary -- setup.py falls back to serial nvcc)'), wheel + packaging importable"

# ---------------------------------------------------------------------------
echo "=== [3/4] source ==="
SRC="$BUILD_ROOT/flash-attention"
if [ ! -d "$SRC" ]; then
    git clone --depth 1 --branch "$FA_VERSION" --recurse-submodules \
        https://github.com/Dao-AILab/flash-attention.git "$SRC"
else
    echo "  reusing $SRC"
fi

# ---------------------------------------------------------------------------
# One arch only. Building the full default set is several times the work for
# kernels no card here can run, and the point of this wheel is that it matches
# this box.
#
# MAX_JOBS is memory-bound, not core-bound: each nvcc for these kernels peaks
# around 2-4GB. Sized against free RAM rather than nproc, because the failure mode
# of guessing from 224 cores is the OOM killer taking the build at 90%.
# ---------------------------------------------------------------------------
echo "=== [4/4] compile (sm${FA_ARCHS}) ==="
# "80" is a GATE, not just another arch. flash-attn's setup.py reads
#
#     if "80" in cuda_archs():
#         cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
#         if bare_metal_version >= Version("11.8") and "90" in cuda_archs():
#             cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]
#
# -- sm90 is nested inside the sm80 branch. FLASH_ATTN_CUDA_ARCHS="90" on its own
# therefore emits NO -gencode at all, and the build "succeeds" into a wheel with no
# kernels for anything. That is why upstream's default is "80;90;100;120".
#
# So the gate always goes in, and the card's own arch with it. The cost is sm80
# cubins we do not need here; the benefit is one wheel that also runs on the A100
# boxes, which is what the vendored Cornell wheel was for.
if [ "$FA_ARCHS" = "80" ]; then
    _archs="80"
else
    _archs="80;$FA_ARCHS"
fi
export FLASH_ATTN_CUDA_ARCHS="$_archs"
# The dot goes before the LAST digit, not after the first: 90 -> 9.0, 86 -> 8.6,
# but 100 -> 10.0 and 120 -> 12.0. Splitting after the first digit gets Blackwell
# wrong (1.00) in a way nothing downstream would flag.
export TORCH_CUDA_ARCH_LIST=$(echo "$_archs" | tr ';' '\n' | sed 's/\(.*\)\(.\)$/\1.\2/' | paste -sd';')
_free_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
export MAX_JOBS=${MAX_JOBS:-$(( _free_gb / 4 < $(nproc) ? _free_gb / 4 : $(nproc) ))}
[ "${MAX_JOBS:-0}" -lt 1 ] && export MAX_JOBS=1
echo "  FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS=$MAX_JOBS  (${_free_gb}G available, $(nproc) cores)"

cd "$SRC"
# --no-build-isolation so it builds against the venv's torch; an isolated build
# resolves its own `torch` and would compile against the wrong ABI.
# --no-deps so nothing here can move torch: flash-attn declares an unpinned torch
# dependency, which is how this environment once went 2.11.0 -> 2.13.0 and took
# vllm and torchvision with it.
time python -m pip wheel . --no-build-isolation --no-deps \
    -i "${PIP_INDEX_URL:-https://pypi.org/simple/}" -w "$BUILD_ROOT/wheels"

echo
echo "=== done ==="
ls -la "$BUILD_ROOT/wheels"/flash_attn-*.whl
echo
echo "install it when no lane is running (it rewrites the SHARED venv):"
echo "  source simopd_env.sh"
echo "  pip install --no-deps $BUILD_ROOT/wheels/flash_attn-*.whl"
echo "  python -c 'import torch, flash_attn_2_cuda; print(\"ok\")'"
