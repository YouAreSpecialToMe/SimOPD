#!/usr/bin/env bash
# Bring up SimOPD on a fresh PAI-DSW instance (interactive, 8x A100-80G).
#
#   git clone git@github.com:YouAreSpecialToMe/SimOPD.git && cd SimOPD
#   bash deploy/dsw/setup.sh
#
# Reproduces the stack verified on the Cornell cluster 2026-07-31/08-01. Idempotent:
# re-running skips whatever is already in place, so it is safe after a DSW restart
# (only the workspace volume survives those, which is why everything lands under
# $SIMOPD_ROOT rather than in the image).

set -euo pipefail

SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$SIMOPD_ROOT/../simopd_data}
cd "$SIMOPD_ROOT"

# ---------------------------------------------------------------------------
# Mirrors. On by default because this targets a mainland instance; set
# SIMOPD_MIRRORS=0 to go straight upstream. Every one is overridable.
# ---------------------------------------------------------------------------
if [ "${SIMOPD_MIRRORS:-1}" = "1" ]; then
    # uv downloads its own CPython build; without this it pulls from GitHub.
    export UV_PYTHON_INSTALL_MIRROR=${UV_PYTHON_INSTALL_MIRROR:-https://python-standalone.org/mirror/astral-sh/python-build-standalone}
    # DEFAULT_INDEX *replaces* PyPI; UV_INDEX alone only appends one, leaving uv
    # free to reach pypi.org and stall. Both are set so either uv version behaves.
    export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
    export UV_INDEX=${UV_INDEX:-$UV_DEFAULT_INDEX}
    export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
    # A flat wheel listing, not a PEP 503 index, so it goes through --find-links.
    # Verified 2026-08-01 to carry torch/torchvision/torchaudio 2.11.0+cu129 cp312
    # x86_64 -- the only mainland mirror of the three checked that does (Tsinghua
    # 404s on cu129, SJTU redirects away).
    TORCH_FIND_LINKS=${TORCH_FIND_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu129/}
    # uv hardlinks from its cache by default. On DSW the cache sits on local disk
    # and the venv on the /mnt/workspace network volume; hardlinking across
    # filesystems is the kind of thing that half-succeeds and leaves a package
    # directory with no __init__.py. Copying is slower and reliable.
    export UV_LINK_MODE=${UV_LINK_MODE:-copy}
    # github.com release assets and clones are the remaining slow path. Set e.g.
    # GITHUB_PROXY=https://gh-proxy.com/ to route them; left empty by default
    # rather than hardcoding someone else's relay into the setup path.
    GITHUB_PROXY=${GITHUB_PROXY:-}
else
    TORCH_FIND_LINKS=""; GITHUB_PROXY=""
    export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
fi
GH() { echo "${GITHUB_PROXY}$1"; }
echo "mirrors: pypi=${UV_DEFAULT_INDEX:-upstream} hf=$HF_ENDPOINT torch=${TORCH_FIND_LINKS:-upstream} gh_proxy=${GITHUB_PROXY:-none}"

echo "=== [0/6] pre-flight ==="
# Three things that decide whether the rest of this script can work at all. Checked
# up front so a mismatch fails here instead of 40 minutes into a wheel install.
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || {
    echo "no GPU visible -- wrong instance type?"; exit 1; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -lt 525 ]; then
    echo "FATAL: driver $DRV < 525; the cu129 wheels this project pins will not load." >&2
    exit 1
fi
# CUDA family is decided by nvcc, NOT by the driver. The driver only says what can
# RUN; flash-attn has no prebuilt wheel for torch 2.11 and must be compiled, and
# torch refuses to build an extension when its CUDA major differs from nvcc's:
#   RuntimeError: The detected CUDA version (12.8) mismatches the version that was
#   used to compile PyTorch (13.0)
# A >=580 driver on a box whose toolkit is 12.8 therefore still needs cu129 torch.
# (Minor differences are fine -- torch warns for 12.9-vs-12.8 and carries on.)
NVCC_MAJ=""
if command -v nvcc >/dev/null 2>&1; then
    NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
    NVCC_MAJ=${NVCC_VER%%.*}
    echo "nvcc $NVCC_VER (build toolkit)"
else
    echo "WARNING: no nvcc on PATH; flash-attn cannot be compiled here." >&2
fi
if [ "${NVCC_MAJ:-12}" = "13" ] && [ "$DRV" -ge 580 ]; then
    CUDA_FLAVOR=cu130
    echo "driver $DRV + nvcc ${NVCC_VER:-?} -> cu130 (all from the Aliyun mirror, no GitHub)"
else
    CUDA_FLAVOR=cu129
    if [ "${NVCC_MAJ:-}" = "12" ] && [ "$DRV" -ge 580 ]; then
        echo "driver $DRV would allow cu130, but nvcc is ${NVCC_VER} -> cu129, so flash-attn can build"
    else
        echo "driver $DRV, nvcc ${NVCC_VER:-none} -> cu129"
    fi
    if [ "${SIMOPD_MIRRORS:-1}" = "1" ] && [ -z "${GITHUB_PROXY:-}" ] && [ -z "${VLLM_WHEEL:-}" ]; then
        echo "  NOTE: the cu129 vLLM wheel (~400MB) is a GitHub release asset with no" >&2
        echo "  mainland mirror. If it stalls, set GITHUB_PROXY=https://<relay>/ or" >&2
        echo "  download it yourself and pass VLLM_WHEEL=/path/to/the.whl" >&2
    fi
fi

# A managed image often exports PYTHONPATH at its own site-packages. That wins over
# the venv we are about to build, so `import torch` can resolve to the image's copy
# -- which is how a venv with a perfectly good torch still reports
# "module 'torch' has no attribute '__version__'".
if [ -n "${PYTHONPATH:-}" ]; then
    echo "WARNING: PYTHONPATH is set to '$PYTHONPATH'." >&2
    echo "  It takes precedence over the venv. If imports resolve oddly, unset it:" >&2
    echo "    env -u PYTHONPATH bash deploy/dsw/setup.sh" >&2
fi
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free space here: ${AVAIL_GB}G"
# ~50G models + ~17G per run of checkpoints (MAX_CKPT_KEEP=2). A 17-run campaign
# wants ~350G; below 150G you will run out mid-campaign, not at the start.
[ "${AVAIL_GB:-0}" -lt 150 ] && echo "WARNING: under 150G free -- cap runs or raise MAX_CKPT_KEEP=1" >&2

echo "=== [1/6] third-party checkouts ==="
[ -d verl ] || git clone --depth 1 "$(GH https://github.com/volcengine/verl.git)"
# Read-only references for arm provenance checks (PROTOCOL-unified section 2); the
# audit ports their methods, never their harnesses.
[ -d EasyOPD ] || git clone --depth 1 "$(GH https://github.com/lds-ustc/EasyOPD.git)" || true
[ -d OPD ] || git clone --depth 1 "$(GH https://github.com/thunlp/OPD.git)" || true

echo "=== [2/6] python env ==="
# uv is resolved to an absolute path, not assumed to be on PATH: the installer
# script drops it in ~/.local/bin and `pip install uv` puts it in that python's
# scripts dir, neither of which a DSW shell necessarily searches. (`python -m uv`
# is NOT an option -- the wheel ships only a binary, no importable module.)
# pip-from-the-mirror is tried first; astral.sh is the overseas, slower path.
find_uv() {
    command -v uv 2>/dev/null && return 0
    for c in "$(python3 -c 'import sysconfig,os;print(os.path.join(sysconfig.get_path("scripts"),"uv"))' 2>/dev/null)" \
             "$(python3 -m site --user-base 2>/dev/null)/bin/uv" \
             "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
    done
    return 1
}

UV=$(find_uv || true)
if [ -z "$UV" ]; then
    python3 -m pip install -q -i "${UV_DEFAULT_INDEX:-https://pypi.org/simple/}" uv 2>/dev/null || true
    UV=$(find_uv || true)
fi
if [ -z "$UV" ]; then
    curl -LsSf --max-time 180 https://astral.sh/uv/install.sh | sh || true
    UV=$(find_uv || true)
fi
[ -n "$UV" ] || {
    echo "FATAL: could not install uv. Install it by hand, then re-run:" >&2
    echo "  python3 -m pip install -i ${UV_DEFAULT_INDEX:-https://pypi.org/simple/} uv" >&2
    echo "  # or: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}
export PATH="$(dirname "$UV"):$PATH"
echo "uv: $UV ($("$UV" --version 2>&1 | head -1))"

[ -d .venv ] || $UV venv --python 3.12 .venv
source .venv/bin/activate
# From here on uv installs into the activated venv; keep using the same resolved
# command so a PATH-less uv still works.

echo "=== [3/6] torch + vLLM (cu129) ==="
# cu129, not the cu130 PyPI default: cu130 needs driver >= 580, while cu129 runs on
# any 525+ driver through CUDA minor-version compatibility. Check with nvidia-smi.
VLLM_WHEEL=${VLLM_WHEEL:-$(GH https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl)}

# Probe an ATTRIBUTE, not just importability. An interrupted wheel install leaves
# site-packages/torch/ without its __init__.py, and Python then imports it happily
# as an empty namespace package -- `import torch` succeeds and
# `torch.__version__` raises "module has no attribute". A plain `import vllm`
# guard therefore reports "already installed" and skips the whole step, leaving
# torch absent. Touching __version__ is what tells a real package from a stub.
if ! python -c "import torch, vllm; torch.__version__; vllm.__version__" 2>/dev/null; then
    # Clear any half-written tree first; installing over it keeps the stub.
    if python -c "import torch" 2>/dev/null && ! python -c "import torch; torch.__version__" 2>/dev/null; then
        echo "  partial torch detected (namespace stub) -- removing before reinstall"
        $UV pip uninstall torch torchvision torchaudio vllm >/dev/null 2>&1 || true
        # A stub has no dist-info, so uninstall leaves it; remove the trees by hand
        # or the reinstall lands on top of the wreckage.
        _site=$(python -c 'import site;print(site.getsitepackages()[0])')
        rm -rf "${_site:?}/torch" "${_site:?}/vllm" "${_site:?}"/torch-*.dist-info 2>/dev/null || true
    fi

    if [ "$CUDA_FLAVOR" = "cu130" ]; then
        # Plain PyPI builds, both mirrored: nothing here leaves the mainland.
        $UV pip install "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" "vllm==0.26.0"
    elif [ -n "$TORCH_FIND_LINKS" ]; then
        $UV pip install --find-links "$TORCH_FIND_LINKS" \
            "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129"
        $UV pip install "vllm @ $VLLM_WHEEL"
    else
        $UV pip install --index-url https://download.pytorch.org/whl/cu129 \
            "torch==2.11.0+cu129" "torchvision==0.26.0+cu129"
        $UV pip install "torchaudio==2.11.0"
        $UV pip install "vllm @ $VLLM_WHEEL"
    fi
fi

# Verify torch here, not at the first thing that imports it. pip's CUDA libraries
# are separate packages and the resolver can pick an inconsistent set: a 12.9-era
# libcusparse against an older libnvJitLink gives
#   undefined symbol: __nvJitLinkGetErrorLogSize_12_9
# which surfaces wherever torch is first imported -- typically the flash-attn build,
# so it reads like a flash-attn problem and is not one. The known-good env has
# nvidia-nvjitlink-cu12 12.9.86 with cusparse 12.5.10.65; the symbol name tells you
# the floor, so raise nvjitlink to meet it and re-check.
if ! python -c "import torch; torch.__version__" 2>/tmp/torchimp.txt; then
    if grep -q "nvJitLink" /tmp/torchimp.txt; then
        echo "  torch import failed on an nvJitLink symbol -- realigning the CUDA runtime packages"
        $UV pip install -U "nvidia-nvjitlink-cu12>=12.9"
        python -c "import torch; print('  torch ok after realignment:', torch.__version__)"
    elif grep -q "no attribute" /tmp/torchimp.txt; then
        # `import torch` works but torch.__version__ does not: something other than
        # a working torch is answering to that name. Which one is the whole question,
        # so print the evidence instead of guessing -- __file__ names the culprit.
        echo "FATAL: torch imports but has no __version__ -- the wrong torch is on the path." >&2
        python - >&2 <<'PYDIAG' || true
import sys, importlib.util
spec = importlib.util.find_spec("torch")
print(f"  torch resolves to : {getattr(spec, 'origin', None)}")
print(f"  search locations  : {getattr(spec, 'submodule_search_locations', None)}")
print(f"  interpreter       : {sys.executable}")
print(f"  user site enabled : {getattr(sys, 'flags', None) and not sys.flags.no_user_site}")
print("  sys.path:")
for p in sys.path:
    print(f"    {p or '<cwd>'}")
PYDIAG
        echo "  Most likely: another torch (image-provided, or a user-site one under" >&2
        echo "  ~/.local/lib) is shadowing the venv. Re-run with PYTHONNOUSERSITE=1, or" >&2
        echo "  \$UV pip install --force-reinstall --no-cache the torch line above." >&2
        exit 1
    else
        cat /tmp/torchimp.txt >&2
        echo "FATAL: torch does not import; nothing downstream can work." >&2
        exit 1
    fi
else
    python -c "import torch; print('torch ok:', torch.__version__, '| cuda', torch.version.cuda)"
fi

echo "=== [4/6] verl + extras ==="
$UV pip install -e ./verl
$UV pip install huggingface_hub math-verify liger-kernel "TransferQueue==0.1.8" wandb pyyaml pandas

echo "=== [5/6] flash-attn ==="
# LAST, deliberately. flash-attn compiles against whatever torch is installed at
# build time, and installing verl/liger-kernel afterwards can re-resolve torch --
# uv treats 2.11.0+cu129 and 2.11.0 as the same version with different local tags,
# so a swap is silent. The compiled extension then fails at import with
# "undefined symbol". Building it after everything else pins it to the final torch.
#
# No prebuilt wheel exists for torch 2.11, so this compiles (~14 min for one arch).
# A100 = sm80. Drop a prebuilt wheel in deploy/dsw/ to skip it -- but only one built
# against this exact torch, or you get the same undefined symbol.
$UV pip install packaging ninja psutil setuptools wheel   # flash-attn build deps; --no-build-isolation means they must already be here
# Probe the compiled extension, not the package: `import flash_attn` can succeed
# while flash_attn_2_cuda is the one carrying the missing symbols. And uninstall
# first -- a broken build of the same version makes `uv pip install` a no-op, so
# a re-run would silently keep it.
python -c "import torch, flash_attn_2_cuda" 2>/dev/null || {   # torch first: it loads libc10.so that the extension links against
    $UV pip uninstall flash-attn >/dev/null 2>&1 || true
    if ls deploy/dsw/flash_attn-*.whl >/dev/null 2>&1; then
        $UV pip install --force-reinstall deploy/dsw/flash_attn-*.whl
    else
        # FORCE_BUILD stops flash-attn's setup.py from first trying to fetch a
        # prebuilt wheel off GitHub -- a request that tends to hang rather than
        # fail from the mainland, so the build never starts.
        FLASH_ATTENTION_FORCE_BUILD=TRUE \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
        FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}" \
        MAX_JOBS="${MAX_JOBS:-32}" NVCC_THREADS=2 \
        $UV pip install --force-reinstall --no-cache flash-attn --no-build-isolation
    fi
}

# Catch an ABI mismatch here, in ten seconds, instead of at the first training step.
python - <<'PYCHK'
import torch, flash_attn
import flash_attn_2_cuda  # the extension whose symbols are the ones that go missing
print(f"flash_attn {flash_attn.__version__} imports cleanly against torch {torch.__version__}")
PYCHK

echo "=== [6/6] models + data ==="
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf_cache}
mkdir -p "$HF_HOME" "$DATA_ROOT"
for m in Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B Qwen/Qwen3-1.7B-Base \
         Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B; do
    hf download "$m" --exclude "*.pth" >/dev/null && echo "  ok $m"
done
[ -f "$DATA_ROOT/simopd_math/train.parquet" ] || \
    python scripts/prep_nemotron_math.py --local_save_dir "$DATA_ROOT/simopd_math"
python - <<'PY'
import datasets
for d, split in [("HuggingFaceH4/MATH-500","test"), ("math-ai/amc23","test"),
                 ("HuggingFaceH4/aime_2024","train"), ("math-ai/aime25","test"),
                 ("math-ai/minervamath","test"), ("google/IFEval","train")]:
    datasets.load_dataset(d, split=split)
print("eval benchmarks cached")
PY

cat <<EOF

=== setup done ===
Put these in your shell (or ~/.bashrc) before running anything:

  cd $SIMOPD_ROOT && source .venv/bin/activate
  export PYTHONPATH=$SIMOPD_ROOT/src        # registers our arm losses in Ray workers
  export HF_ENDPOINT=$HF_ENDPOINT
  export HF_HOME=$HF_HOME
  export DATA_DIR=$DATA_ROOT/simopd_math
  export CKPT_ROOT=$DATA_ROOT/ckpt
  export WANDB_DIR=$DATA_ROOT/wandb

Then verify, then launch:
  python scripts/arm.py check
  bash deploy/dsw/run_parallel.sh --rehearsal      # 3 steps per arm, catches breakage cheap
  bash deploy/dsw/run_parallel.sh                  # the real 300-step campaign
EOF
