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
HF_ENDPOINT_DEFAULT=${HF_ENDPOINT:-https://hf-mirror.com}   # mainland-friendly HF mirror
cd "$SIMOPD_ROOT"

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
echo "driver $DRV: OK for cu129 (needs >= 525 via CUDA minor-version compatibility)"
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free space here: ${AVAIL_GB}G"
# ~50G models + ~17G per run of checkpoints (MAX_CKPT_KEEP=2). A 17-run campaign
# wants ~350G; below 150G you will run out mid-campaign, not at the start.
[ "${AVAIL_GB:-0}" -lt 150 ] && echo "WARNING: under 150G free -- cap runs or raise MAX_CKPT_KEEP=1" >&2

echo "=== [1/6] third-party checkouts ==="
[ -d verl ] || git clone --depth 1 https://github.com/volcengine/verl.git
# Read-only references for arm provenance checks (PROTOCOL-unified section 2); the
# audit ports their methods, never their harnesses.
[ -d EasyOPD ] || git clone --depth 1 https://github.com/lds-ustc/EasyOPD.git || true
[ -d OPD ] || git clone --depth 1 https://github.com/thunlp/OPD.git || true

echo "=== [2/6] python env ==="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.12 .venv
source .venv/bin/activate

echo "=== [3/6] torch + vLLM (cu129) ==="
# cu129, not the cu130 PyPI default: cu130 needs driver >= 580, while cu129 runs on
# any 525+ driver through CUDA minor-version compatibility. Check with nvidia-smi.
python -c "import vllm" 2>/dev/null || {
    uv pip install "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" \
        --index-url https://download.pytorch.org/whl/cu129
    uv pip install "torchaudio==2.11.0"
    uv pip install "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
}

echo "=== [4/6] flash-attn ==="
# No prebuilt wheel exists for torch 2.11, so this compiles (~14 min for one arch).
# A100 = sm80. Drop a prebuilt wheel in deploy/dsw/ to skip it.
python -c "import flash_attn" 2>/dev/null || {
    if ls deploy/dsw/flash_attn-*.whl >/dev/null 2>&1; then
        uv pip install deploy/dsw/flash_attn-*.whl
    else
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
        FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}" \
        MAX_JOBS="${MAX_JOBS:-32}" NVCC_THREADS=2 \
        uv pip install flash-attn --no-build-isolation
    fi
}

echo "=== [5/6] verl + extras ==="
uv pip install -e ./verl
uv pip install huggingface_hub math-verify liger-kernel "TransferQueue==0.1.8" wandb pyyaml pandas

echo "=== [6/6] models + data ==="
export HF_ENDPOINT=$HF_ENDPOINT_DEFAULT
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
  export HF_HOME=$HF_HOME
  export DATA_DIR=$DATA_ROOT/simopd_math
  export CKPT_ROOT=$DATA_ROOT/ckpt
  export WANDB_DIR=$DATA_ROOT/wandb

Then verify, then launch:
  python scripts/arm.py check
  bash deploy/dsw/run_parallel.sh --rehearsal      # 3 steps per arm, catches breakage cheap
  bash deploy/dsw/run_parallel.sh                  # the real 300-step campaign
EOF
