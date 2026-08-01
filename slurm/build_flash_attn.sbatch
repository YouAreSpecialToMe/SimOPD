#!/usr/bin/env bash
#SBATCH -J fa-build
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

# Build flash-attn from source against torch 2.11+cu129, sm80 only.
# No GPU needed for compilation; nvcc 12.8 lives on the compute node.

set -euo pipefail

PROJ_DIR=$HOME/pythonProject/SimOPD
cd "$PROJ_DIR"
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TMPDIR=/scratch/zz865/tmp
mkdir -p "$TMPDIR" "$HOME/wheels"

# sm80 = A100 (nlplarge), sm86 = RTX A6000 (rush partition screening pool)
export FLASH_ATTN_CUDA_ARCHS="80;86"
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export MAX_JOBS=28
export NVCC_THREADS=2

nvcc --version | tail -2

pip wheel flash-attn --no-build-isolation --no-deps -w "$HOME/wheels" -v 2>&1 | tail -20
ls -la "$HOME/wheels"

# install into the shared venv right away
uv pip install "$HOME"/wheels/flash_attn-*.whl
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
