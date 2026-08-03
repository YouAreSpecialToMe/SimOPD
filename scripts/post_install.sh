#!/usr/bin/env bash
# SimOPD W1: post-install fix + data prep + job submission.
# Run after `uv pip install -e ./verl` completes.

set -uo pipefail
cd "$HOME/pythonProject/SimOPD"
source "${SIMOPD_VENV:-simopd}/bin/activate"

echo "== [1/5] torch trio -> cu129 =="
uv pip install "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" \
    --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -3 || exit 1
uv pip install "torchaudio==2.11.0" 2>&1 | tail -2

echo "== [2/5] vllm 0.26.0+cu129 (GitHub wheel) =="
uv pip install "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" 2>&1 | tail -4 || exit 1

echo "== [3/5] verify imports =="
python - <<'EOF' || exit 1
import torch, vllm, transformers, ray, verl
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("ray", ray.__version__)
print("verl import OK")
EOF

echo "== [4/5] gsm8k data prep =="
python verl/examples/data_preprocess/gsm8k.py 2>&1 | tail -2
ls -la "$HOME/data/gsm8k/" || exit 1

echo "== [5/5] submit slurm jobs =="
sbatch slurm/build_flash_attn.sbatch
sbatch slurm/smoke.sbatch
squeue -u zz865 -o "%.10i %.28P %.14j %.8T %.10M %R" 2>/dev/null | head -8

echo "POST-INSTALL DONE"
