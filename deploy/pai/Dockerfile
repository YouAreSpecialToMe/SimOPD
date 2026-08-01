# SimOPD training image for Alibaba PAI-DLC.
# Mirrors the stack verified on the Cornell cluster 2026-07-31
# (torch 2.11.0+cu129 / vLLM 0.26.0+cu129 / verl mainline / flash-attn built from source).
#
# Target confirmed 2026-08-01: PAI quota is A100 = sm80, same arch as the local
# cluster, so the flash-attn wheel we already built is reusable as-is. To skip the
# ~14 min compile, drop it into the build context first:
#   cp ~/wheels/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl deploy/pai/
# The Dockerfile uses it when present and compiles from source otherwise.
# (Other arches, for reference: A10 = 8.6 | L20 = 8.9 | H20/H800/H100 = 9.0.)
#
# Build (on a GPU-less build machine is fine; nvcc does not need a GPU):
#   docker build -t simopd:v1 -f deploy/pai/Dockerfile .
#   docker tag simopd:v1 registry-vpc.cn-<region>.aliyuncs.com/<ns>/simopd:v1
#   docker push registry-vpc.cn-<region>.aliyuncs.com/<ns>/simopd:v1

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG TORCH_CUDA_ARCH_LIST="8.0"
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
ARG MAX_JOBS=32

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    UV_HTTP_TIMEOUT=600 \
    PIP_INDEX_URL=${PIP_INDEX}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt
RUN uv venv --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"

# 1. torch trio pinned to cu129 (matches the driver ceiling we validated; also runs on newer drivers)
RUN uv pip install "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" \
        --index-url https://download.pytorch.org/whl/cu129 \
 && uv pip install "torchaudio==2.11.0"

# 2. vLLM cu129 build (PyPI default is cu130-only and needs driver >= 580)
RUN uv pip install \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

# 3. flash-attn from source — no prebuilt wheel exists for torch 2.11
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    FLASH_ATTN_CUDA_ARCHS="$(echo ${TORCH_CUDA_ARCH_LIST} | tr -d '.' | tr ';' ' ')" \
    MAX_JOBS=${MAX_JOBS} NVCC_THREADS=2 \
    uv pip install flash-attn --no-build-isolation

# 4. verl mainline + our extras
COPY verl /opt/verl
RUN uv pip install -e /opt/verl \
 && uv pip install huggingface_hub math-verify liger-kernel "TransferQueue==0.1.8" wandb

# 5. SimOPD scripts/configs (docs and third-party clones excluded via .dockerignore)
COPY scripts /opt/simopd/scripts
COPY data    /opt/simopd/data

WORKDIR /opt/simopd
RUN python -c "import torch, vllm, verl, flash_attn; \
print('torch', torch.__version__, '| vllm', vllm.__version__, '| fa', flash_attn.__version__)"
