# SimOPD training image for Alibaba PAI-DLC.
# Mirrors the stack verified on the Cornell cluster 2026-07-31
# (torch 2.11.0+cu129 / vLLM 0.26.0+cu129 / verl mainline / flash-attn built from source).
#
# Target confirmed 2026-08-01: PAI quota is A100-80G = sm80, the same arch and the
# same memory as the local cluster, so the flash-attn wheel we already built is
# reusable as-is. To skip the ~14 min compile, drop it into the build context first:
#   cp ~/wheels/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl deploy/pai/
# The build uses it when present and compiles from source otherwise.
# (Other arches, for reference: A10 = 8.6 | L20 = 8.9 | H20/H800/H100 = 9.0.)
#
# Build (a GPU-less build machine is fine; nvcc does not need a GPU):
#   docker build -t simopd:v1 -f deploy/pai/Dockerfile .
#   docker tag  simopd:v1 registry-vpc.cn-<region>.aliyuncs.com/<ns>/simopd:v1
#   docker push registry-vpc.cn-<region>.aliyuncs.com/<ns>/simopd:v1

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG TORCH_CUDA_ARCH_LIST="8.0"
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
# Flat wheel listing (not PEP 503), so it is consumed with --find-links. Verified
# 2026-08-01 to carry torch/torchvision/torchaudio 2.11.0+cu129 cp312 x86_64 --
# the only mainland mirror of three checked that does.
ARG TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu129/
# Prefix for github.com, e.g. https://gh-proxy.com/ . Empty by default rather than
# baking someone else's relay into the image build.
ARG GITHUB_PROXY=
ARG MAX_JOBS=32

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    UV_HTTP_TIMEOUT=600 \
    PIP_INDEX_URL=${PIP_INDEX} \
    UV_DEFAULT_INDEX=${PIP_INDEX} \
    UV_INDEX=${PIP_INDEX} \
    UV_PYTHON_INSTALL_MIRROR=https://python-standalone.org/mirror/astral-sh/python-build-standalone

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt
RUN uv venv --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"

# 1. torch trio pinned to cu129. PyPI's default torch/vLLM are cu130-only and need
#    driver >= 580; cu129 runs on any 525+ driver via CUDA minor-version compatibility.
RUN uv pip install --find-links ${TORCH_FIND_LINKS} \
        "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129"

# 2. vLLM cu129 build (the GitHub release carries the +cu129 variant; PyPI does not)
RUN uv pip install \
    "vllm @ ${GITHUB_PROXY}https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

# 3. flash-attn: no prebuilt wheel exists for torch 2.11, so use the one we already
#    built for sm80 if it was copied into the context, else compile.
COPY deploy/pai/ /tmp/paictx/
RUN if ls /tmp/paictx/flash_attn-*.whl >/dev/null 2>&1; then \
        echo "using prebuilt flash-attn wheel" && uv pip install /tmp/paictx/flash_attn-*.whl; \
    else \
        echo "compiling flash-attn for arch ${TORCH_CUDA_ARCH_LIST}" && \
        FLASH_ATTENTION_FORCE_BUILD=TRUE \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        FLASH_ATTN_CUDA_ARCHS="$(echo ${TORCH_CUDA_ARCH_LIST} | tr -d '.' | tr ';' ' ')" \
        MAX_JOBS=${MAX_JOBS} NVCC_THREADS=2 \
        uv pip install flash-attn --no-build-isolation; \
    fi

# 4. verl mainline + the extras our protocol needs (pyyaml drives the arm registry)
COPY verl /opt/verl
RUN uv pip install -e /opt/verl \
 && uv pip install huggingface_hub math-verify liger-kernel "TransferQueue==0.1.8" wandb pyyaml pandas

# 5. SimOPD itself. src/ is NOT optional: it holds every custom arm loss, and
#    sitecustomize on PYTHONPATH is what registers them inside the Ray workers where
#    the registry is read. Without it every non-stock arm dies on "Unsupported loss
#    mode" (loudly, at least). configs/ is the arm registry the run scripts read.
COPY src     /opt/simopd/src
COPY configs /opt/simopd/configs
COPY scripts /opt/simopd/scripts
COPY data    /opt/simopd/data
ENV PYTHONPATH=/opt/simopd/src

WORKDIR /opt/simopd

# Fail the build, not a 40-hour run, if the stack or the arm registry is broken.
RUN python -c "import torch, vllm, verl, flash_attn; \
print('torch', torch.__version__, '| vllm', vllm.__version__, '| fa', flash_attn.__version__)" \
 && python -c "\
import verl.trainer.distillation.losses as vl; \
from simopd.topk_losses import _dispatching_compute_topk_loss as d; \
stock={'kl','k1','abs','mse','k2','low_var_kl','k3','forward_kl_topk'}; \
arms=sorted(set(vl.DISTILLATION_LOSS_REGISTRY)-stock); \
assert len(arms) == 12, f'expected 12 arm losses, got {len(arms)}: {arms}'; \
assert vl.compute_topk_loss is d, 'top-k dispatch not installed'; \
print('arm losses registered:', len(arms))" \
 && python scripts/arm.py check | head -1
