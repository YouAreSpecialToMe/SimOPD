#!/usr/bin/env bash
# Read-only health check for a SimOPD install. Run it any time; it changes nothing.
#
#   bash deploy/dsw/doctor.sh
#
# Every check here exists because it has actually gone wrong on this project, and
# each failure prints the fix rather than just the symptom. Paste the whole output
# when reporting a problem -- the evidence is the point.

set -uo pipefail

SIMOPD_ROOT=${SIMOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$SIMOPD_ROOT"
# One venv, ./simopd, on every machine. SIMOPD_VENV overrides it.
VENV=${SIMOPD_VENV:-simopd}
PROBLEMS=0
ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; PROBLEMS=$((PROBLEMS + 1)); }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$1"; }
fix()  { printf '       -> %s\n' "$1"; }

echo "=== SimOPD doctor @ $(date -u +%FT%TZ) ==="
echo "    repo: $SIMOPD_ROOT ($(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"

echo; echo "[environment]"
# A managed image often exports PYTHONPATH at its own site-packages, which wins
# over the venv and is the classic cause of "torch has no attribute __version__".
if [ -n "${PYTHONPATH:-}" ]; then
    case ":${PYTHONPATH}:" in
        *":$SIMOPD_ROOT/src:"*) ok "PYTHONPATH contains our src (expected when running arms)" ;;
        *) warn "PYTHONPATH='$PYTHONPATH' -- takes precedence over the venv"
           fix "if imports resolve oddly: env -u PYTHONPATH bash deploy/dsw/setup.sh" ;;
    esac
else
    ok "PYTHONPATH unset"
fi
# PAI images export this, and verl acts on it at import: modelscope's patch_hub()
# reroutes every huggingface_hub call, and ModelScope's default branch is 'master',
# so asking for HF's 'main' fails on models that are already downloaded. It kills the
# run at rollout-worker startup, minutes in, with a traceback about a missing revision.
case "$(printf '%s' "${VERL_USE_MODELSCOPE:-False}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes) bad "VERL_USE_MODELSCOPE=$VERL_USE_MODELSCOPE -- verl will route HF downloads through ModelScope"
                fix "export VERL_USE_MODELSCOPE=False   (setup.sh now writes this into simopd_env.sh)" ;;
    *)          ok "VERL_USE_MODELSCOPE off -- hub calls stay on HF/\${HF_ENDPOINT}" ;;
esac
# vLLM puts its worker IPC and NCCL buffers in shared memory. Containers default
# /dev/shm to 64MB, which does not fail loudly: the rollout worker dies and Ray
# reports it as `ActorUnavailableError ... rpc_code: 14`, which names neither shm
# nor the worker's actual error.
SHM=$(df -m /dev/shm 2>/dev/null | tail -1 | awk '{print $2}')
if [ "${SHM:-0}" -ge 8192 ]; then ok "/dev/shm ${SHM}M"
elif [ "${SHM:-0}" -gt 0 ]; then bad "/dev/shm is only ${SHM}M -- vLLM workers will die at startup"
     fix "restart the container with --shm-size=32g (DSW: raise it in the instance spec)"
else warn "could not read /dev/shm"; fi
# A Ray head left behind by a crashed run is not inert: the next ray.init() attaches
# to it, and its workers inherit the environment from when IT started -- so a variable
# you have since fixed in your shell is still wrong inside them.
if pgrep -f "raylet|gcs_server" >/dev/null 2>&1; then
    warn "a Ray cluster is already running -- a new run will attach to it, stale env and all"
    fix "ray stop --force   # single-lane machines only; on 4 lanes use the per-lane cleanup in _lane.sh"
else
    ok "no leftover Ray cluster"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    MAJ=${DRV%%.*}
    if [ "${MAJ:-0}" -ge 580 ]; then ok "driver $DRV, ${NGPU} GPU -- cu130 path available (no GitHub needed)"
    elif [ "${MAJ:-0}" -ge 525 ]; then ok "driver $DRV, ${NGPU} GPU -- cu129 path"
    else bad "driver $DRV is below 525; the pinned wheels cannot load"; fi
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/       /'
elif [ -n "${SIMOPD_EXPECT_GPU:-}" ]; then
    bad "nvidia-smi not found"
else
    warn "nvidia-smi not found (expected on a login node; on DSW this is a problem)"
fi
# The build toolkit, not the driver, is what has to agree with torch: flash-attn is
# compiled here, and torch refuses across a CUDA major mismatch.
NVCC_VER=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
if [ -n "$NVCC_VER" ]; then ok "nvcc $NVCC_VER -> torch must be cu${NVCC_VER%%.*}x"
else warn "no nvcc on PATH -- flash-attn cannot be compiled here"; fi
AVAIL=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${AVAIL:-0}" -ge 150 ] && ok "${AVAIL}G free here" || warn "${AVAIL:-?}G free -- a 17-run campaign wants ~350G (MAX_CKPT_KEEP=1 to halve it)"

echo; echo "[python]"
if [ -d "$VENV" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    ok "venv activated: $(python -V 2>&1) at $(command -v python)"
else
    bad "no venv at ./$VENV"; fix "bash deploy/dsw/setup.sh"
fi

probe() {  # probe <label> <module> <attribute>
    local label=$1 mod=$2 attr=$3 out
    out=$(python - "$mod" "$attr" <<'PY' 2>&1
import importlib, importlib.util, sys
mod, attr = sys.argv[1], sys.argv[2]
spec = importlib.util.find_spec(mod)
if spec is None:
    print("MISSING"); raise SystemExit
try:
    m = importlib.import_module(mod)
    print(f"{getattr(m, attr, '<no ' + attr + '>')}\t{getattr(m, '__file__', spec.origin)}")
except Exception as e:
    print(f"IMPORTERROR\t{type(e).__name__}: {e}")
PY
)
    printf '%s' "$out"
}

echo; echo "[torch]"
T=$(probe torch torch __version__)
case "$T" in
    MISSING*)      bad "torch not installed"; fix "bash deploy/dsw/setup.sh" ;;
    IMPORTERROR*)  bad "torch import fails: ${T#*$'\t'}"
                   case "$T" in
                     *nvJitLink*) fix "\$UV pip install -U 'nvidia-nvjitlink-cu12>=12.9'  (pip's CUDA libs are out of step)" ;;
                     *) fix "see the traceback above; nothing downstream can work until torch imports" ;;
                   esac ;;
    "<no __version__>"*)
                   ORIGIN=${T#*$'\t'}
                   if [ "$ORIGIN" = "None" ]; then
                       # spec.origin is None <=> namespace package <=> a bare directory
                       # with no __init__.py. Almost always an install that died partway
                       # (the dependencies land first, the package body last).
                       bad "torch is a namespace package -- an install that never finished"
                       fix "the dependency wheels are there but torch's own body is not; force a clean reinstall:"
                       fix "  rm -rf \"\$(python -c 'import site;print(site.getsitepackages()[0])')/torch\""
                       fix "  bash deploy/dsw/setup.sh          # detects and repairs this case"
                   else
                       bad "torch imports but has no __version__ -- a different 'torch' is answering"
                       fix "it resolves to: $ORIGIN"
                       fix "usually a PYTHONPATH or user-site copy shadowing the venv"
                   fi ;;
    *)             TC=$(python -c 'import torch;print(torch.version.cuda or "none")' 2>/dev/null)
                   ok "torch ${T%%$'\t'*}  (cuda $TC)"
                   printf '       from %s\n' "${T#*$'\t'}"
                   if [ -n "$NVCC_VER" ] && [ "${TC%%.*}" != "${NVCC_VER%%.*}" ]; then
                       bad "torch is cuda $TC but nvcc is $NVCC_VER -- CUDA majors differ"
                       fix "extensions cannot be compiled against this pair; flash-attn will fail with"
                       fix "  'The detected CUDA version mismatches the version used to compile PyTorch'"
                       fix "install the torch matching nvcc:  bash deploy/dsw/repair_torch.sh"
                   fi ;;
esac
# The pip CUDA stack must be internally consistent; a 12.9-era cusparse against an
# older nvJitLink is the mismatch that surfaces wherever torch is first imported.
python -c "
from importlib.metadata import version, PackageNotFoundError
for p in ('nvidia-nvjitlink-cu12','nvidia-cusparse-cu12','nvidia-cublas-cu12'):
    try: print(f'       {p}: {version(p)}')
    except PackageNotFoundError: pass" 2>/dev/null

echo; echo "[vllm / flash-attn / verl]"
for spec in "vllm vllm __version__" "flash_attn flash_attn __version__" "verl verl __version__"; do
    # shellcheck disable=SC2086
    set -- $spec
    R=$(probe "$1" "$2" "$3")
    case "$R" in
        MISSING*)     bad "$1 not installed" ;;
        IMPORTERROR*) bad "$1: ${R#*$'\t'}" ;;
        *)            ok "$1 ${R%%$'\t'*}" ;;
    esac
done
# flash-attn's compiled extension is where "undefined symbol" lives; the package
# itself can import fine while this fails.
# torch first, always: the extension links against libc10.so, which only exists
# in the process once torch has been imported. Probing it alone always "fails".
if python -c "import torch, flash_attn_2_cuda" 2>/tmp/fa.txt; then
    ok "flash_attn_2_cuda extension loads (ABI matches this torch)"
else
    bad "flash_attn_2_cuda fails: $(tail -1 /tmp/fa.txt)"
    fix "rebuild it AGAINST THE CURRENT torch -- it must be installed last:"
    fix "  SITE=\$(python -c 'import site;print(site.getsitepackages()[0])')"
    fix "  rm -rf \$SITE/flash_attn \$SITE/flash_attn_2_cuda*.so \$SITE/flash_attn-*.dist-info"
    fix "  FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=8.0 uv pip install --force-reinstall --no-cache flash-attn --no-build-isolation"
    fix "  (the top-level .so is separate from the package dir -- removing only the dir leaves it)"
fi

echo; echo "[simopd arms]"
if python -c "import yaml" 2>/dev/null; then
    python scripts/arm.py check 2>&1 | head -1 | sed 's/^/  /'
else
    bad "pyyaml missing (the arm registry needs it)"
fi
ARMS=$(PYTHONPATH="$SIMOPD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -c "
import verl.trainer.distillation.losses as vl
stock={'kl','k1','abs','mse','k2','low_var_kl','k3','forward_kl_topk'}
print(len(set(vl.DISTILLATION_LOSS_REGISTRY)-stock))" 2>/dev/null)
if [ "${ARMS:-0}" = "12" ]; then
    ok "12 custom arm losses register through sitecustomize"
else
    bad "expected 12 custom arm losses, got '${ARMS:-<import failed>}'"
    fix "PYTHONPATH must include $SIMOPD_ROOT/src; verl must import cleanly"
fi

echo; echo "[data]"
# Look wherever the data actually is -- DSW puts it beside the repo, the Cornell
# box under ~/data. Guessing one layout and reporting the other missing is noise.
_dd=${DATA_DIR:-}
for _c in "$_dd" "$SIMOPD_ROOT/../simopd_data/simopd_math" "$HOME/data/simopd_math"; do
    [ -n "$_c" ] && [ -f "$_c/train.parquet" ] && { _dd=$_c; break; }
done
_dd=${_dd:-$SIMOPD_ROOT/../simopd_data/simopd_math}
# One source of truth for what "the assets are present" means, shared with setup.
if python "$SIMOPD_ROOT/scripts/fetch_assets.py" --check --data-dir "$_dd" >/tmp/assets.txt 2>&1; then
    ok "$(grep -c cached /tmp/assets.txt) assets cached (models, eval sets, transfer benches, training data)"
else
    bad "missing assets:"
    grep -E "MISSING|FAILED" /tmp/assets.txt | sed 's/^/       /' >&2
    fix "bash deploy/dsw/setup.sh   # fetches only what is absent"
fi
# Deliberately not run here: the code harness self-check executes 542 canonical
# solutions and takes minutes, which is not what a doctor is for.
python - <<'PY' 2>/dev/null || fix "python scripts/transfer_eval.py --selfcheck   # verify the code sandbox on this machine"
import importlib.util as u, sys
sys.exit(0 if all(u.find_spec(m) for m in ("evalplus", "nltk", "langdetect", "immutabledict")) else 1)
PY

echo; echo "[mirrors]"
for u in "${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}" "${HF_ENDPOINT:-https://hf-mirror.com}"; do
    c=$(timeout 12 curl -s -o /dev/null -w "%{http_code}" "$u" 2>/dev/null)
    [ "$c" = "200" ] && ok "reachable ($c) $u" || warn "$c $u"
done

echo
if [ "$PROBLEMS" -eq 0 ]; then
    echo "=== no problems found ==="
    echo "next:  bash deploy/dsw/run_parallel.sh --rehearsal"
else
    echo "=== $PROBLEMS problem(s) -- see the -> lines above ==="
fi
exit 0
