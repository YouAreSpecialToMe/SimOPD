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
# One venv, ./simopd, on every machine. SIMOPD_VENV overrides it.
VENV=${SIMOPD_VENV:-simopd}

# ---------------------------------------------------------------------------
# Mirrors. On by default because this targets a mainland instance; set
# SIMOPD_MIRRORS=0 to go straight upstream. Every one is overridable.
# ---------------------------------------------------------------------------
# Captured before defaults are filled in, so the race below can tell "the caller
# chose this" from "this is just the default" and never overrides an explicit pick.
_USER_INDEX=${UV_DEFAULT_INDEX:-}
_USER_WHEELS=${TORCH_FIND_LINKS:-}
_USER_HF_ENDPOINT=${HF_ENDPOINT:-}
if [ "${SIMOPD_MIRRORS:-1}" = "1" ]; then
    # uv downloads its own CPython build; without this it pulls from GitHub.
    export UV_PYTHON_INSTALL_MIRROR=${UV_PYTHON_INSTALL_MIRROR:-https://python-standalone.org/mirror/astral-sh/python-build-standalone}
    # DEFAULT_INDEX *replaces* PyPI; UV_INDEX alone only appends one, leaving uv
    # free to reach pypi.org and stall. Both are set so either uv version behaves.
    export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
    export UV_INDEX=${UV_INDEX:-$UV_DEFAULT_INDEX}
    export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
    # HF moved large files to Xet-backed storage. A mirror can serve the metadata
    # but file reconstruction still goes to cas-server.xethub.hf.co, which the
    # mirror cannot proxy -- the result is
    #   CAS Client Error: HTTP status client error (401 Unauthorized)
    # partway through a download. Disabling Xet falls back to the plain HTTP path,
    # which mirrors do serve.
    export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
    # A flat wheel listing, not a PEP 503 index, so it goes through --find-links.
    # Verified 2026-08-01 to carry torch/torchvision/torchaudio 2.11.0+cu129 cp312
    # x86_64 -- the only mainland mirror of the three checked that does (Tsinghua
    # 404s on cu129, SJTU redirects away).
    TORCH_FIND_LINKS=${TORCH_FIND_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu129/}
    # As environment variables, not just wrapper flags: pip reads these on every
    # invocation, including the ones it spawns itself and any nested pip during a
    # build. A flag only covers the call that carries it, which is how a mirror
    # ends up applying to some downloads and not others.
    export PIP_INDEX_URL=${PIP_INDEX_URL:-$UV_DEFAULT_INDEX}
    export PIP_FIND_LINKS=${PIP_FIND_LINKS:-$TORCH_FIND_LINKS}
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    # This box sits behind a TLS-intercepting proxy: curl succeeds because the
    # system trusts the proxy CA, while pip ships its own certifi bundle and fails
    # with "self-signed certificate in certificate chain". Point pip at the system
    # store so it trusts what curl trusts.
    for _ca in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt; do
        [ -r "$_ca" ] && { export PIP_CERT=${PIP_CERT:-$_ca}
                           export SSL_CERT_FILE=${SSL_CERT_FILE:-$_ca}
                           export REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-$_ca}
                           break; }
    done
    # uv hardlinks from its cache by default. On DSW the cache sits on local disk
    # and the venv on the /mnt/workspace network volume; hardlinking across
    # filesystems is the kind of thing that half-succeeds and leaves a package
    # directory with no __init__.py. Copying is slower and reliable.
    export UV_LINK_MODE=${UV_LINK_MODE:-copy}
    # uv checks TLS against bundled roots, not the system store; behind a
    # TLS-intercepting proxy that gives "invalid peer certificate: UnknownIssuer"
    # while curl and git work fine. This points uv at the system CA bundle.
    export UV_NATIVE_TLS=${UV_NATIVE_TLS:-1}
    # github.com release assets and clones are the remaining slow path. Set e.g.
    # GITHUB_PROXY=https://gh-proxy.com/ to route them; left empty by default
    # rather than hardcoding someone else's relay into the setup path.
    GITHUB_PROXY=${GITHUB_PROXY:-}
else
    TORCH_FIND_LINKS=""; GITHUB_PROXY=""
    export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
fi
GH() { echo "${GITHUB_PROXY}$1"; }

# The mirror block above picks hf-mirror.com sight unseen. That is right where the
# hub itself is unreachable and wrong where it is not: on this cluster hf-mirror
# answers the model API with a 308 that huggingface_hub does not follow, so every
# asset fails with LocalEntryNotFoundError -- "check your connection" -- while
# huggingface.co serves the same call with a 200. The mirror is an availability
# workaround, so use it only when it is actually the one that works.
#
# Probe the API path, not the landing page: hf-mirror.com/ returns 200 either way.
# An explicit HF_ENDPOINT is honoured untouched.
if [ -z "${_USER_HF_ENDPOINT:-}" ] && command -v curl >/dev/null 2>&1; then
    _probe=api/models/Qwen/Qwen3-1.7B-Base
    _code=$(curl -s -o /dev/null -w '%{http_code}' -m 15 "$HF_ENDPOINT/$_probe" 2>/dev/null || echo 000)
    if [ "$_code" != "200" ]; then
        _alt=https://huggingface.co
        _acode=$(curl -s -o /dev/null -w '%{http_code}' -m 15 "$_alt/$_probe" 2>/dev/null || echo 000)
        if [ "$_acode" = "200" ]; then
            echo "  hf: $HF_ENDPOINT answers $_code on the model API; $_alt answers 200 -> using it"
            export HF_ENDPOINT=$_alt
        else
            echo "  WARNING: neither $HF_ENDPOINT ($_code) nor $_alt ($_acode) serves the model API;" >&2
            echo "  asset download will probably fail. Set HF_ENDPOINT to something reachable." >&2
        fi
    fi
fi
echo "mirrors: pypi=${UV_DEFAULT_INDEX:-upstream} hf=$HF_ENDPOINT torch=${TORCH_FIND_LINKS:-upstream} gh_proxy=${GITHUB_PROXY:-none}"

echo "=== [0/6] pre-flight ==="
# Three things that decide whether the rest of this script can work at all. Checked
# up front so a mismatch fails here instead of 40 minutes into a wheel install.
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || {
    echo "no GPU visible -- wrong instance type?"; exit 1; }
# --id=0 rather than piping all eight rows into `head -1`. Under `set -euo pipefail`
# that pipe is a race the script loses about 40% of the time on an 8-GPU box: head
# exits after the first line, nvidia-smi gets SIGPIPE and returns 141, pipefail
# promotes it to the pipeline's status, and set -e kills the script -- with no error
# text, because nothing wrote any. It reads as "setup stopped after printing the GPU
# table" and it succeeds on the next attempt, which is what makes it expensive.
# Measured on dsw251: 2 of 5 runs returned 141.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0 | cut -d. -f1)
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
    # `|| true` for the same SIGPIPE reason as DRV above; this one has a fallback
    # (${NVCC_VER:-?}) so an empty value is survivable, a killed script is not.
    NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)
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

# Race the candidate sources once and take the fastest, instead of assuming a
# mainland box wants a mainland mirror. On this DSW box the proxy reaches GitHub at
# ~25 MB/s while the Aliyun pypi mirror gives ~2.4 MB/s, so the assumption is
# actively wrong here. Raced per source, not per package: the host is the
# bottleneck, and probing each package would add a round trip per package.
# SIMOPD_RACE=0 skips it; an explicit UV_DEFAULT_INDEX/TORCH_FIND_LINKS also wins.
# Rank the candidates by measured throughput, then VERIFY the winner can actually
# serve a package before committing to it. Throughput alone is not enough: Tsinghua
# served its index page fast on this box and then returned 403 for the package
# files behind it, which fails only after the real install has started.
# SIMOPD_RACE=0 skips the whole thing; an explicit setting is never overridden.
rank_sources() {   # rank_sources <name;pick;probe>... -> picks, fastest first, on stdout
    local out="" spd name pick probe
    for cand in "$@"; do
        IFS=';' read -r name pick probe <<< "$cand"
        spd=$(timeout 20 curl -s -o /dev/null -w '%{speed_download}' \
                --max-time 15 -r 0-4194303 "$probe" 2>/dev/null || echo 0)
        spd=${spd%%.*}; spd=${spd:-0}
        printf '    %-12s %6.1f MB/s\n' "$name" "$(awk "BEGIN{print $spd/1048576}")" >&2
        out="$out$spd $name $pick\n"
    done
    printf "$out" | sort -rn | awk '{print $2";"$3}'
}

index_works() {   # index_works <index_url> -- will it actually hand over a wheel?
    # curl, not `pip download`: this races before the venv exists, so there is no
    # pip yet -- an earlier version used one and "failed" every candidate for that
    # reason alone. Fetch the simple page, take a wheel link, and ask for one byte
    # of it. That exercises the file-serving path, which is what returns 403 on a
    # mirror whose index pages are perfectly fast.
    local idx=${1%/} pkg page url code
    # Several packages, not one: Tsinghua handed over cachetools and then 403'd on
    # setuptools, so a single-package check happily certifies a broken mirror.
    for pkg in setuptools cachetools numpy; do
        page=$(timeout 15 curl -sL "$idx/$pkg/" 2>/dev/null) || return 1
        url=$(printf '%s' "$page" | grep -oE 'href="[^"]+\.whl[^"]*"' | tail -1 | sed 's/^href="//; s/"$//')
        [ -z "$url" ] && return 1
        case "$url" in
            http*) : ;;
            /*)    url="$(printf '%s' "$idx" | sed -E 's|(https?://[^/]+).*|\1|')$url" ;;
            *)     url="$idx/$pkg/$url" ;;
        esac
        code=$(timeout 20 curl -s -o /dev/null -w '%{http_code}' -r 0-0 "${url%%#*}" 2>/dev/null)
        case "$code" in 200|206) ;; *) return 1 ;; esac
    done
    return 0
}

# The two torch-wheel candidates are different KINDS of URL, and pip needs a
# different flag for each:
#
#   mirrors.aliyun.com/pytorch-wheels/cu129/  flat directory of .whl files
#                                             -> --find-links scrapes it, correct
#   download.pytorch.org/whl/cu129            PEP 503 index ROOT; the page lists
#                                             package NAMES, not wheels
#
# Handing the second one to --find-links finds zero wheels, silently falls back to
# the pypi index, and reports "No matching distribution found for
# torch==2.11.0+cu129" -- with the wheel sitting right there at
# /whl/cu129/torch/torch-2.11.0+cu129-cp311-...whl. So the flag has to follow
# whichever URL is actually in hand, rather than being fixed at --find-links.
#
# --extra-index-url rather than --index-url: torch's own dependency closure
# (sympy, filelock, networkx...) resolves from pypi, and the +cu129 local version
# label exists only on the pytorch index, so the pinned spec still lands there
# deterministically.
torch_src_flag() {   # torch_src_flag <url>
    case "$1" in
        *download.pytorch.org/whl/*) echo --extra-index-url ;;
        *)                           echo --find-links ;;
    esac
}

if [ "${SIMOPD_MIRRORS:-1}" = "1" ] && [ "${SIMOPD_RACE:-1}" = "1" ] && command -v curl >/dev/null 2>&1; then
    _tw=torch-2.11.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl
    echo "racing package sources (SIMOPD_RACE=0 to skip)"

    if [ -n "$_USER_WHEELS" ]; then
        echo "  torch wheels: using your TORCH_FIND_LINKS, not racing"
    else
        echo "  torch wheels:"
        _first=$(rank_sources \
            "aliyun;https://mirrors.aliyun.com/pytorch-wheels/cu129/;https://mirrors.aliyun.com/pytorch-wheels/cu129/$_tw" \
            "pytorch.org;https://download.pytorch.org/whl/cu129;https://download.pytorch.org/whl/cu129/$_tw" | head -1 || true)
        [ -n "$_first" ] && TORCH_FIND_LINKS=${_first#*;}
        echo "    -> ${_first%%;*}"
    fi

    if [ -n "$_USER_INDEX" ]; then
        echo "  pypi index: using your UV_DEFAULT_INDEX, not racing"
    else
        echo "  pypi index:"
        _ranked=$(rank_sources \
            "aliyun;https://mirrors.aliyun.com/pypi/simple/;https://mirrors.aliyun.com/pypi/simple/torch/" \
            "pypi.org;https://pypi.org/simple/;https://pypi.org/simple/torch/")
        _chosen=""
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            if index_works "${line#*;}"; then
                _chosen=${line#*;}; echo "    -> ${line%%;*} (serves packages)"; break
            fi
            echo "    x  ${line%%;*} is fast but will not serve packages (403 or similar); trying next" >&2
        done <<< "$_ranked"
        if [ -n "$_chosen" ]; then
            export UV_DEFAULT_INDEX=$_chosen UV_INDEX=$_chosen PIP_INDEX_URL=$_chosen
        else
            echo "  no candidate index could serve a package; keeping $PIP_INDEX_URL" >&2
        fi
    fi
    # Only a flat wheel directory belongs in PIP_FIND_LINKS; pointing it at an index
    # root just makes every later pip call scrape a page with no wheels on it.
    if [ "$(torch_src_flag "$TORCH_FIND_LINKS")" = --find-links ]; then
        export PIP_FIND_LINKS=$TORCH_FIND_LINKS
    fi
    echo "  index=$PIP_INDEX_URL"
    echo "  wheels=$TORCH_FIND_LINKS ($(torch_src_flag "$TORCH_FIND_LINKS"))"
fi

echo "=== [1/6] third-party checkouts ==="
[ -d verl ] || git clone --depth 1 "$(GH https://github.com/volcengine/verl.git)"
# Read-only references for arm provenance checks (PROTOCOL-unified section 2); the
# audit ports their methods, never their harnesses.
[ -d EasyOPD ] || git clone --depth 1 "$(GH https://github.com/lds-ustc/EasyOPD.git)" || true
[ -d OPD ] || git clone --depth 1 "$(GH https://github.com/thunlp/OPD.git)" || true

echo "=== [2/6] python env ==="
# Installer: plain venv + pip by default, uv only if it is already usable.
# Both problems this box hit are uv-specific and simply do not exist with pip:
# uv validates TLS against its own bundled roots (hence "invalid peer certificate:
# UnknownIssuer" behind an intercepting proxy) and hardlinks from its cache (which
# half-succeeds across the local-disk/NAS boundary on /mnt/workspace). pip uses the
# system CA store and always copies. It is slower; it is predictable.
# SIMOPD_USE_UV=1 opts back into uv when you know it works.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
UV=""
if [ "${SIMOPD_USE_UV:-0}" = "1" ]; then
    UV=$(command -v uv 2>/dev/null || true)
    [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
fi
if [ -n "$UV" ]; then
    export UV_LINK_MODE=${UV_LINK_MODE:-copy}
    export UV_NATIVE_TLS=${UV_NATIVE_TLS:-1}
    echo "installer: uv ($UV)"
else
    echo "installer: python -m pip (no uv needed)"
fi

# Flag names differ between the two, so wrap rather than sprinkle conditionals.
# Notably pip's uninstall needs -y, and its no-cache flag is --no-cache-dir.
pipi() {   # pipi [flags...] <spec...>
    if [ -n "$UV" ]; then
        "$UV" pip install "$@"
        return $?
    fi
    _pip_try "$PIP_INDEX_URL" "$@" && return 0
    # A mirror that worked a minute ago can start refusing (403, rate limit) partway
    # through a long install. Retry the same command against the other index rather
    # than making you re-run the script.
    for alt in https://pypi.org/simple/ https://mirrors.aliyun.com/pypi/simple/; do
        [ "$alt" = "$PIP_INDEX_URL" ] && continue
        echo "  index $PIP_INDEX_URL failed; retrying via $alt" >&2
        _pip_try "$alt" "$@" && { export PIP_INDEX_URL=$alt UV_DEFAULT_INDEX=$alt; return 0; }
    done
    return 1
}
_pip_try() {   # _pip_try <index> [flags...] <spec...>
    local idx=$1; shift
    if true; then
        local args=()
        for a in "$@"; do
            case "$a" in
                --no-cache) args+=(--no-cache-dir) ;;
                *) args+=("$a") ;;
            esac
        done
        python -m pip install -i "$idx" "${args[@]}"
    fi
}
pipu() {   # pipu <package...>
    if [ -n "$UV" ]; then "$UV" pip uninstall "$@"; else python -m pip uninstall -y "$@"; fi
}

# venv from the stdlib: one less moving part, and it is what pip expects.
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"

# `uv venv` creates an environment with NO pip in it -- uv does not need one. A
# .venv left over from an earlier uv-based attempt therefore has no pip, and the
# pip path above would die on "No module named pip". Bootstrap it if absent.
if ! python -m pip --version >/dev/null 2>&1; then
    echo "venv has no pip (created by uv?) -- bootstrapping"
    python -m ensurepip --upgrade >/dev/null 2>&1 \
        || curl -sS https://bootstrap.pypa.io/get-pip.py | python - \
        || { echo "FATAL: could not get pip into the venv." >&2
             echo "  simplest fix: rm -rf "$VENV" && bash deploy/dsw/setup.sh" >&2; exit 1; }
fi
python -m pip install -q --upgrade pip -i "${UV_DEFAULT_INDEX:-https://pypi.org/simple/}" 2>/dev/null || true
echo "python: $(command -v python)  pip: $(python -m pip --version 2>&1 | cut -d' ' -f2)"

echo "=== [3/6] torch + vLLM (cu129) ==="
# cu129, not the cu130 PyPI default: cu130 needs driver >= 580, while cu129 runs on
# any 525+ driver through CUDA minor-version compatibility. Check with nvidia-smi.
# GitHub release assets are the one hop with no mainland mirror; curl (system CA
# store) is the fallback when uv's bundled roots reject an intercepting proxy.
vllm_install() {
    case "$1" in
        /*|file://*) pipi "$1"; return $? ;;
    esac
    pipi "vllm @ $1" && return 0
    echo "  uv could not fetch the wheel; retrying via curl" >&2
    local tmp; tmp=$(mktemp -d)/vllm.whl
    curl -fL --retry 3 --retry-delay 5 -o "$tmp" "$1" || {
        echo "  curl failed too -- download it elsewhere and pass VLLM_WHEEL=/path/to/the.whl" >&2
        return 1
    }
    pipi "$tmp"
}

# The pinned matrix (docs/INFRA-NOTES.md). Single-sourced so the check below and
# the installs below cannot drift apart.
PIN_TORCH=2.11.0
PIN_TV=0.26.0
PIN_VLLM=0.26.0

VLLM_WHEEL=${VLLM_WHEEL:-$(GH https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl)}

# Probe an ATTRIBUTE, not just importability. An interrupted wheel install leaves
# site-packages/torch/ without its __init__.py, and Python then imports it happily
# as an empty namespace package -- `import torch` succeeds and
# `torch.__version__` raises "module has no attribute". A plain `import vllm`
# guard therefore reports "already installed" and skips the whole step, leaving
# torch absent. Touching __version__ is what tells a real package from a stub.
# Check the VERSION, not just importability. A wrong-but-importable torch passes an
# import test and the whole step gets skipped -- which is how a torch silently
# upgraded to 2.13.0 by flash-attn would survive a re-run of this script and keep
# vllm broken. Re-running now repairs that instead of stepping over it.
if ! python - "$PIN_TORCH" "$PIN_VLLM" <<'PYCHK' 2>/dev/null; then
import sys
want_torch, want_vllm = sys.argv[1], sys.argv[2]
import torch, vllm
assert torch.__version__.startswith(want_torch), f"torch {torch.__version__} != {want_torch}"
assert vllm.__version__.startswith(want_vllm), f"vllm {vllm.__version__} != {want_vllm}"
PYCHK
    _have=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo none)
    echo "  torch is '$_have', want ${PIN_TORCH}+cu129 -- reinstalling"
    # Clear any half-written tree first; installing over it keeps the stub.
    if python -c "import torch" 2>/dev/null && ! python -c "import torch; torch.__version__" 2>/dev/null; then
        echo "  partial torch detected (namespace stub) -- removing before reinstall"
        pipu torch torchvision torchaudio vllm >/dev/null 2>&1 || true
        # A stub has no dist-info, so uninstall leaves it; remove the trees by hand
        # or the reinstall lands on top of the wreckage.
        _site=$(python -c 'import site;print(site.getsitepackages()[0])')
        rm -rf "${_site:?}/torch" "${_site:?}/vllm" "${_site:?}"/torch-*.dist-info 2>/dev/null || true
    fi

    if [ "$CUDA_FLAVOR" = "cu130" ]; then
        # Plain PyPI builds, both mirrored: nothing here leaves the mainland.
        pipi "torch==${PIN_TORCH}" "torchvision==${PIN_TV}" "torchaudio==${PIN_TORCH}" "vllm==${PIN_VLLM}"
    elif [ -n "$TORCH_FIND_LINKS" ]; then
        # Flag chosen by URL kind, not hardcoded -- see torch_src_flag. All three
        # +cu129 wheels exist for cp310..cp314 on either source.
        pipi "$(torch_src_flag "$TORCH_FIND_LINKS")" "$TORCH_FIND_LINKS" \
            "torch==${PIN_TORCH}+cu129" "torchvision==${PIN_TV}+cu129" "torchaudio==${PIN_TORCH}+cu129"
        vllm_install "$VLLM_WHEEL"
    else
        pipi --index-url https://download.pytorch.org/whl/cu129 \
            "torch==${PIN_TORCH}+cu129" "torchvision==${PIN_TV}+cu129"
        pipi "torchaudio==${PIN_TORCH}"
        vllm_install "$VLLM_WHEEL"
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
        pipi -U "nvidia-nvjitlink-cu12>=12.9"
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
        echo "  \pipi --force-reinstall --no-cache the torch line above." >&2
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
pipi -e ./verl
pipi huggingface_hub math-verify liger-kernel "TransferQueue==0.1.8" wandb pyyaml pandas

# Cross-domain transfer column (METRICS §2). --no-deps on evalplus is not tidiness:
# its dependency closure pulls google-generativeai, which drags protobuf back to 5.x
# and breaks vllm/ray. We use evalplus for data + unit-test execution only, never its
# own generation backends, so its real runtime deps are listed by hand.
# IFEval's checker has no PyPI release and is vendored under third_party/ instead.
pipi --no-deps "evalplus==0.3.1"
pipi appdirs tempdir wget termcolor tree-sitter tree-sitter-python multipledispatch \
     langdetect nltk immutabledict

echo "=== [5/6] flash-attn ==="
# Installed ONLY from a prebuilt wheel, never built here. Building it declares an
# unpinned `torch` dependency, which is how this environment's torch silently went
# 2.11.0 -> 2.13.0 and took vllm and torchvision down with it. --no-deps on a
# prebuilt wheel cannot move anything.
#
# deploy/dsw/flash_attn-*.whl is the wheel built on the Cornell box against exactly
# this stack -- torch 2.11.0+cu129, cp312, cxx11abi TRUE, sm80+sm86 -- so it drops
# straight in on an A100. There is no official prebuilt wheel for torch 2.11
# (Dao-AILab ships cp312 wheels only up to torch 2.10), which is why it is vendored.
if python -c "import torch, flash_attn_2_cuda" 2>/dev/null; then
    echo "  present and matching this torch"
elif ls "$SIMOPD_ROOT"/deploy/dsw/flash_attn-*.whl >/dev/null 2>&1; then
    _site=$(python -c 'import site;print(site.getsitepackages()[0])')
    # A stale build leaves a top-level .so with no dist-info, which an install will
    # not replace; clear all three pieces first.
    rm -rf "${_site:?}"/flash_attn "${_site:?}"/flash_attn_2_cuda*.so "${_site:?}"/flash_attn-*.dist-info 2>/dev/null || true
    # `|| true` because this script runs under `set -e` and the wheel is allowed to
    # be inapplicable. It is tagged cp312, so on a cp311 interpreter pip refuses it
    # outright ("not a supported wheel on this platform") -- which killed the whole
    # setup at 5/6, after torch and vLLM were in and before models, data and
    # simopd_env.sh. The `else` branch below already treats a mismatched wheel as a
    # warning; a REJECTED wheel is the same situation and must not be fatal either.
    pipi --no-deps "$SIMOPD_ROOT"/deploy/dsw/flash_attn-*.whl || true
    if python -c "import torch, flash_attn_2_cuda" 2>/dev/null; then
        echo "  installed from the vendored wheel"
    else
        echo "  vendored wheel does not apply here (wrong cp tag or torch) --" >&2
        echo "  run arms with USE_REMOVE_PADDING=False, and note it enters the fingerprint" >&2
    fi
else
    echo "  no wheel available; run arms with USE_REMOVE_PADDING=False (slower, same results)" >&2
fi

echo "=== [6/6] models + data ==="
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf_cache}
mkdir -p "$HF_HOME" "$DATA_ROOT"
# Resolves every asset offline first and only fetches what is genuinely absent.
# `hf download` is incremental but still contacts the hub to compare each file --
# which is exactly where a mirror + Xet setup 401s, on assets already fully
# downloaded. Re-running this step on a complete machine now touches no network.
python "$SIMOPD_ROOT/scripts/fetch_assets.py" --data-dir "$DATA_ROOT/simopd_math" || {
    echo "  some assets are missing; see above. Re-run this script to retry." >&2
}

# Write the environment out and hook it into ~/.bashrc, so a fresh shell is ready
# without anyone remembering six exports. Idempotent, and fenced by markers so it
# can be removed with a single sed.
ENV_FILE="$SIMOPD_ROOT/simopd_env.sh"
cat > "$ENV_FILE" <<ENVEOF
# Generated by deploy/dsw/setup.sh -- safe to source repeatedly.
export SIMOPD_ROOT="$SIMOPD_ROOT"
export HF_HOME="$HF_HOME"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# One hub, chosen explicitly. Either value is legitimate -- for Qwen models
# ModelScope is a first-party source and on an Alibaba instance usually the faster
# one -- but it must be SET rather than left to the image, because verl and
# fetch_assets.py both branch on it and the failure mode is that they disagree:
# fetch_assets reports every asset present (HF cache) while verl reads a different,
# corrupt copy (ModelScope cache). Both true, neither useful.
#   VERL_USE_MODELSCOPE=true bash deploy/dsw/setup.sh   # to run on ModelScope
export VERL_USE_MODELSCOPE=$( [ "${SIMOPD_USE_MODELSCOPE:-0}" = 1 ] && echo True || echo False )
# vLLM has its OWN flag, read at import of vllm.transformers_utils, which calls
# modelscope's patch_hub() before anything else happens. Setting only verl's leaves
# this one live -- which is what actually routed a DSW run through ModelScope while
# VERL_USE_MODELSCOPE was False, and produced 400s against modelscope.cn/api/v1/login
# followed by a config class that was not a class.
export VLLM_USE_MODELSCOPE=$( [ "${SIMOPD_USE_MODELSCOPE:-0}" = 1 ] && echo True || echo False )
export DATA_DIR="$DATA_ROOT/simopd_math"
export CKPT_ROOT="$DATA_ROOT/ckpt"
export WANDB_DIR="$DATA_ROOT/wandb"
# Where the offline evaluators write their per-problem parquet artifacts. Their
# argparse defaults are Cornell paths that exist on no other cluster, and unlike
# CKPT_ROOT there was nothing here to override them -- so eval_offline.py,
# d6_matrix.py and informativeness.py now all read this first.
export SIMOPD_EVAL_ROOT="$DATA_ROOT/evals"
# Offline unless credentials exist. A lane under nohup cannot answer wandb's login
# prompt, and an instance behind the GFW may not reach wandb.ai at all; offline
# records everything locally for `wandb sync` later.
if [ -z "\${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "\$HOME/.netrc"; then
    export WANDB_MODE=\${WANDB_MODE:-offline}
fi

# Our arm losses register through sitecustomize on PYTHONPATH. Guarded so repeated
# sourcing does not stack duplicates onto the path.
case ":\${PYTHONPATH:-}:" in
    *":\$SIMOPD_ROOT/src:"*) ;;
    *) export PYTHONPATH="\$SIMOPD_ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}" ;;
esac

# Activate the venv unless some other one is already active -- clobbering an
# active environment from a shell rc file is a nasty surprise.
if [ -z "\${VIRTUAL_ENV:-}" ] && [ -f "\$SIMOPD_ROOT/simopd/bin/activate" ]; then
    . "\$SIMOPD_ROOT/simopd/bin/activate"
fi
ENVEOF
echo "wrote $ENV_FILE"

BASHRC="$HOME/.bashrc"
if ! grep -q "^# >>> simopd >>>" "$BASHRC" 2>/dev/null; then
    {
        echo ""
        echo "# >>> simopd >>>"
        echo "[ -f \"$ENV_FILE\" ] && . \"$ENV_FILE\""
        echo "# <<< simopd <<<"
    } >> "$BASHRC"
    echo "hooked into $BASHRC (remove with: sed -i '/# >>> simopd >>>/,/# <<< simopd <<</d' $BASHRC)"
else
    echo "$BASHRC already hooked"
fi

cat <<EOF

=== setup done ===
A new shell now has the venv active and every variable set. For this one:

  source $ENV_FILE

Then:
  bash deploy/dsw/doctor.sh                        # health check
  bash deploy/dsw/run_parallel.sh --rehearsal      # 3 steps per arm
  bash deploy/dsw/run_parallel.sh                  # the real campaign
EOF
