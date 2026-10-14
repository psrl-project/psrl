#!/bin/bash
# Build the AIRS-Bench agent sandbox image.
# The build uses a reachable Docker mirror and host-created Conda artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${IMAGE_TAG:-psrl/airs-bench-agent:latest}"
MIRROR="${DOCKERHUB_MIRROR:-mirror.ccs.tencentyun.com}"
APT_MIRROR="${APT_MIRROR:-mirrors.tencentyun.com}"
BASE_IMAGE="${MIRROR}/library/ubuntu:jammy"
MINICONDA_SH="${SCRIPT_DIR}/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
CONDA_PACK="${SCRIPT_DIR}/mlgym_generic.tar.gz"
PSRL_CONDA="${PSRL_CONDA:-/apdcephfs_zwfy10/share_303541817/lhy/anaconda3/bin/conda}"

if [[ ! -f "${MINICONDA_SH}" ]]; then
    echo "=== Downloading Miniconda installer ==="
    wget -q --show-progress -O "${MINICONDA_SH}" "${MINICONDA_URL}"
fi

if [[ ! -f "${CONDA_PACK}" ]]; then
    echo "=== Pre-creating mlgym_generic conda env (requires psrl conda 24.9.2) ==="
    if [[ ! -x "${PSRL_CONDA}" ]]; then
        echo "ERROR: PSRL_CONDA=${PSRL_CONDA} not found. Cannot pre-create the env." >&2
        echo "Set PSRL_CONDA to the path of a conda 24.x binary, then re-run." >&2
        exit 1
    fi
    ENV_DIR="$(mktemp -d)/mlgym_generic"
    "${PSRL_CONDA}" create -y -p "${ENV_DIR}" python=3.11
    "${PSRL_CONDA}-pack" -p "${ENV_DIR}" -o "${CONDA_PACK}" --compress-level 1
    rm -rf "$(dirname "${ENV_DIR}")"
    echo "=== Conda env packed to ${CONDA_PACK} ==="
fi

echo "=== Pulling base image ${BASE_IMAGE} ==="
docker pull "${BASE_IMAGE}"

echo "=== Building ${IMAGE_TAG} ==="
docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "APT_MIRROR=${APT_MIRROR}" \
    --build-arg "http_proxy=${http_proxy:-}" \
    --build-arg "https_proxy=${https_proxy:-}" \
    -t "${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

echo "=== Verifying the image ==="
docker run --rm "${IMAGE_TAG}" bash -lc \
    "conda env list && /home/agent/miniconda3/envs/mlgym_generic/bin/python -c 'import torch, sklearn, xgboost; print(\"deps ok\")'"

docker images "${IMAGE_TAG}"
echo "=== Build complete: ${IMAGE_TAG} ==="
