#!/usr/bin/env bash
# Build the Rethlas base image if needed, then the math RL derivative.
# Usage: `build_image.sh` with MATH_RL_ASSETS_ROOT set.
set -euo pipefail

: "${MATH_RL_ASSETS_ROOT:?Set MATH_RL_ASSETS_ROOT to the directory that contains Rethlas.}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_IMAGE=${RETHLAS_CODEX_IMAGE:-rethlas-codex:local}
TARGET_IMAGE=${MATH_RL_SANDBOX_IMAGE:-math-rl-codex:local}

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    docker build \
        --tag "$BASE_IMAGE" \
        --file "${MATH_RL_ASSETS_ROOT}/Rethlas/docker/Dockerfile" \
        "${MATH_RL_ASSETS_ROOT}/Rethlas"
fi

docker build --tag "$TARGET_IMAGE" --file "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
echo "Built ${TARGET_IMAGE} from ${BASE_IMAGE}."
