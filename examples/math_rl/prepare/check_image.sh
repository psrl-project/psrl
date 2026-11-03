#!/usr/bin/env bash
# Report whether an image already ships Codex and which settings to use.
# Usage: `check_image.sh IMAGE`
set -euo pipefail

IMAGE=${1:?Usage: check_image.sh IMAGE}

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not on PATH, so the probe cannot run here." >&2
    exit 2
fi

if docker run --rm --entrypoint sh "$IMAGE" -c 'command -v codex'; then
    echo "Codex is in the image."
    echo "Use MATH_RL_RUNTIME_IN_IMAGE=true and MATH_RL_HARNESS_RUNTIME_MOUNT=/usr/local."
else
    echo "Codex is not in the image."
    echo "Use MATH_RL_RUNTIME_IN_IMAGE=false and set PSRL_HARNESS_RUNTIME_ROOT."
fi
