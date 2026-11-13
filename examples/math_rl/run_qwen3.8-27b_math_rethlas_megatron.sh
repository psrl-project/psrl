#!/usr/bin/env bash
# Train Qwen3.8-27B on the Rethlas harness with GRPO.
#
# This is the Rethlas entry point. It selects the math_rethlas loop, applies the
# harness defaults, and delegates the rest of the recipe to the shared launcher.
# Usage: run_qwen3.8-27b_math_rethlas_megatron.sh [launcher overrides]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'USAGE'
Train Qwen3.8-27B on the Rethlas harness.

Required:
  MATH_RL_RETHLAS_REPO       Host path to the Rethlas checkout.

Common:
  MATH_RL_VERIFIER_URL       External verifier endpoint including /verify.
  MATH_RL_VERIFIER_VERSION   Harness version, for example V3.
  MATH_RL_VERIFIER_MODEL     Model name, for example glm-5.3.
  MATH_RL_VERIFIER_PROXY     Proxy for the verifier call.
  MATH_RL_RETHLAS_PROXY      Proxy used inside the sandbox for package installs.
  MATH_RL_RETHLAS_MAX_ITERATIONS  Generation rounds per episode, default 3.
  MATH_RL_RETHLAS_MAX_CALLS       Verifier calls per episode, default 4.
  TOTAL_TRAINING_STEPS       Training steps, default 200.
  TRAIN_BSZ / ROLLOUT_N      Batch and rollouts, default 32 and 8.
  TRAIN_MINI_BSZ             Prompts per optimizer update.
  GROUP_POST_PROCESS_ENABLE  Set false for a degenerate smoke.

The Rethlas script defaults the verifier stage timeout to 3600 seconds and the
trajectory timeout to 7200 seconds. Override them with
MATH_RL_VERIFIER_STAGE_TIMEOUT_S and TRAJECTORY_TIMEOUT.
USAGE
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

if [[ -z "${MATH_RL_RETHLAS_REPO:-}" ]]; then
    echo "ERROR: set MATH_RL_RETHLAS_REPO to the Rethlas checkout." >&2
    exit 1
fi
if [[ ! -f "${MATH_RL_RETHLAS_REPO}/agents/generation/tests/run_example.sh" ]]; then
    echo "ERROR: ${MATH_RL_RETHLAS_REPO} does not look like a Rethlas checkout." >&2
    echo "       agents/generation/tests/run_example.sh is missing." >&2
    exit 1
fi
if [[ -z "${MATH_RL_RETHLAS_PROXY:-}" ]]; then
    echo "WARNING: MATH_RL_RETHLAS_PROXY is unset. The first episode builds the MCP" >&2
    echo "         wheelhouse on the worker and may need a package proxy." >&2
fi

export MATH_RL_LOOP=math_rethlas
export MATH_RL_EXPERIMENT_NAME=GRPO-Qwen3.8-27B-math-rethlas-score-centering
export MATH_RL_VERIFIER_STAGE_TIMEOUT_S="${MATH_RL_VERIFIER_STAGE_TIMEOUT_S:-3600}"
export TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-7200}"

exec bash "${SCRIPT_DIR}/run_qwen3.8-27b_math_codex_megatron.sh" "$@"
