#!/usr/bin/env bash
# Run rollout, reward, trainer and validation on `train_pool` through NIXL.
# Usage: `bash mode2_colocated.sh [smoke_test=0] [extra hydra overrides...]`
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=colocated
export PSRL_DEPLOY_EXPERIMENT=mode2_bs_128_colocated_16
export PSRL_DEPLOY_STALENESS=0
export PSRL_DEPLOY_RM_ASYNC=True
export PSRL_DEPLOY_NNODES=2
export PSRL_DEPLOY_TRAIN_NNODES=2
export PSRL_DEPLOY_TRAIN_NGPUS=8
# Replicas share `train_pool`, so no separate shared pool is needed.
export PSRL_DEPLOY_SHARED_NNODES=0
export PSRL_DEPLOY_SHARED_NGPUS=0
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
