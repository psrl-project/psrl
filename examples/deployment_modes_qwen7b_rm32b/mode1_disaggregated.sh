#!/usr/bin/env bash
# Mode 1 runs eight TP1 rollout instances, two TP4 RM replicas, and a 16-GPU trainer in separate pools.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=disaggregated
export PSRL_DEPLOY_EXPERIMENT=mode1_bs_128_disaggregated_interrupt_10s
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=True
export PSRL_DEPLOY_NNODES=7
export PSRL_DEPLOY_TRAIN_NNODES=2
export PSRL_DEPLOY_TRAIN_NGPUS=8
export PSRL_DEPLOY_SHARED_NNODES=1
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=8
export PSRL_DEPLOY_SMOKE=${1:-0}
export PSRL_DEPLOY_DATASET=${PSRL_DEPLOY_DATASET:-dapo}
export PSRL_NEED_VALIDATION=${PSRL_NEED_VALIDATION:-1}
shift || true

launch_qwen7b_rm32b_deployment_mode "$@"
