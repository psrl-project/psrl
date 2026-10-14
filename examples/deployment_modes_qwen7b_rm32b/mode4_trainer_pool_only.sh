#!/usr/bin/env bash
# Mode 4: independent main pools plus fixed replicas on idle trainer GPUs.
# Defaults place 4 TP1 rollout and 3 TP4 RM instances on the 16-GPU train pool.
# Usage: bash mode4_trainer_pool_only.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

IDLE_ROLLOUT_INSTANCES=${IDLE_ROLLOUT_INSTANCES:-4}
IDLE_RM_INSTANCES=${IDLE_RM_INSTANCES:-1}
TRAIN_POOL_GPUS=16
IDLE_GPU_DEMAND=$((IDLE_ROLLOUT_INSTANCES * PSRL_DEPLOY_ROLLOUT_TP + IDLE_RM_INSTANCES * PSRL_DEPLOY_RM_TP))
if (( IDLE_GPU_DEMAND > TRAIN_POOL_GPUS )); then
    echo "Idle replicas require ${IDLE_GPU_DEMAND} GPUs, but train_pool has ${TRAIN_POOL_GPUS}." >&2
    exit 1
fi

export PSRL_DEPLOY_MODE=trainer_pool_only
export PSRL_DEPLOY_EXPERIMENT=mode4_bs_128_roll_8_rm_4_trainer_pool_only
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=False
export PSRL_DEPLOY_NNODES=4
export PSRL_DEPLOY_TRAIN_NNODES=1
export PSRL_DEPLOY_TRAIN_NGPUS=8
export PSRL_DEPLOY_SHARED_NNODES=1
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=4
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

export PSRL_DEPLOY_EXTRA="\
+psrl.deployment.trainer_pool_idle_rollout_instances=${IDLE_ROLLOUT_INSTANCES} \
+psrl.deployment.trainer_pool_idle_rm_instances=${IDLE_RM_INSTANCES} \
"

launch_qwen7b_rm32b_deployment_mode "$@"
