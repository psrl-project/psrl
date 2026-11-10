#!/usr/bin/env bash
# Mode 1 uses independent rollout and RM pools of 16 GPUs each, plus 32 trainer GPUs.
#
# Usage: bash mode1_disaggregated.sh [smoke_test=0] [extra hydra overrides...]
#   smoke_test=1 -> 2 steps, small batch (quick launch check).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=disaggregated
# export PSRL_DEPLOY_EXPERIMENT=psrl_roll_3_rm_9_mode1_disaggregated_rollout7b_rm8b
export PSRL_DEPLOY_EXPERIMENT=mode1_roll_4_rm_4_disaggregated
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=True
export PSRL_DEPLOY_NNODES=4
export PSRL_DEPLOY_TRAIN_NNODES=2
export PSRL_DEPLOY_TRAIN_NGPUS=8
# Select with TRAIN_BACKEND=megatron. The default remains fsdp2.
export PSRL_DEPLOY_TRAIN_BACKEND=${PSRL_DEPLOY_TRAIN_BACKEND:-${TRAIN_BACKEND:-megatron}}
export PSRL_DEPLOY_MEGATRON_TP=${PSRL_DEPLOY_MEGATRON_TP:-${MEGATRON_TP:-2}}
export PSRL_DEPLOY_MEGATRON_PP=${PSRL_DEPLOY_MEGATRON_PP:-${MEGATRON_PP:-2}}
export PSRL_DEPLOY_MEGATRON_CP=${PSRL_DEPLOY_MEGATRON_CP:-${MEGATRON_CP:-1}}
export PSRL_DEPLOY_MEGATRON_EP=${PSRL_DEPLOY_MEGATRON_EP:-${MEGATRON_EP:-8}}
export PSRL_DEPLOY_MEGATRON_ETP=${PSRL_DEPLOY_MEGATRON_ETP:-${MEGATRON_ETP:-1}}
export PSRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU=${PSRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU:-${MEGATRON_MICRO_BSZ_PER_GPU:-1}}
# Independent rollout pool (2 nodes x 8 GPUs -> 4 TP4 rollout instances).
export PSRL_DEPLOY_SHARED_NNODES=1
export PSRL_DEPLOY_SHARED_NGPUS=8
# Independent gen-RM pool (2 nodes x 8 GPUs -> 4 TP4 RM replicas).
export PSRL_DEPLOY_RM_NUM_REPLICAS=2
export PSRL_DEPLOY_SMOKE=${1:-1}
shift || true

launch_deployment_mode "$@"
