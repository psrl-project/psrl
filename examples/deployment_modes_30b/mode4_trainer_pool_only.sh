#!/usr/bin/env bash
# Mode 4 wakes fixed extra rollout and RM replicas on idle trainer GPUs, then sleeps them before training.
#
# Main rollout and RM pools each use four GPUs. Extra replicas occupy disjoint bundles on eight trainer GPUs.
#
# Require `nixl_cpu` or `nixl_gpu` and disjoint idle replica bundles that fit within `PSRL_DEPLOY_TRAIN_NGPUS`.
#
# Usage: bash mode4_trainer_pool_only.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

# Rollout idle bundles are allocated before RM bundles. Set `IDLE_ROLLOUT_INSTANCES` and `IDLE_RM_INSTANCES`.
IDLE_ROLLOUT_INSTANCES=${IDLE_ROLLOUT_INSTANCES:-4}
IDLE_RM_INSTANCES=${IDLE_RM_INSTANCES:-4}

export PSRL_DEPLOY_MODE=trainer_pool_only
export PSRL_DEPLOY_EXPERIMENT=mode4_bs_128_roll_4_rm_4_trainer_pool_only
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
# Asynchronous reward lets independent rollout and RM pools run concurrently.
export PSRL_DEPLOY_RM_ASYNC=False
export PSRL_DEPLOY_NNODES=8
export PSRL_DEPLOY_TRAIN_NNODES=4
export PSRL_DEPLOY_TRAIN_NGPUS=8
# Select with TRAIN_BACKEND=megatron. The default remains fsdp2.
export PSRL_DEPLOY_TRAIN_BACKEND=${PSRL_DEPLOY_TRAIN_BACKEND:-${TRAIN_BACKEND:-fsdp2}}
export PSRL_DEPLOY_MEGATRON_TP=${PSRL_DEPLOY_MEGATRON_TP:-${MEGATRON_TP:-2}}
export PSRL_DEPLOY_MEGATRON_PP=${PSRL_DEPLOY_MEGATRON_PP:-${MEGATRON_PP:-2}}
export PSRL_DEPLOY_MEGATRON_CP=${PSRL_DEPLOY_MEGATRON_CP:-${MEGATRON_CP:-1}}
export PSRL_DEPLOY_MEGATRON_EP=${PSRL_DEPLOY_MEGATRON_EP:-${MEGATRON_EP:-8}}
export PSRL_DEPLOY_MEGATRON_ETP=${PSRL_DEPLOY_MEGATRON_ETP:-${MEGATRON_ETP:-1}}
export PSRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU=${PSRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU:-${MEGATRON_MICRO_BSZ_PER_GPU:-1}}
export PSRL_DEPLOY_SHARED_NNODES=2
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=4
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

# Mode 4 drives fixed idle replicas through coordinator commands. Elastic policy settings do not apply.
export PSRL_DEPLOY_EXTRA="\
+psrl.deployment.trainer_pool_idle_rollout_instances=${IDLE_ROLLOUT_INSTANCES} \
+psrl.deployment.trainer_pool_idle_rm_instances=${IDLE_RM_INSTANCES} \
"

launch_deployment_mode "$@"
