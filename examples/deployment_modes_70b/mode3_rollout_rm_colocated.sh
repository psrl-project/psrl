#!/usr/bin/env bash
# Mode 3 alternates rollout and reward on the shared pool while a separate trainer pipelines buffers.
#
# Requirements: psrl.ps_mode in {nixl_cpu, nixl_gpu}. launch_reward_fn_async=True.
#   staleness=0.
#
# Usage: bash mode3_rollout_rm_colocated.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=rollout_rm_colocated
export PSRL_DEPLOY_EXPERIMENT=mode3_bs_128_rollout_rm_colocated
export PSRL_DEPLOY_STALENESS=2
export PSRL_DEPLOY_RM_ASYNC=True
export PSRL_DEPLOY_NNODES=16
export PSRL_DEPLOY_TRAIN_NNODES=8
export PSRL_DEPLOY_TRAIN_NGPUS=8
# Select with TRAIN_BACKEND=megatron. The default remains fsdp2.
export PSRL_DEPLOY_TRAIN_BACKEND=${PSRL_DEPLOY_TRAIN_BACKEND:-${TRAIN_BACKEND:-fsdp2}}
export PSRL_DEPLOY_MEGATRON_TP=${PSRL_DEPLOY_MEGATRON_TP:-${MEGATRON_TP:-2}}
export PSRL_DEPLOY_MEGATRON_PP=${PSRL_DEPLOY_MEGATRON_PP:-${MEGATRON_PP:-2}}
export PSRL_DEPLOY_MEGATRON_CP=${PSRL_DEPLOY_MEGATRON_CP:-${MEGATRON_CP:-1}}
export PSRL_DEPLOY_MEGATRON_EP=${PSRL_DEPLOY_MEGATRON_EP:-${MEGATRON_EP:-8}}
export PSRL_DEPLOY_MEGATRON_ETP=${PSRL_DEPLOY_MEGATRON_ETP:-${MEGATRON_ETP:-1}}
export PSRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU=${PSRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU:-${MEGATRON_MICRO_BSZ_PER_GPU:-1}}
export PSRL_DEPLOY_SHARED_NNODES=8
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
