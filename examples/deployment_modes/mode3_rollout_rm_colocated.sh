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
export PSRL_DEPLOY_EXPERIMENT=mode3_bs_128_rollout_rm_colocated_share_real_8
export PSRL_DEPLOY_STALENESS=2
export PSRL_DEPLOY_RM_ASYNC=True
export PSRL_DEPLOY_NNODES=2
export PSRL_DEPLOY_TRAIN_NNODES=1
export PSRL_DEPLOY_TRAIN_NGPUS=8
export PSRL_DEPLOY_SHARED_NNODES=1
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
