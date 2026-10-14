#!/usr/bin/env bash
# Mode 4 — trainer-pool-only (fixed idle replicas, no elastic auto-scaling).
# Three independent pools (rollout / rm / trainer) running concurrently, PLUS a
# FIXED number of extra rollout / rm replicas that are woken on train_pool GPUs
# while the trainer is idle, and SLEPT when the training step runs.
#
# Topology (2 nodes): rollout pool (4 GPUs, 4 main instances) + rm pool (4 GPUs,
# 4 main RM replicas) + train pool (8 GPUs, actor + idle rollout/rm replicas on
# DISJOINT bundles, time-multiplexed with actor via NIXL sleep/wake).
#
# Requirements: psrl.ps_mode in {nixl_cpu, nixl_gpu}. Idle rollout and idle rm
# replicas each occupy one GPU bundle on train_pool and must NOT share bundles;
# ensure IDLE_ROLLOUT_INSTANCES + IDLE_RM_INSTANCES <= PSRL_DEPLOY_TRAIN_NGPUS.
#
# Usage: bash mode4_trainer_pool_only.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

# Fixed extra replicas woken on train_pool while the trainer is idle.
# Rollout idle bundles are allocated first, then rm (disjoint). Default 4+4
# fills an 8-GPU train_pool; override via IDLE_ROLLOUT_INSTANCES / IDLE_RM_INSTANCES.
IDLE_ROLLOUT_INSTANCES=${IDLE_ROLLOUT_INSTANCES:-4}
IDLE_RM_INSTANCES=${IDLE_RM_INSTANCES:-4}

export PSRL_DEPLOY_MODE=trainer_pool_only
export PSRL_DEPLOY_EXPERIMENT=mode4_bs_128_roll_4_rm_4_trainer_pool_only
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
# Mode 4 is a disaggregated concurrent pipeline (like mode 1) PLUS fixed idle
# replicas on train_pool. 
# With sync reward the rollout worker would block on the rm per request and the
# "running concurrently" intent would be lost.
export PSRL_DEPLOY_RM_ASYNC=False
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
export PSRL_DEPLOY_SHARED_NNODES=4
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=4
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

# Mode 4 deliberately does NOT set the elastic policy / cooldown / throughput_optimal /
# coordinator_command_timeout_s block that mode 5 uses: resolve_deployment_mode
# forces elastic.enable=False + enable_policy=False for trainer_pool_only, so
# no ElasticExecutor / ScalingPolicy / monitor loop is created and those knobs
# are inert. Mode 4 drives its idle replicas directly via coordinator
# exec_command (SLEEP/WAKE_UP) + actor NIXL sleep/wake, gated by the two counts
# below. Override PSRL_DEPLOY_EXTRA to add any extra hydra flags if needed.
export PSRL_DEPLOY_EXTRA="\
+psrl.deployment.trainer_pool_idle_rollout_instances=${IDLE_ROLLOUT_INSTANCES} \
+psrl.deployment.trainer_pool_idle_rm_instances=${IDLE_RM_INSTANCES} \
"

launch_deployment_mode "$@"
