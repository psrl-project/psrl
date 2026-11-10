#!/usr/bin/env bash
# Elastic RL with Megatron TP2/PP1 training for Qwen2.5-1.5B and GLM-Z1-9B-0414 reward.
# Usage: `bash examples/elastic/qwen2.5_1.5b_glm_z1_9b_megatron.sh [smoke_test=0] [overrides...]`
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

export PSRL_DEPLOY_TRAIN_BACKEND=megatron
export PSRL_DEPLOY_MEGATRON_TP=${PSRL_DEPLOY_MEGATRON_TP:-2}
export PSRL_DEPLOY_MEGATRON_PP=${PSRL_DEPLOY_MEGATRON_PP:-1}
export PSRL_DEPLOY_MEGATRON_CP=${PSRL_DEPLOY_MEGATRON_CP:-1}
export PSRL_DEPLOY_MEGATRON_EP=${PSRL_DEPLOY_MEGATRON_EP:-1}
export PSRL_DEPLOY_MEGATRON_ETP=${PSRL_DEPLOY_MEGATRON_ETP:-1}
export PSRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU=${PSRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU:-1}

# Actor weight arenas require FSDP2 parameter bindings.
exec bash "${SCRIPT_DIR}/qwen2.5_1.5b_glm_z1_9b.sh" "${1:-0}" \
    ++psrl.nixl.weight_arena.actor_enabled=false "${@:2}"
