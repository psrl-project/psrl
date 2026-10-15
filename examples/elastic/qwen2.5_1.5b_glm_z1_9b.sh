#!/usr/bin/env bash
# Elastic RL for Qwen2.5-1.5B rollout and GLM-Z1-9B-0414 reward.
# Usage: `bash qwen2.5_1.5b_glm_z1_9b.sh [smoke_test=0] [extra hydra overrides...]`
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_elastic.sh"

export PSRL_DEPLOY_EXPERIMENT=elastic_rl
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=False
export PSRL_DEPLOY_NNODES=2
export PSRL_DEPLOY_TRAIN_NNODES=1
export PSRL_DEPLOY_TRAIN_NGPUS=4
export PSRL_DEPLOY_SHARED_NNODES=1
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_MODEL_NAME=Qwen2.5-1.5B
export PSRL_DEPLOY_RM_MODEL_NAME=GLM-Z1-9B-0414
export PSRL_DEPLOY_ROLLOUT_TP=1
export PSRL_DEPLOY_RM_TP=1
export PSRL_DEPLOY_VAL_TP=1
export PSRL_DEPLOY_ROLLOUT_COST_MODEL=qwen2.5_1.5b.json
export PSRL_DEPLOY_RM_COST_MODEL=glm_z1_9b_0414.json
export PSRL_DEPLOY_GSM8K_SUBDIR=gsm8k
export PSRL_DEPLOY_RM_COEF=1.0
export PSRL_DEPLOY_GEN_GPU_MEMORY_UTILIZATION=0.7
export PSRL_DEPLOY_RM_GPU_MEMORY_UTILIZATION=0.8
export PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION=0.6
export PSRL_DEPLOY_RM_MAX_NUM_SEQS=1024
export PSRL_DEPLOY_RM_MAX_NUM_BATCHED_TOKENS=$((1024 * 40))
export PSRL_DEPLOY_FULL_STEPS=100
export PSRL_DEPLOY_SMOKE_STEPS=10
export PSRL_DEPLOY_TEST_FREQ=5
export PSRL_DEPLOY_SAVE_FREQ=-1
export PSRL_DEPLOY_MAX_SCALE_INSTANCES=-1
export PSRL_DEPLOY_OPTIMIZER_OFFLOAD=${PSRL_DEPLOY_OPTIMIZER_OFFLOAD:-True}
export PSRL_DEPLOY_VAL_BEFORE_TRAIN=False
export PSRL_DEPLOY_SMOKE=${1:-0}
export PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION=${PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION:-true}
export PSRL_CANDIDATE_EVALUATION_BACKEND=${PSRL_CANDIDATE_EVALUATION_BACKEND:-cpp}
export PSRL_CANDIDATE_EVALUATION_MAX_WORKERS=${PSRL_CANDIDATE_EVALUATION_MAX_WORKERS:-64}
export PSRL_CANDIDATE_EVALUATION_CPP_BINARY=${PSRL_CANDIDATE_EVALUATION_CPP_BINARY:-null}
export PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S=${PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S:-30.0}
export PSRL_EXCLUSIVE_REBALANCE_MIGRATION_QUEUE=True
shift || true

export PSRL_DEPLOY_EXTRA="\
psrl.tms.range=all \
psrl.tms.enable_nixl=True \
++psrl.nixl.weight_arena.actor_enabled=true \
++psrl.nixl.weight_arena.actor_materialization=repack \
++psrl.nixl.weight_arena.rollout_enabled=true \
++psrl.nixl.weight_arena.rollout_materialization=repack \
++psrl.nixl.weight_arena.reward_enabled=true \
++psrl.nixl.weight_arena.reward_materialization=repack \
++psrl.nixl.weight_arena.reward_cpu_cache_mode=node_shared \
++psrl.nixl.weight_arena.reward_cpu_cache_pin_memory=false \
++psrl.nixl.weight_arena.max_chunk_gb=4 \
++psrl.nixl.weight_arena.alignment_bytes=256 \
+reward.managers.gen_qwen3_8b.rollout.engine_kwargs.vllm.hf_overrides.rope_scaling.type=yarn \
+reward.managers.gen_qwen3_8b.rollout.engine_kwargs.vllm.hf_overrides.rope_scaling.rope_type=yarn \
+reward.managers.gen_qwen3_8b.rollout.engine_kwargs.vllm.hf_overrides.rope_scaling.factor=1.25 \
+reward.managers.gen_qwen3_8b.rollout.engine_kwargs.vllm.hf_overrides.rope_scaling.original_max_position_embeddings=32768 \
"

launch_elastic "$@"
