#!/usr/bin/env bash
# Mode 5: ITL elastic scaling on a 24-GPU pool shared by rollout and RM.
# Usage: bash mode5_elastic_rl.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=elastic_rl
export PSRL_DEPLOY_EXPERIMENT=mode5_bs_128_share_48_elastic_staleness_2_migration
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=False
export PSRL_DEPLOY_NNODES=4
export PSRL_DEPLOY_TRAIN_NNODES=1
export PSRL_DEPLOY_TRAIN_NGPUS=8
export PSRL_DEPLOY_SHARED_NNODES=3
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_SMOKE=${1:-0}
export PSRL_DEPLOY_DATASET=${PSRL_DEPLOY_DATASET:-mixed}
export PSRL_NEED_VALIDATION=${PSRL_NEED_VALIDATION:-0}
export PSRL_DEPLOY_OPTIMIZER_OFFLOAD=${PSRL_DEPLOY_OPTIMIZER_OFFLOAD:-True}
export PSRL_WEIGHT_VERIFY=${PSRL_WEIGHT_VERIFY:-False}
export PSRL_WEIGHT_VERIFY_SAMPLE_EVERY=${PSRL_WEIGHT_VERIFY_SAMPLE_EVERY:-1}
export PSRL_WEIGHT_VERIFY_FULL_EVERY=${PSRL_WEIGHT_VERIFY_FULL_EVERY:-0}
export PSRL_WEIGHT_VERIFY_SAMPLE_COUNT=${PSRL_WEIGHT_VERIFY_SAMPLE_COUNT:-16}
export PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION=${PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION:-true}
export PSRL_CANDIDATE_EVALUATION_BACKEND=${PSRL_CANDIDATE_EVALUATION_BACKEND:-cpp}
export PSRL_CANDIDATE_EVALUATION_MAX_WORKERS=${PSRL_CANDIDATE_EVALUATION_MAX_WORKERS:-64}
export PSRL_CANDIDATE_EVALUATION_CPP_BINARY=${PSRL_CANDIDATE_EVALUATION_CPP_BINARY:-null}
export PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S=${PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S:-30.0}
export PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN=True
export PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE=False
export PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION=${PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION:-0.5}
export PSRL_DEPLOY_VAL_BEFORE_TRAIN=False
shift || true

# Parent elastic_rl routing points at qwen2.5_1.5b.json. This folder runs
# Qwen2.5-7B (TP1) and Qwen3-30B-A3B-Thinking-2507 (TP4).
COST_MODEL_DIR=$(cd -- "${SCRIPT_DIR}/../../psrl/trainer/config/cost_model" &>/dev/null && pwd)
ROLLOUT_COST_MODEL="${COST_MODEL_DIR}/qwen2.5_7b.json"
RM_COST_MODEL="${COST_MODEL_DIR}/qwen3_30b_a3b_thinking_2507.json"
for _cost_model in "${ROLLOUT_COST_MODEL}" "${RM_COST_MODEL}"; do
    if [[ ! -f "${_cost_model}" ]]; then
        echo "Cost model not found: ${_cost_model}" >&2
        exit 1
    fi
done

export PSRL_DEPLOY_EXTRA="\
+psrl.nixl.weight_verification.enable=${PSRL_WEIGHT_VERIFY} \
+psrl.nixl.weight_verification.transfer_chain=True \
+psrl.nixl.weight_verification.trainer_sleep_wake=True \
+psrl.nixl.weight_verification.sample_every_n_versions=${PSRL_WEIGHT_VERIFY_SAMPLE_EVERY} \
+psrl.nixl.weight_verification.full_every_n_versions=${PSRL_WEIGHT_VERIFY_FULL_EVERY} \
+psrl.nixl.weight_verification.sample_count_per_tensor=${PSRL_WEIGHT_VERIFY_SAMPLE_COUNT} \
psrl.deployment.elastic.enable_policy=True \
psrl.deployment.elastic.scaling_policy_variant=throughput-optimal \
psrl.deployment.elastic.enable_trainer_pool=True \
psrl.deployment.elastic.min_awake_per_role=0 \
psrl.deployment.elastic.cooldown_ms=10000 \
psrl.deployment.elastic.throughput_optimal.min_gain=0.05 \
psrl.deployment.elastic.monitor_interval_ms=1000 \
psrl.deployment.elastic.wakeup_immunity_ms=10000 \
psrl.deployment.elastic.throughput_optimal.decision_window_s=60.0 \
psrl.deployment.elastic.throughput_optimal.throughput_objective=sum \
psrl.deployment.elastic.throughput_optimal.max_scale_instances_per_action=-1 \
psrl.deployment.elastic.throughput_optimal.router_waiting_top_t=-1 \
psrl.deployment.elastic.throughput_optimal.role_throughput_weight_enable=true \
psrl.deployment.elastic.throughput_optimal.role_throughput_weight_basis=request_count \
psrl.deployment.elastic.throughput_optimal.role_throughput_weight_mode=raw \
psrl.deployment.elastic.throughput_optimal.enable_heterogeneous_parallelism_candidates=true \
psrl.deployment.elastic.throughput_optimal.enable_request_level_candidate_evaluation=${PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION} \
psrl.deployment.elastic.throughput_optimal.candidate_evaluation_backend=${PSRL_CANDIDATE_EVALUATION_BACKEND} \
psrl.deployment.elastic.throughput_optimal.candidate_evaluation_max_workers=${PSRL_CANDIDATE_EVALUATION_MAX_WORKERS} \
psrl.deployment.elastic.throughput_optimal.candidate_evaluation_cpp_binary=${PSRL_CANDIDATE_EVALUATION_CPP_BINARY} \
psrl.deployment.elastic.throughput_optimal.candidate_evaluation_cpp_timeout_s=${PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S} \
psrl.deployment.elastic.throughput_optimal.exclusive_rebalance_migration_queue=True \
+reward.managers.gen_qwen3_8b.routing_strategy.method=itl \
+reward.managers.gen_qwen3_8b.routing_strategy.cost_model_path=${RM_COST_MODEL} \
+reward.managers.gen_qwen3_8b.routing_strategy.delta_throughput_threshold=0.005 \
+reward.managers.gen_qwen3_8b.routing_strategy.request_budget=1024 \
+reward.managers.gen_qwen3_8b.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
+reward.managers.gen_qwen3_8b.routing_strategy.max_concurrent_seqs_per_instance=128 \
"

launch_qwen7b_rm32b_deployment_mode \
    psrl.rollout_coordination.routing_strategy.cost_model_path=${ROLLOUT_COST_MODEL} \
    "$@"
