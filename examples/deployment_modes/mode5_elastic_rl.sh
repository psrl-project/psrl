#!/usr/bin/env bash
# Mode 5 — elastic RL (current elastic auto-scaling between rollout and rm).
# rollout + RM share shared_rollout_pool with a ScalingPolicy that auto-scales the
# awake instance counts; trainer on a separate train_pool. This is the existing
# elastic path (elastic.enable=True + enable_policy=True).
#
# Two nodes: shared_rollout_pool (8 GPUs, rollout+rm elastic) + train_pool (8 GPUs).
#
# Usage: bash mode5_elastic_rl.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=elastic_rl
export PSRL_DEPLOY_EXPERIMENT=elastic_rl
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=False
export PSRL_DEPLOY_NNODES=2
export PSRL_DEPLOY_TRAIN_NNODES=1
export PSRL_DEPLOY_TRAIN_NGPUS=4
export PSRL_DEPLOY_SHARED_NNODES=1
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_VAL_BEFORE_TRAIN=False
export PSRL_DEPLOY_OPTIMIZER_OFFLOAD=${PSRL_DEPLOY_OPTIMIZER_OFFLOAD:-True}
export PSRL_DEPLOY_SMOKE=${1:-0}
export PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION=${PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION:-true}
export PSRL_CANDIDATE_EVALUATION_BACKEND=${PSRL_CANDIDATE_EVALUATION_BACKEND:-cpp}
export PSRL_CANDIDATE_EVALUATION_MAX_WORKERS=${PSRL_CANDIDATE_EVALUATION_MAX_WORKERS:-64}
export PSRL_CANDIDATE_EVALUATION_CPP_BINARY=${PSRL_CANDIDATE_EVALUATION_CPP_BINARY:-null}
export PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S=${PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S:-30.0}
shift || true

# ---- Elastic auto-scaling config ----
# enable_policy=True keeps the ElasticExecutor monitor loop running; the
# scaling_policy_variant selects the ScalingPolicy implementation. Below is a
# complete, tunable elastic block (modeled on examples/elastic_rm/*_fsdp.sh):
# cooldown plus throughput-optimal policy tuning.
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
psrl.deployment.elastic.throughput_optimal.enable_request_level_candidate_evaluation=${PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION} \
psrl.deployment.elastic.throughput_optimal.candidate_evaluation_backend=${PSRL_CANDIDATE_EVALUATION_BACKEND} \
psrl.deployment.elastic.throughput_optimal.candidate_evaluation_max_workers=${PSRL_CANDIDATE_EVALUATION_MAX_WORKERS} \
psrl.deployment.elastic.throughput_optimal.candidate_evaluation_cpp_binary=${PSRL_CANDIDATE_EVALUATION_CPP_BINARY} \
psrl.deployment.elastic.throughput_optimal.candidate_evaluation_cpp_timeout_s=${PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S} \
psrl.deployment.elastic.throughput_optimal.exclusive_rebalance_migration_queue=True \
+reward.managers.gen_qwen3_8b.routing_strategy.method=itl \
+reward.managers.gen_qwen3_8b.routing_strategy.cost_model_path=/apdcephfs_zwfy10/share_303541817/yfzhao/psrl/psrl/trainer/config/cost_model/glm_z1_9b_0414.json \
+reward.managers.gen_qwen3_8b.routing_strategy.delta_throughput_threshold=0.005 \
+reward.managers.gen_qwen3_8b.routing_strategy.request_budget=1024 \
+reward.managers.gen_qwen3_8b.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
+reward.managers.gen_qwen3_8b.routing_strategy.max_concurrent_seqs_per_instance=128 \
"


launch_deployment_mode "$@"
