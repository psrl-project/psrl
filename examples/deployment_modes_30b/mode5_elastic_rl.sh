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
export PSRL_DEPLOY_EXPERIMENT=__debug__mode5_share_16_elastic_rl
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
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
export PSRL_DEPLOY_SHARED_NNODES=4
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_OPTIMIZER_OFFLOAD=${PSRL_DEPLOY_OPTIMIZER_OFFLOAD:-True}
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

# ---- Elastic auto-scaling config ----
# enable_policy=True keeps the ElasticExecutor monitor loop running; the
# scaling_policy_variant selects the ScalingPolicy implementation. Below is a
# complete, tunable elastic block (modeled on examples/elastic_rm/*_fsdp.sh):
# cooldown plus throughput-optimal policy tuning.
export PSRL_DEPLOY_EXTRA="\
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
psrl.deployment.elastic.throughput_optimal.max_scale_instances_per_action=32 \
psrl.deployment.elastic.throughput_optimal.router_waiting_top_t=-1 \
psrl.deployment.elastic.throughput_optimal.role_throughput_weight_enable=true \
psrl.deployment.elastic.throughput_optimal.role_throughput_weight_basis=request_count \
psrl.deployment.elastic.throughput_optimal.role_throughput_weight_mode=raw \
+reward.managers.gen_qwen3_8b.routing_strategy.method=itl \
+reward.managers.gen_qwen3_8b.routing_strategy.cost_model_path=/apdcephfs_zwfy10/share_303541817/yfzhao/psrl/psrl/trainer/config/cost_model/qwen3_30b_a3b_thinking_2507.json \
+reward.managers.gen_qwen3_8b.routing_strategy.delta_throughput_threshold=0.005 \
+reward.managers.gen_qwen3_8b.routing_strategy.request_budget=1024 \
+reward.managers.gen_qwen3_8b.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
+reward.managers.gen_qwen3_8b.routing_strategy.max_concurrent_seqs_per_instance=128 \
"


launch_deployment_mode "$@"
