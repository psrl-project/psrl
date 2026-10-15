#!/usr/bin/env bash
# Mode 6 ablates elastic RL with Qwen2.5-7B and Qwen3-30B-A3B on two trainer and two shared nodes.
#
# Set `POLICY`, `USE_COST_MODEL`, `USE_REBALANCE`, and `INTERRUPT_VLLM_WAITING` to select ablations.
#
# USE_REBALANCE=1 requires POLICY=throughput-optimal. USE_REBALANCE=0 only skips
# post-scale-up ABORT. ITL candidate scoring still uses request-level evaluation.
#
# Run from the repository root: `bash examples/deployment_modes_qwen7b_rm32b/mode6_ablation_study_copy.sh`.
set -euo pipefail

_normalize_bool() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) echo 1 ;;
        0|false|FALSE|no|NO|off|OFF) echo 0 ;;
        *)
            echo "Invalid boolean '${1:-}'; expected 1/0 (or true/false)." >&2
            return 1
            ;;
    esac
}

_normalize_policy() {
    case "${1:-}" in
        throughput-optimal) echo throughput-optimal ;;
        rule-based) echo rule-based ;;
        *)
            echo "Invalid POLICY='${1:-}'. Expected throughput-optimal or rule-based." >&2
            return 1
            ;;
    esac
}

POLICY=$(_normalize_policy "${POLICY:-rule-based}")
USE_COST_MODEL=$(_normalize_bool "${USE_COST_MODEL:-0}")
USE_REBALANCE=$(_normalize_bool "${USE_REBALANCE:-0}")
INTERRUPT_VLLM_WAITING=$(_normalize_bool "${INTERRUPT_VLLM_WAITING:-1}")

if [[ "${USE_REBALANCE}" == "1" && "${POLICY}" != "throughput-optimal" ]]; then
    echo "USE_REBALANCE=1 requires POLICY=throughput-optimal (request-level rebalance is ITL-only)." >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=elastic_rl
export PSRL_DEPLOY_EXPERIMENT="mode6_ablation_${POLICY}_cost${USE_COST_MODEL}_rebalance${USE_REBALANCE}_interrupt${INTERRUPT_VLLM_WAITING}"
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=False
export PSRL_DEPLOY_NNODES=4
export PSRL_DEPLOY_TRAIN_NNODES=2
export PSRL_DEPLOY_TRAIN_NGPUS=8
export PSRL_DEPLOY_SHARED_NNODES=2
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_SMOKE=${1:-0}
export PSRL_NEED_VALIDATION=${PSRL_NEED_VALIDATION:-0}
shift || true

COST_MODEL_DIR=/apdcephfs_zwfy10/share_303541817/yfzhao/psrl/psrl/trainer/config/cost_model
# Rollout routing in parent _common_deployment.sh points at qwen2.5_1.5b.json.
# This folder's model is Qwen2.5-7B. pass the matching file last so it wins.
ROLLOUT_COST_MODEL="${COST_MODEL_DIR}/qwen2.5_7b.json"
RM_COST_MODEL="${COST_MODEL_DIR}/qwen3_30b_a3b_thinking_2507.json"

if [[ "${USE_COST_MODEL}" == "1" ]]; then
    COST_MODEL_OVERRIDE="psrl.deployment.elastic.throughput_optimal.cost_model_path=${COST_MODEL_DIR}"
else
    COST_MODEL_OVERRIDE="psrl.deployment.elastic.throughput_optimal.cost_model_path=null"
fi

REQUEST_LEVEL_OVERRIDE="true"
if [[ "${USE_REBALANCE}" == "1" ]]; then
    REBALANCE_OVERRIDE="true"
else
    REBALANCE_OVERRIDE="false"
fi

export PSRL_DEPLOY_EXTRA="\
psrl.deployment.elastic.enable_policy=True \
psrl.deployment.elastic.scaling_policy_variant=${POLICY} \
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
psrl.deployment.elastic.throughput_optimal.enable_heterogeneous_parallelism_candidates=true \
psrl.deployment.elastic.throughput_optimal.enable_request_level_candidate_evaluation=${REQUEST_LEVEL_OVERRIDE} \
psrl.deployment.elastic.throughput_optimal.rebalance_after_scale_up=${REBALANCE_OVERRIDE} \
${COST_MODEL_OVERRIDE} \
+reward.managers.gen_qwen3_8b.routing_strategy.method=itl \
+reward.managers.gen_qwen3_8b.routing_strategy.cost_model_path=${RM_COST_MODEL} \
+reward.managers.gen_qwen3_8b.routing_strategy.delta_throughput_threshold=0.005 \
+reward.managers.gen_qwen3_8b.routing_strategy.request_budget=1024 \
+reward.managers.gen_qwen3_8b.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
+reward.managers.gen_qwen3_8b.routing_strategy.max_concurrent_seqs_per_instance=128 \
"

echo "mode6 ablation: POLICY=${POLICY} USE_COST_MODEL=${USE_COST_MODEL} USE_REBALANCE=${USE_REBALANCE} INTERRUPT_VLLM_WAITING=${INTERRUPT_VLLM_WAITING}"

launch_deployment_mode \
    psrl.rollout_coordination.routing_strategy.cost_model_path=${ROLLOUT_COST_MODEL} \
    "$@"
