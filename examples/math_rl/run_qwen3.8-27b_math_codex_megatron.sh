#!/usr/bin/env bash
# Train Qwen3.8-27B on the math RL slice with GRPO and score centering.
# Usage: `run_qwen3.8-27b_math_codex_megatron.sh`, with the verifier and p3b env set.
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PSRL_PATH=$(cd "${SCRIPT_DIR}/../.." && pwd)

# The workspace env sets cluster transport variables and may change `python`.
# Keep the interpreter this script was launched with, matching the other recipes.
SCRIPT_PYTHON=$(command -v python || true)
if [[ -f "${PSRL_WORKSPACE:-}/env/env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PSRL_WORKSPACE}/env/env.sh"
elif [[ -f "${PSRL_WORKSPACE:-}/env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PSRL_WORKSPACE}/env.sh"
fi
if [[ -n "${SCRIPT_PYTHON}" ]]; then
    export PATH="$(dirname "${SCRIPT_PYTHON}"):$PATH"
fi

export PYTHONPATH="${PSRL_PATH}:${PYTHONPATH:-}"

LOCAL_IP=${LOCAL_IP:-127.0.0.1}
STALENESS=${STALENESS:-0}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
NIXL_PORT=${NIXL_PORT:-23456}

HF_MODEL_PATH=${HF_MODEL_PATH:-${PSRL_WORKSPACE}/models/Qwen3.8-27B}
MATH_RL_DATA_DIR=${MATH_RL_DATA_DIR:-${PSRL_PATH}/examples/math_rl/data}
AGENT_CONFIG=${PSRL_PATH}/examples/math_rl/config/math_harness_config.yaml
REWARD_SCRIPT=${PSRL_PATH}/examples/math_rl/reward.py

MATH_RL_SANDBOX_IMAGE=${MATH_RL_SANDBOX_IMAGE:-math-rl-codex:local}
MATH_RL_RUNTIME_IN_IMAGE=${MATH_RL_RUNTIME_IN_IMAGE:-true}
if [[ "${MATH_RL_RUNTIME_IN_IMAGE}" == "true" ]]; then
    MATH_RL_HARNESS_RUNTIME_MOUNT=${MATH_RL_HARNESS_RUNTIME_MOUNT:-/usr/local}
else
    MATH_RL_HARNESS_RUNTIME_MOUNT=${MATH_RL_HARNESS_RUNTIME_MOUNT:-/opt/harness}
fi
MATH_RL_VERIFIER_MODE=${MATH_RL_VERIFIER_MODE:-api}
MATH_RL_VERIFIER_VERSION=${MATH_RL_VERIFIER_VERSION:-V3}
MATH_RL_VERIFIER_MODEL=${MATH_RL_VERIFIER_MODEL:-glm-5.3}
MATH_RL_VERIFIER_EFFORT=${MATH_RL_VERIFIER_EFFORT:-max}
MATH_RL_VERIFIER_PROXY=${MATH_RL_VERIFIER_PROXY:-${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}}
MATH_RL_VERIFIER_TRANSPORT=${MATH_RL_VERIFIER_TRANSPORT:-stream}
MATH_RL_VERIFIER_STAGE_TIMEOUT_S=${MATH_RL_VERIFIER_STAGE_TIMEOUT_S:-900}
MATH_RL_VERIFIER_MAX_RETRIES=${MATH_RL_VERIFIER_MAX_RETRIES:-1}
MATH_RL_VERIFIER_MAX_CONCURRENCY=${MATH_RL_VERIFIER_MAX_CONCURRENCY:-16}
MATH_RL_REWARD_MODE=${MATH_RL_REWARD_MODE:-graded}
GRADING_TIMEOUT=${GRADING_TIMEOUT:-2900}
export MATH_RL_SANDBOX_IMAGE MATH_RL_RUNTIME_IN_IMAGE MATH_RL_HARNESS_RUNTIME_MOUNT
export MATH_RL_VERIFIER_MODE MATH_RL_VERIFIER_VERSION MATH_RL_VERIFIER_MODEL MATH_RL_VERIFIER_EFFORT
export MATH_RL_VERIFIER_PROXY MATH_RL_VERIFIER_TRANSPORT MATH_RL_VERIFIER_STAGE_TIMEOUT_S MATH_RL_VERIFIER_MAX_RETRIES
export MATH_RL_VERIFIER_MAX_CONCURRENCY
export MATH_RL_REWARD_MODE
export PYTHONDONTWRITEBYTECODE=1

TRAIN_FILE=${MATH_RL_DATA_DIR}/train.parquet
TEST_FILE=${MATH_RL_DATA_DIR}/val.parquet
LOG_DIR=${LOG_DIR:-${PSRL_WORKSPACE}/output/math_rl/$(date +%Y%m%d_%H%M%S)}

mkdir -p "${LOG_DIR}"

# --- Mock verifier ---
if [[ "${MATH_RL_VERIFIER_MODE}" == "mock" ]]; then
    MOCK_PORT=${MATH_RL_MOCK_VERIFIER_PORT:-18080}
    MOCK_POLICY=${MATH_RL_MOCK_VERIFIER_POLICY:-hashed}
    python -m examples.math_rl.mock_verifier_server \
        --host 0.0.0.0 --port "${MOCK_PORT}" --policy "${MOCK_POLICY}" \
        --version "${MATH_RL_VERIFIER_VERSION}" \
        >"${LOG_DIR}/mock_verifier.log" 2>&1 &
    MOCK_PID=$!
    trap 'kill ${MOCK_PID} 2>/dev/null || true' EXIT
    export MATH_RL_VERIFIER_URL="http://${LOCAL_IP}:${MOCK_PORT}/verify"
    export MATH_RL_VERIFIER_MODEL=mock-verifier
    MOCK_READY=0
    for _ in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${MOCK_PORT}/healthz" >/dev/null; then
            MOCK_READY=1
            break
        fi
        sleep 1
    done
    if [[ "${MOCK_READY}" -ne 1 ]]; then
        echo "ERROR: mock verifier did not start on port ${MOCK_PORT}." >&2
        exit 1
    fi
    echo "Mock verifier is listening on ${MATH_RL_VERIFIER_URL}."
fi

: "${MATH_RL_VERIFIER_URL:?Set MATH_RL_VERIFIER_URL, or use MATH_RL_VERIFIER_MODE=mock.}"
: "${PSRL_P3B_ENDPOINT:?Set PSRL_P3B_ENDPOINT to the sandbox control plane.}"

# --- Pre-flight ---
echo "=== Pre-flight checks ==="
[[ -f "${HF_MODEL_PATH}/config.json" ]] || { echo "ERROR: model not found at ${HF_MODEL_PATH}." >&2; exit 1; }
[[ -f "${TRAIN_FILE}" ]] || { echo "ERROR: training parquet not found at ${TRAIN_FILE}." >&2; exit 1; }
[[ -f "${TEST_FILE}" ]] || { echo "ERROR: validation parquet not found at ${TEST_FILE}." >&2; exit 1; }
[[ -f "${AGENT_CONFIG}" ]] || { echo "ERROR: agent config not found at ${AGENT_CONFIG}." >&2; exit 1; }

if [[ "${MATH_RL_RUNTIME_IN_IMAGE}" == "false" ]]; then
    : "${PSRL_HARNESS_RUNTIME_ROOT:?Set PSRL_HARNESS_RUNTIME_ROOT when the image has no Codex.}"
    export PSRL_HARNESS_RUNTIME_ROOT
    [[ -x "${PSRL_HARNESS_RUNTIME_ROOT}/codex/bin/codex" ]] || {
        echo "ERROR: Codex runtime not found under ${PSRL_HARNESS_RUNTIME_ROOT}/codex." >&2
        exit 1
    }
fi

if command -v docker >/dev/null 2>&1; then
    docker image inspect "${MATH_RL_SANDBOX_IMAGE}" >/dev/null 2>&1 \
        || echo "WARNING: image ${MATH_RL_SANDBOX_IMAGE} is not present on this host."
fi
ray status >/dev/null 2>&1 || echo "WARNING: ray status failed. Start the cluster before training."
python -m psrl.sandbox.reclaimer --once >/dev/null 2>&1 \
    || echo "WARNING: reclaiming sandboxes from a previous run failed."

VERIFIER_HEALTH_URL="${MATH_RL_VERIFIER_URL%/verify/stream}"
VERIFIER_HEALTH_URL="${VERIFIER_HEALTH_URL%/verify}/health"
python - "${VERIFIER_HEALTH_URL}" "${MATH_RL_VERIFIER_VERSION}" "${MATH_RL_VERIFIER_MODEL}" <<'PY'
import json
import sys
import urllib.request

url, version, model = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        health = json.load(response)
except Exception as error:
    raise SystemExit(f"The verifier health check failed at {url}: {error}")
served = health.get("verifier_harness_version") or health.get("version") or ""
if served != version:
    raise SystemExit(f"The verifier serves {served!r}, but the recipe asks for {version!r}.")
models = health.get("models") or {}
if models and model not in models:
    raise SystemExit(f"The verifier does not serve model {model!r}. Configured: {sorted(models)}.")
if models and not models[model].get("configured", False):
    raise SystemExit(f"The verifier reports model {model!r} as not configured.")
print(
    f"Verifier healthy: version {served}, model {model}, "
    f"web_search {health.get('web_search')}, network_access {health.get('network_access')}."
)
PY
echo "=== Pre-flight done ==="

# Resolve the harness YAML on the driver, because Ray workers do not inherit these variables.
RESOLVED_AGENT_CONFIG=${LOG_DIR}/math_harness_config.yaml
MATH_RL_AGENT_CONFIG=${AGENT_CONFIG} MATH_RL_RESOLVED_AGENT_CONFIG=${RESOLVED_AGENT_CONFIG} python - <<'PY'
import os

import hydra  # noqa: F401
from omegaconf import OmegaConf

source = os.environ["MATH_RL_AGENT_CONFIG"]
target = os.environ["MATH_RL_RESOLVED_AGENT_CONFIG"]
OmegaConf.save(OmegaConf.load(source), target, resolve=True)
PY
chmod 600 "${RESOLVED_AGENT_CONFIG}"
AGENT_CONFIG=${RESOLVED_AGENT_CONFIG}

# --- Cluster layout: 2 nodes, 8 GPUs each ---
GEN_TP=2
GEN_PP=1
VAL_TP=2
TRAIN_TP=4
TRAIN_PP=2
TRAIN_CP=1

NNODES=2
NGPUS_PER_NODE=8

GEN_NNODES=1
GEN_NGPUS_PER_NODE=${NGPUS_PER_NODE}
GEN_INSTANCES=$(( (GEN_NNODES * GEN_NGPUS_PER_NODE) / (GEN_TP * GEN_PP) ))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( GEN_TP * GEN_PP ))

TRAIN_NNODES=1
TRAIN_NGPUS_PER_NODE=${NGPUS_PER_NODE}
VAL_INSTANCES=$(( (TRAIN_NNODES * TRAIN_NGPUS_PER_NODE) / VAL_TP ))
VAL_NGPUS_PER_NODE_PER_INSTANCE=${VAL_TP}

# --- Batch and sequence budgets ---
train_prompt_bsz=${TRAIN_BSZ:-32}
n_resp_per_prompt=${ROLLOUT_N:-8}
n_resp_per_prompt_val=1
train_prompt_mini_bsz=${TRAIN_MINI_BSZ:-${train_prompt_bsz}}
max_prompt_length=4096
max_response_length=16384
packing_length=$((max_prompt_length + max_response_length))
max_turns=60
trajectory_timeout=3600

# --- Algorithm ---
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode=token-mean
rollout_is=token
rollout_is_threshold=2.0

# --- Sampling ---
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# --- Training hyperparameters ---
actor_lr=1e-6
lr_warmup_steps=${LR_WARMUP_STEPS:-10}
offload=True

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo \
    --config-path="${PSRL_PATH}/psrl/trainer/config" \
    --config-name=ppo_megatron_trainer \
    psrl.ps_manager_ip="${LOCAL_IP}" \
    psrl.rollout_n="${n_resp_per_prompt}" \
    psrl.staleness="${STALENESS}" \
    psrl.staleness_buffer_entries="${train_prompt_bsz}" \
    psrl.ps_mode=nixl_cpu \
    psrl.lmcache.enable=False \
    psrl.lmcache.enable_p2p=False \
    psrl.colocate_validate_and_train=False \
    psrl.logging_path="${LOG_DIR}/psrl" \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.agentic_rl.batch_agg_mode=request \
    psrl.agentic_rl.trajectory_output.enable=True \
    psrl.agentic_rl.trajectory_output.dir="${LOG_DIR}/trajectories" \
    psrl.deployment.n_rollout_instances="${GEN_INSTANCES}" \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance="${GEN_NGPUS_PER_NODE_PER_INSTANCE}" \
    psrl.deployment.n_validate_instances="${VAL_INSTANCES}" \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance="${VAL_NGPUS_PER_NODE_PER_INSTANCE}" \
    psrl.deployment.train_nnodes="${TRAIN_NNODES}" \
    psrl.deployment.train_ngpus_per_node="${TRAIN_NGPUS_PER_NODE}" \
    psrl.deployment.total_nnodes="${NNODES}" \
    psrl.deployment.sandbox_placement.enabled=false \
    psrl.nixl.server_port="${NIXL_PORT}" \
    psrl.rollout_gateway.trajectory_id_strategy=auto \
    psrl.rollout_coordination.routing_strategy.method=cache_aware_v1 \
    psrl.rollout_coordination.routing_strategy.cache_aware_policy.lmcache_overlap_weight=0.5 \
    psrl.rollout_coordination.routing_strategy.enable_trajectory_sticky=True \
    psrl.rollout_coordination.routing_strategy.enable_group_sticky=True \
    psrl.rollout_coordination.routing_strategy.kv_transfer.enable=False \
    psrl.rollout_coordination.routing_strategy.kv_transfer.transfer_mode=async \
    psrl.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption=0 \
    psrl.rollout_coordination.session_strategy.thunder_agent.enable=True \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.enable=False \
    psrl.rollout_gateway.tito_tool_normalization=codex \
    psrl.rollout_gateway.tito_debug=false \
    psrl.rollout_gateway.tool_call_parser=qwen_xml \
    psrl.group_post_process.enable=True \
    psrl.group_post_process.name=dynamic_sampling_filter \
    \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=math_codex \
    gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG}" \
    gen_actor_rollout_ref.rollout.agent.sandbox.default_backend=p3b \
    gen_actor_rollout_ref.rollout.agent.sandbox.backends.p3b.endpoint="${PSRL_P3B_ENDPOINT}" \
    ~gen_actor_rollout_ref.rollout.agent.sandbox.backends.docker \
    gen_actor_rollout_ref.rollout.agent.overlong_filtering=True \
    gen_actor_rollout_ref.rollout.agent.num_workers="${NNODES}" \
    gen_actor_rollout_ref.rollout.agent.trajectory_timeout="${trajectory_timeout}" \
    gen_actor_rollout_ref.rollout.agent.trajectory_timeout_action=keep \
    gen_actor_rollout_ref.rollout.agent.grading_timeout="${GRADING_TIMEOUT}" \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size="${GEN_PP}" \
    gen_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens="${packing_length}" \
    gen_actor_rollout_ref.rollout.max_model_len="${packing_length}" \
    gen_actor_rollout_ref.rollout.temperature="${temperature}" \
    gen_actor_rollout_ref.rollout.top_p="${top_p}" \
    gen_actor_rollout_ref.rollout.top_k="${top_k}" \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns="${max_turns}" \
    gen_actor_rollout_ref.rollout.sampler_distribution=topk \
    gen_actor_rollout_ref.rollout.sampler_max_tokens=32 \
    gen_actor_rollout_ref.rollout.sampler_compact_support=True \
    gen_actor_rollout_ref.rollout.logprobs_mode=processed_logprobs \
    \
    train_actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    train_actor_rollout_ref.model.use_remove_padding=True \
    train_actor_rollout_ref.model.use_fused_kernels=False \
    train_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${packing_length}" \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size="${VAL_TP}" \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens="${packing_length}" \
    train_actor_rollout_ref.rollout.max_model_len="${packing_length}" \
    train_actor_rollout_ref.rollout.multi_turn.max_turns="${max_turns}" \
    train_actor_rollout_ref.rollout.agent.trajectory_timeout="${trajectory_timeout}" \
    train_actor_rollout_ref.rollout.agent.trajectory_timeout_action=keep \
    train_actor_rollout_ref.rollout.agent.default_agent_loop=math_codex \
    train_actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG}" \
    train_actor_rollout_ref.rollout.agent.sandbox.default_backend=p3b \
    train_actor_rollout_ref.rollout.agent.sandbox.backends.p3b.endpoint="${PSRL_P3B_ENDPOINT}" \
    ~train_actor_rollout_ref.rollout.agent.sandbox.backends.docker \
    train_actor_rollout_ref.rollout.val_kwargs.temperature="${temperature}" \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    train_actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
    train_actor_rollout_ref.rollout.val_kwargs.top_k="${top_k}" \
    train_actor_rollout_ref.rollout.val_kwargs.n="${n_resp_per_prompt_val}" \
    train_actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    train_actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    train_actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    train_actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    train_actor_rollout_ref.actor.clip_ratio_c=10.0 \
    train_actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    train_actor_rollout_ref.actor.optim.lr_warmup_steps="${lr_warmup_steps}" \
    train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
    train_actor_rollout_ref.actor.optim.clip_grad=1.0 \
    train_actor_rollout_ref.actor.use_dynamic_bsz=True \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${packing_length}" \
    train_actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
    train_actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    train_actor_rollout_ref.actor.megatron.param_offload=False \
    train_actor_rollout_ref.actor.megatron.optimizer_offload="${offload}" \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    train_actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
    train_actor_rollout_ref.actor.megatron.use_mbridge=True \
    ++train_actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
    \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.rollout_correction.rollout_is="${rollout_is}" \
    algorithm.rollout_correction.rollout_is_threshold="${rollout_is_threshold}" \
    algorithm.rollout_correction.score_centering=True \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=reinforce \
    algorithm.group_baseline_valid_only=True \
    algorithm.filter_groups.metric=reward_score \
    \
    reward.active_managers='[naive]' \
    +reward.managers.naive.reward_model_name=null \
    reward.managers.naive.reward_fn.0.path="${REWARD_SCRIPT}" \
    reward.managers.naive.reward_fn.0.name=compute_score \
    \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${TEST_FILE}']" \
    data.reward_model_dicts.0.reward_loop_type=naive \
    data.reward_model_dicts.0.reward_fn=compute_score \
    data.reward_model_dicts.0.reward_model_name=null \
    data.prompt_key=prompt \
    data.truncation=error \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.train_batch_size="${train_prompt_bsz}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    \
    trainer.logger='["console"]' \
    trainer.project_name=math_rl \
    trainer.experiment_name="GRPO-Qwen3.8-27B-math-codex-score-centering" \
    trainer.default_local_dir="${LOG_DIR}/checkpoints" \
    trainer.val_before_train=False \
    trainer.log_val_generations=5 \
    trainer.test_freq=5 \
    trainer.save_freq=50 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@" \
    2>&1 | tee "${LOG_DIR}/train.log"
