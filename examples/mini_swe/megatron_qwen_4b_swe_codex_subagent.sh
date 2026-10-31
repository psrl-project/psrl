#!/usr/bin/env bash
set -xeuo pipefail

staleness=${1:-1}
project_name=psrl_swe_smith
experiment_name=GRPO-Qwen-4B-swe_smith-codex-subagent-megatron-staleness_${staleness}

SCRIPT_PYTHON=$(command -v python || true)
if [[ -n "${PSRL_WORKSPACE:-}" && -f "${PSRL_WORKSPACE}/env/env.sh" ]]; then
    source "${PSRL_WORKSPACE}/env/env.sh"
elif [[ -n "${PSRL_WORKSPACE:-}" && -f "${PSRL_WORKSPACE}/env.sh" ]]; then
    source "${PSRL_WORKSPACE}/env.sh"
fi
if [[ -n "$SCRIPT_PYTHON" ]]; then
    export PATH="$(dirname "$SCRIPT_PYTHON"):$PATH"
fi

# Read-only harness runtime trees (native Claude Code / Codex, no Node).
# The canonical trees live on the shared filesystem and the p3b nodes allowlist
# this exact path; each sandbox mounts <root>/codex read-only.
export PSRL_HARNESS_RUNTIME_ROOT=${PSRL_HARNESS_RUNTIME_ROOT:-/apdcephfs_zwfy10_303541817/share_303541817/ls/harness_runtimes}

export SWE_STRICT_NO_TEST_PATCH=1
export SWE_STRICT_NO_CONFIG_PATCH=1
export SWE_TEST_PATCH_POLICY_SCOPE=all_tests

HOME=${PSRL_WORKSPACE}
# Run the checkout this script lives in. The shared environment has an editable
# `psrl` install that may point at another worktree; a validation run must not
# silently execute that copy.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PSRL_PATH=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$PSRL_PATH${PYTHONPATH:+:$PYTHONPATH}"

# Staged runtime libraries and NIXL backend plugins. The wheel's compiled
# default is a build path and the wheel's own plugins may be built against a
# different NIXL API; this file points at the shared runtime tree.
source "${PSRL_PATH}/scripts/nixl_env.sh"
# The cluster's NIXL library and the staged plugins are the API-1 pair. Put the
# cluster lib first so the wheel's API-2 copy cannot win through RUNPATH.
export LD_LIBRARY_PATH="/apdcephfs_zwfy10/share_303541817/lhy/anaconda3/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/apdcephfs_zwfy10/share_303541817/lhy/anaconda3/lib/libnixl.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export NIXL_PLUGIN_DIR=${PSRL_NIXL_PLUGIN_DIR:-/apdcephfs_zwfy10/share_303541817/lhy/nixl_runtime/plugins}
if [[ -n "$SCRIPT_PYTHON" ]]; then
    export PATH="$(dirname "$SCRIPT_PYTHON"):$PATH"
fi

# --- Pre-flight checks ---
echo "=== Pre-flight checks ==="
python -c "from minisweagent.agents.default import DefaultAgent; print('mini-swe-agent: OK')"
python -c "from examples.mini_swe.grading.payload import grader_zip_bytes; print('grading payload', len(grader_zip_bytes()), 'bytes: OK')"
python -c "from examples.mini_swe.swebench_grader import grade_fresh_container; print('swebench_grader: OK')"
ray status 2>/dev/null | head -5 || echo "WARNING: ray status failed"
echo "=== Pre-flight done ==="

# --- Reclaim sandboxes a previous run left behind ---
# A run's sandboxes are normally removed by its own teardown. A run that was killed skips
# that, and its sandboxes stay up with their harness agents inside, retrying against
# whatever binds the gateway port next. This run would then open on a burst of
# model_not_found from a previous run's agents, and the node would not get their memory
# back. So the previous run's sandboxes go before this one starts.
#
# The sweep removes a container only when its owner lease is gone, so a run that is still
# alive is untouched, and the lock it takes keeps it from racing that run's own collector.
#
# One window remains by design: an owner whose heartbeat is younger than the lease TTL is
# not yet provably dead, so a run killed less than that ago is reclaimed by the node
# collector a minute later rather than here.
echo "=== Reclaiming sandboxes from previous runs ==="
# The nodes come from the control plane's own fleet declaration, so this follows the
# deployment rather than repeating a host list that can drift from it.
SANDBOX_NODES=${SANDBOX_NODES:-$(python - "$PSRL_PATH" <<'PY'
import json
import os
import sys

try:
    with open(os.path.join(sys.argv[1], "p3b", "deploy_control.json"), encoding="utf-8") as handle:
        fleet = json.load(handle).get("fleet_nodes", [])
except OSError:
    fleet = []
print(" ".join(node["address"].split(":")[0] for node in fleet))
PY
)}
for node in ${SANDBOX_NODES:-$LOCAL_IP}; do
    if [ "$node" = "$LOCAL_IP" ]; then
        (cd "$PSRL_PATH" && python -m psrl.sandbox.reclaimer --once) ||
            echo "WARNING: reclaiming sandboxes on $node failed"
    else
        ssh -n -o StrictHostKeyChecking=no "root@$node" \
            ". /root/env/env.sh && cd $PSRL_PATH && /root/venv/bin/python -m psrl.sandbox.reclaimer --once" ||
            echo "WARNING: reclaiming sandboxes on $node failed"
    fi
done
echo "=== Sandbox reclamation done ==="

# --- Model ---
# Qwen3.5-4B has max_position_embeddings=262144, so do not patch config.json.
HF_MODEL_PATH=${HF_MODEL_PATH:-/jizhicfs/lhy/models/Qwen3.5-4B}
CUSTOM_CHAT_TEMPLATE=${CUSTOM_CHAT_TEMPLATE:-${PSRL_PATH}/examples/mini_swe/config/qwen35_no_think_strip.jinja}

# --- Data ---
# Both splits are SkyRL-v0-293: 293 train and 23 validation SWE-bench Verified rows.
# The parquets live on the shared filesystem so every worker resolves the same paths.
SWE_GYM_293_ROOT=${SWE_GYM_293_ROOT:-/apdcephfs_zwfy10_303541817/share_303541817/ls/docker_images/swe_gym_293}
TRAIN_FILE=${SWE_GYM_293_ROOT}/train.parquet
TEST_FILE=${SWE_GYM_293_ROOT}/val.parquet

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: Training data not found at $TRAIN_FILE"
    echo "Run the data preparation commands at the top of this script."
    exit 1
fi

if [[ ! -f "$TEST_FILE" ]]; then
    echo "ERROR: Validation data not found at $TEST_FILE"
    echo "Run the data preparation commands at the top of this script."
    exit 1
fi

train_files="['$TRAIN_FILE']"
test_files="['$TEST_FILE']"

CKPT_ROOT=${CKPT_ROOT:-$PWD}
default_local_dir=$CKPT_ROOT/checkpoint/$experiment_name

# --- Agent loop config (swebench-tuned: cwd=/testbed, swebench-style submit) ---
# The subagent variant enables Codex's native subagent tools. The training policy
# below keeps only the main agent, so subagent turns are replayed as context and
# never produce a training row. It requires the companion smg-subagent binding,
# which carries turn sequence numbers and the compact snapshot format.
agent_loop_config_path=${PSRL_PATH}/examples/mini_swe/config/swebench_harness_config_codex_subagent.yaml

# --- Cluster layout (16 GPUs: one node of rollout engines, one node of training ranks) ---
# Rollout: TP=1, so one engine per GPU.
# Train: TP=4, CP=2 by default; a model whose head layout does not divide by
# TP*CP needs its own values here.
GEN_TP=${GEN_TP:-1}
GEN_PP=${GEN_PP:-1}

VAL_TP=${VAL_TP:-1}

TRAIN_TP=${TRAIN_TP:-4}
TRAIN_PP=${TRAIN_PP:-1}
TRAIN_CP=${TRAIN_CP:-2}

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

# --- Algorithm (GRPO / DAPO) ---
# Aligned with OpenClaw swe-rl 4B defaults:
#
#   GRPO + DAPO asymmetric clip (0.2/0.28), KL effectively off, entropy=0.
# Dynamic sampling filter: drops rollout groups with std=0 reward.
enable_dynamic_sampling_filter=False
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28

# --- Sequence lengths ---
#
# One multi-turn trajectory shares a 65536-token budget between prompt and response,
# and max_model_len matches it so the CLI compaction trigger stays trainable.
#
# Codex takes no turn cap of its own, so this is declared as the session's turn budget
# and the session router refuses the turn after it is spent. The numbers match the
# Claude Code recipe: both bound the parent agent's turns, and a subagent's calls are
# charged separately rather than to the same budget.
max_turns=80
val_max_turns=160
trajectory_timeout=${TRAJECTORY_TIMEOUT:-1800}
# An episode can still exhaust the trajectory budget before it exhausts its turns, and
# one that does is stopped at the deadline with its partial patch and trajectory kept and
# graded, instead of being cancelled and re-run with all of its turns discarded.
trajectory_timeout_action=keep
max_prompt_length=32768
max_response_length=32768
packing_length=$((max_prompt_length + max_response_length))
max_model_len=$((max_prompt_length + max_response_length))

# --- Training hyperparameters (from OpenClaw 4B) ---
# Optimizer: lr=1e-6, weight_decay=0.01, adam_beta2=0.999
actor_lr=1e-6
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 30))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
train_prompt_bsz=${TRAIN_BSZ:-8}
n_resp_per_prompt=${ROLLOUT_N:-16}
n_resp_per_prompt_val=1
train_prompt_mini_bsz=${TRAIN_MINI_BSZ:-$train_prompt_bsz}

# --- Sampling (from OpenClaw 4B) ---
temperature=1
top_p=1
top_k=-1
val_top_p=0.7

# --- Reward ---
# Reward mode: binary | partial_credit | test_ratio | shaped
reward_mode=binary_01

# --- TIS ---
rollout_is=token
rollout_is_threshold=2.0

# --- Performance ---
use_dynamic_bsz=True
offload=True

# The in-process docker backend is dropped below rather than left as a fallback. It keeps
# its own mirror of the node envelope (SandboxCapacityCoordinator) and charges each sandbox
# its declared 30 GiB against it, which caps a step at 58 sandboxes per node, so a sandbox
# that lands there never sees the service's measured pricing. With p3b alone every sandbox
# goes through one ledger -- the one that prices from measurement -- and a step runs 256
# concurrent sandboxes with no admission queue.
PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=${n_resp_per_prompt} \
    psrl.staleness=${staleness} \
    psrl.staleness_buffer_entries=${train_prompt_bsz} \
    psrl.ps_mode=nixl_cpu \
    psrl.lmcache.enable=False \
    psrl.lmcache.enable_p2p=False \
    psrl.colocate_validate_and_train=False \
    psrl.rollout_coordination.routing_strategy.method=cache_aware_v1 \
    psrl.rollout_coordination.routing_strategy.cache_aware_policy.lmcache_overlap_weight=0.5 \
    psrl.rollout_coordination.routing_strategy.enable_trajectory_sticky=True \
    psrl.rollout_coordination.routing_strategy.enable_group_sticky=True \
    psrl.rollout_coordination.routing_strategy.kv_transfer.enable=False \
    psrl.rollout_coordination.routing_strategy.kv_transfer.transfer_mode=async \
    psrl.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption=0 \
    psrl.rollout_coordination.session_strategy.thunder_agent.enable=True \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.enable=False \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.indicator=request_num \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.threshold=10 \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_indicator=request_num \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_threshold=20 \
    psrl.logging_path=${PSRL_PATH}/examples/mini_swe/megatron_psrl_log/${experiment_name} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.agentic_rl.batch_agg_mode=request \
    psrl.agentic_rl.subagent_training.mode=main_only \
    psrl.agentic_rl.subagent_training.val_mode=main_only \
    psrl.rollout_gateway.tito_snapshot_format=compact \
    psrl.agentic_rl.trajectory_output.enable=True \
    psrl.agentic_rl.trajectory_output.dir=${PSRL_PATH}/examples/mini_swe/megatron_psrl_log/${experiment_name}/trajectories \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.deployment.total_nnodes=${NNODES} \
    psrl.deployment.sandbox_placement.enabled=false \
    psrl.nixl.server_port=23456 \
    psrl.rollout_gateway.trajectory_id_strategy=auto \
    psrl.rollout_gateway.tito_tool_normalization=codex \
    psrl.rollout_gateway.tito_debug=false \
    psrl.rollout_gateway.tool_call_parser=qwen_xml \
    \
    gen_actor_rollout_ref.rollout.agent.sandbox.backends.p3b.endpoint=${P3B_ENDPOINT:-29.162.247.148:9000} \
    gen_actor_rollout_ref.rollout.agent.overlong_filtering=True \
    \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=mini_swe_codex_subagent \
    gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj \
    gen_actor_rollout_ref.rollout.agent.sandbox.default_backend=p3b \
    ~gen_actor_rollout_ref.rollout.agent.sandbox.backends.docker \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=${packing_length} \
    gen_actor_rollout_ref.rollout.temperature=${temperature} \
    gen_actor_rollout_ref.rollout.top_p=${top_p} \
    gen_actor_rollout_ref.rollout.top_k=${top_k} \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=$max_turns \
    gen_actor_rollout_ref.rollout.agent.trajectory_timeout=$trajectory_timeout \
    gen_actor_rollout_ref.rollout.agent.trajectory_timeout_action=$trajectory_timeout_action \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    gen_actor_rollout_ref.rollout.agent.env.name=mini_swe_env \
    gen_actor_rollout_ref.rollout.agent.data.name=mini_swe_agent_data \
    gen_actor_rollout_ref.rollout.agent.num_workers=${NNODES} \
    gen_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    \
    train_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    train_actor_rollout_ref.model.custom_chat_template=${CUSTOM_CHAT_TEMPLATE} \
    train_actor_rollout_ref.model.use_fused_kernels=False \
    train_actor_rollout_ref.model.use_remove_padding=True \
    train_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${packing_length} \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=${packing_length} \
    train_actor_rollout_ref.rollout.multi_turn.max_turns=$val_max_turns \
    train_actor_rollout_ref.rollout.agent.trajectory_timeout=$trajectory_timeout \
    train_actor_rollout_ref.rollout.agent.trajectory_timeout_action=$trajectory_timeout_action \
    train_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    train_actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    train_actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    train_actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
    train_actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    train_actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    train_actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    train_actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    train_actor_rollout_ref.actor.clip_ratio_c=10.0 \
    train_actor_rollout_ref.actor.optim.lr=${actor_lr} \
    train_actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
    train_actor_rollout_ref.actor.optim.clip_grad=1.0 \
    train_actor_rollout_ref.actor.optim.betas='[0.9,0.98]' \
    train_actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${packing_length} \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    train_actor_rollout_ref.actor.megatron.param_offload=False \
    train_actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.filter_groups.metric=reward_score \
    algorithm.group_baseline_valid_only=True \
    psrl.group_post_process.enable=${enable_dynamic_sampling_filter} \
    psrl.group_post_process.name=dynamic_sampling_filter \
    \
    reward.active_managers='[dapo]' \
    reward.managers.dapo.reward_fn.0.path=${PSRL_PATH}/examples/mini_swe/reward.py \
    reward.managers.dapo.reward_fn.0.name=compute_score \
    +reward.managers.dapo.reward_fn.0.reward_kwargs.reward_mode=${reward_mode} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.log=False \
    reward.managers.dapo.reward_kwargs.max_resp_len=${max_response_length} \
    \
    data.train_files="$train_files" \
    data.reward_model_dicts.0.reward_loop_type=dapo \
    data.reward_model_dicts.0.reward_fn=compute_score \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.val_before_train=False \
    trainer.log_val_generations=5 \
    trainer.test_freq=5 \
    trainer.save_freq=50 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=${TOTAL_STEPS:-200} 2>&1 | tee ${experiment_name}.log
