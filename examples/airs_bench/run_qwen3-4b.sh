#!/usr/bin/env bash
# AIRS-Bench RL recipe, Qwen3-4B on two nodes.
# The environment node uses CPUs while the compute node runs rollout and training.
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export RAY_prestart_worker_first_driver=false
export RAY_num_workers_soft_limit=0
export RAY_memory_monitor_refresh_ms=0

source /apdcephfs_zwfy10/share_303541817/lhy/env/psrl.sh

PSRL_PATH="${PSRL_PATH:-$(python3 -c "import os, psrl; print(os.path.dirname(os.path.dirname(psrl.__file__)))")}"
cd "${PSRL_PATH}"

HF_MODEL_PATH="${HF_MODEL_PATH:-/jizhicfs/johnnyslin/models/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-${PSRL_PATH}/examples/airs_bench/data}"
export AIRS_DATA_ROOT="${AIRS_DATA_ROOT:-/apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data}"
ENV_NODE_IP="${ENV_NODE_IP:-28.49.16.220}"
export SANDBOX_IMAGE="${SANDBOX_IMAGE:-psrl/airs-bench-agent:latest}"

TRAIN_FILE="${DATA_DIR}/train.parquet"
TEST_FILE="${DATA_DIR}/val.parquet"

for path in "${HF_MODEL_PATH}" "${TRAIN_FILE}" "${TEST_FILE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required path not found: ${path}" >&2
        exit 1
    fi
done

# Experiment output directories.
project_name=airs_bench
experiment_name=qwen3_4b_airs_bench
OUTPUT_DIR="${OUTPUT_DIR:-${PSRL_PATH}/outputs/airs_bench}"
CKPTS_DIR="${OUTPUT_DIR}/ckpts/${project_name}/${experiment_name}"
PSRL_LOG_DIR="${OUTPUT_DIR}/psrl_logs/${experiment_name}"
mkdir -p "${CKPTS_DIR}" "${PSRL_LOG_DIR}"

# --- Cluster layout ---
# `total_nnodes=1` keeps PSRL workers off the dedicated environment node.
NNODES=1
GEN_TP=2
GEN_PP=1
GEN_NNODES=1
GEN_NGPUS_PER_NODE=4
GEN_INSTANCES=$(((GEN_NNODES * GEN_NGPUS_PER_NODE) / (GEN_TP * GEN_PP)))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$((GEN_TP * GEN_PP))

TRAIN_TP=2
TRAIN_PP=1
TRAIN_CP=1
TRAIN_NNODES=1
TRAIN_NGPUS_PER_NODE=4

VAL_TP=2
VAL_PP=1
VAL_INSTANCES=$(((TRAIN_NNODES * TRAIN_NGPUS_PER_NODE) / (VAL_TP * VAL_PP)))
VAL_NGPUS_PER_NODE_PER_INSTANCE=$((VAL_TP * VAL_PP))

# --- Algorithm ---
# Grouped responses provide the advantage signal across the 14 training tasks.
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-30720}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
max_model_len=32768
max_num_batched_tokens=32768
max_tokens_per_gpu=9216
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-14}"

actor_lr=1e-6
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.3

REWARD_PATH="${PSRL_PATH}/examples/airs_bench/reward.py"

echo "=== AIRS-Bench RL ==="
echo "model:    ${HF_MODEL_PATH}"
echo "train:    ${TRAIN_FILE}"
echo "val:      ${TEST_FILE}"
echo "env node: ${ENV_NODE_IP}"
echo "image:    ${SANDBOX_IMAGE}"
echo "rollouts: ${N_RESP_PER_PROMPT} per prompt, ${MAX_TURNS} max turns"

PYTHONUNBUFFERED=1 python3 -m psrl.trainer.main_ppo \
    --config-path="${PSRL_PATH}/psrl/trainer/config" \
    --config-name=ppo_trainer model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.reward_model_dicts.0.reward_loop_type=dapo \
    data.reward_model_dicts.0.reward_fn=compute_score \
    train_actor_rollout_ref.nccl_timeout=6000 \
    train_actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    train_actor_rollout_ref.actor.optim.lr=${actor_lr} \
    train_actor_rollout_ref.actor.optim.lr_decay_style=constant \
    train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
    train_actor_rollout_ref.actor.optim.betas='[0.9,0.98]' \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_dynamic_bsz=True \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_tokens_per_gpu} \
    train_actor_rollout_ref.actor.rollout_n=${N_RESP_PER_PROMPT} \
    train_actor_rollout_ref.actor.clip_ratio=0.2 \
    train_actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    train_actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    train_actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    train_actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    train_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.loss_agg_mode=token-mean \
    train_actor_rollout_ref.actor.megatron.param_offload=False \
    train_actor_rollout_ref.actor.megatron.grad_offload=True \
    train_actor_rollout_ref.actor.megatron.optimizer_offload=True \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    train_actor_rollout_ref.actor.megatron.sequence_parallel=True \
    train_actor_rollout_ref.actor.megatron.use_mbridge=True \
    train_actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    train_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    train_actor_rollout_ref.rollout.val_kwargs.n=1 \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_tokens_per_gpu} \
    train_actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    train_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_tokens_per_gpu} \
    train_actor_rollout_ref.ref.megatron.param_offload=False \
    gen_actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    gen_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    gen_actor_rollout_ref.rollout.temperature=1.0 \
    gen_actor_rollout_ref.rollout.top_p=1.0 \
    gen_actor_rollout_ref.rollout.top_k=-1 \
    gen_actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    gen_actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=${MAX_TURNS} \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=mlgym_agent \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path="${PSRL_PATH}/examples/airs_bench/config/mlgym_agent_config.yaml" \
    gen_actor_rollout_ref.rollout.agent.env.name=mlgym_env \
    gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj \
    gen_actor_rollout_ref.rollout.agent.num_workers=8 \
    psrl.ps_manager_ip="${LOCAL_IP:-127.0.0.1}" \
    psrl.rollout_n=${N_RESP_PER_PROMPT} \
    psrl.staleness=0 \
    psrl.staleness_buffer_entries=${TRAIN_BATCH_SIZE} \
    psrl.agentic_rl.batch_agg_mode=request \
    psrl.agentic_rl.trajectory_output.enable=False \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path="${PSRL_LOG_DIR}" \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.total_nnodes=${NNODES} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.env_worker.enable=True \
    psrl.env_worker.placement=dedicated \
    psrl.env_worker.dedicated_node_ips="['${ENV_NODE_IP}']" \
    psrl.env_worker.cpu_slots_per_worker=8 \
    psrl.env_worker.gpu_slots_per_worker=0 \
    psrl.env_worker.sandbox_cpus=8.0 \
    psrl.env_worker.sandbox_memory=32g \
    psrl.env_worker.routing.method=least_loaded \
    psrl.group_post_process.processors="['dynamic_sampling_filter']" \
    psrl.rollout_gateway.trajectory_id_strategy=auto \
    psrl.rollout_coordination.routing_strategy.method=cache_aware \
    psrl.rollout_coordination.routing_strategy.enable_group_sticky=True \
    psrl.rollout_coordination.routing_strategy.enable_trajectory_sticky=False \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.enable=True \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.indicator=request_num \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.threshold=1000 \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_indicator=request_num \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_threshold=1000 \
    psrl.rollout_coordination.partial_rollout.enable=True \
    psrl.colocate_validate_and_train=False \
    reward.launch_reward_fn_async=True \
    reward.active_managers='[dapo]' \
    reward.managers.dapo.reward_fn.0.path="${REWARD_PATH}" \
    reward.managers.dapo.reward_fn.0.name=compute_score \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=False \
    reward.managers.dapo.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.val_before_train=False \
    trainer.total_epochs="${TOTAL_EPOCHS:-30}" \
    trainer.save_freq="${SAVE_FREQ:-10}" \
    trainer.test_freq="${TEST_FREQ:-5}" \
    "$@" 2>&1 | tee "${experiment_name}.log"
