#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export RAY_prestart_worker_first_driver=false
export RAY_num_workers_soft_limit=0
export RAY_memory_monitor_refresh_ms=0

PSRL_PATH=${PSRL_PATH:-$(python3 -c "import os, psrl; print(os.path.dirname(os.path.dirname(psrl.__file__)))")}

# Model and data
HF_MODEL_PATH=/jizhicfs/johnnyslin/models/Qwen3-4B
train_files=/jizhicfs/johnnyslin/data/hotpotqa/hotpotqa_train_32k.parquet
test_files=/jizhicfs/johnnyslin/data/hotpotqa/hotpotqa_dev.parquet

if [[ ! -d "${HF_MODEL_PATH}" ]]; then
    echo "ERROR: model directory not found: ${HF_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${train_files}" ]]; then
    echo "ERROR: training parquet not found: ${train_files}" >&2
    exit 1
fi
if [[ ! -f "${test_files}" ]]; then
    echo "ERROR: validation parquet not found: ${test_files}" >&2
    exit 1
fi

# Experiment
project_name=mem_agent
experiment_name=qwen3_4b_mem_agent_grpo
OUTPUT_DIR=${OUTPUT_DIR:-${PSRL_PATH}/outputs/mem_agent}
CKPTS_DIR=${OUTPUT_DIR}/ckpts/${project_name}/${experiment_name}
LOG_DIR=${OUTPUT_DIR}/logs/${project_name}/${experiment_name}/$(date +%Y%m%d_%H%M%S)
PSRL_LOG_DIR=${OUTPUT_DIR}/psrl_logs/${experiment_name}
mkdir -p "${CKPTS_DIR}" "${LOG_DIR}" "${PSRL_LOG_DIR}"

# MemAgent configuration from the VIME Qwen3-4B recipe.
export MEM_CHUNK_TOKENS=${MEM_CHUNK_TOKENS:-2048}
export MEM_MAX_MEMORY=${MEM_MAX_MEMORY:-1024}
export MEM_MAX_FINAL=${MEM_MAX_FINAL:-256}
export MEM_MAX_CHUNKS=${MEM_MAX_CHUNKS:-64}
export MEM_ALLOW_CONTEXT_TRUNCATION=${MEM_ALLOW_CONTEXT_TRUNCATION:-false}
max_turns=$((MEM_MAX_CHUNKS + 1))

agent_loop_config_path=${PSRL_PATH}/examples/mem_agent/config/mem_agent_loop.yaml
reward_path=${PSRL_PATH}/examples/mem_agent/reward.py

# Batch and sequence lengths
train_batch_size=8
rollout_N=8
max_prompt_length=4096
max_response_length=1024
max_model_len=32768
max_tokens_per_gpu=9216
max_num_batched_tokens=32768

# PSRL deployment: one rollout node and one training node.
NNODES=2
NGPUS_PER_NODE=8

GEN_TP=2
GEN_PP=1
GEN_NNODES=1
GEN_NGPUS_PER_NODE=${NGPUS_PER_NODE}
GEN_INSTANCES=$(((GEN_NNODES * GEN_NGPUS_PER_NODE) / (GEN_TP * GEN_PP)))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$((GEN_TP * GEN_PP))

TRAIN_TP=2
TRAIN_PP=1
TRAIN_CP=1
TRAIN_NNODES=1
TRAIN_NGPUS_PER_NODE=${NGPUS_PER_NODE}

VAL_TP=2
VAL_PP=1
VAL_INSTANCES=$(((TRAIN_NNODES * TRAIN_NGPUS_PER_NODE) / (VAL_TP * VAL_PP)))
VAL_NGPUS_PER_NODE_PER_INSTANCE=$((VAL_TP * VAL_PP))

# GRPO and optimizer
actor_lr=1e-6
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.3
total_training_steps=100
save_freq=50
test_freq=5

PYTHONUNBUFFERED=1 python3 -m psrl.trainer.main_ppo \
    --config-path="${PSRL_PATH}/psrl/trainer/config" \
    --config-name=ppo_trainer model_engine=megatron \
    psrl.ps_manager_ip=${LOCAL_IP:-127.0.0.1} \
    psrl.rollout_n=${rollout_N} \
    psrl.staleness=0 \
    psrl.rollout_gateway.trajectory_id_strategy=auto \
    psrl.staleness_buffer_entries=${train_batch_size} \
    psrl.agentic_rl.batch_agg_mode=request \
    psrl.agentic_rl.trajectory_output.enable=False \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_LOG_DIR} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.deployment.total_nnodes=${NNODES} \
    psrl.nixl.server_port=27237 \
    psrl.group_post_process.enable=False \
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
    \
    gen_actor_rollout_ref.rollout.name=vllm \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    gen_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    gen_actor_rollout_ref.rollout.n=${rollout_N} \
    gen_actor_rollout_ref.rollout.temperature=1.0 \
    gen_actor_rollout_ref.rollout.top_p=1.0 \
    gen_actor_rollout_ref.rollout.top_k=-1 \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns} \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=mem_agent \
    gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj \
    \
    train_actor_rollout_ref.nccl_timeout=6000 \
    train_actor_rollout_ref.model.path=${HF_MODEL_PATH} \
    train_actor_rollout_ref.actor.optim.lr=${actor_lr} \
    train_actor_rollout_ref.actor.optim.lr_decay_style=constant \
    train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
    train_actor_rollout_ref.actor.optim.betas='[0.9,0.98]' \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${train_batch_size} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_dynamic_bsz=True \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_tokens_per_gpu} \
    train_actor_rollout_ref.actor.rollout_n=${rollout_N} \
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
    \
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
    \
    reward.launch_reward_fn_async=True \
    reward.active_managers='[dapo]' \
    reward.managers.dapo.reward_fn.0.path=${reward_path} \
    reward.managers.dapo.reward_fn.0.name=compute_score \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=False \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=${max_response_length} \
    reward.managers.dapo.reward_kwargs.max_resp_len=${max_response_length} \
    \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.train_batch_size=${train_batch_size} \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.reward_model_dicts.0.reward_loop_type=dapo \
    data.reward_model_dicts.0.reward_fn=compute_score \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=${total_training_steps} \
    "$@" 2>&1 | tee "${experiment_name}.log"
