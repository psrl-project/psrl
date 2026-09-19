#!/bin/bash
set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

HF_MODEL_PATH=${PSRL_PATH}/models/Qwen3-8B
train_files=${PSRL_PATH}/data/retool_dapo/train.parquet
test_files=${PSRL_PATH}/data/retool_aime2024/train.parquet

OUTPUT_DIR=${PSRL_PATH}/output
mkdir -p "$OUTPUT_DIR"

staleness=${1:-3}
project_name=psrl_example
experiment_name=PPO-Qwen3-8B-megatron-staleness_${staleness}
CKPTS_DIR=${OUTPUT_DIR}/ckpts/"${project_name}"/"${experiment_name}"
LOG_DIR=${OUTPUT_DIR}/logs/$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"
mkdir -p "$CKPTS_DIR"


train_batch_size=64
rollout_N=8

nnodes=1

TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}

GEN_TP=1
GEN_PP=1

VAL_TP=1
VAL_PP=1

GEN_NNODES=1
GEN_NGPUS_PER_NODE=4
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) )) # Number of generation instances
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} )) # Number of GPUs per node for generation per instance

TRAIN_NNODES=1
TRAIN_NGPUS_PER_NODE=4

VAL_INSTANCES=$(( (${TRAIN_NNODES} * ${TRAIN_NGPUS_PER_NODE}) / ( ${VAL_TP} * ${VAL_PP} ) )) # Number of validation instances
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} )) # Number of GPUs per node for validation per instance


# data length config
max_prompt_length=2048
max_response_length=8192

k=${k:-1}
max_token_len_per_gpu=$(( k * (max_prompt_length + max_response_length) ))

# train rollout params
train_temperature=1.2
train_top_p=1.0
train_top_k=-1

# val rollout params
val_temperature=0
val_rollot_n=1
val_top_p=1.0
val_top_k=-1
val_do_sample=False


python3 -m psrl.trainer.main_ppo --config-path=./config \
    --config-name='ppo_trainer' model_engine=megatron \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=${rollout_N} \
    psrl.staleness=${staleness} \
    psrl.staleness_buffer_entries=${train_batch_size} \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${LOG_DIR} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.nixl.server_port=27837 \
    psrl.group_post_process.enable=False \
    psrl.group_post_process.name=dynamic_sampling_filter \
    \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='error' \
    data.return_multi_modal_inputs=False \
    data.shuffle=True \
    \
    train_actor_rollout_ref.nccl_timeout=6000 \
    train_actor_rollout_ref.model.path=$HF_MODEL_PATH \
    train_actor_rollout_ref.model.use_remove_padding=True \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_kl_loss=False \
    train_actor_rollout_ref.actor.kl_loss_coef=0.01 \
    train_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.rollout_n=$rollout_N \
    train_actor_rollout_ref.actor.use_dynamic_bsz=False \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    train_actor_rollout_ref.actor.megatron.param_offload=False \
    train_actor_rollout_ref.actor.megatron.grad_offload=True \
    train_actor_rollout_ref.actor.megatron.optimizer_offload=True \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    train_actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    train_actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    train_actor_rollout_ref.actor.megatron.use_mbridge=True \
    train_actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    train_actor_rollout_ref.actor.megatron.use_remove_padding=True \
    \
    train_actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    train_actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    train_actor_rollout_ref.rollout.val_kwargs.n=${val_rollot_n} \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    \
    gen_actor_rollout_ref.rollout.name=vllm \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    gen_actor_rollout_ref.rollout.temperature=${train_temperature} \
    gen_actor_rollout_ref.rollout.top_p=${train_top_p} \
    gen_actor_rollout_ref.rollout.top_k=${train_top_k} \
    \
    critic.enable=True \
    critic.optim.lr=1e-5 \
    critic.model.path=$HF_MODEL_PATH \
    critic.model.use_remove_padding=True \
    critic.ppo_mini_batch_size=$train_batch_size \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.use_dynamic_bsz=False \
    critic.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    critic.megatron.param_offload=False \
    critic.megatron.grad_offload=True \
    critic.megatron.optimizer_offload=True \
    critic.megatron.tensor_model_parallel_size=$TP \
    critic.megatron.pipeline_model_parallel_size=$PP \
    critic.megatron.context_parallel_size=$CP \
    critic.megatron.expert_model_parallel_size=$EP \
    critic.megatron.expert_tensor_parallel_size=$ETP \
    critic.megatron.use_mbridge=True \
    critic.megatron.vanilla_mbridge=True \
    critic.megatron.use_remove_padding=True \
    \
    reward.launch_reward_fn_async=True \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=False \
    trainer.save_freq=200 \
    trainer.test_freq=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_training_steps=200 \
    trainer.total_epochs=10 $@ 2>&1 | tee "${experiment_name}.log"
