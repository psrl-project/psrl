#!/bin/bash

project_name='psrl_nixl'
experiment_name='PPO-Qwen2.5-3b-gsm8k-fsdp2-batch-nixl-staleness_1'

source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct
GLOBAL_BATCH_SIZE=512

GEN_TP=2 # TP in the generation side
GEN_PP=1 # PP in the generation side
VAL_TP=2 # TP in the training side for validation

NNODES=2
NGPUS_PER_NODE=8

GEN_NNODES=${NNODES} # Number of nodes for generation
GEN_NGPUS_PER_NODE=4 # Number of GPUs per node for generation
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) )) # Number of generation instances
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} )) # Number of GPUs per node for generation per instance

TRAIN_NNODES=${NNODES} # Number of nodes for training
TRAIN_NGPUS_PER_NODE=$(( ${NGPUS_PER_NODE} - ${GEN_NGPUS_PER_NODE} )) # Number of GPUs per node for training

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

# TIS
rollout_is=null
rollout_is_threshold=null

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=1 \
    psrl.staleness=1 \
    psrl.staleness_buffer_entries=${GLOBAL_BATCH_SIZE} \
    psrl.gen_mode=batch \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_PATH}/examples/precision_test/ppo/fsdp_psrl_log/${experiment_name} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.nixl.server_mode=meta_server \
    psrl.nixl.server_port=23456 \
    psrl.validate_on_psrl=False \
    psrl.tms.range=null \
    psrl.tms.enable_nixl=False \
    \
    gen_actor_rollout_ref.rollout.mode=sync \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    \
    train_actor_rollout_ref.model.path="$MODEL_PATH" \
    train_actor_rollout_ref.model.use_remove_padding=True \
    train_actor_rollout_ref.model.enable_gradient_checkpointing=True \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${GLOBAL_BATCH_SIZE} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.fsdp_config.param_offload=False \
    train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    train_actor_rollout_ref.actor.use_kl_loss=False \
    \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="$MODEL_PATH" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${GLOBAL_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.total_training_steps=500 \
    trainer.save_freq=500 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 2>&1 | tee ${experiment_name}.log