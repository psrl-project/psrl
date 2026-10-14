#!/bin/bash

project_name='psrl_nixl'
experiment_name='PPO-Qwen2.5-3b-gsm8k-mcore-batch-nixl-staleness_0'

source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

HF_MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct
DIST_CKPT_PATH=${PSRL_WORKSPACE}/models/mcore_ckpt/Qwen2.5-3B-Instruct
python ${PSRL_PATH}/scripts/convert_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH

GLOBAL_BATCH_SIZE=512
GEN_TP=2 # TP in the generation side
GEN_PP=1 # PP in the generation side

VAL_TP=2 # TP in the training side for validation
TRAIN_TP=2 # TP in the training side 
TRAIN_PP=1 # PP in the training side 
TRAIN_CP=2 # CP in the training side

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

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=1 \
    psrl.staleness=0 \
    psrl.staleness_buffer_entries=${GLOBAL_BATCH_SIZE} \
    psrl.gen_mode=batch \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_PATH}/examples/precision_test/ppo/megatron_psrl_log/${experiment_name} \
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
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    \
    train_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${GLOBAL_BATCH_SIZE} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_kl_loss=False \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    train_actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    train_actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    train_actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    train_actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP} \
    train_actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    train_actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    \
    critic.optim.lr=1e-5 \
    critic.model.path="$HF_MODEL_PATH" \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    critic.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    critic.megatron.context_parallel_size=${TRAIN_CP} \
    critic.megatron.use_dist_checkpointing=True \
    critic.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    \
    reward.launch_reward_fn_async=True \
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
