#!/bin/bash

source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}

MODEL_NAME=Qwen3-30B-A3B

MODEL_PATH=${PSRL_WORKSPACE}/models/${MODEL_NAME}
DIST_CKPT_PATH=${PSRL_WORKSPACE}/models/mcore_ckpt/${MODEL_NAME}
python ${PSRL_PATH}/scripts/converter_hf_to_mcore.py  --hf_model_path $MODEL_PATH --output_path $DIST_CKPT_PATH

GLOBAL_BATCH_SIZE=768

GEN_TP=1 # TP in the generation side
GEN_PP=1 # PP in the generation side
GEN_EP=1 # EP in the generation side
VAL_TP=4 # TP in the training side for validation
VAL_EP=1 # EP in the training side for validation

TP=4  # TP in the training side
PP=4  # PP in the training side
CP=1  # CP in the training side
EP=4  # EP in the training side
ETP=1  # ETP in the training side

NNODES=3
NGPUS_PER_NODE=8

GEN_NNODES=1
GEN_NGPUS_PER_NODE=8
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) )) # Number of generation instances
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} )) # Number of GPUs per node for generation per instance

TRAIN_NNODES=2
TRAIN_NGPUS_PER_NODE=8

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

offload=True
optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.}

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=1 \
    psrl.staleness=2 \
    psrl.staleness_buffer_entries=${GLOBAL_BATCH_SIZE} \
    psrl.gen_mode=stream \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_PATH}/examples/ppo_trainer/megatron/psrl_log \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.log_prob.enable_train_engine_recompute_log_prob=True \
    psrl.log_prob.mode=recompute \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.nixl.server_mode=meta_server \
    psrl.nixl.server_port=23456 \
    \
    gen_actor_rollout_ref.rollout.mode=psrl_async \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    \
    train_actor_rollout_ref.model.path="$MODEL_PATH" \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.expert_parallel_size=${VAL_EP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    train_actor_rollout_ref.rollout.max_num_seqs=1024 \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_kl_loss=False \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    train_actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
    train_actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
    train_actor_rollout_ref.actor.megatron.use_mbridge=False \
    train_actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    train_actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    train_actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP} \
    train_actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP} \
    train_actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
    train_actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP} \
    train_actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
    +train_actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +train_actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +train_actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +train_actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    \
    critic.optim.lr=1e-5 \
    critic.model.path="$MODEL_PATH" \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.megatron.use_mbridge=False \
    critic.megatron.tensor_model_parallel_size=${TP} \
    critic.megatron.pipeline_model_parallel_size=${PP} \
    critic.megatron.context_parallel_size=${CP} \
    critic.megatron.expert_model_parallel_size=${EP} \
    critic.megatron.expert_tensor_parallel_size=${ETP} \
    \
    train_actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    train_actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    train_actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    train_actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    critic.megatron.use_dist_checkpointing=True \
    critic.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${GLOBAL_BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='psrl_megatron_ppo_test' \
    trainer.experiment_name='moe' \
    trainer.total_training_steps=100 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 2>&1 | tee psrl_megatron_ppo_test-moe.log

