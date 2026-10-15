#!/bin/bash

project_name='psrl_gspo'
experiment_name="GSPO-Qwen2.5-3b-gsm8k-fsdp2-batch-nixl-staleness_1"

source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct
GLOBAL_BATCH_SIZE=128

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

adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
offload=false # it's a small model, offloading will just slow-down training
rollout_mode=sync # can be async to speedup large scale xps
gpu_memory_utilization=0.8
reward_manager=dapo
shuffle_dataset=true
first_time_dataset_prep=true # prepare dataset

test_freq=10
save_freq=10
total_epochs=10
total_training_steps=500
val_before_train=false

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1
train_batch_size=512
ppo_mini_batch_size=128 # maintain 4 mini-batches as recommended by the paper, see Sec. 5.1
ppo_micro_batch_size_per_gpu=8 # setup depending on your GPU memory
n_resp_per_prompt=16

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
# dapo reward manager params
enable_overlong_buffer=false # true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# TIS
rollout_is=null
rollout_is_threshold=null

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=false
entropy_checkpointing=true # This enables entropy recomputation specifically for the entropy calculation, lowering memory usage during training.

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.staleness=1 \
    psrl.staleness_buffer_entries=${GLOBAL_BATCH_SIZE} \
    psrl.gen_mode=batch \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_PATH}/examples/precision_test/grpo/fsdp_psrl_log/${experiment_name} \
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
    gen_actor_rollout_ref.rollout.mode=${rollout_mode} \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    gen_actor_rollout_ref.rollout.enable_chunked_prefill=true \
    gen_actor_rollout_ref.rollout.temperature=${temperature} \
    gen_actor_rollout_ref.rollout.top_p=${top_p} \
    gen_actor_rollout_ref.rollout.top_k=${top_k} \
    \
    train_actor_rollout_ref.model.path="$MODEL_PATH" \
    train_actor_rollout_ref.model.use_remove_padding=True \
    train_actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    train_actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    train_actor_rollout_ref.model.enable_gradient_checkpointing=True \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.mode=sync \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    train_actor_rollout_ref.rollout.enable_chunked_prefill=true \
    train_actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    train_actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    train_actor_rollout_ref.rollout.val_kwargs.n=1 \
    train_actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    train_actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    train_actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    train_actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    train_actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    train_actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    train_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.grad_clip=1.0 \
    train_actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    train_actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    train_actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    train_actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    train_actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.shuffle=$shuffle_dataset \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=True \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    reward.reward_manager.name=${reward_manager} \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.val_before_train=${val_before_train} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.log_val_generations=2 2>&1 | tee ${experiment_name}.log
