#!/usr/bin/env bash
set -xeuo pipefail

PSRL_WORKSPACE=/jizhicfs/pkuhetu/yfzhao/psrl

project_name='psrl_validation'
experiment_name='math-only-multi-datasets-stream-megatron-qwen-7b'

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
HF_MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen2.5-7B
DIST_CKPT_PATH=${PSRL_WORKSPACE}/models/mcore_ckpt/Qwen2.5-7B
# python ${PSRL_PATH}/scripts/convert_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH

GSM8K_TRAIN=${PSRL_WORKSPACE}/data/gsm8k_verl/train.parquet
GSM8K_TEST=${PSRL_WORKSPACE}/data/gsm8k_verl/test.parquet
DAPO_TRAIN=${PSRL_WORKSPACE}/data/dapo/dapo-math-17k.parquet
DAPO_VAL=${PSRL_WORKSPACE}/data/dapo/aime-2024.parquet
_DS_NAIVE="reward_fn_key:data_source,reward_loop_type:naive,reward_fn:default,reward_model_name:null,reward_coef:1.0"
_DS_DAPO="reward_fn_key:data_source,reward_loop_type:dapo,reward_fn:default,reward_model_name:null,reward_coef:1.0"
_DS_GEN="reward_fn_key:data_source,reward_loop_type:gen,reward_fn:default,reward_model_name:Qwen3-8B,reward_coef:1.0"
_ROW_TRAIN_GSM8K="{file:${GSM8K_TRAIN},data_source_name:openai/gsm8k,prompt_key:prompt,reward_model_dicts:[{${_DS_NAIVE}},{${_DS_GEN}}]}"
_ROW_TRAIN_DAPO="{file:${DAPO_TRAIN},data_source_name:dapo/dapo-math-17k,prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}},{${_DS_GEN}}]}"
_ROW_VAL_GSM8K="{file:${GSM8K_TEST},prompt_key:prompt,reward_model_dicts:[{${_DS_NAIVE}}]}"
_ROW_VAL_DAPO="{file:${DAPO_VAL},prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}}]}"
TRAIN_DATAS="[${_ROW_TRAIN_GSM8K},${_ROW_TRAIN_DAPO}]"
VAL_DATAS="[${_ROW_VAL_GSM8K},${_ROW_VAL_DAPO}]"

# dapo-reward
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# rollout / reward settings
ROLL_TP=1
ROLL_PP=1
ROLL_INSTANCES=3
ROLL_NGPUS_PER_NODE_PER_INSTANCE=$((ROLL_TP * ROLL_PP))

# training settings
TRAIN_TP=2
TRAIN_PP=2
TRAIN_CP=1
TRAIN_NGPUS=4
NNODES=1

VAL_TP=4 # TP in the training side for validation

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
tis_imp_ratio_cap=2.0
clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 5))
loss_agg_mode="token-mean"
train_prompt_bsz=128
n_resp_per_prompt=8
train_prompt_mini_bsz=32

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# TIS
rollout_is=token
rollout_is_threshold=2.0

# NOTE(lhy): parameters of the actor cannot be offloaded when using nixl_cpu mode
# May support this in the future
offload=True

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path="${PSRL_PATH}/psrl/trainer/config" --config-name='ppo_megatron_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=${n_resp_per_prompt} \
    psrl.staleness=2 \
    psrl.staleness_buffer_entries=${train_prompt_bsz} \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_PATH}/logs/multi_datasets_log/${experiment_name} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.n_rollout_instances=${ROLL_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${ROLL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS} \
    psrl.nixl.server_port=23456 \
    \
    gen_actor_rollout_ref.rollout.mode=psrl_async \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLL_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${ROLL_PP} \
    gen_actor_rollout_ref.rollout.enable_chunked_prefill=False \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    gen_actor_rollout_ref.rollout.temperature=${temperature} \
    gen_actor_rollout_ref.rollout.top_p=${top_p} \
    gen_actor_rollout_ref.rollout.top_k=${top_k} \
    \
    train_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    train_actor_rollout_ref.rollout.enable_chunked_prefill=False \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    train_actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    train_actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    train_actor_rollout_ref.rollout.val_kwargs.n=1 \
    train_actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    train_actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    train_actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    train_actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    train_actor_rollout_ref.actor.clip_ratio_c=10.0 \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
    train_actor_rollout_ref.actor.optim.clip_grad=1.0 \
    train_actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    train_actor_rollout_ref.actor.megatron.param_offload=False \
    train_actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    train_actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    train_actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    train_actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    \
    reward.reward_normalization=batch \
    reward.active_managers='[naive,dapo,gen_qwen3_8b]' \
    \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.train_datas="${TRAIN_DATAS}" \
    data.val_datas="${VAL_DATAS}" \
    data.train_datasets_ratios=[0.5,0.5] \
    \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    \
    trainer.logger='["console", "wandb"]' \
    +trainer.wandb_proxy=http://star-proxy.oa.com:3128 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=200 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 2>&1 | tee ${PSRL_WORKSPACE}/logs/${experiment_name}_${LOCAL_IP}.log

    # reward.active_managers='[dapo]' \
    # reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    # reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    # reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    # reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.log=False \
    # reward.managers.dapo.reward_kwargs.max_resp_len=${max_response_length} \

