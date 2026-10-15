#!/usr/bin/env bash
set -xeuo pipefail

staleness=${1:-2}
project_name=psrl_example
experiment_name=DAPO-Qwen2.5-7B-AIME-fsdp2-retool-staleness_${staleness}

source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
MODEL_PATH=${PSRL_WORKSPACE}/models/multiturn-sft-qwen-2.5-7b-instruct
TRAIN_FILE=${PSRL_WORKSPACE}/data/dapo/dapo-math-17k.parquet
# aime_2024=${PSRL_WORKSPACE}/data/dapo/aime-2024-raw.parquet
aime_2025=${PSRL_WORKSPACE}/data/dapo/aime-2025.parquet

train_files="['$TRAIN_FILE']"
# test_files="['$aime_2025', '$aime_2024']"
test_files="['$aime_2025']"

CKPT_ROOT=${CKPT_ROOT:-$PWD}
default_local_dir=$CKPT_ROOT/checkpoint/$experiment_name

tool_config_path=${PSRL_PATH}/examples/retool/sandbox_fusion_tool_config.yaml

GEN_DP=1 # DP in the generation side
GEN_TP=1 # TP in the generation side
GEN_PP=1 # PP in the generation side

VAL_DP=1 # DP in the training side for validation
VAL_TP=1 # TP in the training side for validation
VAL_PP=1 # PP in the training side for validation

TRAIN_SP=2 # SP in the training side
TRAIN_FSDP=8 # FSDP in the training side

NNODES=4
NGPUS_PER_NODE=8

GEN_NNODES=2 # Number of nodes for generation
GEN_NGPUS_PER_NODE=${NGPUS_PER_NODE} # Number of GPUs per node for generation
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) )) # Number of generation instances
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} )) # Number of GPUs per node for generation per instance

TRAIN_NNODES=2 # Number of nodes for training
TRAIN_NGPUS_PER_NODE=${NGPUS_PER_NODE}

VAL_INSTANCES=$(( (${TRAIN_NNODES} * ${TRAIN_NGPUS_PER_NODE}) / ( ${VAL_TP} * ${VAL_PP} ) )) # Number of validation instances
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} )) # Number of GPUs per node for validation per instance

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
max_turns=64
max_prompt_length=2048
max_response_length=16384
actor_lr=1e-6
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 10))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
train_prompt_bsz=64
n_resp_per_prompt=8
n_resp_per_prompt_val=8
train_prompt_mini_bsz=64

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1.0

# TIS
rollout_is=token
rollout_is_threshold=2.0

# Performance Related Parameter
use_dynamic_bsz=True
packing_length=$(( (max_prompt_length + max_response_length) * 1 ))
# NOTE(lhy): Actor parameters cannot be offloaded in `nixl_cpu` mode.
offload=False

# gen_actor_rollout_ref.rollout.trace.backend=weave
# gen_actor_rollout_ref.rollout.trace.token2text=True

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path="${PSRL_PATH}/psrl/trainer/config" --config-name='ppo_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=${n_resp_per_prompt} \
    psrl.staleness=${staleness} \
    psrl.staleness_buffer_entries=${train_prompt_bsz} \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_PATH}/examples/retool/fsdp_psrl_log/${experiment_name} \
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
    \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    gen_actor_rollout_ref.rollout.data_parallel_size=${GEN_DP} \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    gen_actor_rollout_ref.rollout.temperature=${temperature} \
    gen_actor_rollout_ref.rollout.top_p=${top_p} \
    gen_actor_rollout_ref.rollout.top_k=${top_k} \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=$max_turns \
    gen_actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    gen_actor_rollout_ref.rollout.multi_turn.format=hermes \
    gen_actor_rollout_ref.rollout.agent.env.name=tool_env \
    gen_actor_rollout_ref.rollout.agent.data.name=tool_agent_data \
    \
    train_actor_rollout_ref.model.path="$MODEL_PATH" \
    train_actor_rollout_ref.model.use_remove_padding=True \
    +train_actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    train_actor_rollout_ref.model.enable_gradient_checkpointing=True \
    train_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${packing_length} \
    train_actor_rollout_ref.rollout.data_parallel_size=${VAL_DP} \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    train_actor_rollout_ref.rollout.temperature=${temperature} \
    train_actor_rollout_ref.rollout.top_p=${top_p} \
    train_actor_rollout_ref.rollout.top_k=${top_k} \
    train_actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    train_actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    train_actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    train_actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    train_actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    train_actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    train_actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    train_actor_rollout_ref.actor.clip_ratio_c=10.0 \
    train_actor_rollout_ref.actor.optim.lr=$actor_lr \
    train_actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
    train_actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${packing_length} \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    train_actor_rollout_ref.actor.strategy=fsdp2 \
    train_actor_rollout_ref.actor.fsdp_config.param_offload=False \
    train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    train_actor_rollout_ref.actor.ulysses_sequence_parallel_size=${TRAIN_SP} \
    train_actor_rollout_ref.actor.fsdp_config.fsdp_size=${TRAIN_FSDP} \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.grad_clip=1.0 \
    train_actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    \
    reward.active_managers='[naive]' \
    reward.managers.naive.reward_fn.0.path=${PSRL_PATH}/examples/retool/retool.py \
    reward.managers.naive.reward_fn.0.name=compute_score \
    \
    +data.train_files="$train_files" \
    data.reward_model_dicts.0.reward_loop_type=naive \
    data.reward_model_dicts.0.reward_fn=compute_score \
    +data.val_files="$test_files" \
    +data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    +data.custom_cls.path=${PSRL_PATH}/examples/retool/retool.py \
    +data.custom_cls.name=CustomRLHFDataset \
    \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.test_freq=5 \
    trainer.save_freq=1000 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 2>&1 | tee ${experiment_name}.log
