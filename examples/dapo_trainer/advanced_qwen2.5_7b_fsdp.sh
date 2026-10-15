#!/bin/bash
set -xeuo pipefail

staleness=${1:-2}
project_name=psrl_example
experiment_name=DAPO-Qwen2.5-7B-fsdp-staleness_${staleness}
fix_weight=${2:-False}
disable_attn=${3:-False}
source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
HF_MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen2.5-Math-7B
TRAIN_FILE=${PSRL_WORKSPACE}/data/gsm8k/train.parquet
TEST_FILE=${PSRL_WORKSPACE}/data/gsm8k/test.parquet

GEN_DP=1 # DP in the generation side
GEN_TP=1 # TP in the generation side
GEN_PP=1 # PP in the generation side

VAL_DP=1 # DP in the training side for validation
VAL_TP=1 # TP in the training side for validation
VAL_PP=1 # PP in the training side for validation

NNODES=1
NGPUS_PER_NODE=4

GEN_NNODES=1 # Number of nodes for generation
GEN_NGPUS_PER_NODE=${NGPUS_PER_NODE} # Number of GPUs per node for generation
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) )) # Number of generation instances
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} )) # Number of GPUs per node for generation per instance

TRAIN_NNODES=1 # Number of nodes for training
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
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 10))
packing_length=$((1024 * 24))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
train_prompt_bsz=64
redundant_train_prompt_bsz=64
n_resp_per_prompt=8
redundant_n_resp_per_prompt=8
train_prompt_mini_bsz=16

# Algorithm
temperature=1
top_p=1
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
filter_groups_metric=acc

# NOTE(lhy): Actor parameters cannot be offloaded in `nixl_cpu` mode.
offload=True
use_dynamic_bsz=True

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path="${PSRL_PATH}/psrl/trainer/config" --config-name='ppo_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.reward_service_ip=${LOCAL_IP} \
    psrl.rollout_n=${n_resp_per_prompt} \
    psrl.staleness=${staleness} \
    psrl.staleness_buffer_entries=${train_prompt_bsz} \
    psrl.ps_mode=nixl_cpu \
    psrl.profile.disable_attn=${disable_attn} \
    psrl.profile.fix_weight=${fix_weight} \
    psrl.logging_path=${PSRL_PATH}/examples/dapo_trainer/logs/${experiment_name} \
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
    psrl.group_post_process.enable=False \
    psrl.group_post_process.name=dynamic_sampling_filter \
    \
    psrl.rollout_coordination.redundant_rollout.enable=True \
    psrl.rollout_coordination.redundant_rollout.redundant_global_batch_size=${redundant_train_prompt_bsz} \
    psrl.rollout_coordination.redundant_rollout.redundant_rollout_n=${redundant_n_resp_per_prompt} \
    \
    psrl.rollout_coordination.partial_rollout.enable=True \
    \
    psrl.rollout_coordination.routing_strategy.method="throughput_optimal" \
    psrl.rollout_coordination.routing_strategy.candidate_sort_indicator=reserve_capability \
    psrl.rollout_coordination.routing_strategy.enable_multi_priority_queue=True \
    psrl.rollout_coordination.routing_strategy.enable_group_sticky=False \
    psrl.rollout_coordination.routing_strategy.cost_model_path=${PSRL_PATH}/psrl/trainer/config/cost_model/qwen_7b.json \
    psrl.rollout_coordination.routing_strategy.delta_throughput_threshold=0.2 \
    psrl.rollout_coordination.routing_strategy.request_budget=1024 \
    psrl.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
    psrl.rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance=512 \
    \
    psrl.rollout_coordination.sync_and_mig_strategy.method="status_based" \
    psrl.rollout_coordination.sync_and_mig_strategy.sync.indicator="kv_cache" \
    psrl.rollout_coordination.sync_and_mig_strategy.sync.threshold=0.6 \
    psrl.rollout_coordination.sync_and_mig_strategy.sync.check_req_before_sync=False \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.enable=True \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.indicator="throughput" \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.threshold=5 \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_indicator="request_num" \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.stop_threshold=10 \
    \
    psrl.rollout_coordination.proactive_filter_strategy.method="retry" \
    psrl.rollout_coordination.proactive_filter_strategy.threshold=4 \
    \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    gen_actor_rollout_ref.rollout.data_parallel_size=${GEN_DP} \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=${packing_length} \
    gen_actor_rollout_ref.rollout.temperature=${temperature} \
    gen_actor_rollout_ref.rollout.top_p=${top_p} \
    gen_actor_rollout_ref.rollout.top_k=${top_k} \
    gen_actor_rollout_ref.rollout.disable_log_stats=false \
    \
    train_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    train_actor_rollout_ref.model.use_remove_padding=True \
    +train_actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    train_actor_rollout_ref.model.enable_gradient_checkpointing=True \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=${packing_length} \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${packing_length} \
    train_actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    train_actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    train_actor_rollout_ref.rollout.val_kwargs.n=1 \
    train_actor_rollout_ref.rollout.data_parallel_size=${VAL_DP} \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    train_actor_rollout_ref.actor.strategy=fsdp2 \
    train_actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    train_actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    train_actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    train_actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    train_actor_rollout_ref.actor.clip_ratio_c=10.0 \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
    train_actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${packing_length} \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    train_actor_rollout_ref.actor.fsdp_config.param_offload=False \
    train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.grad_clip=1.0 \
    train_actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    reward.active_managers='[dapo]' \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.log=False \
    reward.managers.dapo.reward_kwargs.max_resp_len=${max_response_length} \
    \
    +data.train_files="${TRAIN_FILE}" \
    data.reward_model_dicts.0.reward_loop_type=dapo \
    data.reward_model_dicts.0.reward_fn=compute_score \
    +data.val_files="${TEST_FILE}" \
    +data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=200 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 2>&1 | tee ${experiment_name}.log
