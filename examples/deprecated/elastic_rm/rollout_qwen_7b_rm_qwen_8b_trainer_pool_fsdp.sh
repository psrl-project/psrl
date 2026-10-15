#!/usr/bin/env bash
# Elastic RM with trainer-pool reuse: shared_rollout_pool + train_pool elastic replicas.
# Based on rollout_qwen_7b_rm_qwen_8b_fsdp.sh with enable_trainer_pool enabled.
set -xeuo pipefail

PSRL_WORKSPACE=/jizhicfs/pkuhetu/yfzhao/psrl

staleness=${1:-2}
fix_weight=${2:-False}
disable_attn=${3:-False}
min_awake_per_role=${4:-0}
smoke_test=${5:-0}

if [[ "${smoke_test}" == "1" ]]; then
    experiment_suffix="_smoke"
    total_training_steps=2
    test_freq=9999
    save_freq=9999
    train_prompt_bsz=32
    train_prompt_mini_bsz=8
else
    experiment_suffix=""
    total_training_steps=200
    test_freq=200
    save_freq=200
    train_prompt_bsz=128
    train_prompt_mini_bsz=32
fi

project_name='psrl_elastic_rm'
experiment_name=debug_router_share_8_train_8_rm_qwen_8b_rollout_qwen_7b_staleness_${staleness}${experiment_suffix}

source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
MODEL_NAME='Qwen2.5-7B'
HF_MODEL_PATH=${PSRL_WORKSPACE}/models/${MODEL_NAME}
RM_MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen3-8B

# Dataset specs follow `data.train_datas` / `data.val_datas`.
GSM8K_TRAIN="${PSRL_WORKSPACE}/data/gsm8k_verl/train.parquet"
GSM8K_TEST="${PSRL_WORKSPACE}/data/gsm8k_verl/test.parquet"
DAPO_TRAIN="${PSRL_WORKSPACE}/data/dapo/dapo-math-17k.parquet"
DAPO_VAL="${PSRL_WORKSPACE}/data/dapo/aime-2024.parquet"

# One reward-dict in OmegaConf inline form: { reward_fn_key, reward_loop_type, ... }
_DS_NAIVE='reward_fn_key:data_source,reward_loop_type:naive,reward_fn:default,reward_model_name:null,reward_coef:1.0'
_DS_DAPO='reward_fn_key:data_source,reward_loop_type:dapo,reward_fn:default,reward_model_name:null,reward_coef:1.0'
_DS_GEN='reward_fn_key:data_source,reward_loop_type:gen,reward_fn:default,reward_model_name:Qwen3-8B,reward_coef:1.0'

_ROW_TRAIN_GSM8K="{file:${GSM8K_TRAIN},data_source_name:openai/gsm8k,prompt_key:prompt,reward_model_dicts:[{${_DS_NAIVE}},{${_DS_GEN}}]}"
_ROW_TRAIN_DAPO="{file:${DAPO_TRAIN},data_source_name:dapo/dapo-math-17k,prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}},{${_DS_GEN}}]}"
TRAIN_DATAS="[${_ROW_TRAIN_GSM8K},${_ROW_TRAIN_DAPO}]"

_ROW_VAL_GSM8K="{file:${GSM8K_TEST},prompt_key:prompt,reward_model_dicts:[{${_DS_NAIVE}}]}"
_ROW_VAL_DAPO="{file:${DAPO_VAL},prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}}]}"
VAL_DATAS="[${_ROW_VAL_GSM8K},${_ROW_VAL_DAPO}]"

TRAIN_DATASETS_RATIOS='[0.5,0.5]'

NNODES=2
NGPUS_PER_NODE=8

# shared_rollout_pool: rollout + RM elastic replicas on the inference side
SHARED_NNODES=1
SHARED_NGPUS_PER_NODE=8

# train_pool: actor training + trainer-pool rollout/RM replicas (sleep/wake with trainer)
TRAIN_NNODES=1
TRAIN_NGPUS_PER_NODE=8

# rollout settings
GEN_TP=1
GEN_PP=1
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} ))

# validation settings (separate validate pools; does not use trainer-pool elastic path)
VAL_TP=4
VAL_PP=1
VAL_INSTANCES=$(( (${TRAIN_NNODES} * ${TRAIN_NGPUS_PER_NODE}) / ( ${VAL_TP} * ${VAL_PP} ) ))
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} ))

sp_size=1
fsdp_size=8
use_dynamic_bsz=True

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 10))
packing_length=$(((max_prompt_length + max_response_length) * 2))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 8))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
n_resp_per_prompt=8

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
filter_groups_metric=acc

rollout_is="token"
rollout_is_threshold=2.0

enable_elastic=True
enable_trainer_pool=False
scaling_policy_variant=throughput-optimal

offload=True

REWARD_MODELS=(
    reward.active_managers='[naive,dapo,gen_qwen3_8b]'
    reward.launch_reward_fn_async=False
    reward.reward_normalization=batch
    reward.managers.naive.reward_loop_type=naive
    reward.managers.dapo.reward_loop_type=dapo
    reward.managers.dapo.reward_kwargs.max_resp_len=5120
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=True
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=3072
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.log=False
    reward.managers.gen_qwen3_8b.reward_loop_type=gen
    reward.managers.gen_qwen3_8b.reward_model_name=Qwen3-8B
    reward.managers.gen_qwen3_8b.enable_resource_pool=True
    reward.managers.gen_qwen3_8b.rollout_ngpus_per_instance_per_node=1
    reward.managers.gen_qwen3_8b.rollout_nnodes_per_instance=1
    reward.managers.gen_qwen3_8b.max_concurrent_requests_per_instance=128
    reward.managers.gen_qwen3_8b.model.path=${RM_MODEL_PATH}
    reward.managers.gen_qwen3_8b.model.trust_remote_code=False
    reward.managers.gen_qwen3_8b.rollout._target_=psrl.workers.config.RolloutConfig
    reward.managers.gen_qwen3_8b.rollout.name=vllm
    reward.managers.gen_qwen3_8b.rollout.mode=psrl_async
    reward.managers.gen_qwen3_8b.rollout.disable_attn=${disable_attn}
    reward.managers.gen_qwen3_8b.rollout.dtype=bfloat16
    reward.managers.gen_qwen3_8b.rollout.gpu_memory_utilization=0.8
    reward.managers.gen_qwen3_8b.rollout.enforce_eager=true
    reward.managers.gen_qwen3_8b.rollout.free_cache_engine=true
    reward.managers.gen_qwen3_8b.rollout.data_parallel_size=1
    reward.managers.gen_qwen3_8b.rollout.expert_parallel_size=1
    reward.managers.gen_qwen3_8b.rollout.tensor_model_parallel_size=1
    reward.managers.gen_qwen3_8b.rollout.pipeline_model_parallel_size=1
    reward.managers.gen_qwen3_8b.rollout.max_num_batched_tokens=$((1024 * 11 + 1024 * 10))
    reward.managers.gen_qwen3_8b.rollout.max_num_seqs=1024
    reward.managers.gen_qwen3_8b.rollout.enable_chunked_prefill=false
    reward.managers.gen_qwen3_8b.rollout.enable_prefix_caching=false
    reward.managers.gen_qwen3_8b.rollout.disable_log_stats=false
    reward.managers.gen_qwen3_8b.rollout.skip_tokenizer_init=false
    reward.managers.gen_qwen3_8b.rollout.prompt_length=$((1024 * 11))
    reward.managers.gen_qwen3_8b.rollout.response_length=$((1024 * 10))
    reward.managers.gen_qwen3_8b.rollout.max_model_len=$((1024 * 11 + 1024 * 10))
    reward.managers.gen_qwen3_8b.rollout.runner=generate
    reward.managers.gen_qwen3_8b.rollout.task=generate
    reward.managers.gen_qwen3_8b.sampling_config.temperature=1.0
    reward.managers.gen_qwen3_8b.sampling_config.top_p=-1
    reward.managers.gen_qwen3_8b.sampling_config.top_k=1
)

mkdir -p "${PSRL_WORKSPACE}/logs/${project_name}"

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.reward_service_ip=${LOCAL_IP} \
    psrl.rollout_n=${n_resp_per_prompt} \
    psrl.staleness=${staleness} \
    psrl.staleness_buffer_entries=${train_prompt_bsz} \
    psrl.ps_mode=nixl_cpu \
    psrl.profile.disable_attn=${disable_attn} \
    psrl.profile.fix_weight=${fix_weight} \
    psrl.logging_path=${PSRL_PATH}/logs/${project_name}/${experiment_name} \
    \
    psrl.deployment.elastic.enable=${enable_elastic} \
    psrl.deployment.elastic.enable_trainer_pool=${enable_trainer_pool} \
    psrl.deployment.elastic.shared_nnodes=${SHARED_NNODES} \
    psrl.deployment.elastic.shared_ngpus_per_node=${SHARED_NGPUS_PER_NODE} \
    psrl.deployment.elastic.min_awake_per_role=${min_awake_per_role} \
    psrl.deployment.elastic.scaling_policy_variant=${scaling_policy_variant} \
    \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.deployment.total_nnodes=${NNODES} \
    \
    psrl.colocate_validate_and_train=True \
    psrl.fuse_rollout_with_validate=False \
    \
    psrl.nixl.server_port=23456 \
    psrl.group_post_process.enable=False \
    psrl.group_post_process.name=dynamic_sampling_filter \
    \
    psrl.rollout_coordination.redundant_rollout.enable=False \
    \
    psrl.rollout_coordination.partial_rollout.enable=True \
    \
    psrl.rollout_coordination.routing_strategy.method="throughput_optimal" \
    psrl.rollout_coordination.routing_strategy.candidate_sort_indicator=reserve_capability \
    psrl.rollout_coordination.routing_strategy.enable_multi_priority_queue=True \
    psrl.rollout_coordination.routing_strategy.cost_model_path=${PSRL_PATH}/psrl/trainer/config/cost_model/qwen_7b.json \
    psrl.rollout_coordination.routing_strategy.delta_throughput_threshold=0.2 \
    psrl.rollout_coordination.routing_strategy.request_budget=1024 \
    psrl.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
    psrl.rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance=512 \
    \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
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
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
    "${REWARD_MODELS[@]}" \
    \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.train_datas="${TRAIN_DATAS}" \
    data.val_datas="${VAL_DATAS}" \
    data.train_datasets_ratios="${TRAIN_DATASETS_RATIOS}" \
    \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    psrl.rollout_coordination.proactive_filter_strategy.method="retry" \
    psrl.rollout_coordination.proactive_filter_strategy.threshold=4 \
    \
    trainer.logger='["console", "wandb"]' \
    +trainer.wandb_proxy=http://star-proxy.oa.com:3128 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${total_training_steps} \
    2>&1 | tee ${PSRL_WORKSPACE}/logs/${experiment_name}_${LOCAL_IP}.log
