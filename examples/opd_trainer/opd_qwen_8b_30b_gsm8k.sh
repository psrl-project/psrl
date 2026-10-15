#!/usr/bin/env bash
set -xeuo pipefail

############################ Quick Config ############################

# Activate PSRL conda env.
source ./env/env_311.sh

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
PSRL_WORKSPACE=${PSRL_WORKSPACE:-$PSRL_PATH}
HOME=${PSRL_WORKSPACE}

ROLLOUT_NAME="vllm"
DATA_PATH="${PSRL_WORKSPACE}/data"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-models/Qwen2.5-7B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-models/Qwen2.5-32B-Instruct}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen2.5-32B-Instruct}"

USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"
USE_FUSED_KERNELS=False

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

PROJECT_NAME='psrl_opd'

# DAPO-math / AIME lengths (aligned with elastic_stream_fsdp_qwen_30b_a3b.sh)
MAX_PROMPT=$((1024 * 1))
MAX_RESPONSE_LENGTH=$((1024 * 8))
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE_LENGTH))
# Batched-token budget for vLLM. must be > max_model_len when chunked prefill is on.
MAX_NUM_TOKENS=$((1024 * 10))
TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=${MAX_NUM_TOKENS}
USE_DYNAMIC_BSZ=True

# DAPO reward overlong buffer
DAPO_MAX_RESP_LEN=5120
DAPO_OVERLONG_BUFFER_LEN=$((1024 * 4))
DAPO_OVERLONG_PENALTY_FACTOR=1.0

STUDENT_WORLD_SIZE=2
TEACHER_WORLD_SIZE=4

# Teacher RM (OPD) vLLM parallel . gen_worker requires world_size == TP*PP (dp=1).
# EP is passed to vLLM via expert_parallel_size, not counted in world_size.
TEACHER_TP=2
TEACHER_EP=1
TEACHER_PP=1
TEACHER_DP=1
TEACHER_GPUS_PER_INSTANCE=$(( TEACHER_TP * TEACHER_PP ))
TEACHER_NUM_REPLICAS=$(( TEACHER_WORLD_SIZE / TEACHER_GPUS_PER_INSTANCE ))
TEACHER_ROLLOUT_NGPUS_PER_NODE=${TEACHER_GPUS_PER_INSTANCE}

launch_reward_fn_async=True

NNODES=1
NGPUS_PER_NODE=8
VAL_TP=1
VAL_PP=1
VAL_INSTANCES=$(( STUDENT_WORLD_SIZE / (VAL_TP * VAL_PP) ))
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( VAL_TP * VAL_PP ))

ENFORCE_EAGER=True # true for faster debugging

EXP_NAME="psrl_opd_qwen_dapo_aime"

############################ Paths ############################

DAPO_TRAIN="${DATA_PATH}/dapo/dapo-math-17k.parquet"
DAPO_VAL="${DATA_PATH}/dapo/aime-2024.parquet"
LOG_DIR="${PSRL_PATH}/examples/opd_trainer/logs"

mkdir -p "${LOG_DIR}"

############################ Reward model (OPD teacher) ############################

_DS_DAPO='reward_fn_key:data_source,reward_loop_type:dapo,reward_fn:default,reward_model_name:null,reward_coef:1.0'
_DS_OPD="reward_fn_key:data_source,reward_loop_type:opd,reward_fn:default,reward_model_name:${TEACHER_MODEL_NAME},reward_coef:1.0"
_ROW_TRAIN_DAPO="{file:${DAPO_TRAIN},data_source_name:dapo/dapo-math-17k,prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}},{${_DS_OPD}}]}"
_ROW_VAL_AIME="{file:${DAPO_VAL},prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}}]}"
TRAIN_DATAS="[${_ROW_TRAIN_DAPO}]"
VAL_DATAS="[${_ROW_VAL_AIME}]"

# Use the named DAPO manager and repurpose the generative manager as the OPD teacher.
REWARD_MODELS=(
    reward.active_managers='[dapo,gen_qwen3_8b]'
    reward.launch_reward_fn_async=${launch_reward_fn_async}
    reward.managers.dapo.reward_kwargs.max_resp_len=${DAPO_MAX_RESP_LEN}
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=True
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=${DAPO_OVERLONG_BUFFER_LEN}
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.penalty_factor=${DAPO_OVERLONG_PENALTY_FACTOR}
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.log=False
    reward.managers.gen_qwen3_8b.reward_model_name=${TEACHER_MODEL_NAME}
    reward.managers.gen_qwen3_8b.reward_loop_type=opd
    reward.managers.gen_qwen3_8b.reward_manager.name=opd
    reward.managers.gen_qwen3_8b.enable_resource_pool=True
    reward.managers.gen_qwen3_8b.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    reward.managers.gen_qwen3_8b.nnodes=1
    reward.managers.gen_qwen3_8b.num_replicas=${TEACHER_NUM_REPLICAS}
    reward.managers.gen_qwen3_8b.rollout_ngpus_per_instance_per_node=${TEACHER_ROLLOUT_NGPUS_PER_NODE}
    reward.managers.gen_qwen3_8b.rollout_nnodes_per_instance=1
    reward.managers.gen_qwen3_8b.max_concurrent_requests_per_instance=128
    +reward.managers.gen_qwen3_8b.reward_kwargs.topk=64
    +reward.managers.gen_qwen3_8b.reward_kwargs.teacher_key=default
    reward.managers.gen_qwen3_8b.model.path="${TEACHER_MODEL_PATH}"
    reward.managers.gen_qwen3_8b.model.trust_remote_code=False
    reward.managers.gen_qwen3_8b.rollout._target_=psrl.workers.config.RolloutConfig
    reward.managers.gen_qwen3_8b.rollout.name=vllm
    reward.managers.gen_qwen3_8b.rollout.mode=psrl_async
    reward.managers.gen_qwen3_8b.rollout.disable_attn=False
    reward.managers.gen_qwen3_8b.rollout.dtype=bfloat16
    reward.managers.gen_qwen3_8b.rollout.gpu_memory_utilization=0.8
    reward.managers.gen_qwen3_8b.rollout.enforce_eager=${ENFORCE_EAGER}
    reward.managers.gen_qwen3_8b.rollout.free_cache_engine=true
    reward.managers.gen_qwen3_8b.rollout.data_parallel_size=${TEACHER_DP}
    reward.managers.gen_qwen3_8b.rollout.expert_parallel_size=${TEACHER_EP}
    reward.managers.gen_qwen3_8b.rollout.tensor_model_parallel_size=${TEACHER_TP}
    reward.managers.gen_qwen3_8b.rollout.pipeline_model_parallel_size=${TEACHER_PP}
    reward.managers.gen_qwen3_8b.rollout.max_num_batched_tokens=${MAX_NUM_TOKENS}
    reward.managers.gen_qwen3_8b.rollout.max_num_seqs=1024
    reward.managers.gen_qwen3_8b.rollout.enable_chunked_prefill=false
    reward.managers.gen_qwen3_8b.rollout.enable_prefix_caching=false
    reward.managers.gen_qwen3_8b.rollout.disable_log_stats=false
    reward.managers.gen_qwen3_8b.rollout.skip_tokenizer_init=false
    reward.managers.gen_qwen3_8b.rollout.prompt_length=${MAX_PROMPT}
    reward.managers.gen_qwen3_8b.rollout.response_length=${MAX_RESPONSE_LENGTH}
    reward.managers.gen_qwen3_8b.rollout.max_model_len=${MAX_MODEL_LEN}
    reward.managers.gen_qwen3_8b.rollout.runner=generate
    reward.managers.gen_qwen3_8b.rollout.task=generate
    reward.managers.gen_qwen3_8b.sampling_config.temperature=1.0
    reward.managers.gen_qwen3_8b.sampling_config.top_p=-1
    reward.managers.gen_qwen3_8b.sampling_config.top_k=1
)

############################ Launch ############################

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path="${PSRL_PATH}/psrl/trainer/config" --config-name='ppo_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.reward_service_ip=${LOCAL_IP} \
    psrl.ps_mode=nixl_cpu \
    psrl.rollout_n=1 \
    psrl.staleness=0 \
    psrl.staleness_buffer_entries=${TRAIN_PROMPT_BSZ} \
    psrl.logging_path=${LOG_DIR}/${EXP_NAME} \
    psrl.log_prob.enable_rollout_engine_log_prob=False \
    psrl.deployment.n_rollout_instances=${STUDENT_WORLD_SIZE} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=1 \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${NNODES} \
    psrl.deployment.train_ngpus_per_node=${STUDENT_WORLD_SIZE} \
    psrl.deployment.total_nnodes=${NNODES} \
    \
    "${REWARD_MODELS[@]}" \
    \
    data.train_datas="${TRAIN_DATAS}" \
    data.val_datas="${VAL_DATAS}" \
    data.train_datasets_ratios=[1.0] \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_PROMPT_BSZ} \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.shuffle=False \
    \
    train_actor_rollout_ref.model.enable_gradient_checkpointing=True \
    train_actor_rollout_ref.model.use_remove_padding=True \
    train_actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS} \
    train_actor_rollout_ref.actor.use_torch_compile=True \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_BSZ} \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${STUDENT_MICRO_BATCH_SIZE_PER_GPU} \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${STUDENT_MAX_TOKEN_LEN_PER_GPU} \
    train_actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    train_actor_rollout_ref.actor.fsdp_config.param_offload=False \
    train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    train_actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    \
    train_actor_rollout_ref.model.path="${STUDENT_MODEL_PATH}" \
    gen_actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    gen_actor_rollout_ref.rollout.calculate_log_probs=False \
    gen_actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_TOKENS} \
    gen_actor_rollout_ref.rollout.max_num_seqs=1024 \
    gen_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    gen_actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    \
    train_actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_TOKENS} \
    train_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${STUDENT_MICRO_BATCH_SIZE_PER_GPU} \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${STUDENT_MAX_TOKEN_LEN_PER_GPU} \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    \
    distillation.enabled=True \
    distillation.teacher_key=default \
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE} \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT} \
    distillation.distillation_loss.loss_max_clamp=${DISTILLATION_LOSS_MAX_CLAMP} \
    distillation.distillation_loss.log_prob_min_clamp=${DISTILLATION_LOG_PROB_MIN_CLAMP} \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${STUDENT_WORLD_SIZE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=200 \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.log_val_generations=5 \
    "$@" 2>&1 | tee "${LOG_DIR}/${EXP_NAME}_${DISTILLATION_LOSS_MODE}_${USE_POLICY_GRADIENT}.log"
