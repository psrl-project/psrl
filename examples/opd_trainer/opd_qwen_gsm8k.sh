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

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-models/Qwen/Qwen2.5-0.5B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-models/Qwen/Qwen2.5-3B-Instruct}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen2.5-3B-Instruct}"

# USE_POLICY_GRADIENT=False
# DISTILLATION_LOSS_MODE="k3"
# DISTILLATION_LOSS_MODE="forward_kl_topk"
# USE_FUSED_KERNELS=False
USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"
USE_FUSED_KERNELS=False

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

PROJECT_NAME='psrl_opd'

MAX_PROMPT=256
MAX_RESPONSE_LENGTH=512
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=2
TEACHER_WORLD_SIZE=4

NNODES=1
NGPUS_PER_NODE=8
VAL_TP=1
VAL_PP=1
VAL_INSTANCES=$(( STUDENT_WORLD_SIZE / (VAL_TP * VAL_PP) ))
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( VAL_TP * VAL_PP ))

ENFORCE_EAGER=True # true for faster debugging

EXP_NAME="psrl_opd_qwen_gsm8k"

############################ Paths ############################

gsm8k_train_path="${DATA_PATH}/gsm8k_verl/train.parquet"
gsm8k_test_path="${DATA_PATH}/gsm8k_verl/test.parquet"
LOG_DIR="${PSRL_PATH}/examples/opd_trainer/logs"

mkdir -p "${LOG_DIR}"

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
    data.train_datas="[{file:${gsm8k_train_path},data_source_name:openai/gsm8k,prompt_key:prompt,reward_model_dicts:[{reward_fn_key:data_source,reward_loop_type:naive,reward_fn:default,reward_model_name:null,reward_coef:1.0},{reward_fn_key:data_source,reward_loop_type:opd,reward_fn:default,reward_model_name:${TEACHER_MODEL_NAME},reward_coef:1.0}]}]" \
    data.val_datas="[{file:${gsm8k_test_path},prompt_key:prompt,reward_model_dicts:[{reward_fn_key:data_source,reward_loop_type:naive,reward_fn:default,reward_model_name:null,reward_coef:1.0}]}]" \
    data.train_datasets_ratios=[1.0] \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_PROMPT_BSZ} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
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
    gen_actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS} \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_TOKENS} \
    gen_actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_TOKENS} \
    gen_actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
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
    "$@" 2>&1 | tee "${LOG_DIR}/psrl_opd_qwen_gsm8k_$(basename "${STUDENT_MODEL_PATH}")_$(basename "${TEACHER_MODEL_PATH}")_${DISTILLATION_LOSS_MODE}_${USE_POLICY_GRADIENT}.log"
