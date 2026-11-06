#!/usr/bin/env bash
# Distill Qwen3.5-35B-A3B into Qwen3.5-4B on Geo3K with FSDP2.
# Usage: `bash examples/opd_trainer/run_qwen3_5_4b_fsdp.sh [--dry-run] [Hydra overrides]`
set -eo pipefail

PSRL_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${PSRL_PATH}"
source env/psrl.sh
set -u

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

STUDENT_MODEL=${STUDENT_MODEL:-${PSRL_WORKSPACE}/models/Qwen3.5-4B}
TEACHER_MODEL=${TEACHER_MODEL:-${PSRL_WORKSPACE}/models/Qwen3.5-35B-A3B}
TRAIN_FILE=${TRAIN_FILE:-${PSRL_WORKSPACE}/data/geo3k/train.parquet}
VAL_FILE=${VAL_FILE:-${PSRL_WORKSPACE}/data/geo3k/test.parquet}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEACHER_TP=${TEACHER_TP:-4}
TEACHER_EP=${TEACHER_EP:-4}
VAL_TP=${VAL_TP:-2}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
STALENESS=${STALENESS:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

# Student rollout and training use separate GPU pools in PSRL.
TRAIN_WORLD_SIZE=$((NNODES * NGPUS_PER_NODE))
ROLLOUT_WORLD_SIZE=$((NNODES * ROLLOUT_GPUS_PER_NODE))
TEACHER_TOTAL_GPUS=$((NNODES * TEACHER_WORLD_SIZE))
if (( ROLLOUT_TP <= 0 || TEACHER_TP <= 0 || TEACHER_EP <= 0 || VAL_TP <= 0 \
    || ROLLOUT_TP > ROLLOUT_GPUS_PER_NODE || TEACHER_TP > TEACHER_WORLD_SIZE || VAL_TP > NGPUS_PER_NODE \
    || ROLLOUT_WORLD_SIZE % ROLLOUT_TP != 0 || TEACHER_TOTAL_GPUS % TEACHER_TP != 0 \
    || TRAIN_WORLD_SIZE % VAL_TP != 0 )); then
    echo "GPU counts must fit complete rollout, teacher, and validation replicas within each node." >&2
    exit 1
fi
ROLLOUT_INSTANCES=$((ROLLOUT_WORLD_SIZE / ROLLOUT_TP))
TEACHER_INSTANCES=$((TEACHER_TOTAL_GPUS / TEACHER_TP))
VAL_INSTANCES=$((TRAIN_WORLD_SIZE / VAL_TP))

PROJECT_NAME=${PROJECT_NAME:-psrl_distill_geo3k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_4b_from_qwen3_5_35b_vllm_fsdp}
START_TIME=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-${PSRL_PATH}/logs/${EXPERIMENT_NAME}/${START_TIME}}
HYDRA_ARGS=()
if [[ ${1:-} == --dry-run ]]; then
    HYDRA_ARGS=(--cfg job --resolve)
    shift
fi

DEPLOYMENT=(
    psrl.deployment.mode=disaggregated
    psrl.ps_manager_ip="${LOCAL_IP:-127.0.0.1}"
    psrl.reward_service_ip="${LOCAL_IP:-127.0.0.1}"
    psrl.ps_mode=nixl_cpu
    psrl.rollout_n=1
    psrl.staleness="${STALENESS}"
    psrl.staleness_buffer_entries="${TRAIN_BATCH_SIZE}"
    psrl.logging_path="${LOG_DIR}"
    psrl.log_prob.enable_rollout_engine_log_prob=False
    psrl.deployment.train_nnodes="${NNODES}"
    psrl.deployment.train_ngpus_per_node="${NGPUS_PER_NODE}"
    psrl.deployment.n_rollout_instances="${ROLLOUT_INSTANCES}"
    psrl.deployment.rollout_nnodes_per_instance=1
    psrl.deployment.rollout_ngpus_per_node_per_instance="${ROLLOUT_TP}"
    psrl.deployment.n_validate_instances="${VAL_INSTANCES}"
    psrl.deployment.validate_nnodes_per_instance=1
    psrl.deployment.validate_ngpus_per_node_per_instance="${VAL_TP}"
)

TASK_REWARD='reward_loop_type:naive,reward_fn:default,reward_coef:1.0'
TEACHER_REWARD='reward_loop_type:opd,reward_fn:default,reward_model_name:teacher,teacher_weight:1.0'
DATA=(
    "data.train_datas=[{file:'${TRAIN_FILE}',prompt_key:prompt,reward_model_dicts:[{${TASK_REWARD}},{${TEACHER_REWARD}}]}]"
    "data.val_datas=[{file:'${VAL_FILE}',prompt_key:prompt,reward_model_dicts:[{${TASK_REWARD}}]}]"
    data.train_datasets_ratios=[1.0]
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.image_key=images
    data.shuffle=True
)

MODEL=(
    train_actor_rollout_ref.model.path="${STUDENT_MODEL}"
    train_actor_rollout_ref.model.use_remove_padding=True
    train_actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    train_actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}"
    train_actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    train_actor_rollout_ref.actor.use_dynamic_bsz=False
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    train_actor_rollout_ref.actor.use_torch_compile=False
    train_actor_rollout_ref.actor.strategy=fsdp2
    train_actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    train_actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    train_actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=False
    train_actor_rollout_ref.actor.fsdp_config.offload_policy=False
    train_actor_rollout_ref.actor.fsdp_config.param_offload=False
    train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
    train_actor_rollout_ref.ref.strategy=fsdp2
    train_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    train_actor_rollout_ref.ref.fsdp_config.offload_policy=False
    train_actor_rollout_ref.ref.fsdp_config.param_offload=True
    train_actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    train_actor_rollout_ref.ref.fsdp_config.entropy_from_logits_with_chunking=False
    train_actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
    gen_actor_rollout_ref.rollout.name=vllm
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    gen_actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.4}"
    gen_actor_rollout_ref.rollout.n=1
    gen_actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    gen_actor_rollout_ref.rollout.calculate_log_probs=True
    gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj
    train_actor_rollout_ref.rollout.tensor_model_parallel_size="${VAL_TP}"
    train_actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.4}"
    train_actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
)

TEACHER=(
    reward.active_loops=[naive,opd]
    reward.reward_loops.opd.reward_model_name=teacher
    reward.reward_loops.opd.model.path="${TEACHER_MODEL}"
    reward.reward_loops.opd.nnodes="${NNODES}"
    reward.reward_loops.opd.n_gpus_per_node="${TEACHER_WORLD_SIZE}"
    reward.reward_loops.opd.num_replicas="${TEACHER_INSTANCES}"
    reward.reward_loops.opd.rollout_ngpus_per_instance_per_node="${TEACHER_TP}"
    reward.reward_loops.opd.rollout_nnodes_per_instance=1
    reward.reward_loops.opd.rollout.name=vllm
    reward.reward_loops.opd.rollout.load_format=auto
    reward.reward_loops.opd.rollout.tensor_model_parallel_size="${TEACHER_TP}"
    reward.reward_loops.opd.rollout.expert_parallel_size="${TEACHER_EP}"
    reward.reward_loops.opd.rollout.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL:-0.4}"
    reward.reward_loops.opd.rollout.enforce_eager=True
    reward.reward_loops.opd.rollout.max_num_batched_tokens=8192
    reward.reward_loops.opd.rollout.enable_chunked_prefill=True
    reward.reward_loops.opd.rollout.enable_prefix_caching=True
    reward.reward_loops.opd.rollout.prompt_length="$((MAX_MODEL_LEN - 1))"
    reward.reward_loops.opd.rollout.response_length=1
    reward.reward_loops.opd.rollout.max_model_len="${MAX_MODEL_LEN}"
)

DISTILLATION=(
    distillation.enabled=True
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE:-k1}"
    distillation.distillation_loss.topk="${DISTILLATION_TOPK:-64}"
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient="${USE_POLICY_GRADIENT:-True}"
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

TRAINER=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    trainer.balance_batch=True
    trainer.logger=[console,wandb]
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.save_freq="${SAVE_FREQ:-200}"
    trainer.test_freq="${TEST_FREQ:-5}"
    trainer.total_epochs="${TOTAL_EPOCHS:-15}"
)

mkdir -p "${LOG_DIR}"
echo "Training log: ${LOG_DIR}/console.log"
echo "GPU pools: train=${TRAIN_WORLD_SIZE}, rollout=${ROLLOUT_WORLD_SIZE}, teacher=${TEACHER_TOTAL_GPUS}."
python -m psrl.trainer.main_ppo "${HYDRA_ARGS[@]}" \
    "${DEPLOYMENT[@]}" "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" \
    "${ROLLOUT[@]}" "${TEACHER[@]}" "${DISTILLATION[@]}" "${TRAINER[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/console.log"
