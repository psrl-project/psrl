#!/usr/bin/env bash
# Launch script demonstrating the five deployment modes (psrl.deployment.mode).
#
# Usage:
#   bash examples/tx/run_deployment_modes.sh <mode> [extra hydra overrides...]
#
# where <mode> in {disaggregated, colocated, rollout_rm_colocated,
#                  trainer_pool_only, elastic_rl}.
#
# Mode-specific required overrides are applied automatically below. Any extra
# hydra overrides passed after <mode> are forwarded to main_ppo.
#
# Notes:
# - Modes 2/3/4 require psrl.ps_mode in {nixl_cpu, nixl_gpu} (NIXL sleep/wake).
# - Modes 2/3 require reward.launch_reward_fn_async=true and
#   psrl.staleness=0 (per-buffer sequential rollout->reward phasing).
# - Mode 4 enables fixed extra rollout/rm replicas on train_pool via
#   trainer_pool_idle_{rollout,rm}_instances.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export RAY_prestart_worker_first_driver=false
export RAY_num_workers_soft_limit=0
export RAY_memory_monitor_refresh_ms=0

MODE="${1:?usage: $0 <mode> [overrides...]}"
shift || true

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
HF_MODEL_PATH=${HF_MODEL_PATH:-/jizhicfs/pkuhetu/zym/Qwen3-8B}
train_files=${TRAIN_FILES:-/jizhicfs/pkuhetu/zym/verl/data/retool_dapo/train.parquet}
test_files=${TEST_FILES:-/jizhicfs/pkuhetu/zym/verl/data/retool_aime2024/train.parquet}

OUTPUT_DIR=${OUTPUT_DIR:-/jizhicfs/pkuhetu/zym/psrl-new/psrl/tx-output}
project_name=tx
experiment_name="deployment_${MODE}"
CKPTS_DIR=${OUTPUT_DIR}/ckpts/"${project_name}"/"${experiment_name}"
LOG_DIR=${OUTPUT_DIR}/logs/${MODE}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR" "$CKPTS_DIR"

train_batch_size=64
rollout_N=8
GEN_TP=1
GEN_PP=1
GEN_NNODES=1
GEN_NGPUS_PER_NODE=8
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) ))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} ))
TRAIN_NNODES=1
TRAIN_NGPUS_PER_NODE=8
VAL_TP=1
VAL_PP=1
VAL_INSTANCES=$(( (${TRAIN_NNODES} * ${TRAIN_NGPUS_PER_NODE}) / ( ${VAL_TP} * ${VAL_PP} ) ))
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} ))

# ---- Mode-specific required overrides ----
MODE_OVERRIDES=""
case "$MODE" in
  disaggregated)
    # Mode 1: three independent pools, concurrent async pipeline. No extra requirements.
    MODE_OVERRIDES="+psrl.deployment.mode=disaggregated psrl.staleness=2"
    ;;
  colocated)
    # Mode 2: all three roles share train_pool, time-multiplexed per step.
    MODE_OVERRIDES="+psrl.deployment.mode=colocated \
      psrl.ps_mode=nixl_cpu \
      psrl.staleness=0 \
      reward.launch_reward_fn_async=True"
    ;;
  rollout_rm_colocated)
    # Mode 3: rollout+rm share shared_rollout_pool; trainer on train_pool.
    # shared pool carries the rollout/rm replicas; train_pool carries the actor.
    MODE_OVERRIDES="+psrl.deployment.mode=rollout_rm_colocated \
      psrl.ps_mode=nixl_cpu \
      psrl.staleness=0 \
      psrl.deployment.elastic.shared_nnodes=1 \
      psrl.deployment.elastic.shared_ngpus_per_node=8 \
      reward.launch_reward_fn_async=True"
    ;;
  trainer_pool_only)
    # Mode 4: three independent pools + fixed extra replicas on train_pool while idle.
    # Tune the idle counts to fit train_pool GPUs (must be divisible by instance world size).
    MODE_OVERRIDES="+psrl.deployment.mode=trainer_pool_only \
      psrl.ps_mode=nixl_cpu \
      psrl.staleness=2 \
      +psrl.deployment.trainer_pool_idle_rollout_instances=2 \
      +psrl.deployment.trainer_pool_idle_rm_instances=1"
    ;;
  elastic_rl)
    # Mode 5: current elastic auto-scaling.
    MODE_OVERRIDES="+psrl.deployment.mode=elastic_rl \
      psrl.ps_mode=nixl_cpu \
      psrl.staleness=2 \
      psrl.deployment.elastic.enable=True \
      psrl.deployment.elastic.enable_policy=True \
      psrl.deployment.elastic.shared_nnodes=1 \
      psrl.deployment.elastic.shared_ngpus_per_node=8 \
      reward.launch_reward_fn_async=True"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Valid modes: disaggregated | colocated | rollout_rm_colocated | trainer_pool_only | elastic_rl" >&2
    exit 1
    ;;
esac

case "$MODE" in
  elastic_rl|colocated|rollout_rm_colocated)
    colocate_validate_and_train=True
    fuse_rollout_with_validate=False
    ;;
  *)
    colocate_validate_and_train=False
    fuse_rollout_with_validate=True
    ;;
esac

python3 -m psrl.trainer.main_ppo --config-path="${PSRL_PATH}/psrl/trainer/config" \
    --config-name='ppo_megatron_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP:-127.0.0.1} \
    psrl.rollout_n=${rollout_N} \
    psrl.staleness_buffer_entries=${train_batch_size} \
    psrl.logging_path=${LOG_DIR} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.nixl.server_port=27237 \
    psrl.group_post_process.enable=False \
    algorithm.adv_estimator=grpo \
    +data.train_files="$train_files" \
    +data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=10240 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    train_actor_rollout_ref.nccl_timeout=6000 \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_kl_loss=False \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    +train_actor_rollout_ref.actor.rollout_n=$rollout_N \
    train_actor_rollout_ref.actor.use_dynamic_bsz=True \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=45056 \
    train_actor_rollout_ref.actor.megatron.param_offload=False \
    train_actor_rollout_ref.actor.megatron.grad_offload=True \
    train_actor_rollout_ref.actor.megatron.optimizer_offload=True \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    train_actor_rollout_ref.rollout.val_kwargs.n=1 \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=45056 \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    train_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    gen_actor_rollout_ref.rollout.name=vllm \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    gen_actor_rollout_ref.rollout.n=$rollout_N \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    train_actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    train_actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=45056 \
    train_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.ref.megatron.param_offload=False \
    psrl.rollout_coordination.routing_strategy.method="request_num_balance" \
    psrl.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption=10000 \
    psrl.rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance=1024 \
    psrl.rollout_coordination.sync_and_mig_strategy.method="greedy" \
    psrl.rollout_coordination.partial_rollout.enable=True \
    psrl.colocate_validate_and_train=${colocate_validate_and_train} \
    psrl.fuse_rollout_with_validate=${fuse_rollout_with_validate} \
    reward.launch_reward_fn_async=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=1 \
    $MODE_OVERRIDES \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
