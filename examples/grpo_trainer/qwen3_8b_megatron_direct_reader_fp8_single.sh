#!/bin/bash
# Direct supply with reader-side block-FP8 quantization.
#
# The trainer stays BF16 and allocates no FP8 publish copy. Each rollout reads the
# live BF16 shards, stages them, and writes its own 128x128 block FP8 parameters plus
# `_scale_inv`. The write gate protects the live BF16 reads: it closes the version
# before the optimizer writes, preempts in-flight readers, and lets a conflicting
# reader retry the next version.
#
# This is the phase-1 edge of design 7.2. Peer diffusion is disabled for this mode so
# a follower cannot read an uninitialized holder scale; every reader quantizes its own
# target shard. The canonical-chunk peer cache is phase 2.
#
# All other settings are aligned with `qwen3_8b_megatron_direct_fp8gen.sh` so the
# reader-quantized arm can be compared with the publisher arm on data, batch sizes,
# step count, placement, and validation.
#
# Training and rollout each take a full 8-GPU node. Start the Ray cluster with eight
# GPUs per node (`PSRL_RAY_GPUS=8 scripts/start_nixl_cluster.sh <head> <worker>`) and set
# `LOCAL_IP` to a cluster node, because `PSManager` is pinned to `psrl.ps_manager_ip`. The
# preflight below refuses to start on a layout that would let both roles share a node.
#
# Ten training steps, with validation before training and every five steps after, as in
# the BF16 arm.
set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

HF_MODEL_PATH=${HF_MODEL_PATH:-${PSRL_PATH}/models/Qwen3-8B}
train_files=${TRAIN_FILES:-/apdcephfs_zwfy10_303541817/share_303541817/ls/data/dapo/dapo-math-17k.parquet}
test_files=${TEST_FILES:-/apdcephfs_zwfy10_303541817/share_303541817/ls/data/dapo/aime-2024.parquet}

OUTPUT_DIR=${PSRL_PATH}/output
mkdir -p "$OUTPUT_DIR"

# `STALENESS` rather than `$1`: a trailing `$@` forwards extra Hydra overrides, and a
# positional read would consume the first of them and leave it in `$@`.
staleness=${STALENESS:-3}
project_name=psrl_example

# FP8 training is opt-in. Persistent FP8 parameters (`fp8_param`) are not available:
# they need the model to be built inside TE's `fp8_model_init`, which this engine does
# not do, and enabling the flag alone corrupts the first update (config validation
# refuses it). Hybrid autocast keeps the parameters in bf16.
TRAIN_FP8=${TRAIN_FP8:-false}
TRAIN_FP8_RECIPE=${TRAIN_FP8_RECIPE:-tensorwise}
fp8_training_overrides=()
train_tag=""
if [ "${TRAIN_FP8}" = "true" ]; then
    fp8_training_overrides=(
        "+train_actor_rollout_ref.actor.megatron.override_transformer_config.fp8=hybrid"
        "+train_actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=${TRAIN_FP8_RECIPE}"
        "+train_actor_rollout_ref.actor.megatron.override_transformer_config.fp8_param=False"
    )
    train_tag="-fp8train"
fi

experiment_name=GRPO-Qwen3-8B-megatron-direct-reader-fp8gen-single-staleness_${staleness}
CKPTS_DIR=${OUTPUT_DIR}/ckpts/"${project_name}"/"${experiment_name}"
LOG_DIR=${OUTPUT_DIR}/logs/$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"
mkdir -p "$CKPTS_DIR"

train_batch_size=${TRAIN_BATCH_SIZE:-16}
rollout_N=${ROLLOUT_N:-8}

total_training_steps=${TOTAL_TRAINING_STEPS:-10}
test_freq=${TEST_FREQ:-5}

TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}

GEN_TP=${GEN_TP:-1}
GEN_PP=1

VAL_TP=1
VAL_PP=1

GEN_NNODES=${GEN_NNODES:-1}
GEN_NGPUS_PER_NODE=${GEN_NGPUS_PER_NODE:-4}
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) ))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} ))

TRAIN_NNODES=${TRAIN_NNODES:-1}
TRAIN_NGPUS_PER_NODE=${TRAIN_NGPUS_PER_NODE:-4}

VAL_INSTANCES=${VAL_INSTANCES:-0}
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} ))

# data length config
max_prompt_length=2048
max_response_length=8192

k=${k:-1}
max_token_len_per_gpu=$(( k * (max_prompt_length + max_response_length) ))

train_temperature=1.2
train_top_p=1.0
train_top_k=-1

val_temperature=0
val_rollot_n=1
val_top_p=1.0
val_top_k=-1
val_do_sample=False


python3 -m psrl.trainer.main_ppo --config-path=./config \
    --config-name='ppo_megatron_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=${rollout_N} \
    psrl.staleness=${staleness} \
    psrl.staleness_buffer_entries=${train_batch_size} \
    psrl.ps_mode=nixl_direct \
    psrl.colocate_validate_and_train=false \
    psrl.checkpoint.use_dcp_save=False \
    psrl.logging_path=${LOG_DIR} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=0 \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.nixl.server_port=27837 \
    psrl.group_post_process.enable=False \
    psrl.group_post_process.name=dynamic_sampling_filter \
    \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.train_max_samples=${TRAIN_MAX_SAMPLES:-512} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='error' \
    data.return_multi_modal_inputs=False \
    data.shuffle=True \
    \
    train_actor_rollout_ref.nccl_timeout=6000 \
    train_actor_rollout_ref.model.path=$HF_MODEL_PATH \
    train_actor_rollout_ref.model.use_remove_padding=True \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_kl_loss=False \
    train_actor_rollout_ref.actor.kl_loss_coef=0.01 \
    train_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.rollout_n=$rollout_N \
    train_actor_rollout_ref.actor.use_dynamic_bsz=False \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    train_actor_rollout_ref.actor.megatron.param_offload=False \
    train_actor_rollout_ref.actor.megatron.optimizer_offload=True \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    train_actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    train_actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    train_actor_rollout_ref.actor.megatron.use_mbridge=True \
    train_actor_rollout_ref.actor.megatron.use_remove_padding=True \
    ${fp8_training_overrides[@]+"${fp8_training_overrides[@]}"} \
    \
    train_actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    train_actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    train_actor_rollout_ref.rollout.val_kwargs.n=${val_rollot_n} \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu} \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    \
    gen_actor_rollout_ref.rollout.name=vllm \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=generate_only_agent \
    gen_actor_rollout_ref.rollout.temperature=${train_temperature} \
    gen_actor_rollout_ref.rollout.top_p=${train_top_p} \
    gen_actor_rollout_ref.rollout.top_k=${train_top_k} \
    psrl.nixl.weight_quantization.scheme=fp8 \
    psrl.nixl.reader_quantization=true \
    \
    reward.launch_reward_fn_async=True \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=False \
    trainer.save_freq=${SAVE_FREQ:--1} \
    trainer.test_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_training_steps=${total_training_steps} \
    trainer.total_epochs=1 $@ 2>&1 | tee "${experiment_name}.log"
