PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

HF_MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen3-32B
DIST_CKPT_PATH=${PSRL_WORKSPACE}/models/mcore_ckpt/Qwen3-32B
python ${PSRL_PATH}/scripts/convert_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH
TRAIN_FILE=${PSRL_WORKSPACE}/data/dapo/dapo-math-17k.parquet
TEST_FILE=${PSRL_WORKSPACE}/data/dapo/aime-2024.parquet

OUTPUT_DIR=${PSRL_PATH}/examples/dapo_trainer/output
mkdir -p "$OUTPUT_DIR"
project_name=tx
experiment_name=qwen3_32b_megatron_resp_10240_lr1e-6_grpo-psrl-staleness2-5+3-partial
CKPTS_DIR=${OUTPUT_DIR}/ckpts/"${project_name}"/"${experiment_name}"
LOG_DIR=${OUTPUT_DIR}/logs/$(date +%Y%m%d_%H%M%S)


mkdir -p "$LOG_DIR"
mkdir -p "$CKPTS_DIR"


# train_batch_size=64
train_batch_size=80
tensor_model_parallel_size=8
pipeline_model_parallel_size=1
rollout_tensor_model_parallel_size=4
rollout_N=4
nnodes=6

GEN_TP=4
GEN_PP=1

VAL_TP=4
VAL_PP=1

GEN_NNODES=2
GEN_NGPUS_PER_NODE=8
GEN_INSTANCES=$(( (${GEN_NNODES} * ${GEN_NGPUS_PER_NODE}) / ( ${GEN_TP} * ${GEN_PP} ) )) # Number of generation instances
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} )) # Number of GPUs per node for generation per instance

TRAIN_NNODES=4
TRAIN_NGPUS_PER_NODE=8

VAL_INSTANCES=$(( (${TRAIN_NNODES} * ${TRAIN_NGPUS_PER_NODE}) / ( ${VAL_TP} * ${VAL_PP} ) )) # Number of validation instances
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} )) # Number of GPUs per node for validation per instance

# Requires vLLM 0.11 or newer, Megatron-LM 0.13 or newer, and the `qwen3vl_cp` MBridge branch.

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path=./config \
    --config-name='ppo_trainer' model_engine=megatron \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.reward_service_ip=${LOCAL_IP} \
    psrl.rollout_n=${rollout_N} \
    psrl.staleness=2 \
    psrl.staleness_buffer_entries=${train_batch_size} \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_PATH}/examples/dapo_trainer/megatron_psrl_log/${experiment_name} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.deployment.n_rollout_instances=${GEN_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.n_validate_instances=${VAL_INSTANCES} \
    psrl.deployment.validate_nnodes_per_instance=1 \
    psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${TRAIN_NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    psrl.nixl.server_port=26837 \
    psrl.group_post_process.enable=False \
    psrl.group_post_process.name=dynamic_sampling_filter \
    \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=10240 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    train_actor_rollout_ref.model.path=$HF_MODEL_PATH \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.actor.use_kl_loss=False \
    train_actor_rollout_ref.actor.kl_loss_coef=0.01 \
    train_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    train_actor_rollout_ref.actor.entropy_coeff=0 \
    train_actor_rollout_ref.actor.rollout_n=$rollout_N \
    train_actor_rollout_ref.actor.use_dynamic_bsz=True \
    train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22528 \
    +train_actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=0 \
    +train_actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    +train_actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=False \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +train_actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    train_actor_rollout_ref.actor.megatron.param_offload=False \
    train_actor_rollout_ref.actor.megatron.grad_offload=True \
    train_actor_rollout_ref.actor.megatron.optimizer_offload=True \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$tensor_model_parallel_size \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$pipeline_model_parallel_size \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    train_actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    train_actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    \
    train_actor_rollout_ref.rollout.val_kwargs.n=1 \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${VAL_PP} \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=22528 \
    train_actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    train_actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    \
    gen_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    gen_actor_rollout_ref.rollout.name=vllm \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    gen_actor_rollout_ref.rollout.n=$rollout_N \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    \
    train_actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    train_actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=22528 \
    train_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    \
    psrl.rollout_coordination.routing_strategy.method="request_num_balance" \
    psrl.rollout_coordination.routing_strategy.enable_group_sticky=True \
    psrl.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption=10000 \
    psrl.rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance=1024 \
    \
    psrl.rollout_coordination.sync_and_mig_strategy.method="greedy" \
    \
    psrl.rollout_coordination.partial_rollout.enable=True \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_training_steps=55 \
    trainer.total_epochs=15 $@ 2>&1 | tee "train_5.log"