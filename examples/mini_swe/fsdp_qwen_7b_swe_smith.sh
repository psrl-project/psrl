set -xeuo pipefail

staleness=${1:-1}
project_name=psrl_swe_smith
experiment_name=GRPO-SWE-agent-LM-7B-swe_smith-fsdp2-staleness_${staleness}

source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

# --- Pre-flight checks ---
echo "=== Pre-flight checks ==="
python -c "from minisweagent.agents.default import DefaultAgent; print('mini-swe-agent: OK')"
python -c "from examples.mini_swe.grading.payload import grader_zip_bytes; print('grading payload', len(grader_zip_bytes()), 'bytes: OK')"
python -c "from examples.mini_swe.swebench_grader import grade_fresh_container; print('swebench_grader: OK')"
ray status 2>/dev/null | head -5 || echo "WARNING: ray status failed"
echo "=== Pre-flight done ==="

# --- Model ---
# NOTE(lhy): Modify max_position_embeddings in config.json to 32768 after downloading.
MODEL_PATH=${PSRL_WORKSPACE}/models/SWE-agent-LM-7B

# --- Data ---

TRAIN_FILE=${PSRL_PATH}/examples/mini_swe/data/swe_smith_py_1k/train.parquet
TEST_FILE=${PSRL_PATH}/examples/mini_swe/data/verified_subset_80/train.parquet

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: Training data not found at $TRAIN_FILE"
    echo "Run the data preparation commands at the top of this script."
    exit 1
fi

if [[ ! -f "$TEST_FILE" ]]; then
    echo "ERROR: Validation data not found at $TEST_FILE"
    echo "Run the data preparation commands at the top of this script."
    exit 1
fi

train_files="['$TRAIN_FILE']"
test_files="['$TEST_FILE']"

CKPT_ROOT=${CKPT_ROOT:-$PWD}
default_local_dir=$CKPT_ROOT/checkpoint/$experiment_name

# --- Agent loop config (full SWE-agent format for SWE-agent-LM-7B) ---
agent_loop_config_path=${PSRL_PATH}/examples/mini_swe/config/swebench_agent_config_full_sweagent.yaml

# --- Cluster layout (4 nodes x 8 GPUs, same as dapo script) ---
GEN_TP=2
GEN_PP=1

VAL_TP=2
VAL_PP=1

TRAIN_SP=2
TRAIN_FSDP=8

NNODES=4
NGPUS_PER_NODE=8

GEN_NNODES=2
GEN_NGPUS_PER_NODE=${NGPUS_PER_NODE}
GEN_INSTANCES=$(( (GEN_NNODES * GEN_NGPUS_PER_NODE) / (GEN_TP * GEN_PP) ))
GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( GEN_TP * GEN_PP ))

TRAIN_NNODES=2
TRAIN_NGPUS_PER_NODE=${NGPUS_PER_NODE}

VAL_INSTANCES=$(( (TRAIN_NNODES * TRAIN_NGPUS_PER_NODE) / (VAL_TP * VAL_PP) ))
VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( VAL_TP * VAL_PP ))

# --- Algorithm (GRPO / DAPO) ---

# Drop groups with zero reward variance to avoid zero-gradient updates.
enable_dynamic_sampling_filter=False
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# --- Sequence lengths ---

# Match evaluation's 50-turn ceiling to bound agent loops.
max_turns=50
max_prompt_length=2048
max_response_length=30000

# --- Training hyperparameters ---
actor_lr=1e-6
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 10))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
train_prompt_bsz=64
n_resp_per_prompt=8
n_resp_per_prompt_val=8
train_prompt_mini_bsz=32

# --- Sampling ---

# Preserve the model's sampling distribution while allowing GRPO exploration.
temperature=0.7
top_p=0.95
top_k=-1
val_top_p=0.7

# --- Reward ---
reward_mode=binary

# --- TIS ---
rollout_is=token
rollout_is_threshold=2.0

# --- Performance ---
use_dynamic_bsz=True
packing_length=$(( (max_prompt_length + max_response_length) * 1 ))
offload=True

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path=./config --config-name='ppo_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=${n_resp_per_prompt} \
    psrl.staleness=${staleness} \
    psrl.staleness_buffer_entries=${train_prompt_bsz} \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_PATH}/examples/mini_swe/fsdp_psrl_log/${experiment_name} \
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
    psrl.nixl.server_port=23456 \
    \
    gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    gen_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    gen_actor_rollout_ref.rollout.temperature=${temperature} \
    gen_actor_rollout_ref.rollout.top_p=${top_p} \
    gen_actor_rollout_ref.rollout.top_k=${top_k} \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=$max_turns \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    gen_actor_rollout_ref.rollout.agent.env.name=mini_swe_env \
    gen_actor_rollout_ref.rollout.agent.data.name=mini_swe_agent_data \
    gen_actor_rollout_ref.rollout.agent.num_workers=${NNODES} \
    \
    train_actor_rollout_ref.model.path="$MODEL_PATH" \
    train_actor_rollout_ref.model.use_remove_padding=True \
    +train_actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    train_actor_rollout_ref.model.enable_gradient_checkpointing=True \
    train_actor_rollout_ref.rollout.enable_chunked_prefill=True \
    train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${packing_length} \
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
    train_actor_rollout_ref.rollout.multi_turn.enable=True \
    train_actor_rollout_ref.rollout.multi_turn.max_turns=$max_turns \
    train_actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    train_actor_rollout_ref.rollout.agent.env.name=mini_swe_env \
    train_actor_rollout_ref.rollout.agent.data.name=mini_swe_agent_data \
    train_actor_rollout_ref.rollout.agent.num_workers=${NNODES} \
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
    +train_actor_rollout_ref.actor.use_rollout_log_probs=True \
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
    psrl.group_post_process.enable=${enable_dynamic_sampling_filter} \
    psrl.group_post_process.name=dynamic_sampling_filter \
    algorithm.filter_groups.metric=reward_score \
    \
    reward.active_managers='[dapo]' \
    reward.managers.dapo.reward_fn.0.path=${PSRL_PATH}/examples/mini_swe/reward.py \
    reward.managers.dapo.reward_fn.0.name=compute_score \
    +reward.managers.dapo.reward_fn.0.reward_kwargs.reward_mode=${reward_mode} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.log=False \
    reward.managers.dapo.reward_kwargs.max_resp_len=${max_response_length} \
    \
    data.train_files="$train_files" \
    data.reward_model_dicts.0.reward_loop_type=dapo \
    data.reward_model_dicts.0.reward_fn=compute_score \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.val_before_train=True \
    trainer.log_val_generations=5 \
    trainer.test_freq=5 \
    trainer.save_freq=50 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=200 2>&1 | tee ${experiment_name}.log
