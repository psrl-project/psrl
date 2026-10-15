#!/usr/bin/env bash
set -xeuo pipefail

PSRL_WORKSPACE=/jizhicfs/pkuhetu/yfzhao/psrl

project_name='psrl_reward_model'
experiment_name='RM-Qwen2.5-Math-7B-GenLoop-dapo-skywork'

source ${PSRL_WORKSPACE}/env/psrl.sh

PSRL_WORKSPACE=/jizhicfs/pkuhetu/yfzhao/psrl


HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl, os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
HF_MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen2.5-Math-7B
DIST_CKPT_PATH=${PSRL_WORKSPACE}/models/mcore_ckpt/Qwen2.5-Math-7B
python ${PSRL_PATH}/scripts/convert_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH

TRAIN_FILE=${PSRL_WORKSPACE}/data/dapo/dapo-math-17k.parquet
TEST_FILE=${PSRL_WORKSPACE}/data/dapo/aime-2024.parquet

# rollout / reward settings
ROLL_TP=1
ROLL_PP=1
ROLL_INSTANCES=2
ROLL_NGPUS_PER_NODE_PER_INSTANCE=$((ROLL_TP * ROLL_PP))

RM_MODEL_PATH=${PSRL_WORKSPACE}/models/Skywork-Reward-V2-Qwen3-8B
# RM_MODEL_PATH=${PSRL_WORKSPACE}/../../qwen3-30B-A3B
RM_GEN_REWARD_FUNCTION=skywork
# RM_GEN_REWARD_FUNCTION=default
RM_VLLM_TASK=classify
RM_TP=1
RM_PP=1
RM_INSTANCES=2
RM_NGPUS_PER_NODE_PER_INSTANCE=$((RM_TP * RM_PP))

# training settings
TRAIN_TP=2
TRAIN_PP=2
TRAIN_CP=1
TRAIN_NGPUS=4

NNODES=1
NGPUS_PER_NODE=8

temperature=0.8
top_p=0.9
val_top_p=0.7
train_prompt_bsz=128
train_prompt_mini_bsz=64
n_resp_per_prompt=8
max_prompt_length=2048
max_response_length=4096

export PSRL_LOGGING_LEVEL=INFO

PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
    psrl.ps_manager_ip=${LOCAL_IP} \
    psrl.rollout_n=${n_resp_per_prompt} \
    psrl.staleness=0 \
    psrl.staleness_buffer_entries=${train_prompt_bsz} \
    psrl.gen_mode=stream \
    psrl.ps_mode=nixl_cpu \
    psrl.logging_path=${PSRL_PATH}/logs/reward_rm_log/${experiment_name} \
    psrl.log_prob.enable_rollout_engine_log_prob=True \
    psrl.log_prob.enable_train_engine_recompute_log_prob=True \
    psrl.log_prob.mode=rollout \
    psrl.deployment.n_rollout_instances=${ROLL_INSTANCES} \
    psrl.deployment.rollout_nnodes_per_instance=1 \
    psrl.deployment.rollout_ngpus_per_node_per_instance=${ROLL_NGPUS_PER_NODE_PER_INSTANCE} \
    psrl.deployment.train_nnodes=${NNODES} \
    psrl.deployment.train_ngpus_per_node=${TRAIN_NGPUS} \
    \
    reward_model.enable=True \
    reward_model.use_reward_loop=True \
    reward_model.reward_manager=gen \
    reward_model.model.path="${RM_MODEL_PATH}" \
    reward_model.num_replicas=${RM_INSTANCES} \
    reward_model.enable_resource_pool=True \
    reward_model.n_gpus_per_node=${RM_NGPUS_PER_NODE_PER_INSTANCE} \
    reward_model.nnodes=${NNODES} \
    reward_model.rollout_ngpus_per_instance_per_node=${RM_NGPUS_PER_NODE_PER_INSTANCE} \
    reward_model.rollout_nnodes_per_instance=1 \
    reward_model.rollout.tensor_model_parallel_size=${RM_TP} \
    reward_model.rollout.pipeline_model_parallel_size=${RM_PP} \
    reward_model.rollout.name=vllm \
    reward_model.rollout.dtype=bfloat16 \
    reward_model.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length + max_response_length)) \
    reward_model.rollout.max_model_len=10240 \
    reward_model.gen_reward_function=${RM_GEN_REWARD_FUNCTION} \
    reward_model.rollout.task=${RM_VLLM_TASK} \
    \
    gen_actor_rollout_ref.rollout.mode=psrl_async \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLL_TP} \
    gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${ROLL_PP} \
    gen_actor_rollout_ref.rollout.temperature=${temperature} \
    gen_actor_rollout_ref.rollout.top_p=${top_p} \
    gen_actor_rollout_ref.rollout.top_k=-1 \
    gen_actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    \
    train_actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    train_actor_rollout_ref.rollout.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.rollout.pipeline_model_parallel_size=${TRAIN_PP} \
    train_actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    train_actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    train_actor_rollout_ref.rollout.val_kwargs.n=1 \
    train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    train_actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    train_actor_rollout_ref.actor.optim.lr=1e-6 \
    train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    train_actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    train_actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.val_before_train=False \
    trainer.test_freq=200 \
    trainer.save_freq=200 \
    trainer.total_training_steps=200 2>&1 | tee ${PSRL_WORKSPACE}/logs/${experiment_name}.log

