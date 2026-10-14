#!/usr/bin/env bash
set -xeuo pipefail

# Run the real-data rollout benchmark with vLLM AsyncLLM.
source ${PSRL_WORKSPACE}/env/psrl.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

# Model configuration
HF_MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen2.5-32B

# Data configuration
TRAIN_FILE=${PSRL_WORKSPACE}/data/dapo/dapo-math-17k.parquet
TEST_FILE=${PSRL_WORKSPACE}/data/dapo/aime-2024.parquet

# Simplified vLLM configuration.
GEN_TP=4  # Tensor parallel size for generation
GEN_PP=1  # Pipeline parallel size for generation

# Node configuration
NNODES=1  # Simplified to single node
NGPUS_PER_NODE=8

# Test parameters
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
batch_size=512
num_iterations=1
warmup_iterations=1
test_mode="real_data"  # Use real data mode

# Generation parameters
temperature=1.0
top_p=1.0
top_k=-1

# Run the simplified rollout performance test with real data
PYTHONUNBUFFERED=1 python -m psrl.bench.rollout.main_rollout \
    psrl.logging_path=${PSRL_PATH}/examples/bench/rollout/summary \
    \
    model.path="$HF_MODEL_PATH" \
    +model.override_config.max_position_embeddings=32768 \
    \
    rollout.gpu_memory_utilization=0.5 \
    rollout.tensor_parallel_size=${GEN_TP} \
    rollout.pipeline_parallel_size=${GEN_PP} \
    rollout.enable_chunked_prefill=False \
    rollout.max_num_seqs=${batch_size} \
    rollout.max_num_batched_tokens=$((max_prompt_length * 32)) \
    rollout.temperature=${temperature} \
    rollout.top_p=${top_p} \
    rollout.top_k=${top_k} \
    rollout.disable_log_stats=false \
    rollout.ignore_eos=false \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${batch_size} \
    \
    rollout_test.batch_size=${batch_size} \
    rollout_test.num_iterations=${num_iterations} \
    rollout_test.warmup_iterations=${warmup_iterations} \
    rollout_test.mode=${test_mode} \
    rollout_test.profile_logs_dir=${PSRL_WORKSPACE}/psrl/examples/bench/rollout/details \
    rollout_test.profile_log_file=Real_TP${GEN_TP}_PP${GEN_PP}_B${batch_size}_P${max_prompt_length}_R${max_response_length} \
    2>&1 | tee rollout_test.log
