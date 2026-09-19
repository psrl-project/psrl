#!/bin/bash
set -xeuo pipefail

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
export PSRL_LOGGING_PATH=${PSRL_PATH}/tests/nixl/log
export PSRL_LOGGING_LEVEL=INFO
cd ${PSRL_PATH}/tests/nixl

# Pick a CASE via env override, e.g. `CASE=4 bash scripts/test_nixl_e2e.sh`.
# Default is the original Qwen2.5-3B HSDP smoke case.
CASE=${CASE:-0}

# NOTE(lhy): This unit test uses FSDP2, so HSDP/FSDP precision differs from Megatron.
# The all-ones initialization and pull-equality verification are precision-independent.

# CASE 0: Qwen2.5-3B-Instruct, HSDP train (1×8) and vLLM gen (TP=2 PP=2), 16 GPUs.
if [ $CASE -eq 0 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=fsdp_hybrid \
        test.fsdp_hybrid.ddp_size=1 \
        test.fsdp_hybrid.fsdp_size=8 \
        test.gen.tensor_parallel_size=2 \
        test.gen.pipeline_parallel_size=2 \
        model.path=${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 1: Qwen2.5-32B, Megatron train (TP=4 PP=2) and vLLM gen (TP=4), 16 GPUs.
if [ $CASE -eq 1 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=4 \
        test.megatron.pipeline_model_parallel_size=1 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        model.path=${PSRL_WORKSPACE}/models/Qwen2.5-32B \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 2: Qwen2.5-3B-Instruct, HSDP train (4×8) and vLLM gen (TP=2), 64 GPUs.
if [ $CASE -eq 2 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=32 \
        test.num_gen=32 \
        test.train_engine_type=fsdp_hybrid \
        test.fsdp_hybrid.ddp_size=4 \
        test.fsdp_hybrid.fsdp_size=8 \
        test.gen.tensor_parallel_size=2 \
        test.gen.pipeline_parallel_size=1 \
        model.path=${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 3: Qwen2.5-32B, Megatron train (TP=8 PP=2) and vLLM gen (TP=4), 64 GPUs.
if [ $CASE -eq 3 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        nixl.max_pinned_temp_memory_slots=4 \
        test.num_train=32 \
        test.num_gen=32 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=8 \
        test.megatron.pipeline_model_parallel_size=2 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        model.path=${PSRL_WORKSPACE}/models/Qwen2.5-32B \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 4: Qwen3-1.7B dense, HSDP train (2×4) and vLLM gen (TP=2 DP=4), 8 GPUs.
# Exercises dense Qwen3 projection splits without MoE or GDN.
if [ $CASE -eq 4 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=fsdp_hybrid \
        test.fsdp_hybrid.ddp_size=2 \
        test.fsdp_hybrid.fsdp_size=4 \
        test.gen.tensor_parallel_size=2 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.expert_parallel_size=1 \
        test.gen.data_parallel_size=4 \
        model.path=${PSRL_WORKSPACE}/models/Qwen3-1.7B \
        model.train_dtype=float32 \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 5: Qwen3-30B-A3B-Instruct-2507 with 128 routed experts on 8 GPUs.
# Exercises Qwen3-MoE projection splits with expert parallelism on both sides.
if [ $CASE -eq 5 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=2 \
        test.megatron.pipeline_model_parallel_size=1 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.megatron.expert_model_parallel_size=4 \
        test.megatron.expert_tensor_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.expert_parallel_size=4 \
        model.path=${PSRL_WORKSPACE}/models/Qwen3-30B-A3B-Instruct-2507 \
        model.train_dtype=bfloat16 \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 6: Qwen3.5-4B dense, Megatron train (TP=2 PP=2 DP=2) and vLLM gen (TP=2 DP=4), 8 GPUs.
# Exercises full and linear attention projection layouts on GDN layers.
if [ $CASE -eq 6 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=2 \
        test.megatron.pipeline_model_parallel_size=2 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.megatron.expert_model_parallel_size=1 \
        test.megatron.expert_tensor_parallel_size=1 \
        test.gen.tensor_parallel_size=2 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.data_parallel_size=4 \
        model.path=${PSRL_WORKSPACE}/models/Qwen3.5-4B \
        model.trust_remote_code=true \
        model.train_dtype=bfloat16 \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 7: Qwen3.5-35B-A3B with 256 experts on 8 GPUs.
# Exercises fused MoE, QKV, and GDN layouts with expert parallelism.
if [ $CASE -eq 7 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=1 \
        test.megatron.pipeline_model_parallel_size=1 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.megatron.expert_model_parallel_size=8 \
        test.megatron.expert_tensor_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.expert_parallel_size=4 \
        test.gen.data_parallel_size=2 \
        model.path=${PSRL_WORKSPACE}/models/Qwen3.5-35B-A3B \
        model.trust_remote_code=true \
        model.train_dtype=bfloat16 \
        2>&1 | tee test_nixl_e2e.log
fi

# CASE 8: Moonlight-16B-A3B, Megatron train (TP=2 PP=1 DP=4 EP=4) and vLLM gen (TP=4 DP=2 EP=4).
# Exercises DeepSeek MLA and fused MoE layouts with shared experts.
if [ $CASE -eq 8 ]; then
    PYTHONUNBUFFERED=1 python test_nixl_e2e.py \
        test.num_train=8 \
        test.num_gen=8 \
        test.train_engine_type=megatron \
        test.megatron.tensor_model_parallel_size=2 \
        test.megatron.pipeline_model_parallel_size=1 \
        test.megatron.virtual_pipeline_model_parallel_size=1 \
        test.megatron.context_parallel_size=1 \
        test.megatron.expert_model_parallel_size=4 \
        test.megatron.expert_tensor_parallel_size=1 \
        test.gen.tensor_parallel_size=4 \
        test.gen.pipeline_parallel_size=1 \
        test.gen.data_parallel_size=2 \
        test.gen.expert_parallel_size=4 \
        model.path=${PSRL_WORKSPACE}/models/Moonlight-16B-A3B \
        model.trust_remote_code=true \
        model.train_dtype=bfloat16 \
        2>&1 | tee test_nixl_e2e.log
fi
