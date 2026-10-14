#!/usr/bin/env bash
# Run chunked prefill microbenchmarks.
# Optional arguments select parallelism, model, budget, experiment, and GPU utilization.
set -xeuo pipefail

source ${PSRL_WORKSPACE}/env/psrl.sh

PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

GEN_TP=${1:-1}
HF_MODEL_PATH=${2:-${PSRL_WORKSPACE}/models/SWE-agent-LM-7B}
BUDGET=${3:-65536}
EXPERIMENT=${4:-all}
GPU_UTIL=${5:-0.90}

RESULTS_DIR=${PSRL_PATH}/examples/bench/rollout/exp/chunked_prefill
mkdir -p "${RESULTS_DIR}"

PYTHONUNBUFFERED=1 python -m psrl.bench.chunked_prefill.main_micro \
    psrl.logging_path="${PSRL_PATH}/examples/bench/rollout/exp/chunked_prefill/logs" \
    model.path="${HF_MODEL_PATH}" \
    rollout.tensor_parallel_size=${GEN_TP} \
    rollout.gpu_memory_utilization=${GPU_UTIL} \
    rollout.max_num_batched_tokens=${BUDGET} \
    micro.experiment=${EXPERIMENT} \
    micro.output_dir="${RESULTS_DIR}" \
    micro.output_prefix="micro_TP${GEN_TP}_N${BUDGET}" \
    2>&1 | tee "${RESULTS_DIR}/micro_TP${GEN_TP}_N${BUDGET}_${EXPERIMENT}.log"
