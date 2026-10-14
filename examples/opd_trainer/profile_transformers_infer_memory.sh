#!/usr/bin/env bash
# Launch Transformers TP/EP inference memory profiler.
#
# Run `bash examples/opd_trainer/profile_transformers_infer_memory.sh` with `MODEL_PATH`, `SEQ_LEN`, `TP`, and `EP`.
#
# Requires: source env (conda) and torchrun with NPROC == TP.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source env/env_311.sh

MODEL_PATH="${MODEL_PATH:-models/Qwen3-235B-A22B}"
SEQ_LEN="${SEQ_LEN:-16384}"
PROMPT_LEN="${PROMPT_LEN:-0}"
RESPONSE_LEN="${RESPONSE_LEN:-}"
TOPK="${TOPK:-16}"
TP="${TP:-8}"
EP="${EP:-8}"
DTYPE="${DTYPE:-bfloat16}"
NUM_WARMUP="${NUM_WARMUP:-1}"
NUM_ITERS="${NUM_ITERS:-3}"
LOG_DIR="${LOG_DIR:-${PSRL_ROOT}/examples/opd_trainer/logs}"
LOG_PREFIX="${LOG_PREFIX:-profile_transformers_infer}"
TORCH_PROFILER="${TORCH_PROFILER:-true}"
TORCH_PROFILER_EVERY_RANK="${TORCH_PROFILER_EVERY_RANK:-false}"
TORCH_PROFILER_TOP_EVENTS="${TORCH_PROFILER_TOP_EVENTS:-200}"
TORCH_PROFILER_DIR="${TORCH_PROFILER_DIR:-torch_profiler_traces}"

USE_KV_CACHE_FLAG=()
if [[ "${USE_KV_CACHE:-false}" == "true" ]]; then
  USE_KV_CACHE_FLAG=(--use-kv-cache)
fi

TRUST_FLAG=()
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then
  TRUST_FLAG=(--trust-remote-code)
fi

TORCH_PROFILER_FLAG=()
if [[ "${TORCH_PROFILER}" == "true" ]]; then
  TORCH_PROFILER_FLAG=(--torch-profiler --torch-profiler-dir "${TORCH_PROFILER_DIR}" --torch-profiler-top-events "${TORCH_PROFILER_TOP_EVENTS}")
  if [[ "${TORCH_PROFILER_EVERY_RANK}" == "true" ]]; then
    TORCH_PROFILER_FLAG+=(--torch-profiler-every-rank)
  fi
fi

mkdir -p "${LOG_DIR}"

cd "${PSRL_ROOT}"

torchrun \
  --standalone \
  --nproc_per_node="${TP}" \
  examples/opd_trainer/profile_transformers_infer_memory.py \
  --model-path "${MODEL_PATH}" \
  --seq-len "${SEQ_LEN}" \
  --prompt-len "${PROMPT_LEN}" \
  ${RESPONSE_LEN:+--response-len "${RESPONSE_LEN}"} \
  --topk "${TOPK}" \
  --tp-size "${TP}" \
  --ep-size "${EP}" \
  --dtype "${DTYPE}" \
  --num-warmup "${NUM_WARMUP}" \
  --num-iters "${NUM_ITERS}" \
  --log-dir "${LOG_DIR}" \
  --log-prefix "${LOG_PREFIX}" \
  "${USE_KV_CACHE_FLAG[@]}" \
  "${TRUST_FLAG[@]}" \
  "${TORCH_PROFILER_FLAG[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_PREFIX}_${DTYPE}_${TP}x${EP}.log"
