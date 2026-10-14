#!/usr/bin/env bash
# Profile vLLM TP/EP inference memory (OPD teacher: prompt_logprobs + max_tokens=1).
#
# Run `bash examples/opd_trainer/profile_vllm_infer_memory.sh` with `MODEL_PATH`, `SEQ_LEN`, `TP`, and `EP`.
#
# vLLM manages TP/EP in a single driver process (no torchrun).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${PSRL_ROOT}/env/env_311.sh"

MODEL_PATH="${MODEL_PATH:-models/Qwen3-235B-A22B}"
SEQ_LEN="${SEQ_LEN:-16384}"
PROMPT_LEN="${PROMPT_LEN:-0}"
RESPONSE_LEN="${RESPONSE_LEN:-}"
TOPK="${TOPK:-16}"
TP="${TP:-8}"
EP="${EP:-8}"
PP="${PP:-1}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
NUM_WARMUP="${NUM_WARMUP:-1}"
NUM_ITERS="${NUM_ITERS:-3}"
LOG_DIR="${LOG_DIR:-${PSRL_ROOT}/examples/opd_trainer/logs}"
LOG_PREFIX="${LOG_PREFIX:-profile_vllm_infer}"
VLLM_PROFILER="${VLLM_PROFILER:-true}"
VLLM_PROFILER_DIR="${VLLM_PROFILER_DIR:-torch_profiler_traces_vllm_chunked_prefill}"
VLLM_PROFILER_WAIT="${VLLM_PROFILER_WAIT:-0}"
VLLM_PROFILER_WARMUP="${VLLM_PROFILER_WARMUP:-1}"
VLLM_PROFILER_ACTIVE="${VLLM_PROFILER_ACTIVE:-1}"
VLLM_PROFILER_IGNORE_FRONTEND="${VLLM_PROFILER_IGNORE_FRONTEND:-true}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"

TRUST_FLAG=()
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then
  TRUST_FLAG=(--trust-remote-code)
fi

ENFORCE_EAGER_FLAG=()
if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  ENFORCE_EAGER_FLAG=(--enforce-eager)
else
  ENFORCE_EAGER_FLAG=(--no-enforce-eager)
fi

mkdir -p "${LOG_DIR}"
cd "${PSRL_ROOT}"

python examples/opd_trainer/profile_vllm_infer_memory.py \
  --model-path "${MODEL_PATH}" \
  --seq-len "${SEQ_LEN}" \
  --prompt-len "${PROMPT_LEN}" \
  ${RESPONSE_LEN:+--response-len "${RESPONSE_LEN}"} \
  --topk "${TOPK}" \
  --tp-size "${TP}" \
  --ep-size "${EP}" \
  --pp-size "${PP}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --num-warmup "${NUM_WARMUP}" \
  --num-iters "${NUM_ITERS}" \
  --log-dir "${LOG_DIR}" \
  --log-prefix "${LOG_PREFIX}" \
  $( [[ "${VLLM_PROFILER}" == "true" ]] && echo "--vllm-profiler" || true ) \
  --vllm-profiler-dir "${VLLM_PROFILER_DIR}" \
  --vllm-profiler-wait "${VLLM_PROFILER_WAIT}" \
  --vllm-profiler-warmup "${VLLM_PROFILER_WARMUP}" \
  --vllm-profiler-active "${VLLM_PROFILER_ACTIVE}" \
  $( [[ "${ENABLE_CHUNKED_PREFILL}" == "true" ]] && echo "--enable-chunked-prefill" || echo "--no-enable-chunked-prefill" ) \
  $( [[ "${VLLM_PROFILER_IGNORE_FRONTEND}" == "true" ]] && echo "--vllm-profiler-ignore-frontend" || echo "--no-vllm-profiler-ignore-frontend" ) \
  "${ENFORCE_EAGER_FLAG[@]}" \
  "${TRUST_FLAG[@]}" \
  2>&1 | tee "${LOG_DIR}/${LOG_PREFIX}_${DTYPE}_${TP}x${EP}.log"
