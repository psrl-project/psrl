#!/usr/bin/env bash
# Evaluate an already exported PSRL Hugging Face checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PSRL_PATH="${PSRL_PATH:-$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH must be a PSRL checkpoint in Hugging Face format." >&2
    echo "Example: MODEL_PATH=/path/to/global_step_*/actor/model/huggingface bash $0" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: MODEL_PATH is not a complete Hugging Face checkpoint (missing config.json): ${MODEL_PATH}" >&2
    exit 1
fi

LENGTH="${LENGTH:-50 200 800}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data/hotpotqa}"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/results}"
default_save_file="$(basename "${MODEL_PATH%/}")"
if [[ "${default_save_file}" == "huggingface" ]]; then
    checkpoint_step="$(basename "$(dirname "$(dirname "$(dirname "${MODEL_PATH%/}")")")")"
    if [[ "${checkpoint_step}" == global_step_* ]]; then
        default_save_file="${checkpoint_step}"
    fi
fi
SAVE_FILE="${SAVE_FILE:-${default_save_file}}"

TP="${TP:-1}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8000}"
N_PROC="${N_PROC:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
FORCE="${FORCE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"

export MEM_CHUNK_TOKENS="${MEM_CHUNK_TOKENS:-2048}"
export MEM_MAX_MEMORY="${MEM_MAX_MEMORY:-1024}"
export MEM_MAX_FINAL="${MEM_MAX_FINAL:-256}"
export MEM_MAX_CHUNKS="${MEM_MAX_CHUNKS:-64}"
export VLLM_SERVE_HOST="${SERVE_HOST}"
export VLLM_SERVE_PORT="${SERVE_PORT}"
export DATA_ROOT="${DATA_ROOT}"

mkdir -p "${SAVE_DIR}"
LENGTHS="${LENGTH}" DATA_DIR="${DATA_ROOT}" bash "${SCRIPT_DIR}/prepare-eval-data.sh"


STAMP="$(date +%Y%m%d_%H%M%S)"
VLLM_LOG="${SAVE_DIR}/vllm_server_${STAMP}.log"
VLLM_PID=""

cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        echo "Stopping vLLM (pid=${VLLM_PID})"
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local url="http://${SERVE_HOST}:${SERVE_PORT}/v1/models"
    for _ in {1..120}; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited during startup. See ${VLLM_LOG}" >&2
            tail -n 80 "${VLLM_LOG}" >&2 || true
            exit 1
        fi
        if curl -fsS --max-time 10 "${url}" 2>/dev/null | grep -Fq "${MODEL_PATH}"; then
            return
        fi
        sleep 5
    done
    echo "ERROR: vLLM was not ready after 10 minutes. See ${VLLM_LOG}" >&2
    exit 1
}

echo "Starting vLLM: model=${MODEL_PATH} tp=${TP} port=${SERVE_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" vllm serve "${MODEL_PATH}" \
    --tensor-parallel-size "${TP}" \
    --host "${SERVE_HOST}" \
    --port "${SERVE_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
    --trust-remote-code >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
wait_for_server

common_args=(
    --model "${MODEL_PATH}"
    --tokenizer "${MODEL_PATH}"
    --api recurrent
    --n-proc "${N_PROC}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
)
if [[ "${FORCE}" == "1" ]]; then
    common_args+=(--force)
fi

for length in ${LENGTH}; do
    result_dir="${SAVE_DIR}/ruler_hqa_${length}"
    echo "Evaluating RULER-HQA n_docs=${length}"
    python3 "${SCRIPT_DIR}/eval_ruler_hqa.py" \
        "${common_args[@]}" \
        --length "${length}" \
        --data-root "${DATA_ROOT}" \
        --save-dir "${result_dir}" \
        --save-file "${SAVE_FILE}"
done

echo "Evaluation finished: ${SAVE_DIR}"
