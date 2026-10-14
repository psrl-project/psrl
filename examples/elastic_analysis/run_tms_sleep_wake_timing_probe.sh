#!/usr/bin/env bash
# Run one TMS-managed vLLM engine without starting the full trainer.
# The rollout role also starts a minimal CPU PS and NIXL meta server, loads the
# real checkpoint into PS memory, and measures wake/register/pull end to end.
# Usage: scripts/run_tms_sleep_wake_timing_probe.sh [reward|rollout] [32b|7b] [instances]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
PROBE_ROLE=${1:-reward}
MODEL_SIZE=${2:-32b}
INSTANCE_COUNT=${3:-1}

case "${PROBE_ROLE}" in
    reward|rollout) ;;
    *)
        echo "Invalid role: ${PROBE_ROLE}; expected reward or rollout" >&2
        exit 2
        ;;
esac

if ! [[ "${INSTANCE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid instance count: ${INSTANCE_COUNT}; expected a positive integer" >&2
    exit 2
fi
if [[ "${PROBE_ROLE}" != "rollout" && "${INSTANCE_COUNT}" != "1" ]]; then
    echo "Concurrent instances are currently supported only for the rollout probe" >&2
    exit 2
fi

case "${MODEL_SIZE}" in
    32b)
        TEST_ID='qwen2.5-32b-tp4'
        if [[ "${PROBE_ROLE}" == "reward" ]]; then
            MODEL_PATH=${PSRL_TMS_TIMING_MODEL_PATH:-${REPO_ROOT}/models/DeepSeek-R1-Distill-Qwen-32B}
        else
            MODEL_PATH=${PSRL_TMS_TIMING_MODEL_PATH:-${REPO_ROOT}/models/Qwen2.5-32B}
        fi
        MODEL_ENV=PSRL_TMS_VLLM_32B_MODEL
        ;;
    7b)
        TEST_ID='qwen2.5-7b-tp1'
        MODEL_PATH=${PSRL_TMS_TIMING_MODEL_PATH:-${REPO_ROOT}/models/Qwen2.5-7B}
        MODEL_ENV=PSRL_TMS_VLLM_7B_MODEL
        ;;
    *)
        echo "Invalid model size: ${MODEL_SIZE}; expected 32b or 7b" >&2
        exit 2
        ;;
esac

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model not found: ${MODEL_PATH}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/env/env_311.sh"

OUTPUT_SUFFIX=${PROBE_ROLE}_${MODEL_SIZE}
if [[ "${INSTANCE_COUNT}" != "1" ]]; then
    OUTPUT_SUFFIX=${OUTPUT_SUFFIX}_x${INSTANCE_COUNT}
fi
OUTPUT_DIR=${PSRL_TMS_TIMING_OUTPUT_DIR:-${REPO_ROOT}/logs/tms_sleep_wake_timing_probe/${OUTPUT_SUFFIX}}
mkdir -p "${OUTPUT_DIR}"

export PSRL_RUN_TMS_VLLM_SLEEP_SMOKE=1
export PSRL_TMS_VLLM_PROBE_ROLE="${PROBE_ROLE}"
export PSRL_TMS_VLLM_SLEEP_OUTPUT_DIR="${OUTPUT_DIR}"
export PSRL_LOGGING_LEVEL=${PSRL_LOGGING_LEVEL:-INFO}
export PSRL_TMS_VLLM_MAX_TOKENS=${PSRL_TMS_VLLM_MAX_TOKENS:-1}
export PSRL_TMS_VLLM_NUM_INSTANCES="${INSTANCE_COUNT}"
export PSRL_TMS_VLLM_CYCLES=${PSRL_TMS_VLLM_CYCLES:-2}
# Ray's uv hook walks the driver parent process tree even though this probe is
# not launched with uv. Container PID namespaces can make that walk fail before
# pytest collection, so disable only that unused environment propagation hook.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export "${MODEL_ENV}=${MODEL_PATH}"

if [[ "${PROBE_ROLE}" == "reward" ]]; then
    export PSRL_TMS_VLLM_GPU_MEMORY_UTILIZATION=${PSRL_TMS_VLLM_GPU_MEMORY_UTILIZATION:-0.8}
    export PSRL_TMS_VLLM_MAX_MODEL_LEN=${PSRL_TMS_VLLM_MAX_MODEL_LEN:-36864}
    export PSRL_TMS_VLLM_MAX_NUM_BATCHED_TOKENS=${PSRL_TMS_VLLM_MAX_NUM_BATCHED_TOKENS:-36864}
    export PSRL_TMS_VLLM_MAX_NUM_SEQS=${PSRL_TMS_VLLM_MAX_NUM_SEQS:-512}
    export PSRL_TMS_VLLM_ENFORCE_EAGER=${PSRL_TMS_VLLM_ENFORCE_EAGER:-1}
else
    export PSRL_TMS_VLLM_GPU_MEMORY_UTILIZATION=${PSRL_TMS_VLLM_GPU_MEMORY_UTILIZATION:-0.7}
    export PSRL_TMS_VLLM_MAX_MODEL_LEN=${PSRL_TMS_VLLM_MAX_MODEL_LEN:-32768}
    export PSRL_TMS_VLLM_MAX_NUM_BATCHED_TOKENS=${PSRL_TMS_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
    export PSRL_TMS_VLLM_MAX_NUM_SEQS=${PSRL_TMS_VLLM_MAX_NUM_SEQS:-1024}
    export PSRL_TMS_VLLM_ENFORCE_EAGER=${PSRL_TMS_VLLM_ENFORCE_EAGER:-0}
fi

cd "${REPO_ROOT}"
echo "Starting isolated TMS timing probe: role=${PROBE_ROLE} instances=${INSTANCE_COUNT} model=${MODEL_PATH} output=${OUTPUT_DIR}"
echo "Timing cycles: ${PSRL_TMS_VLLM_CYCLES}; weight arena: ${PSRL_TMS_VLLM_WEIGHT_ARENA:-0}"
if [[ "${PROBE_ROLE}" == "rollout" ]]; then
    echo "Rollout probe includes one CPU PSStorageWorker and real NIXL checkpoint pulls."
fi
if [[ -n "${PSRL_TMS_VLLM_UCX_DEVICE_GROUPS:-}" ]]; then
    echo "UCX device groups: ${PSRL_TMS_VLLM_UCX_DEVICE_GROUPS}"
fi
set -o pipefail
TEST_NAME=test_tms_vllm_level2_sleep_memory
if [[ "${PROBE_ROLE}" == "rollout" && "${INSTANCE_COUNT}" != "1" ]]; then
    export PSRL_RUN_TMS_ROLLOUT_CONCURRENT_PROBE=1
    TEST_NAME=test_tms_rollout_concurrent_sleep_wake
fi
python -m pytest -s -q \
    "unit_tests/workers/gen/test_rm_tms_smoke.py::${TEST_NAME}[${TEST_ID}]" \
    2>&1 | tee "${OUTPUT_DIR}/probe.log"

echo "Timing log: ${OUTPUT_DIR}/probe.log"
echo "Inspect with: rg '\[VLLM_SLEEP_WAKE_TIMING\]' '${OUTPUT_DIR}/probe.log'"
