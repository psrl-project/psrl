#!/usr/bin/env bash
# Start only FSDP actor workers and one CPU PS worker per actor node, then time
# repeated TMS sleep/wake, NIXL re-registration, metadata sync, and PS pull.
# Usage: scripts/run_train_actor_sleep_wake_probe.sh [8|16|32|all]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
WORLD_SIZE_ARG=${1:-all}

case "${WORLD_SIZE_ARG}" in
    8|16|32) WORLD_SIZES=("${WORLD_SIZE_ARG}") ;;
    all) WORLD_SIZES=(8 16 32) ;;
    *)
        echo "Invalid actor size: ${WORLD_SIZE_ARG}; expected 8, 16, 32, or all" >&2
        exit 2
        ;;
esac

MODEL_PATH=${PSRL_TRAIN_ACTOR_PROBE_MODEL_PATH:-${REPO_ROOT}/models/Qwen2.5-32B}
GPUS_PER_NODE=${PSRL_TRAIN_ACTOR_PROBE_GPUS_PER_NODE:-8}
CYCLES=${PSRL_TRAIN_ACTOR_PROBE_CYCLES:-2}
TIMEOUT_S=${PSRL_TRAIN_ACTOR_PROBE_TIMEOUT_S:-1800}
OUTPUT_ROOT=${PSRL_TRAIN_ACTOR_PROBE_OUTPUT_ROOT:-${REPO_ROOT}/logs/train_actor_sleep_wake_probe}

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model not found: ${MODEL_PATH}" >&2
    exit 1
fi
if ! [[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid PSRL_TRAIN_ACTOR_PROBE_GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
    exit 2
fi
if ! [[ "${CYCLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid PSRL_TRAIN_ACTOR_PROBE_CYCLES=${CYCLES}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source ./env/env_311.sh
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PSRL_LOGGING_LEVEL=${PSRL_LOGGING_LEVEL:-INFO}

mkdir -p "${OUTPUT_ROOT}"

for world_size in "${WORLD_SIZES[@]}"; do
    output_dir="${OUTPUT_ROOT}/fsdp${world_size}"
    mkdir -p "${output_dir}"
    echo "Starting trainer actor probe: world_size=${world_size} gpus_per_node=${GPUS_PER_NODE} model=${MODEL_PATH}"
    set -o pipefail
    python -m psrl.tools.trainer_actor_ps_timing_probe \
        --world-size "${world_size}" \
        --gpus-per-node "${GPUS_PER_NODE}" \
        --cycles "${CYCLES}" \
        --operation-timeout-s "${TIMEOUT_S}" \
        --model-path "${MODEL_PATH}" \
        --output "${output_dir}/report.json" \
        "${@:2}" \
        2>&1 | tee "${output_dir}/probe.log"
done

echo "Reports: ${OUTPUT_ROOT}/fsdp{8,16,32}/report.json"
echo "Inspect timings with: rg '\[TRAINER_WAKE_(TIMING|PROBE)\]' '${OUTPUT_ROOT}'"
