#!/usr/bin/env bash
# Run all available isolated shift-down memory probes.
#
# The vLLM cases use the TMS sleep-memory integration test and report
# device-wide snapshots for one isolated role.  The actor cases use the FSDP
# actor probe.  A failed case is recorded and does not stop later cases.
#
# Usage:
#   scripts/run_shift_down_memory_probes.sh
#   PSRL_SHIFT_DOWN_MEMORY_ROLES=reward,rollout scripts/run_shift_down_memory_probes.sh
#
# Model paths can be overridden with the model-specific environment variables
# used below, for example PSRL_TMS_VLLM_7B_MODEL=/path/to/Qwen2.5-7B.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/env/env_311.sh"

OUTPUT_ROOT=${PSRL_SHIFT_DOWN_MEMORY_OUTPUT_ROOT:-${REPO_ROOT}/logs/shift_down_memory_probes}
ROLES=${PSRL_SHIFT_DOWN_MEMORY_ROLES:-reward,rollout,actor}
CYCLES=${PSRL_TMS_VLLM_CYCLES:-2}
INIT_TIMEOUT_S=${PSRL_TMS_VLLM_INIT_TIMEOUT_S:-1800}
OPERATION_TIMEOUT_S=${PSRL_TMS_VLLM_OPERATION_TIMEOUT_S:-900}
ACTOR_CYCLES=${PSRL_TRAIN_ACTOR_PROBE_CYCLES:-2}
ACTOR_GPUS_PER_NODE=${PSRL_TRAIN_ACTOR_PROBE_GPUS_PER_NODE:-8}
ACTOR_TIMEOUT_S=${PSRL_TRAIN_ACTOR_PROBE_TIMEOUT_S:-1800}

mkdir -p "${OUTPUT_ROOT}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
export PSRL_LOGGING_LEVEL=${PSRL_LOGGING_LEVEL:-INFO}

has_role() {
    local requested
    IFS=',' read -r -a requested <<< "${ROLES}"
    for requested_role in "${requested[@]}"; do
        if [[ "${requested_role}" == "$1" ]]; then
            return 0
        fi
    done
    return 1
}

FAILED_CASES=()
SKIPPED_CASES=()

record_failure() {
    FAILED_CASES+=("$1")
}

record_skip() {
    SKIPPED_CASES+=("$1")
}

run_tms_case() {
    local role=$1
    local test_id=$2
    local case_name=$3
    local model_env=$4
    local model_path=$5
    local tp_size=$6
    local label="${role}_${case_name}"
    local output_dir="${OUTPUT_ROOT}/${label}"
    local log_file="${output_dir}/probe.log"
    local status

    if [[ ! -f "${model_path}/config.json" ]]; then
        echo "[SKIP] ${label}: model config not found: ${model_path}"
        record_skip "${label}:missing_model"
        return 0
    fi

    mkdir -p "${output_dir}"
    echo "[START] ${label} model=${model_path} tp=${tp_size} output=${output_dir}"

    export PSRL_RUN_TMS_VLLM_SLEEP_SMOKE=1
    export PSRL_TMS_VLLM_PROBE_ROLE="${role}"
    export PSRL_TMS_VLLM_NUM_INSTANCES=1
    export PSRL_TMS_VLLM_CYCLES="${CYCLES}"
    export PSRL_TMS_VLLM_INIT_TIMEOUT_S="${INIT_TIMEOUT_S}"
    export PSRL_TMS_VLLM_OPERATION_TIMEOUT_S="${OPERATION_TIMEOUT_S}"
    export PSRL_TMS_VLLM_TP_SIZE="${tp_size}"
    # Reward workers validate the node-shared cache configuration even before
    # model construction.  Enable the test arena explicitly so reward probes
    # reach the sleep stage; rollout probes use the ordinary NIXL path.
    if [[ "${role}" == "reward" ]]; then
        export PSRL_TMS_VLLM_WEIGHT_ARENA=1
    else
        export PSRL_TMS_VLLM_WEIGHT_ARENA=0
    fi
    # Keep the JSON report beside the tee log instead of pytest's temporary
    # directory, so the result survives pytest cleanup.
    export PSRL_TMS_VLLM_SLEEP_OUTPUT_DIR="${output_dir}"
    export "${model_env}=${model_path}"

    set +e
    python -m pytest -s -q \
        "unit_tests/workers/gen/test_rm_tms_smoke.py::test_tms_vllm_level2_sleep_memory[${test_id}]" \
        2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -eq 0 ]]; then
        echo "[PASS] ${label}"
    else
        echo "[FAIL] ${label} status=${status}; see ${log_file}"
        record_failure "${label}:${status}"
    fi
}

run_actor_case() {
    local world_size=$1
    local model_path=$2
    local label=$3
    local output_dir="${OUTPUT_ROOT}/${label}"
    local report_file="${output_dir}/report.json"
    local log_file="${output_dir}/probe.log"
    local status

    if [[ ! -f "${model_path}/config.json" ]]; then
        echo "[SKIP] ${label}: model config not found: ${model_path}"
        record_skip "${label}:missing_model"
        return 0
    fi

    mkdir -p "${output_dir}"
    echo "[START] ${label} model=${model_path} output=${output_dir}"
    set +e
    python -m psrl.tools.trainer_actor_ps_timing_probe \
        --world-size "${world_size}" \
        --gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
        --cycles "${ACTOR_CYCLES}" \
        --operation-timeout-s "${ACTOR_TIMEOUT_S}" \
        --model-path "${model_path}" \
        --output "${report_file}" \
        psrl.memory_logger.enable=True \
        2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -eq 0 ]]; then
        echo "[PASS] ${label}"
    else
        echo "[FAIL] ${label} status=${status}; see ${log_file}"
        record_failure "${label}:${status}"
    fi
}

# These are the requested isolated model/parallelism cases.  The parameter IDs
# are implemented by test_rm_tms_smoke.py below the TMS sleep-memory test.
if has_role reward; then
    run_tms_case reward glm-z1-9b-tp1 glm_z1_9b_tp1 PSRL_TMS_VLLM_GLM_9B_MODEL \
        "${PSRL_TMS_VLLM_GLM_9B_MODEL:-${REPO_ROOT}/models/GLM-Z1-9B-0414}" 1
    run_tms_case reward qwen3-30b-a3b-tp4-ep4 qwen3_30b_a3b_tp4_ep4 PSRL_TMS_VLLM_MOE_MODEL \
        "${PSRL_TMS_VLLM_MOE_MODEL:-${REPO_ROOT}/models/Qwen3-30B-A3B-Thinking-2507}" 4
    run_tms_case reward qwen3.5-122b-a10b-tp8-ep8 qwen3_5_122b_a10b_tp8_ep8 PSRL_TMS_VLLM_QWEN3_5_122B_MODEL \
        "${PSRL_TMS_VLLM_QWEN3_5_122B_MODEL:-${REPO_ROOT}/models/Qwen3.5-122B-A10B}" 8
fi

if has_role rollout; then
    run_tms_case rollout qwen2.5-1.5b-tp1 qwen2_5_1p5b_tp1 PSRL_TMS_VLLM_1P5B_MODEL \
        "${PSRL_TMS_VLLM_1P5B_MODEL:-${REPO_ROOT}/models/Qwen2.5-1.5B}" 1
    run_tms_case rollout qwen2.5-7b-tp1 qwen2_5_7b_tp1 PSRL_TMS_VLLM_7B_MODEL \
        "${PSRL_TMS_VLLM_7B_MODEL:-${REPO_ROOT}/models/Qwen2.5-7B}" 1
    run_tms_case rollout qwen2.5-32b-tp4 qwen2_5_32b_tp4 PSRL_TMS_VLLM_32B_MODEL \
        "${PSRL_TMS_VLLM_32B_MODEL:-${REPO_ROOT}/models/Qwen2.5-32B}" 4
    run_tms_case rollout qwen2.5-72b-tp8 qwen2_5_72b_tp8 PSRL_TMS_VLLM_72B_MODEL \
        "${PSRL_TMS_VLLM_72B_MODEL:-${REPO_ROOT}/models/Qwen2.5-72B}" 8
fi

if has_role actor; then
    run_actor_case 16 "${PSRL_TRAIN_ACTOR_1P5B_MODEL:-${REPO_ROOT}/models/Qwen2.5-1.5B}" actor_qwen2_5_1p5b_fsdp16
    run_actor_case 8 "${PSRL_TRAIN_ACTOR_7B_MODEL:-${REPO_ROOT}/models/Qwen2.5-7B}" actor_qwen2_5_7b_fsdp8
    run_actor_case 32 "${PSRL_TRAIN_ACTOR_32B_MODEL:-${REPO_ROOT}/models/Qwen2.5-32B}" actor_qwen2_5_32b_fsdp32
fi

printf '\nSummary\n'
printf 'Output root: %s\n' "${OUTPUT_ROOT}"
if ((${#FAILED_CASES[@]} == 0)); then
    echo 'Failed: none'
else
    printf 'Failed: %s\n' "${FAILED_CASES[*]}"
fi
if ((${#SKIPPED_CASES[@]} == 0)); then
    echo 'Skipped: none'
else
    printf 'Skipped: %s\n' "${SKIPPED_CASES[*]}"
fi

if ((${#FAILED_CASES[@]} > 0)); then
    exit 1
fi
