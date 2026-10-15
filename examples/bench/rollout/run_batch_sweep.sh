#!/usr/bin/env bash
set -euo pipefail

# Each run owns one GPU group. Set `GPU_IDS` to select the devices for the parallel sweep.
#
# Run from the repository root: `bash examples/bench/rollout/run_batch_sweep.sh [tp] [prompt_len] [ep]`.
#
# EP defaults to auto-detect in run_rollout_test.sh (MoE => EP=TP, dense => EP=1).
# Pass a third arg or set GEN_EP to override.
#
# For TP8 with 1024 prompt tokens, pass `8 1024`. Append `8` to enable EP8.

export PSRL_WORKSPACE=/apdcephfs_zwfy10/share_303541817/yfzhao/psrl 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${SCRIPT_DIR}/run_rollout_test.sh}"

if [[ -z "${PSRL_WORKSPACE:-}" ]]; then
    echo "Error: PSRL_WORKSPACE is not set."
    exit 1
fi

if [[ ! -f "${ROLLOUT_SCRIPT}" ]]; then
    echo "Error: ${ROLLOUT_SCRIPT} not found."
    exit 1
fi
chmod +x "${ROLLOUT_SCRIPT}"

GEN_TP="${1:-1}"
MAX_PROMPT_LENGTH="${2:-1024}"
# Empty means let run_rollout_test.sh auto-detect per model.
GEN_EP="${3:-${GEN_EP:-}}"

MODEL_NAMES=(GLM-Z1-9B-0414)
DISABLE_ATTN_VALUES=(false true)
BATCH_SIZE_VALUES=(1 2 4 8 16 32 64)
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

total=$((${#MODEL_NAMES[@]} * ${#DISABLE_ATTN_VALUES[@]} * ${#BATCH_SIZE_VALUES[@]}))
failed=()

echo "=========================================="
echo "Rollout batch sweep"
echo "PSRL_WORKSPACE: ${PSRL_WORKSPACE}"
echo "TP: ${GEN_TP}, EP: ${GEN_EP:-auto}, prompt_len: ${MAX_PROMPT_LENGTH}"
echo "Models: ${MODEL_NAMES[*]}"
echo "disable_attn: ${DISABLE_ATTN_VALUES[*]}"
echo "Batch sizes: ${BATCH_SIZE_VALUES[*]}"
echo "GPU_IDS: ${GPU_IDS}"
echo "Total runs: ${total}"
echo "Start: $(date)"
echo "=========================================="

IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
if ((${#gpu_ids[@]} == 0)); then
    echo "Error: GPU_IDS must contain at least one GPU id." >&2
    exit 1
fi
for gpu_id in "${gpu_ids[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Error: GPU_IDS must be a comma-separated list of numeric ids, got '${GPU_IDS}'." >&2
        exit 1
    fi
done
if [[ ! "${GEN_TP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: TP must be a positive integer, got '${GEN_TP}'." >&2
    exit 1
fi
if ((${#gpu_ids[@]} < GEN_TP)); then
    echo "Error: TP=${GEN_TP} requires at least ${GEN_TP} GPUs, but GPU_IDS has ${#gpu_ids[@]}." >&2
    exit 1
fi
if ((${#gpu_ids[@]} % GEN_TP != 0)); then
    echo "Error: GPU_IDS count (${#gpu_ids[@]}) must be divisible by TP (${GEN_TP})." >&2
    exit 1
fi

# Each TP run exclusively owns its assigned GPU group until completion.
gpu_groups=()
gpu_group_count=$((${#gpu_ids[@]} / GEN_TP))
for ((group_start = 0; group_start < ${#gpu_ids[@]}; group_start += GEN_TP)); do
    group_ids=("${gpu_ids[@]:group_start:GEN_TP}")
    gpu_group=$(IFS=','; printf '%s' "${group_ids[*]}")
    gpu_groups+=("${gpu_group}")
done

job_models=()
job_disable_attn=()
job_batch_sizes=()
for model_name in "${MODEL_NAMES[@]}"; do
    for disable_attn in "${DISABLE_ATTN_VALUES[@]}"; do
        for batch_size in "${BATCH_SIZE_VALUES[@]}"; do
            job_models+=("${model_name}")
            job_disable_attn+=("${disable_attn}")
            job_batch_sizes+=("${batch_size}")
        done
    done
done

profile_root="${PROFILE_ROOT:-${PSRL_WORKSPACE}/examples/bench/rollout/exp}"
profile_logs_dir="${PROFILE_LOGS_DIR:-${profile_root}/details}"
summary_root="${PROFILE_SUMMARY_DIR:-${profile_root}/summary}"
run_logs_dir="${RUN_LOGS_DIR:-.}"
mkdir -p "${profile_logs_dir}" "${summary_root}" "${run_logs_dir}"

status_dir=$(mktemp -d "${TMPDIR:-/tmp}/psrl_batch_sweep.XXXXXX")
cleanup_status_dir() {
    rm -rf -- "${status_dir}"
}
trap cleanup_status_dir EXIT

run_worker() {
    local worker_id=$1
    local gpu_group=$2
    local job_index
    local model_name
    local disable_attn
    local batch_size
    local run_label
    local -a rollout_args
    local run_summary_dir
    local gpu_label
    local run_log
    local exit_code

    gpu_label=${gpu_group//,/_}
    for ((job_index = worker_id; job_index < total; job_index += gpu_group_count)); do
        model_name=${job_models[job_index]}
        disable_attn=${job_disable_attn[job_index]}
        batch_size=${job_batch_sizes[job_index]}
        run_label="model=${model_name}, disable_attn=${disable_attn}, batch_size=${batch_size}, ep=${GEN_EP:-auto}"
        run_summary_dir="${summary_root}/run_${job_index}_gpu_${gpu_label}"
        run_log="${run_logs_dir}/rollout_test_run${job_index}_gpu${gpu_label}_${model_name}_tp${GEN_TP}_ep${GEN_EP:-auto}_b${batch_size}_p${MAX_PROMPT_LENGTH}_r8192_disable_attn_${disable_attn}.log"

        echo ""
        echo "=========================================="
        echo "[GPU ${gpu_group}] Run $((job_index + 1))/${total}: ${run_label}"
        echo "Start: $(date)"
        echo "=========================================="

        rollout_args=("${GEN_TP}" "${MAX_PROMPT_LENGTH}" "${batch_size}" "${disable_attn}")
        if [[ -n "${GEN_EP}" ]]; then
            rollout_args+=("${GEN_EP}")
        fi

        if CUDA_VISIBLE_DEVICES="${gpu_group}" \
            MODEL_NAME="${model_name}" \
            PROFILE_LOGS_DIR="${profile_logs_dir}" \
            PROFILE_SUMMARY_DIR="${run_summary_dir}" \
            RUN_LOG="${run_log}" \
            "${ROLLOUT_SCRIPT}" "${rollout_args[@]}"; then
            echo "[GPU ${gpu_group}] Run $((job_index + 1))/${total} (${run_label}) succeeded."
            printf 'success\t%s\n' "${job_index}" >> "${status_dir}/worker_${worker_id}.status"
        else
            exit_code=$?
            echo "[GPU ${gpu_group}] Run $((job_index + 1))/${total} (${run_label}) failed (exit ${exit_code})."
            printf 'failed\t%s\t%s\t%s\n' "${job_index}" "${run_label}" "${exit_code}" >> "${status_dir}/worker_${worker_id}.status"
        fi

        echo "[GPU ${gpu_group}] End: $(date)"
    done
}

worker_pids=()
for ((worker_id = 0; worker_id < gpu_group_count; worker_id++)); do
    run_worker "${worker_id}" "${gpu_groups[worker_id]}" &
    worker_pids+=("$!")
done

for worker_pid in "${worker_pids[@]}"; do
    wait "${worker_pid}"
done

for ((worker_id = 0; worker_id < gpu_group_count; worker_id++)); do
    status_file="${status_dir}/worker_${worker_id}.status"
    [[ -f "${status_file}" ]] || continue
    while IFS=$'\t' read -r status job_index run_label exit_code; do
        [[ -n "${status}" ]] || continue
        if [[ "${status}" == "failed" ]]; then
            failed+=("${run_label} (exit ${exit_code})")
        fi
    done < "${status_file}"
done

echo ""
echo "=========================================="
echo "Batch sweep finished: $(date)"
if ((${#failed[@]} == 0)); then
    echo "All ${total} runs succeeded."
else
    echo "Failed runs (${#failed[@]}/${total}):"
    for run in "${failed[@]}"; do
        echo "  - ${run}"
    done
    exit 1
fi
echo "=========================================="
