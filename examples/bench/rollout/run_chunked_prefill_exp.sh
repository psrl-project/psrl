#!/usr/bin/env bash
# Sweep chunked prefill experiments across tensor parallel sizes.
# Optional arguments select the model, token budget, and GPU utilization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICRO_SCRIPT="${SCRIPT_DIR}/run_chunked_prefill_micro.sh"
MEM_SCRIPT="${SCRIPT_DIR}/run_chunked_prefill_memory_sweep.sh"

for script in "${MICRO_SCRIPT}" "${MEM_SCRIPT}"; do
    if [[ ! -f "${script}" ]]; then
        echo "Error: ${script} not found!"
        exit 1
    fi
    chmod +x "${script}"
done

HF_MODEL_PATH=${1:-${PSRL_WORKSPACE}/models/Qwen2.5-32B}
BUDGET=${2:-32768}
GPU_UTIL=${3:-0.90}

TP_VALUES=(1 2)
EXPERIMENTS=(e1)

total_micro=$((${#TP_VALUES[@]} * ${#EXPERIMENTS[@]}))
total_mem=${#TP_VALUES[@]}
current=0

echo "=========================================="
echo "Chunked prefill experiment sweep"
echo "Model: ${HF_MODEL_PATH}"
echo "Budget: ${BUDGET}"
echo "GPU util: ${GPU_UTIL}"
echo "TP values: ${TP_VALUES[*]}"
echo "E1/E2/E3a experiments: ${EXPERIMENTS[*]}"
echo "Total micro runs: ${total_micro}  Total memory sweeps: ${total_mem}"
echo "=========================================="

# E1 / E2 / E3a runs.
for tp in "${TP_VALUES[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        current=$((current + 1))
        echo ""
        echo "--- Micro run ${current}/${total_micro}: TP=${tp} experiment=${exp} ---"
        echo "Start: $(date)"
        if "${MICRO_SCRIPT}" "${tp}" "${HF_MODEL_PATH}" "${BUDGET}" "${exp}" "${GPU_UTIL}"; then
            echo "✓ TP=${tp} ${exp} completed."
        else
            echo "✗ TP=${tp} ${exp} failed (exit $?), continuing."
        fi
        echo "End: $(date)"
    done
done

# E3b memory sweep (needs one process per N, separate from the micro runs).
for tp in "${TP_VALUES[@]}"; do
    echo ""
    echo "--- E3b memory sweep: TP=${tp} ---"
    echo "Start: $(date)"
    if "${MEM_SCRIPT}" "${tp}" "${HF_MODEL_PATH}" "${GPU_UTIL}"; then
        echo "✓ E3b TP=${tp} sweep completed."
    else
        echo "✗ E3b TP=${tp} sweep failed (exit $?), continuing."
    fi
    echo "End: $(date)"
done

echo ""
echo "=========================================="
echo "All sweeps done. $(date)"
echo "=========================================="
