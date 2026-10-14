#!/usr/bin/env bash
set -xeuo pipefail

# vllm
# customized (for profiling)
export VLLM_DISABLE_ATTN="1" # 0: use attention, 1: forbid the use of the attention

# Sweep TP, prompt length, and batch size across rollout tests.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_SCRIPT="${SCRIPT_DIR}/run_rollout_test.sh"

# Ensure the rollout script exists and is executable
if [[ ! -f "${ROLLOUT_SCRIPT}" ]]; then
    echo "Error: ${ROLLOUT_SCRIPT} not found!"
    exit 1
fi
chmod +x "${ROLLOUT_SCRIPT}"

# Parameter arrays
TP_VALUES=(2 4)
# TP_VALUES=(1)
MAX_PROMPT_LENGTH_VALUES=(128)
BATCH_SIZE_VALUES=(1 2 4 8 16 32 64 128 256)
# BATCH_SIZE_VALUES=(1 2 4)

# Count total experiments
total_experiments=$((${#TP_VALUES[@]} * ${#MAX_PROMPT_LENGTH_VALUES[@]} * ${#BATCH_SIZE_VALUES[@]}))
current_experiment=0

echo "=========================================="
echo "Starting batch experiments"
echo "Total experiments: ${total_experiments}"
echo "TP values: ${TP_VALUES[@]}"
echo "Max prompt length values: ${MAX_PROMPT_LENGTH_VALUES[@]}"
echo "Batch size values: ${BATCH_SIZE_VALUES[@]}"
echo "=========================================="

# Nested loops to iterate through all combinations
for tp in "${TP_VALUES[@]}"; do
    for max_prompt_length in "${MAX_PROMPT_LENGTH_VALUES[@]}"; do
        for batch_size in "${BATCH_SIZE_VALUES[@]}"; do
            current_experiment=$((current_experiment + 1))
            
            echo ""
            echo "=========================================="
            echo "Experiment ${current_experiment}/${total_experiments}"
            echo "TP=${tp}, max_prompt_length=${max_prompt_length}, batch_size=${batch_size}"
            echo "Start time: $(date)"
            echo "=========================================="
            
            # Run the rollout test script
            if "${ROLLOUT_SCRIPT}" "${tp}" "${max_prompt_length}" "${batch_size}"; then
                echo "✓ Experiment completed successfully"
            else
                echo "✗ Experiment failed with exit code $?"
                # Continue with next experiment even if this one fails
            fi
            
            echo "End time: $(date)"
            echo ""
        done
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "=========================================="

