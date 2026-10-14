#!/bin/bash
# Convert a per-rank checkpoint to DCP across all hosts.
# Usage: `INPUT_DIR=<path> OUTPUT_DIR=<path> bash scripts/convert_perrank_to_dcp.sh [hostfile]`.

set -e

# === Log file ===
LOG_DIR="${PSRL_WORKSPACE}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/convert_perrank_to_dcp_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"
echo ""

# === Parallel config (must match training) ===
TP_SIZE=4
PP_SIZE=1
CP_SIZE=1
EP_SIZE=1

# === Node config ===
NPROC_PER_NODE=8
MASTER_PORT=${MASTER_PORT:-29500}

# === Checkpoint paths ===
INPUT_DIR=${INPUT_DIR:-""}
OUTPUT_DIR=${OUTPUT_DIR:-""}

# === Environment + hostfile ===
env_file="${PSRL_WORKSPACE}/env/psrl.sh"
source "${env_file}"

HOSTFILE=${1:-"${PSRL_WORKSPACE}/hosts/16GPUs"}

if [ ! -f "${HOSTFILE}" ]; then
    echo "Error: hostfile not found: ${HOSTFILE}"
    exit 1
fi
if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "Error: INPUT_DIR and OUTPUT_DIR must be set."
    echo "Usage: INPUT_DIR=<path> OUTPUT_DIR=<path> bash scripts/convert_perrank_to_dcp.sh [hostfile]"
    exit 1
fi

mapfile -t hosts < "${HOSTFILE}"
NNODES=${#hosts[@]}
if [ "${NNODES}" -eq 0 ]; then
    echo "Error: empty hostfile: ${HOSTFILE}"
    exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-${hosts[0]}}

echo "Converting checkpoint: ${INPUT_DIR} → ${OUTPUT_DIR}"
echo "Config: TP=${TP_SIZE}, PP=${PP_SIZE}, CP=${CP_SIZE}, EP=${EP_SIZE}"
echo "Nodes (${NNODES}): ${hosts[*]}"
echo "MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# Prefix concurrent SSH output with its source host.
for i in "${!hosts[@]}"; do
    host="${hosts[$i]}"
    echo "Launching on ${host} (NODE_RANK=${i})..."
    ssh "${host}" \
        "source ${env_file} && \
        torchrun \
            --nproc_per_node=${NPROC_PER_NODE} \
            --nnodes=${NNODES} \
            --node_rank=${i} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${PSRL_WORKSPACE}/psrl_agent/scripts/convert_perrank_to_dcp.py \
            --input_dir ${INPUT_DIR} \
            --output_dir ${OUTPUT_DIR} \
            --tp_size ${TP_SIZE} \
            --pp_size ${PP_SIZE} \
            --cp_size ${CP_SIZE} \
            --ep_size ${EP_SIZE}" \
        2>&1 | awk -v h="${host}" '{ print "[" h "] " $0; fflush() }' &
done

wait
echo ""
echo "Done! DCP checkpoint saved to: ${OUTPUT_DIR}"
