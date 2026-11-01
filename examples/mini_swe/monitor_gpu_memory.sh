#!/usr/bin/env bash
# Start the per-node GPU memory samplers for a training run.
#
# Usage: monitor_gpu_memory.sh <output-dir> <duration-seconds> [interval-seconds] [node ...]
#
# With no node arguments it samples the local node only. Each node writes its own
# gpu_mem_<tag>.csv and gpu_proc_<tag>.csv under <output-dir>.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <output-dir> <duration-seconds> [interval-seconds] [node ...]" >&2
    exit 2
fi

OUT=$1
DURATION=$2
INTERVAL=${3:-2}
shift 3 || true
NODES=("$@")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NODE_SCRIPT="${SCRIPT_DIR}/monitor_gpu_memory_node.sh"

mkdir -p "$OUT"

if [ "${#NODES[@]}" -eq 0 ]; then
    NODES=("${LOCAL_IP:?LOCAL_IP is not set}")
fi

for node in "${NODES[@]}"; do
    if [ "$node" = "${LOCAL_IP:-}" ]; then
        nohup bash "$NODE_SCRIPT" "$OUT" "$DURATION" "$INTERVAL" "$node" \
            >/dev/null 2>&1 &
    else
        ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "root@${node}" \
            "nohup bash '${NODE_SCRIPT}' '${OUT}' '${DURATION}' '${INTERVAL}' '${node}' >/dev/null 2>&1 &"
    fi
    echo "monitoring ${node} -> ${OUT}/gpu_mem_${node}.csv"
done
