#!/usr/bin/env bash
# Sample GPU memory on one node while a training run is in flight.
#
# Runs on the node itself, so the samples are taken where the processes live and the
# output can land on the shared filesystem. Each sample is two nvidia-smi queries:
# per-GPU totals, and the per-process memory that lets us tell the trainer ranks from
# the vLLM engines on the same node.
#
# Usage: monitor_gpu_memory_node.sh <output-dir> <duration-seconds> <interval-seconds> <node-tag>
set -euo pipefail

OUT=$1
DURATION=$2
INTERVAL=$3
TAG=$4

mkdir -p "$OUT"
GPU_FILE="$OUT/gpu_mem_${TAG}.csv"
PROC_FILE="$OUT/gpu_proc_${TAG}.csv"
UUID_FILE="$OUT/gpu_uuid_${TAG}.csv"

echo "ts,node,gpu,used_mib,total_mib" > "$GPU_FILE"
echo "ts,node,gpu_uuid,pid,used_mib" > "$PROC_FILE"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits > "$UUID_FILE"

END=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$END" ]; do
    ts=$(date +%s.%N)
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null |
        awk -v ts="$ts" -v node="$TAG" -F', ' '{print ts "," node "," $1 "," $2 "," $3}' >> "$GPU_FILE"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
        awk -v ts="$ts" -v node="$TAG" -F', ' '{print ts "," node "," $1 "," $2 "," $3}' >> "$PROC_FILE"
    sleep "$INTERVAL"
done
