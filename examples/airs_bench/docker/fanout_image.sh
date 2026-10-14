#!/bin/bash
# Distribute the AIRS-Bench image to cluster nodes via the shared filesystem.
#
# Mirrors the approach in examples/retool/docker_scripts/docker_copy.sh: save once
# to shared storage, then docker load on every target node.
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-psrl/airs-bench-agent:latest}"
TAR_DIR="${TAR_DIR:-/apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data/images}"
TAR_PATH="${TAR_DIR}/airs-bench-agent.tar"
HOSTS="${HOSTS:-}"

if [[ -z "${HOSTS}" ]]; then
    echo "ERROR: set HOSTS to a space-separated list of node IPs." >&2
    echo "Example: HOSTS=\"28.49.16.220 29.162.247.148\" bash $0" >&2
    exit 1
fi

mkdir -p "${TAR_DIR}"

if [[ ! -f "${TAR_PATH}" ]]; then
    echo "=== Saving ${IMAGE_TAG} to ${TAR_PATH} ==="
    docker save "${IMAGE_TAG}" -o "${TAR_PATH}"
fi
ls -lh "${TAR_PATH}"

for host in ${HOSTS}; do
    echo "=== Loading image on ${host} ==="
    pssh -H "${host}" -t 3600 -i "docker load -i ${TAR_PATH} && docker images ${IMAGE_TAG}"
done

echo "=== Fan-out complete ==="
