#!/usr/bin/env bash
# Prefetch Docker images for the SWE-Gym Subset (100 instances, 2 repos).
# The subset requires about 100 unique images.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_DIR="$(dirname "$SCRIPT_DIR")"

WORKERS=${SWE_PREFETCH_WORKERS:-1}

bash "$SCRIPT_DIR/prefetch_images.sh" \
    --parquet "$PREPARE_DIR/../data/swe_gym_subset_100/train.parquet" \
    --workers "$WORKERS" \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe_gym_subset \
    --method skopeo \
    --mirrors docker.1ms.run,docker.1panel.live,proxy.vvvv.ee,lispy.org,registry.cyou \
    --no-direct-fallback \
    --retries 10

bash "$SCRIPT_DIR/load_all_nodes.sh" \
    --hosts ${PSRL_WORKSPACE}/hosts/16GPUs \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe_gym_subset
