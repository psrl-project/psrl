#!/usr/bin/env bash
# Reference invocation: pull every image referenced by the 1k smith parquet
# into a shared-FS tar cache, then fan `docker load` out to every node.
#
# Run from this directory (examples/mini_swe/prepare/docker_scripts/):
#   bash prefetch_example.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_DIR="$(dirname "$SCRIPT_DIR")"

bash "$SCRIPT_DIR/prefetch_images.sh" \
    --parquet "$PREPARE_DIR/../data/swe_smith_py_1k/train.parquet" \
    --workers 4 \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe_train \
    --method skopeo \
    --mirrors docker.1ms.run,docker.1panel.live,proxy.vvvv.ee,lispy.org,registry.cyou \
    --no-direct-fallback \
    --retries 10

bash "$SCRIPT_DIR/load_all_nodes.sh" \
    --hosts ${PSRL_WORKSPACE}/hosts/64GPUs \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe_eval