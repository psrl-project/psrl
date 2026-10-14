#!/bin/bash
# set -v

# Copy the configured shared image archive to every Docker node.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=docker_common.sh
. "$SCRIPT_DIR/docker_common.sh"

if [ -z "$DOCKER_NODE_IPS" ] || [ -z "$DOCKER_IMAGE_DIR" ] || [ -z "$DOCKER_IMAGE_FILE" ]; then
    echo "[docker_copy.sh] Required env vars not set."
    echo "Usage:"
    echo "  DOCKER_NODE_IPS=ip1:8,ip2:8 DOCKER_NODE_NUM=8 \\"
    echo "    DOCKER_IMAGE_DIR=/path/to/dir DOCKER_IMAGE_FILE=my.tar \\"
    echo "    ./docker_copy.sh"
    exit 1
fi

docker_cluster_init || exit 1
# pssh -H expects space-separated hosts, not comma-separated
hosts_str="${DOCKER_CLUSTER_HOSTS[*]}"

echo "=== Copying tar to all nodes in parallel ==="
pssh -t 3600 -H "$hosts_str" -i "cp $DOCKER_IMAGE_DIR/$DOCKER_IMAGE_FILE /tmp/"

echo "=== Loading docker image on all nodes in parallel ==="
# Detect the loaded image reference before applying an optional target tag.
REMOTE_CMD='
set -e
TAR=/tmp/'"$DOCKER_IMAGE_FILE"'
NEW_TAG='"$DOCKER_IMAGE_TAG"'

load_out=$(docker load -i "$TAR")
echo "$load_out"
rm "$TAR"

if [ -n "$NEW_TAG" ]; then
    src=$(printf "%s\n" "$load_out" | sed -n "s/^Loaded image: //p; s/^Loaded image ID: //p" | tail -n 1)
    [ -n "$src" ] || { echo "[docker_copy.sh] Could not detect loaded image ref"; exit 1; }
    docker tag "$src" "$NEW_TAG"
    echo "Tagged $src -> $NEW_TAG"
fi
'
pssh -t 3600 -H "$hosts_str" -i "$REMOTE_CMD"

echo "=== Copy completed ==="
