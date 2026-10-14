#!/bin/bash
# set -v

# Save the configured Docker image as a distributable archive.
apply_dockerhub_mirror() {
    local ref="$1"
    local m="${DOCKERHUB_MIRROR:-}"
    [ -n "$m" ] || { printf '%s' "$ref"; return 0; }
    m="${m#/}"; m="${m%/}"
    case "$ref" in
    "$m"/*|docker.io/*) printf '%s' "$ref"; return 0 ;;
    esac
    local first="${ref%%/*}"
    if [[ "$ref" == */* ]]; then
        if [[ "$first" == *.* || "$first" == localhost* ]]; then
            printf '%s' "$ref"
        else
            printf '%s/%s' "$m" "$ref"
        fi
    else
        printf '%s/library/%s' "$m" "$ref"
    fi
}

if [ -z "$DOCKER_IMAGE_DIR" ] || [ -z "$DOCKER_IMAGE_FILE" ] || [ -z "$DOCKER_IMAGE_TAG" ]; then
    echo "[docker_install.sh] Required env vars not set."
    echo "Usage:"
    echo "  DOCKER_IMAGE_DIR=/path/to/dir DOCKER_IMAGE_FILE=my.tar DOCKER_IMAGE_TAG=python:3.11 \\"
    echo "    ./docker_install.sh"
    exit 1
fi

OUT_PATH="$DOCKER_IMAGE_DIR/$DOCKER_IMAGE_FILE"
DOCKER_INSTALL_METHOD=${DOCKER_INSTALL_METHOD:-docker}
PULL_REF=$(apply_dockerhub_mirror "$DOCKER_IMAGE_TAG")

mkdir -p "$DOCKER_IMAGE_DIR" || exit 1

case "$DOCKER_INSTALL_METHOD" in
docker)
    echo "=== docker pull $PULL_REF ==="
    docker pull "$PULL_REF" || exit 1
    echo "=== docker save -> $OUT_PATH ==="
    docker save -o "$OUT_PATH" "$PULL_REF" || exit 1
    ;;
skopeo)
    SKOPEO_BIN="${SKOPEO:-}"
    if [ -z "$SKOPEO_BIN" ] && command -v skopeo >/dev/null 2>&1; then
        SKOPEO_BIN=$(command -v skopeo)
    fi
    if [ -z "$SKOPEO_BIN" ] && [ -x /usr/bin/skopeo ]; then
        SKOPEO_BIN=/usr/bin/skopeo
    fi
    if [ -z "$SKOPEO_BIN" ] || [ ! -x "$SKOPEO_BIN" ]; then
        echo "Error: skopeo not found. Install with: dnf -y install skopeo --nobest"
        echo "       (use --nobest if Docker CE containerd.io is installed), or set SKOPEO=/path/to/skopeo"
        echo "       Current PATH: $PATH"
        exit 1
    fi
    echo "=== skopeo copy docker://$PULL_REF -> docker-archive:$OUT_PATH ==="
    "$SKOPEO_BIN" copy "docker://$PULL_REF" "docker-archive:$OUT_PATH" || exit 1
    ;;
crane)
    if ! command -v crane >/dev/null 2>&1; then
        echo "Error: crane not found. Install crane or use DOCKER_INSTALL_METHOD=docker."
        exit 1
    fi
    echo "=== crane pull $PULL_REF -> $OUT_PATH ==="
    crane pull "$PULL_REF" "$OUT_PATH" || exit 1
    ;;
*)
    echo "Error: unknown DOCKER_INSTALL_METHOD=$DOCKER_INSTALL_METHOD (use docker, skopeo, or crane)"
    exit 1
    ;;
esac

echo "=== Install completed ==="
