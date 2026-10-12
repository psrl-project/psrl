#!/usr/bin/env bash
# rebake_harness_image.sh: force a re-bake of the git-purged derivative images.
#
# `bake_harness_image.sh` tags a derivative as
# `psrl/swebench-harness:<sha12(base-image)>` and SKIPS any image already baked,
#
# so a change to the bake steps alone does not replace an existing derivative.
# This script deletes the derivative(s) first and then delegates to the original
#
# bake, the supported way to roll out a changed purge.
#
# It reuses `bake_harness_image.sh` verbatim. This is a thin wrapper, not a
# second implementation of the bake steps.
#
# Usage (on each worker host that creates sandboxes):
#   bash rebake_harness_image.sh <image> (one image)
#
#   bash rebake_harness_image.sh --parquet <path>
#
# Removing a tag only untags the image. A running sandbox keeps its container.
# Multi-host clusters are node-local: re-run this on every worker host, or push
#
# the result from one host and pull it on the others.
#
# Env overrides:
#   PSRL_BAKE_IMAGE_GLOB images to delete before re-baking (default
#
# `psrl/swebench-harness:*`). The delete is by tag, so stale tags from any
# earlier formula are removed too.
#
#   PSRL_REBAKE_VERIFY=1 after each bake, run the git-leak probe from the
#   README against the new image and fail on a leak.
#
#   PSRL_REBAKE_DRY_RUN=1 print what would be removed/re-baked, change nothing.
#   PSRL_BAKE_SKIP_GIT_CLEAN, PSRL_BAKE_WORKDIR, PSRL_HARNESS_IMAGE_TAG are
#
#   forwarded to bake_harness_image.sh. An explicit PSRL_HARNESS_IMAGE_TAG
#   disables the removal step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAKE_SCRIPT="$SCRIPT_DIR/bake_harness_image.sh"

IMAGE_GLOB="${PSRL_BAKE_IMAGE_GLOB:-psrl/swebench-harness:*}"
VERIFY="${PSRL_REBAKE_VERIFY:-0}"
DRY_RUN="${PSRL_REBAKE_DRY_RUN:-0}"

[ -f "$BAKE_SCRIPT" ] || { echo "missing bake script: $BAKE_SCRIPT" >&2; exit 1; }
if [ "$DRY_RUN" != "1" ]; then
    command -v docker >/dev/null 2>&1 || { echo "docker not found on PATH" >&2; exit 1; }
fi

run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf 'would run: %s\n' "$*"
    else
        "$@"
    fi
}

# Delete every tag matching $IMAGE_GLOB. Deleting by tag (not by image id) also
# clears tags left behind by an older digest formula.
#
# No current base image maps onto those leftover tags.
remove_derivatives() {
    [ -z "${PSRL_HARNESS_IMAGE_TAG:-}" ] || return 0
    local repo="${IMAGE_GLOB%%:*}"
    local tags
    if [ "$DRY_RUN" = "1" ]; then
        echo "would remove every tag matching $IMAGE_GLOB"
        return 0
    fi
    tags="$(docker image ls --no-trunc --format '{{.Repository}}:{{.Tag}}' | grep "^${repo}:" || true)"
    if [ -z "$tags" ]; then
        echo "no derivative images match $IMAGE_GLOB"
        return 0
    fi
    while IFS= read -r tag; do
        [ -z "$tag" ] && continue
        docker image rm --force "$tag" >/dev/null
        echo "removed $tag"
    done <<<"$tags"
}

# Re-run the README probe: a baked derivative must have no remotes and no
# unreachable/refs-only objects outside HEAD.
verify_image() {
    local tag="$1"
    local workdir="${PSRL_BAKE_WORKDIR:-/testbed}"
    echo "verifying $tag"
    docker run --rm -e "PSRL_BAKE_WORKDIR=$workdir" "$tag" bash -lc \
        'cd "$PSRL_BAKE_WORKDIR" || exit 0
         remotes="$(git remote)"
         leaked="$(git rev-list --count --all --reflog --not HEAD 2>/dev/null || echo 0)"
         echo "  remotes=[$remotes] leaked=$leaked"
         [ -z "$remotes" ] && [ "$leaked" = "0" ]' \
        || { echo "leak probe FAILED for $tag" >&2; return 1; }
}

# Print the derivative tag for a base image, matching runner.py's
# `_image_digest` and the baker's `digest`.
derivative_tag() {
    printf 'psrl/swebench-harness:%s' "$(printf '%s' "$1" | sha256sum | cut -c1-12)"
}

bake_one() {
    local base_image="$1"
    local tag="${PSRL_HARNESS_IMAGE_TAG:-$(derivative_tag "$base_image")}"
    if [ "$DRY_RUN" = "1" ]; then
        echo "would rebake $base_image"
    else
        echo "rebaking $base_image"
    fi
    run bash "$BAKE_SCRIPT" "$base_image"
    if [ "$VERIFY" = "1" ] && [ "$DRY_RUN" != "1" ]; then
        verify_image "$tag"
    fi
}

case "${1:-}" in
    --parquet)
        parquet="${2:?--parquet requires a parquet path}"
        remove_derivatives
        run bash "$BAKE_SCRIPT" --parquet "$parquet"
        ;;
    "" | -*)
        echo "usage: rebake_harness_image.sh <image> | --parquet <path>" >&2
        exit 2
        ;;
    *)
        remove_derivatives
        bake_one "$1"
        ;;
esac
