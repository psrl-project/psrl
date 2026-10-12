#!/usr/bin/env bash
# bake_harness_image.sh: bake per-task git-purged derivative images.
#
# Each SWE task uses its OWN per-problem base image (the parquet's
# `sandbox_overrides.environment.image`), so a single global baked image is
#
# meaningless. From one base image, or from every unique image in a parquet,
# this script derives a derivative image:
#
#     psrl/swebench-harness:<sha12(base-image)>
#
# The derivative purges leaked git metadata (remotes / refs / reflog /
# unreachable objects), so a sandbox that starts from it can never read a future
#
# fix commit. Runner (`examples/mini_swe/runner.py`) selects the derivative when
# it exists locally and otherwise falls back to the original image plus the
#
# runtime git sanitization, so a missing bake never blocks training.
#
# The harness executable is NOT baked here. The harness loop mounts it read-only
# from the host runtime tree, so no sandbox ever installs into (or mutates) the
#
# task image's global toolchain. Build that tree once with
# `build_harness_runtimes.sh`.
#
# Usage (once per worker host that creates sandboxes):
#   bash bake_harness_image.sh <image> (one image)
#
#   bash bake_harness_image.sh --parquet <path> (all unique images)
#
# Env overrides:
#   PSRL_BAKE_WORKDIR repo workdir inside the image to git-purge (default /testbed)
#
#   PSRL_BAKE_SKIP_GIT_CLEAN=1 skip the git purge (no-op derivative)
#   PSRL_HARNESS_IMAGE_TAG explicit output tag (single-image mode only)
#
# The tag keys on the base image alone, so this script SKIPS an image already
# baked. After changing the bake steps, run rebake_harness_image.sh instead to
#
# replace the stale derivative.
set -euo pipefail

BAKE_WORKDIR="${PSRL_BAKE_WORKDIR:-/testbed}"
SKIP_GIT_CLEAN="${PSRL_BAKE_SKIP_GIT_CLEAN:-0}"

digest() { printf '%s' "$1" | sha256sum | cut -c1-12; }

# Purge leaked git metadata so the derivative can never expose a future fix
# commit. Must stay in sync with examples/mini_swe/utils/git_sanitize.py.
#
# The workdir placeholder is substituted on the host. A missing repo or a
# non-worktree is a no-op, so this is safe for images that do not bake a repo at
#
# BAKE_WORKDIR.
git_clean_script='set -u
workdir="__PSRL_BAKE_WORKDIR__"
cd "$workdir" 2>/dev/null || { echo "git-clean: $workdir missing; skipping" >&2; exit 0; }
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "git-clean: $workdir is not a git worktree; skipping" >&2
  exit 0
fi
git -c advice.detachedHead=false checkout --detach HEAD
for remote in $(git remote); do
  git remote remove "$remote"
done
git for-each-ref --format="%(refname)" refs/heads refs/remotes refs/tags refs/stash refs/notes 2>/dev/null \
  | while IFS= read -r ref; do
      [ -z "$ref" ] && continue
      git update-ref -d "$ref" || true
    done
git reflog expire --expire=now --expire-unreachable=now --all
git gc --prune=now --quiet
echo "git-clean: purged $workdir" >&2'
git_clean_script="${git_clean_script/__PSRL_BAKE_WORKDIR__/$BAKE_WORKDIR}"

bake_one() {
    local base_image="$1"
    local tag="${PSRL_HARNESS_IMAGE_TAG:-psrl/swebench-harness:$(digest "$base_image")}"
    if docker image inspect "$tag" >/dev/null 2>&1; then
        echo "skip (already baked): $tag"
        return 0
    fi

    # Serialize bakes of the same base image across concurrent invocations on
    # this host so N workers do not repack the same image at once.
    local lock_dir="/tmp/psrl-bake-$(digest "$base_image").lock"
    local waited=0
    while ! mkdir "$lock_dir" 2>/dev/null; do
        if docker image inspect "$tag" >/dev/null 2>&1; then
            echo "skip (baked by peer): $tag"
            return 0
        fi
        sleep 3
        waited=$((waited + 3))
        if [ "$waited" -gt 3600 ]; then
            echo "timed out waiting for bake lock: $lock_dir" >&2
            return 1
        fi
    done

    (
        set -e
        trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

        if docker image inspect "$tag" >/dev/null 2>&1; then
            echo "skip (baked by peer): $tag"
            exit 0
        fi
        if ! docker image inspect "$base_image" >/dev/null 2>&1; then
            echo "base image not present locally; attempting docker pull: $base_image" >&2
            docker pull "$base_image" || { echo "cannot obtain base image: $base_image" >&2; exit 1; }
        fi
        if [ "$SKIP_GIT_CLEAN" = "1" ]; then
            echo "baking $base_image (git purge skipped) -> $tag"
        else
            echo "baking $base_image + git purge -> $tag"
        fi
        container_name="psrl-harness-bake-$$"
        cid="$(docker create --name "$container_name" "$base_image" tail -f /dev/null)"
        docker start "$cid" >/dev/null
        if [ "$SKIP_GIT_CLEAN" != "1" ]; then
            if ! docker exec "$cid" bash -lc "$git_clean_script"; then
                docker rm -f "$container_name" >/dev/null 2>&1 || true
                echo "git purge failed for $base_image" >&2
                exit 1
            fi
        fi
        docker commit "$container_name" "$tag" >/dev/null
        docker rm -f "$container_name" >/dev/null 2>&1 || true
        echo "baked: $tag"
    )
}

if [ "${1:-}" = "--parquet" ]; then
    parquet="${2:?--parquet requires a parquet path}"
    images="$(
        python3 -c '
import pandas as pd, sys
df = pd.read_parquet(sys.argv[1], columns=["extra_info"])
seen = set()
for ei in df["extra_info"]:
    img = (ei.get("sandbox_overrides", {}) or {}).get("environment", {}).get("image")
    if img and img not in seen:
        seen.add(img)
        print(img)
' "$parquet"
    )"
    echo "found $(wc -l <<<"$images") unique images in $parquet"
    while IFS= read -r img; do
        [ -z "$img" ] && continue
        bake_one "$img"
    done <<<"$images"
else
    base_image="${1:-${BASE_IMAGE:-}}"
    [ -n "$base_image" ] || { echo "usage: bake_harness_image.sh <image> | --parquet <path>" >&2; exit 2; }
    bake_one "$base_image"
fi
