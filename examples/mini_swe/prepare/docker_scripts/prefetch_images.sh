#!/usr/bin/env bash
# Pull Docker images referenced by a PSRL dataset.
# Usage: `prefetch_images.sh SOURCE [options]`
set -euo pipefail

PARQUET=""
IMAGES_FILE=""
ONLY_LIST=""
WORKERS=4
DRY_RUN=0
METHOD=""
IMAGE_DIR=""
DO_LOAD=0
FORCE=0
MIRROR_OVERRIDE=""
MIRRORS_LIST=""
NO_DIRECT_FALLBACK=0
RETRIES=3
LOG_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet)   PARQUET="$2";         shift 2 ;;
        --images)    IMAGES_FILE="$2";     shift 2 ;;
        --only)      ONLY_LIST="$2";       shift 2 ;;
        --workers)   WORKERS="$2";         shift 2 ;;
        --method)    METHOD="$2";          shift 2 ;;
        --image-dir) IMAGE_DIR="$2";       shift 2 ;;
        --load)      DO_LOAD=1;            shift ;;
        --mirror)    MIRROR_OVERRIDE="$2"; shift 2 ;;
        --mirrors)   MIRRORS_LIST="$2";    shift 2 ;;
        --no-direct-fallback) NO_DIRECT_FALLBACK=1; shift ;;
        --retries)   RETRIES="$2";         shift 2 ;;
        --log-dir)   LOG_DIR_OVERRIDE="$2"; shift 2 ;;
        --force)     FORCE=1;              shift ;;
        --dry-run)   DRY_RUN=1;            shift ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Validate image source (exactly one of parquet/images/only).
SOURCES=0
[[ -n "$PARQUET"     ]] && SOURCES=$((SOURCES+1))
[[ -n "$IMAGES_FILE" ]] && SOURCES=$((SOURCES+1))
[[ -n "$ONLY_LIST"   ]] && SOURCES=$((SOURCES+1))
if [[ "$SOURCES" -ne 1 ]]; then
    echo "ERROR: exactly one of --parquet / --images / --only is required." >&2
    exit 1
fi

# Load proxy settings when the shared environment exists.
source "${PSRL_WORKSPACE:-$HOME}/env/env_311.sh" 2>/dev/null || true

# Prefer `skopeo` because it supports user space proxying.
if [[ -z "$METHOD" ]]; then
    if command -v skopeo >/dev/null 2>&1; then
        METHOD=skopeo
    else
        METHOD=docker
    fi
fi

case "$METHOD" in
    skopeo|docker) ;;
    *) echo "ERROR: --method must be skopeo or docker (got: $METHOD)" >&2; exit 1 ;;
esac

# Mirror priority is the explicit list, single override, environment, then Docker Hub.
# An empty entry selects direct Docker Hub access.
MIRRORS_ARR=()
if [[ -n "$MIRRORS_LIST" ]]; then
    IFS=',' read -r -a MIRRORS_ARR <<< "$MIRRORS_LIST"
elif [[ -n "$MIRROR_OVERRIDE" ]]; then
    MIRRORS_ARR=("$MIRROR_OVERRIDE")
elif [[ -n "${DOCKERHUB_MIRROR:-}" ]]; then
    MIRRORS_ARR=("$DOCKERHUB_MIRROR")
fi
if [[ "$NO_DIRECT_FALLBACK" -ne 1 ]]; then
    MIRRORS_ARR+=("")
fi
if [[ ${#MIRRORS_ARR[@]} -eq 0 ]]; then
    MIRRORS_ARR=("")  # at least one attempt (direct docker.io)
fi
# Keep the legacy single-mirror env var set to the first non-empty mirror,
# so helper functions that only read $DOCKERHUB_MIRROR still DWIM.
export DOCKERHUB_MIRROR="${MIRRORS_ARR[0]:-}"
# Encode the list as a newline-separated string for export to worker shells.
MIRRORS_ENC=$(printf '%s\n' "${MIRRORS_ARR[@]}")
export MIRRORS_ENC

if [[ -n "$IMAGE_DIR" ]]; then
    if [[ "$METHOD" != "skopeo" ]]; then
        echo "ERROR: --image-dir requires --method skopeo." >&2
        exit 1
    fi
    mkdir -p "$IMAGE_DIR"
fi

if [[ $DO_LOAD -eq 1 && -z "$IMAGE_DIR" ]]; then
    echo "WARN: --load has no effect without --image-dir (skopeo writes directly to dockerd in that mode)." >&2
fi

export METHOD IMAGE_DIR DO_LOAD FORCE RETRIES

# Per-image log dir. Default: `<prepare>/_prefetch_logs/` (one level up from
# this script's `docker_scripts/` dir). Override with --log-dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -n "$LOG_DIR_OVERRIDE" ]]; then
    LOG_DIR="$LOG_DIR_OVERRIDE"
else
    LOG_DIR="$PREPARE_DIR/_prefetch_logs"
fi
mkdir -p "$LOG_DIR"
export LOG_DIR

echo "=== prefetch_images ==="
if [[ -n "$PARQUET" ]]; then
    echo "  source  : parquet=$PARQUET"
elif [[ -n "$IMAGES_FILE" ]]; then
    echo "  source  : images=$IMAGES_FILE"
else
    echo "  source  : --only (inline)"
fi
echo "  method  : $METHOD"
echo "  workers : $WORKERS"
_m_pretty=""
for _m in "${MIRRORS_ARR[@]}"; do
    _m_pretty+="${_m:-docker.io}, "
done
echo "  mirrors : ${_m_pretty%, }"
unset _m _m_pretty
echo "  out-dir : ${IMAGE_DIR:-<dockerd>}"
echo "  log-dir : $LOG_DIR"
echo "  retries : $RETRIES"
echo "  load    : $DO_LOAD"
echo "  force   : $FORCE"
echo

if [[ -n "$PARQUET" ]]; then
    echo "--- reading parquet: $PARQUET ---"
    IMAGES=$(python - "$PARQUET" <<'EOF'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_parquet(path)
images = set()
for row in df.itertuples():
    ei = row.extra_info
    if isinstance(ei, dict):
        so = ei.get("sandbox_overrides", {})
        img = so.get("environment", {}).get("image", "")
    else:
        img = ""
    if img:
        images.add(img)
for img in sorted(images):
    print(img)
EOF
)
elif [[ -n "$IMAGES_FILE" ]]; then
    if [[ "$IMAGES_FILE" == "-" ]]; then
        echo "--- reading images from stdin ---"
        raw=$(cat)
    else
        echo "--- reading images file: $IMAGES_FILE ---"
        raw=$(cat "$IMAGES_FILE")
    fi
    # Strip comments/blank lines, dedupe, keep stable order.
    IMAGES=$(printf '%s\n' "$raw" | sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | awk 'NF && !seen[$0]++')
else
    echo "--- reading images from --only ---"
    IMAGES=$(printf '%s\n' "$ONLY_LIST" | tr ',' '\n' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | awk 'NF && !seen[$0]++')
fi

# Normalize Docker Hub prefixes before deduplication.
# Equivalent references then share one cache key.
IMAGES=$(printf '%s\n' "$IMAGES" | sed -E 's#^(docker\.io|index\.docker\.io)/##' | awk 'NF && !seen[$0]++')

TOTAL=$(echo "$IMAGES" | grep -c . || true)
echo "Found $TOTAL unique images."

if [[ $DRY_RUN -eq 1 ]]; then
    echo "$IMAGES"
    echo "(dry-run, no images pulled)"
    exit 0
fi

# --- Parallel worker helpers ---

# Normalize default registry prefixes before mirror rewriting.
normalize_image_ref() {
    local ref="$1"
    case "$ref" in
        docker.io/*)       ref="${ref#docker.io/}" ;;
        index.docker.io/*) ref="${ref#index.docker.io/}" ;;
    esac
    printf '%s' "$ref"
}

apply_dockerhub_mirror() {
    local ref="$1"
    local m="${DOCKERHUB_MIRROR:-}"
    # Normalize default registry prefixes before detecting custom registries.
    ref=$(normalize_image_ref "$ref")
    [[ -n "$m" ]] || { printf '%s' "$ref"; return 0; }
    m="${m#/}"; m="${m%/}"
    case "$ref" in
        "$m"/*) printf '%s' "$ref"; return 0 ;;
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

# Turn an image reference into a safe tar basename.
image_to_tar_name() {
    local img="$1"
    local name="${img//\//__}"
    name="${name//:/__}"
    printf '%s.tar' "$name"
}

# Detect whether this skopeo supports `--retry-times` (added in v1.2).
SKOPEO_RETRY_FLAG=""
if command -v skopeo >/dev/null 2>&1; then
    if skopeo copy --help 2>&1 | grep -q -- '--retry-times'; then
        SKOPEO_RETRY_FLAG="--retry-times 3"
    fi
fi
export SKOPEO_RETRY_FLAG

export -f normalize_image_ref apply_dockerhub_mirror image_to_tar_name

# Retry a command while appending output and returning the command's status.
run_with_log() {
    # Append all retries to `LOG_FILE`.
    # Remove `RW_PRE_CLEANUP` before attempts so partial archives cannot poison retries.
    local log="$1"; shift
    local attempt rc
    local n="${RETRIES:-3}"
    for (( attempt = 1; attempt <= n; attempt++ )); do
        if [[ -n "${RW_PRE_CLEANUP:-}" ]]; then
            rm -f "$RW_PRE_CLEANUP"
        fi
        {
            echo "----- attempt $attempt/$n @ $(date -Iseconds) : $* -----"
            if [[ -n "${RW_PRE_CLEANUP:-}" ]]; then
                echo "      (pre-attempt cleanup: rm -f $RW_PRE_CLEANUP)"
            fi
        } >> "$log"
        # pipefail needed so `cmd | tee` surfaces cmd's failure.
        ( set -o pipefail; "$@" 2>&1 | tee -a "$log" ) && return 0
        rc=$?
        echo "  [retry $attempt/$n] rc=$rc  (see $log)" >&2
        sleep $(( attempt * 2 ))
    done
    return "$rc"
}
export -f run_with_log

# Require a nonempty Docker archive containing `manifest.json`.
verify_docker_archive() {
    local tar="$1"
    [[ -s "$tar" ]] || return 1
    # Read the full archive so truncation errors propagate from `tar`.
    local names
    names=$(tar -tf "$tar" 2>/dev/null) || return 1
    # Must contain the docker-archive manifest.
    grep -qx 'manifest.json' <<< "$names" || return 1
    return 0
}
export -f verify_docker_archive

# Reset the log before each `pull_image` invocation.
init_log() {
    # usage: init_log LOG_FILE [HEADER_LINE...]
    local log="$1"; shift
    : > "$log"
    if [[ $# -gt 0 ]]; then
        printf '%s\n' "$@" >> "$log"
    fi
}
export -f init_log

# Rewrite IMG using a specific mirror host. Empty host means docker.io (no rewrite).
ref_for_mirror() {
    local img="$1" m="$2"
    DOCKERHUB_MIRROR="$m" apply_dockerhub_mirror "$img"
}
export -f ref_for_mirror

# Read the mirror list exported by the parent shell.
_mirrors_from_env() {
    local arr=()
    while IFS= read -r line; do arr+=("$line"); done <<< "$MIRRORS_ENC"
    printf '%s\n' "${arr[@]}"
}
export -f _mirrors_from_env

pull_image() {
    set -o pipefail
    local img="$1"
    local tar_name
    tar_name=$(image_to_tar_name "$img")
    local log_path="$LOG_DIR/${tar_name%.tar}.log"

    local tar_path=""
    if [[ -n "${IMAGE_DIR:-}" ]]; then
        tar_path="$IMAGE_DIR/$tar_name"
    fi

    # Reuse only structurally valid archives.
    # Invalid archives are removed and pulled again.
    if [[ "$METHOD" == "skopeo" && -n "$tar_path" && -e "$tar_path" && "$FORCE" -ne 1 ]]; then
        if verify_docker_archive "$tar_path"; then
            echo "  [cached-tar] $img  ($tar_path)"
            init_log "$log_path" \
                "========================================================" \
                "Already cached $img." \
                "  tar      : $tar_path ($(du -h "$tar_path" | cut -f1))" \
                "  verified : tar -tf passed, manifest.json present" \
                "  checked  : $(date -Iseconds)" \
                "========================================================"
            if [[ "$DO_LOAD" -eq 1 ]]; then
                if docker image inspect "$img" >/dev/null 2>&1; then
                    echo "  [loaded-cached] $img"
                    echo "(image already loaded into dockerd, skipping docker load)" >> "$log_path"
                    return 0
                fi
                echo "  [loading] $img  <-  $tar_path"
                init_log "$log_path.load" \
                    "========================================================" \
                    "  docker load $img  <-  $tar_path" \
                    "  started @ $(date -Iseconds)" \
                    "========================================================"
                if ! run_with_log "$log_path.load" docker load -i "$tar_path"; then
                    echo "[FAILED]: Docker load failed for $img. Log: $log_path.load." >&2
                    return 1
                fi
                echo "  [loaded]  $img"
            fi
            return 0
        else
            echo "[corrupt] $img ($tar_path). Deleting and pulling again." >&2
            local sz
            sz=$(du -h "$tar_path" 2>/dev/null | cut -f1)
            rm -f "$tar_path"
            # Fall through to the normal pull path. Log reset will include
            # a note about the corruption.
            CORRUPT_NOTE="previous tar was corrupt (size=$sz), removed and re-pulling"
        fi
    fi

    # Reset the main log file before real work so we do not mix stale content
    # from previous runs.
    init_log "$log_path" \
        "========================================================" \
        "  prefetch $img" \
        "  started @ $(date -Iseconds), method=$METHOD" \
        "========================================================"
    if [[ -n "${CORRUPT_NOTE:-}" ]]; then
        echo "  NOTE: $CORRUPT_NOTE" >> "$log_path"
        unset CORRUPT_NOTE
    fi

    # Try each mirror in order until one succeeds.
    local mirrors=()
    while IFS= read -r m; do mirrors+=("$m"); done < <(_mirrors_from_env)

    local mi=0 total_mirrors=${#mirrors[@]}
    echo "(attempting $total_mirrors mirror(s))" >> "$log_path"

    # Remove partial archives if the worker is interrupted.
    # Clear the trap after a successful or explicit cleanup.
    if [[ "$METHOD" == "skopeo" && -n "$tar_path" ]]; then
        # Exit after cleaning an archive interrupted by a signal.
        # shellcheck disable=SC2064
        trap "rm -f '$tar_path'; echo '[interrupted] $img (tar cleaned)' >&2; exit 130" INT TERM
    fi

    for m in "${mirrors[@]}"; do
        mi=$((mi+1))
        local src
        src=$(ref_for_mirror "$img" "$m")
        local tag="[${mi}/${total_mirrors} mirror=${m:-docker.io}]"
        {
            echo
            echo "-------------------- $tag src=$src --------------------"
        } >> "$log_path"

        case "$METHOD" in
            skopeo)
                if [[ -n "$tar_path" ]]; then
                    echo "  [pulling] $img  $tag  ->  $tar_path"
                    # Remove partial archives before each retry.
                    if RW_PRE_CLEANUP="$tar_path" run_with_log "$log_path" \
                            skopeo copy $SKOPEO_RETRY_FLAG \
                            "docker://$src" "docker-archive:$tar_path:$img"; then
                        echo "  [saved]   $img  ($tar_path)  via ${m:-docker.io}"
                        break
                    fi
                    rm -f "$tar_path"
                    echo "  (cleaned up partial tar $tar_path)" >> "$log_path"
                else
                    echo "  [pulling] $img  $tag  (skopeo -> docker-daemon)"
                    if run_with_log "$log_path" \
                            skopeo copy $SKOPEO_RETRY_FLAG \
                            "docker://$src" "docker-daemon:$img"; then
                        echo "  [done]    $img  via ${m:-docker.io}"
                        return 0
                    fi
                fi
                ;;
            docker)
                echo "  [pulling] $img  $tag  (docker pull $src)"
                if run_with_log "$log_path" docker pull "$src"; then
                    [[ "$src" != "$img" ]] && docker tag "$src" "$img"
                    echo "  [done]    $img  via ${m:-docker.io}"
                    return 0
                fi
                ;;
        esac
        echo "  [miss]    $img  via ${m:-docker.io}"
    done

    # Validate the archive before optional loading.
    if [[ "$METHOD" == "skopeo" && -n "$tar_path" ]]; then
        if [[ ! -s "$tar_path" ]] || ! verify_docker_archive "$tar_path"; then
            # Remove incomplete archives so later runs cannot treat them as cached.
            if [[ -e "$tar_path" ]]; then
                echo "  (scrubbing incomplete/corrupt tar $tar_path)" >> "$log_path"
                rm -f "$tar_path"
            fi
            trap - INT TERM
            echo "[FAILED]: All mirrors failed for $img. Log: $log_path." >&2
            return 1
        fi
        # The verified archive no longer needs a cleanup trap.
        trap - INT TERM

        if [[ "$DO_LOAD" -eq 1 ]]; then
            echo "  [loading] $img  <-  $tar_path"
            init_log "$log_path.load" \
                "========================================================" \
                "  docker load $img  <-  $tar_path" \
                "  started @ $(date -Iseconds)" \
                "========================================================"
            if ! run_with_log "$log_path.load" docker load -i "$tar_path"; then
                # docker load failed but the tar is known-good (verified
                # above), so KEEP the tar for manual retry / inspection.
                echo "[FAILED]: Docker load failed for $img. Archive: $tar_path. Log: $log_path.load." >&2
                return 1
            fi
            docker image inspect "$img" >/dev/null 2>&1 || \
                docker tag "$(ref_for_mirror "$img" "")" "$img" 2>/dev/null || true
            echo "  [loaded]  $img"
        fi
        return 0
    fi

    echo "[FAILED]: All ${total_mirrors} mirrors failed for $img. Log: $log_path." >&2
    return 1
}
export -f pull_image

# --- Cache check ---

is_cached() {
    local img="$1"
    [[ "$FORCE" -eq 1 ]] && return 1
    if [[ -n "${IMAGE_DIR:-}" ]]; then
        local tar_path="$IMAGE_DIR/$(image_to_tar_name "$img")"
        # Only treat as cached if the tar is structurally complete. A
        # truncated/EOF'd tar from a previous failed run must not be skipped.
        if verify_docker_archive "$tar_path"; then
            # If we also need it loaded, tar alone is not enough.
            if [[ "$DO_LOAD" -eq 1 ]]; then
                docker image inspect "$img" >/dev/null 2>&1
            else
                return 0
            fi
        else
            return 1
        fi
    else
        docker image inspect "$img" >/dev/null 2>&1
    fi
}

# Dispatch every image through `pull_image` so cached items also receive logs.
TO_PULL=()
CACHED_IMAGES=()
ALL_IMAGES=()
while IFS= read -r img; do
    [[ -z "$img" ]] && continue
    ALL_IMAGES+=("$img")
    if is_cached "$img"; then
        CACHED_IMAGES+=("$img")
    else
        TO_PULL+=("$img")
    fi
done <<< "$IMAGES"

echo
echo "Summary: ${#CACHED_IMAGES[@]} cached, ${#TO_PULL[@]} to pull, ${#ALL_IMAGES[@]} total (${WORKERS} workers)."
if [[ ${#ALL_IMAGES[@]} -eq 0 ]]; then
    echo "No images to process."
    exit 0
fi

if command -v parallel > /dev/null 2>&1; then
    printf '%s\n' "${ALL_IMAGES[@]}" | parallel --jobs "$WORKERS" pull_image {}
else
    printf '%s\n' "${ALL_IMAGES[@]}" | xargs -P "$WORKERS" -I{} bash -c 'pull_image "$@"' _ {}
fi

echo "=== prefetch_images: done ==="
