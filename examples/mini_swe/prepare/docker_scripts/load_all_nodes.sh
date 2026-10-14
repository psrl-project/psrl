#!/usr/bin/env bash
# Load shared-filesystem Docker archives on every host with parallel SSH.
# Usage: `load_all_nodes.sh --hosts FILE --image-dir DIR [options]`
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOSTS=""
IMAGE_DIR=""
IMAGES_LIST=""
PARALLEL_PER_NODE=2
TIMEOUT=7200
SSH_USER=""
OUTDIR=""
SKIP_EXISTING=1
DRY_RUN=0
PSSH_BIN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)              HOSTS="$2"; shift 2 ;;
        --image-dir)          IMAGE_DIR="$2"; shift 2 ;;
        --images-list)        IMAGES_LIST="$2"; shift 2 ;;
        --parallel-per-node)  PARALLEL_PER_NODE="$2"; shift 2 ;;
        --timeout)            TIMEOUT="$2"; shift 2 ;;
        --user)               SSH_USER="$2"; shift 2 ;;
        --outdir)             OUTDIR="$2"; shift 2 ;;
        --skip-existing)      SKIP_EXISTING=1; shift ;;
        --force)              SKIP_EXISTING=0; shift ;;
        --dry-run)            DRY_RUN=1; shift ;;
        --pssh)               PSSH_BIN="$2"; shift 2 ;;
        -h|--help)            sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$HOSTS" && -f "$HOSTS" ]]         || { echo "ERROR: --hosts FILE is required." >&2; exit 1; }
[[ -n "$IMAGE_DIR" && -d "$IMAGE_DIR" ]] || { echo "ERROR: --image-dir DIR is required." >&2; exit 1; }

# Autodetect pssh.
if [[ -z "$PSSH_BIN" ]]; then
    for c in pssh parallel-ssh; do
        if command -v "$c" >/dev/null 2>&1; then PSSH_BIN="$c"; break; fi
    done
fi
[[ -n "$PSSH_BIN" ]] || { echo "ERROR: pssh (or parallel-ssh) not found. Install via 'pip install pssh' or your distro's pssh package." >&2; exit 1; }

# Materialize host list (strip comments / blanks).
CLEAN_HOSTS_FILE=$(mktemp)
trap 'rm -f "$CLEAN_HOSTS_FILE"' EXIT
grep -Ev '^[[:space:]]*(#|$)' "$HOSTS" > "$CLEAN_HOSTS_FILE"
NUM_HOSTS=$(wc -l < "$CLEAN_HOSTS_FILE")
[[ "$NUM_HOSTS" -gt 0 ]] || { echo "ERROR: no hosts in $HOSTS." >&2; exit 1; }

# Keep preparation logs beside other generated artifacts.
if [[ -z "$OUTDIR" ]]; then
    PREPARE_DIR="$(dirname "$HERE")"
    OUTDIR="$PREPARE_DIR/_load_logs/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTDIR"

# Tar list (optionally filtered by --images-list).
mapfile -t ALL_TARS < <(ls "$IMAGE_DIR"/*.tar 2>/dev/null)
[[ ${#ALL_TARS[@]} -gt 0 ]] || { echo "ERROR: no *.tar found in $IMAGE_DIR." >&2; exit 1; }

if [[ -n "$IMAGES_LIST" ]]; then
    # Keep only tars whose basename (minus .tar) matches a line in the list.
    declare -A WANTED
    while IFS= read -r line; do
        line="${line%%#*}"  # strip inline comments
        line="$(echo "$line" | xargs)"  # trim
        [[ -z "$line" ]] && continue
        # Accept either "image ref" (slash/colon form) or tar basename.
        tar_name="${line//\//__}"
        tar_name="${tar_name//:/__}"
        WANTED["${tar_name%.tar}"]=1
    done < "$IMAGES_LIST"

    FILTERED=()
    for t in "${ALL_TARS[@]}"; do
        b=$(basename "$t" .tar)
        [[ -n "${WANTED[$b]:-}" ]] && FILTERED+=("$t")
    done
    ALL_TARS=("${FILTERED[@]}")
    [[ ${#ALL_TARS[@]} -gt 0 ]] || { echo "ERROR: --images-list produced an empty tar set." >&2; exit 1; }
fi

NUM_TARS=${#ALL_TARS[@]}
TOTAL_BYTES=$(du -csb "${ALL_TARS[@]}" 2>/dev/null | tail -1 | cut -f1)
TOTAL_HUMAN=$(numfmt --to=iec --suffix=B "$TOTAL_BYTES" 2>/dev/null || echo "${TOTAL_BYTES}B")

echo "=== load_all_nodes ==="
echo "  pssh         : $PSSH_BIN"
echo "  hosts file   : $HOSTS ($NUM_HOSTS hosts)"
echo "  image dir    : $IMAGE_DIR"
echo "  tar count    : $NUM_TARS ($TOTAL_HUMAN total)"
echo "  per-node jobs: $PARALLEL_PER_NODE"
echo "  skip existing: $SKIP_EXISTING"
echo "  timeout      : ${TIMEOUT}s"
echo "  outdir       : $OUTDIR"
echo

# --- Remote node script ---
read -r -d '' REMOTE_SCRIPT <<'REMOTE_EOF' || true
set -u

: "${IMAGE_DIR_REMOTE:?IMAGE_DIR_REMOTE not set}"
: "${SKIP_EXISTING_REMOTE:=1}"
: "${JOBS_REMOTE:=2}"
: "${IMAGES_FILTER_REMOTE:=}"

HOST=$(hostname -s)
echo "[$HOST] start @ $(date -Iseconds)  image-dir=$IMAGE_DIR_REMOTE  jobs=$JOBS_REMOTE  skip=$SKIP_EXISTING_REMOTE"

if ! command -v docker >/dev/null 2>&1; then
    echo "[$HOST] ERROR: docker not found on PATH" >&2
    exit 127
fi

# Build list of tars (optional filter by wanted basenames).
HAS_FILTER=0
declare -A WANTED
if [[ -n "$IMAGES_FILTER_REMOTE" ]]; then
    HAS_FILTER=1
    while IFS= read -r b; do
        [[ -n "$b" ]] && WANTED["$b"]=1
    done <<< "$IMAGES_FILTER_REMOTE"
fi

mapfile -t TARS < <(ls "$IMAGE_DIR_REMOTE"/*.tar 2>/dev/null || true)
if [[ ${#TARS[@]} -eq 0 ]]; then
    echo "[$HOST] no *.tar in $IMAGE_DIR_REMOTE" >&2
    exit 1
fi
if [[ "$HAS_FILTER" -eq 1 ]]; then
    FILT=()
    for t in "${TARS[@]}"; do
        b=$(basename "$t" .tar)
        [[ -n "${WANTED[$b]:-}" ]] && FILT+=("$t")
    done
    TARS=("${FILT[@]}")
fi

TOTAL=${#TARS[@]}
echo "[$HOST] $TOTAL tar(s) to consider"

# Worker: load one tar.
load_one() {
    local tar="$1"
    local host="$2"
    local skip="$3"
    local base
    base=$(basename "$tar")

    # Pull the first RepoTag out of the archive so we can short-circuit if
    # the image is already present on this node.
    local tag=""
    tag=$(tar -xOf "$tar" manifest.json 2>/dev/null | python3 -c '
import sys, json
try:
    m = json.load(sys.stdin)
    tags = []
    for entry in m:
        tags.extend(entry.get("RepoTags") or [])
    print(tags[0] if tags else "", end="")
except Exception:
    pass
' 2>/dev/null || true)

    if [[ -n "$tag" && "$skip" == "1" ]]; then
        if docker image inspect "$tag" >/dev/null 2>&1; then
            echo "[$host] [skip]   $tag"
            return 0
        fi
    fi

    echo "[$host] [loading] $base${tag:+  (tag=$tag)}"
    if ! out=$(docker load -i "$tar" 2>&1); then
        echo "[$host] [FAILED]  $base" >&2
        echo "$out" | sed "s|^|[$host]   |" >&2
        return 1
    fi
    # Extract loaded image ref for a tidy "loaded" line.
    loaded=$(echo "$out" | sed -n 's/^Loaded image: //p; s/^Loaded image ID: //p' | tail -n 1)
    echo "[$host] [loaded]  ${loaded:-$base}"
}
export -f load_one

# Fan out N docker-loads per node using xargs.
export HOST SKIP_EXISTING_REMOTE
fail=0
printf '%s\n' "${TARS[@]}" | \
    xargs -P "$JOBS_REMOTE" -I{} bash -c \
        'load_one "$1" "$HOST" "$SKIP_EXISTING_REMOTE"' _ {} \
    || fail=1

echo "[$HOST] done  @ $(date -Iseconds)  rc=$fail"
exit $fail
REMOTE_EOF

# Encode the filter list (if any) as a newline-separated string to pass through env.
IMAGES_FILTER=""
if [[ -n "$IMAGES_LIST" ]]; then
    IMAGES_FILTER=$(for t in "${ALL_TARS[@]}"; do basename "$t" .tar; done | sort -u)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "=== DRY RUN: commands for each of $NUM_HOSTS host(s) ==="
    echo
    echo "IMAGE_DIR_REMOTE=$IMAGE_DIR  SKIP_EXISTING_REMOTE=$SKIP_EXISTING  JOBS_REMOTE=$PARALLEL_PER_NODE  bash -s <<(REMOTE_SCRIPT)"
    echo
    echo "--- hosts (first 10) ---"
    head -n 10 "$CLEAN_HOSTS_FILE"
    echo
    echo "--- tars (first 10 of $NUM_TARS) ---"
    printf '  %s\n' "${ALL_TARS[@]:0:10}"
    exit 0
fi

# pssh -x extra-ssh-args: keep it snappy and non-interactive.
SSH_OPTS=(
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=120
)
PSSH_ARGS=(
    -t "$TIMEOUT"
    -h "$CLEAN_HOSTS_FILE"
    -o "$OUTDIR/stdout"
    -e "$OUTDIR/stderr"
    -i
    -x "${SSH_OPTS[*]}"
)
[[ -n "$SSH_USER" ]] && PSSH_ARGS+=(-l "$SSH_USER")

mkdir -p "$OUTDIR/stdout" "$OUTDIR/stderr"

echo "=== launching pssh ==="
set +e
{
    echo "export IMAGE_DIR_REMOTE=$(printf %q "$IMAGE_DIR")"
    echo "export SKIP_EXISTING_REMOTE=$(printf %q "$SKIP_EXISTING")"
    echo "export JOBS_REMOTE=$(printf %q "$PARALLEL_PER_NODE")"
    echo "export IMAGES_FILTER_REMOTE=$(printf %q "$IMAGES_FILTER")"
    printf '%s\n' "$REMOTE_SCRIPT"
} | "$PSSH_BIN" "${PSSH_ARGS[@]}" -I -- bash -s
PSSH_RC=$?
set -e

echo
echo "=== pssh finished (rc=$PSSH_RC) ==="
echo "  per-host stdout : $OUTDIR/stdout/"
echo "  per-host stderr : $OUTDIR/stderr/"

# Quick per-host summary.
echo
echo "--- summary ---"
while IFS= read -r h; do
    so="$OUTDIR/stdout/$h"
    se="$OUTDIR/stderr/$h"
    loaded=0; skipped=0; failed=0
    if [[ -f "$so" ]]; then
        loaded=$(grep -c '\[loaded\]' "$so" 2>/dev/null || true)
        skipped=$(grep -c '\[skip\]'   "$so" 2>/dev/null || true)
    fi
    if [[ -f "$se" ]]; then
        failed=$(grep -c '\[FAILED\]' "$se" 2>/dev/null || true)
    fi
    printf '  %-20s  loaded=%-4d skipped=%-4d failed=%-4d\n' "$h" "$loaded" "$skipped" "$failed"
done < "$CLEAN_HOSTS_FILE"

exit "$PSSH_RC"