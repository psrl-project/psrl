#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

PATCH_DIR_NAMES=()
LIBRARY_NAMES=()

register_library() {
    PATCH_DIR_NAMES+=("$1")
    LIBRARY_NAMES+=("$2")
}

# Register new patch directories here. The second argument must be the
# installed Python distribution name accepted by importlib.metadata.
register_libraries() {
    register_library "lm_cache" "lmcache"
    register_library "megatron_bridge" "megatron-bridge"
    register_library "transfer_queue" "TransferQueue"
    register_library "transformer_engine" "transformer-engine"
    register_library "verl" "verl"
    register_library "vllm" "vllm"
}

show_usage() {
    cat <<EOF
Usage: $0 PATCH_DIR [OPTIONS] [PATCH_FILE]
       $0 --list-libraries

Arguments:
  PATCH_DIR       Registered subdirectory under $SCRIPT_DIR
  PATCH_FILE      Patch file to apply; relative paths are resolved in PATCH_DIR

Options:
  --force         Apply even if the library is not installed editably
  --auto          Select the most recently modified patch file
  --list          List patch files available for PATCH_DIR
  --list-libraries
                  List registered patch directories and distribution names
  -h, --help      Show this help message

Without PATCH_FILE or --auto, patches are matched in this order:
  installed commit, installed version, most recently modified patch
EOF
}

list_libraries() {
    local index

    echo "Registered patch directories:"
    for ((index = 0; index < ${#PATCH_DIR_NAMES[@]}; index++)); do
        printf "  %-20s %s\n" "${PATCH_DIR_NAMES[$index]}" "${LIBRARY_NAMES[$index]}"
    done
}

resolve_library_name() {
    local requested_dir="$1"
    local index

    for ((index = 0; index < ${#PATCH_DIR_NAMES[@]}; index++)); do
        if [ "${PATCH_DIR_NAMES[$index]}" = "$requested_dir" ]; then
            echo "${LIBRARY_NAMES[$index]}"
            return 0
        fi
    done

    return 1
}

list_patches() {
    local patch
    local found=false

    echo "Available patch files in $PATCH_DIR:"
    for patch in "$PATCH_DIR"/*.patch; do
        if [ -f "$patch" ]; then
            echo "  - $(basename "$patch")"
            found=true
        fi
    done

    if [ "$found" = false ]; then
        echo "  (none)"
    fi
}

latest_patch() {
    local patch
    local latest=""

    for patch in "$PATCH_DIR"/*.patch; do
        if [ -f "$patch" ] && { [ -z "$latest" ] || [ "$patch" -nt "$latest" ]; }; then
            latest="$patch"
        fi
    done

    if [ -n "$latest" ]; then
        echo "$latest"
    fi

    return 0
}

find_patch_file() {
    local installed_version="$1"
    local installed_commit="$2"
    local patch
    local patch_name
    local version_core
    local version_prefix

    if [ -n "$installed_commit" ]; then
        echo "Detected $LIBRARY_NAME commit: $installed_commit" >&2
        for patch in "$PATCH_DIR"/*.patch; do
            [ -f "$patch" ] || continue
            patch_name="$(basename "$patch" .patch)"
            if [[ "$patch_name" =~ ^[a-f0-9]{7,40}$ ]] && \
                { [[ "$installed_commit" == "$patch_name"* ]] || [[ "$patch_name" == "$installed_commit"* ]]; }; then
                echo "Found commit match: $(basename "$patch")" >&2
                echo "$patch"
                return 0
            fi
        done
    fi

    if [ -n "$installed_version" ]; then
        echo "Detected $LIBRARY_NAME version: $installed_version" >&2
        version_core="${installed_version%%+*}"
        for version_prefix in "$installed_version" "$version_core"; do
            [ -n "$version_prefix" ] || continue
            for patch in "$PATCH_DIR"/*.patch; do
                [ -f "$patch" ] || continue
                patch_name="$(basename "$patch" .patch)"
                if [[ ! "$patch_name" =~ ^[a-f0-9]{7,40}$ ]] && [[ "$patch_name" == *"$version_prefix"* ]]; then
                    echo "Found version match: $(basename "$patch")" >&2
                    echo "$patch"
                    return 0
                fi
            done
        done

        if [[ "$version_core" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
            for version_prefix in "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}" \
                "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"; do
                for patch in "$PATCH_DIR"/*.patch; do
                    [ -f "$patch" ] || continue
                    patch_name="$(basename "$patch" .patch)"
                    if [[ ! "$patch_name" =~ ^[a-f0-9]{7,40}$ ]] && [[ "$patch_name" == *"$version_prefix"* ]]; then
                        echo "Found version match: $(basename "$patch")" >&2
                        echo "$patch"
                        return 0
                    fi
                done
            done
        fi
    fi

    patch="$(latest_patch)"
    if [ -n "$patch" ]; then
        echo "Using latest patch file: $(basename "$patch")" >&2
        echo "$patch"
        return 0
    fi

    return 1
}

get_library_metadata() {
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Error: Python executable '$PYTHON_BIN' was not found." >&2
        return 1
    fi

    "$PYTHON_BIN" - "$LIBRARY_NAME" <<'PY'
import json
import os
import sys
from importlib import metadata
from urllib.parse import unquote, urlparse

name = sys.argv[1]
try:
    dist = metadata.distribution(name)
except metadata.PackageNotFoundError:
    print("||")
    raise SystemExit

version = dist.version or ""
root = os.path.realpath(str(dist.locate_file("")))
editable = False
direct_url_text = dist.read_text("direct_url.json")
if direct_url_text:
    direct_url = json.loads(direct_url_text)
    editable = direct_url.get("dir_info", {}).get("editable", False)
    parsed_url = urlparse(direct_url.get("url", ""))
    if editable and parsed_url.scheme == "file":
        root = os.path.realpath(unquote(parsed_url.path))

print(f"{version}|{root}|{str(editable).lower()}")
PY
}

register_libraries

if [ "$#" -eq 0 ]; then
    show_usage
    exit 1
fi

case "$1" in
    -h|--help)
        show_usage
        exit 0
        ;;
    --list-libraries)
        list_libraries
        exit 0
        ;;
esac

PATCH_DIR_NAME="$1"
shift
LIBRARY_NAME="$(resolve_library_name "$PATCH_DIR_NAME" || true)"
PATCH_DIR="$SCRIPT_DIR/$PATCH_DIR_NAME"

if [ -z "$LIBRARY_NAME" ]; then
    echo "Error: Patch directory '$PATCH_DIR_NAME' is not registered."
    echo ""
    list_libraries
    exit 1
fi

if [ ! -d "$PATCH_DIR" ]; then
    echo "Error: Registered patch directory $PATCH_DIR does not exist."
    exit 1
fi

FORCE_MODE=false
AUTO_MODE=false
PATCH_FILE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --force)
            FORCE_MODE=true
            ;;
        --auto)
            AUTO_MODE=true
            ;;
        --list)
            list_patches
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *.patch)
            if [ -n "$PATCH_FILE" ]; then
                echo "Error: Multiple patch files specified."
                exit 1
            fi
            PATCH_FILE="$1"
            ;;
        *)
            echo "Error: Unknown option or patch file '$1'."
            show_usage
            exit 1
            ;;
    esac
    shift
done

if [ -n "$PATCH_FILE" ]; then
    if [[ "$PATCH_FILE" = /* ]]; then
        PATCH_FILE_PATH="$PATCH_FILE"
    else
        PATCH_FILE_PATH="$PATCH_DIR/$PATCH_FILE"
    fi
elif [ "$AUTO_MODE" = true ]; then
    PATCH_FILE_PATH="$(latest_patch)"
else
    IFS="|" read -r INSTALLED_VERSION INSTALL_ROOT IS_EDITABLE < <(get_library_metadata)
    if [ -z "$INSTALL_ROOT" ] || [ ! -d "$INSTALL_ROOT" ]; then
        echo "Error: Could not find the installed $LIBRARY_NAME distribution."
        exit 1
    fi
    INSTALLED_COMMIT="$(git -C "$INSTALL_ROOT" rev-parse HEAD 2>/dev/null || true)"
    PATCH_FILE_PATH="$(find_patch_file "$INSTALLED_VERSION" "$INSTALLED_COMMIT" || true)"
fi

if [ -z "$PATCH_FILE_PATH" ] || [ ! -f "$PATCH_FILE_PATH" ]; then
    echo "Error: Could not find patch file '${PATCH_FILE_PATH:-}'."
    echo ""
    list_patches
    exit 1
fi

if [ -z "${INSTALL_ROOT:-}" ]; then
    IFS="|" read -r INSTALLED_VERSION INSTALL_ROOT IS_EDITABLE < <(get_library_metadata)
fi

if [ -z "$INSTALL_ROOT" ] || [ ! -d "$INSTALL_ROOT" ]; then
    echo "Error: Could not find the installed $LIBRARY_NAME distribution."
    exit 1
fi

echo "Using patch file: $(basename "$PATCH_FILE_PATH")"
echo "Found $LIBRARY_NAME installation root: $INSTALL_ROOT"

if [ "$IS_EDITABLE" != true ]; then
    echo "Warning: $LIBRARY_NAME is not installed in editable mode."
    echo "Install it with 'python -m pip install -e /path/to/source' or use --force."
    if [ "$FORCE_MODE" != true ]; then
        exit 1
    fi
    echo "Applying patch forcefully..."
fi

TARGET_FILES=()
while IFS= read -r file; do
    TARGET_FILES+=("$file")
done < <(sed -n 's|^diff --git a/\(.*\) b/.*$|\1|p' "$PATCH_FILE_PATH" | sort -u)

if [ "${#TARGET_FILES[@]}" -eq 0 ]; then
    echo "Error: No target files found in patch file."
    exit 1
fi

echo "Target files:"
for file in "${TARGET_FILES[@]}"; do
    echo "  - $file"
    if [ ! -e "$INSTALL_ROOT/$file" ]; then
        echo "    Warning: $INSTALL_ROOT/$file does not exist; the patch may create it."
    fi
done

echo "Applying patch to $LIBRARY_NAME..."
if ! git -C "$INSTALL_ROOT" apply --check "$PATCH_FILE_PATH"; then
    echo "Error: Patch cannot be applied cleanly."
    echo "It may already be applied, target a different version, or conflict with local changes."
    exit 1
fi

git -C "$INSTALL_ROOT" apply "$PATCH_FILE_PATH"
echo "Patch application completed successfully."
