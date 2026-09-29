#!/usr/bin/env bash
# build_harness_runtimes.sh: materialize the read-only harness runtime trees.
#
# Each harness ships as a self-contained, node-free native executable placed at
#     $PSRL_HARNESS_RUNTIME_ROOT/<kind>/bin/<executable>
#
# The harness loop bind-mounts that tree read-only into every sandbox, so no
# sandbox installs Node/npm or mutates the task image's global toolchain.
#
# Run once per shared filesystem visible to every worker (the mount source is
# resolved on the worker, like the task images).
#
# Idempotent: a tree whose executable already answers --version is left
# untouched. Pass --force to rebuild.
#
#   export PSRL_HARNESS_RUNTIME_ROOT=/shared/psrl/harness-runtimes
#   bash build_harness_runtimes.sh (all harnesses)
#
#   bash build_harness_runtimes.sh claude_code (one harness)
#
# Pinned versions (single source of truth, bump here to upgrade):
#   Claude Code 2.1.233 (linux-x64)
#
#   Codex 0.154.0 (linux-x64)
# Both come from the npm registry as platform packages. NPM_REGISTRY defaults to
#
# registry.npmjs.org. Point it at a mirror (e.g. https://mirrors.tencent.com/npm)
# when the node cannot reach npmjs.org directly.
set -euo pipefail

: "${PSRL_HARNESS_RUNTIME_ROOT:?Set PSRL_HARNESS_RUNTIME_ROOT to the shared directory that will hold the runtime trees}"
NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmjs.org}"

force=0
if [ "${1:-}" = "--force" ]; then force=1; shift; fi

CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.233}"
CLAUDE_CODE_PLATFORM="${CLAUDE_CODE_PLATFORM:-linux-x64}"
# sha256 of the linux-x64 native binary published in the 2.1.233 manifest.
CLAUDE_CODE_SHA256="${CLAUDE_CODE_SHA256:-55d281096f57d411ebbdd94dbf5e9ff3accb7c05713e37348c2c11d4b83bf9d9}"

CODEX_VERSION="${CODEX_VERSION:-0.154.0}"
CODEX_PLATFORM="${CODEX_PLATFORM:-linux-x64}"
CODEX_SHA256="${CODEX_SHA256:-}"

work=""
cleanup() { [ -n "$work" ] && rm -rf "$work"; }
trap cleanup EXIT

runtime_bin() { printf '%s/%s/bin/%s' "$PSRL_HARNESS_RUNTIME_ROOT" "$1" "$2"; }
is_ready() { "$(runtime_bin "$1" "$2")" --version >/dev/null 2>&1; }

fetch_tgz() {
    echo "  fetching $1"
    curl -fsSL --retry 3 --retry-delay 2 -o "$2" "$1"
}

new_workdir() {
    rm -rf "$work"
    work="$(mktemp -d)"
}

# Claude Code: the platform package holds the native binary at package/claude.
build_claude_code() {
    local kind="claude_code" exe="claude"
    local target
    target="$(runtime_bin "$kind" "$exe")"
    if [ "$force" != "1" ] && is_ready "$kind" "$exe"; then
        echo "skip (ready): $kind -> $target"
        return 0
    fi

    local pkg="claude-code-${CLAUDE_CODE_PLATFORM}"
    local url="${NPM_REGISTRY}/@anthropic-ai/claude-code-${CLAUDE_CODE_PLATFORM}/-/${pkg}-${CLAUDE_CODE_VERSION}.tgz"
    new_workdir
    fetch_tgz "$url" "$work/pkg.tgz"
    tar -xzf "$work/pkg.tgz" -C "$work"

    local src="$work/package/claude"
    [ -f "$src" ] || { echo "missing package/claude in $url" >&2; return 1; }

    local actual
    actual="$(sha256sum "$src" | awk '{print $1}')"
    if [ "$actual" != "$CLAUDE_CODE_SHA256" ]; then
        echo "claude binary sha256 mismatch: expected $CLAUDE_CODE_SHA256, got $actual" >&2
        return 1
    fi

    install -D -m 0755 "$src" "$target"
    echo "installed $kind $CLAUDE_CODE_VERSION -> $target ($("$target" --version))"
}

# Codex: the platform package ships a whole self-contained vendor tree
# (bin/codex + sibling resources). Keep that layout under <root>/codex.
build_codex() {
    local kind="codex" exe="codex"
    local target
    target="$(runtime_bin "$kind" "$exe")"
    if [ "$force" != "1" ] && is_ready "$kind" "$exe"; then
        echo "skip (ready): $kind -> $target"
        return 0
    fi

    local triple
    case "$CODEX_PLATFORM" in
        linux-x64) triple="x86_64-unknown-linux-musl" ;;
        linux-arm64) triple="aarch64-unknown-linux-musl" ;;
        *) echo "unsupported CODEX_PLATFORM: $CODEX_PLATFORM" >&2; return 1 ;;
    esac

    local url="${NPM_REGISTRY}/@openai/codex/-/codex-${CODEX_VERSION}-${CODEX_PLATFORM}.tgz"
    new_workdir
    fetch_tgz "$url" "$work/pkg.tgz"
    tar -xzf "$work/pkg.tgz" -C "$work"

    local vendor="$work/package/vendor/$triple"
    [ -f "$vendor/bin/codex" ] || { echo "missing vendor/$triple/bin/codex in $url" >&2; return 1; }

    if [ -n "$CODEX_SHA256" ]; then
        local actual
        actual="$(sha256sum "$vendor/bin/codex" | awk '{print $1}')"
        if [ "$actual" != "$CODEX_SHA256" ]; then
            echo "codex binary sha256 mismatch: expected $CODEX_SHA256, got $actual" >&2
            return 1
        fi
    fi

    rm -rf "${PSRL_HARNESS_RUNTIME_ROOT:?}/$kind"
    mkdir -p "$PSRL_HARNESS_RUNTIME_ROOT/$kind"
    cp -a "$vendor/." "$PSRL_HARNESS_RUNTIME_ROOT/$kind/"
    chmod 0755 "$target"
    echo "installed $kind $CODEX_VERSION -> $target ($("$target" --version))"
}

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
mkdir -p "$PSRL_HARNESS_RUNTIME_ROOT"

kinds=("$@")
if [ "${#kinds[@]}" -eq 0 ]; then
    kinds=(claude_code codex)
fi
for kind in "${kinds[@]}"; do
    case "$kind" in
        claude_code) build_claude_code ;;
        codex) build_codex ;;
        *) echo "unknown harness kind: $kind" >&2; exit 2 ;;
    esac
done
