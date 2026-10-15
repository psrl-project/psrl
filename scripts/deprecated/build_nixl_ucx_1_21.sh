#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NIXL_VERSION="${NIXL_VERSION:-0.10.1}"
MAX_JOBS="${MAX_JOBS:-32}"
BUILD_ROOT="${NIXL_BUILD_ROOT:-/tmp/psrl-nixl-${NIXL_VERSION}}"
SOURCE_DIR="${NIXL_SOURCE_DIR:-${BUILD_ROOT}/nixl}"
UCX_PREFIX="${UCX_PREFIX:-${REPO_ROOT}/third_party/ucx_1_21}"
PREFIX="${NIXL_PREFIX:-${REPO_ROOT}/third_party/nixl}"
DESTDIR="${NIXL_DESTDIR:-}"
WHEEL_DIR="${NIXL_WHEEL_DIR:-${BUILD_ROOT}/wheelhouse}"
BUILD_WHEEL="${NIXL_BUILD_WHEEL:-false}"
PATCH_FILE="${REPO_ROOT}/patch/nixl/nixl-0.10.1-ucx-1.21-build.patch"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/psrl-uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/psrl-xdg-cache}"
export UV_NO_BUILD_ISOLATION="${UV_NO_BUILD_ISOLATION:-1}"

if [[ "${NIXL_VERSION}" != "0.10.1" ]]; then
    echo "The PSRL compatibility patch is pinned to NIXL 0.10.1" >&2
    exit 1
fi
if [[ ! -x "${UCX_PREFIX}/bin/ucx_info" ]]; then
    echo "UCX is not installed at ${UCX_PREFIX}" >&2
    exit 1
fi

mkdir -p "${BUILD_ROOT}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git clone --depth 1 --branch "${NIXL_VERSION}" \
        https://github.com/ai-dynamo/nixl.git "${SOURCE_DIR}"
fi
cd "${SOURCE_DIR}"
git checkout --force "${NIXL_VERSION}" >/dev/null

if git apply --check "${PATCH_FILE}"; then
    git apply "${PATCH_FILE}"
elif ! git apply --reverse --check "${PATCH_FILE}"; then
    echo "NIXL source does not match the expected 0.10.1 patch context" >&2
    exit 1
fi
grep -q 'NIXL_THREAD_SYNC_RW' src/bindings/python/nixl_bindings.cpp
grep -q 'init\["num_workers"\] = str(nixl_conf.num_workers)' src/api/python/_api.py

rm -rf build
meson setup build \
    --prefix="${PREFIX}" \
    -Dbuild_docs=false \
    -Ducx_path="${UCX_PREFIX}" \
    -Dinstall_headers=true \
    -Ddisable_gds_backend=false
meson compile -C build -j "${MAX_JOBS}"
if [[ -z "${DESTDIR}" ]]; then
    meson install -C build --no-rebuild
else
    DESTDIR="${DESTDIR}" meson install -C build --no-rebuild
fi

if [[ "${BUILD_WHEEL}" == "true" ]]; then
    python -c 'import mesonpy' 2>/dev/null || {
        echo "NIXL_BUILD_WHEEL=true requires meson-python in the active environment" >&2
        exit 1
    }
    mkdir -p "${WHEEL_DIR}"
    python -m pip wheel \
        --no-deps \
        --no-build-isolation \
        --wheel-dir "${WHEEL_DIR}" \
        --config-settings=setup-args="-Ducx_path=${UCX_PREFIX}" \
        --config-settings=setup-args="-Dbuild_docs=false" \
        --config-settings=setup-args="-Ddisable_gds_backend=false" \
        .
fi

touch "${DESTDIR}${PREFIX}/.psrl-nixl-${NIXL_VERSION}-complete"

echo "Built NIXL ${NIXL_VERSION} against ${UCX_PREFIX}"
echo "prefix=${PREFIX}"
if [[ -n "${DESTDIR}" ]]; then
    echo "staged_prefix=${DESTDIR}${PREFIX}"
fi
if [[ "${BUILD_WHEEL}" == "true" ]]; then
    echo "wheel_dir=${WHEEL_DIR}"
fi
