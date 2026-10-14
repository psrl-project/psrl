#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UCX_VERSION="${UCX_VERSION:-v1.20.0}"
NIXL_VERSION="${NIXL_VERSION:-0.10.1}"
MAX_JOBS="${MAX_JOBS:-64}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
BUILD_ROOT="${NIXL_STACK_BUILD_ROOT:-${REPO_ROOT}/third_party/psrl-nixl-stack-ucx-1.20}"
UCX_SOURCE_DIR="${BUILD_ROOT}/ucx"
NIXL_SOURCE_DIR="${BUILD_ROOT}/nixl"
UCX_PREFIX="${UCX_PREFIX:-${REPO_ROOT}/third_party/ucx_1_20_rkey65535}"
NIXL_PREFIX="${NIXL_PREFIX:-${REPO_ROOT}/third_party/nixl_ucx_1_20_rkey65535_backlog4096}"
UCX_PATCH_FILE="${REPO_ROOT}/patch/ucx/ucx-1.20.0-rkey-config-uint16-dynamic.patch"
NIXL_PATCH_FILE="${REPO_ROOT}/patch/nixl/nixl.patch"
NIXL_WHEEL_DIR="${NIXL_WHEEL_DIR:-${BUILD_ROOT}/wheelhouse}"
NIXL_BUILD_WHEEL="${NIXL_BUILD_WHEEL:-false}"
NIXL_SUBPROJECT_CACHE_DIR="${NIXL_SUBPROJECT_CACHE_DIR:-}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${BUILD_ROOT}/uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BUILD_ROOT}/xdg-cache}"
export UV_NO_BUILD_ISOLATION="${UV_NO_BUILD_ISOLATION:-1}"

require_command() {
    local command_name="$1"

    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command is unavailable: ${command_name}." >&2
        exit 1
    fi
}

clone_source() {
    local repository="$1"
    local version="$2"
    local destination="$3"

    if [[ -e "${destination}" && ! -d "${destination}/.git" ]]; then
        echo "Source path exists but is not a Git repository: ${destination}." >&2
        exit 1
    fi
    if [[ ! -d "${destination}/.git" ]]; then
        git clone --depth 1 --branch "${version}" "${repository}" "${destination}"
    fi

    local expected_commit
    local actual_commit
    if ! expected_commit="$(git -C "${destination}" rev-parse "${version}^{commit}" 2>/dev/null)"; then
        echo "Existing source does not contain version ${version}: ${destination}." >&2
        exit 1
    fi
    actual_commit="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${expected_commit}" ]]; then
        echo "Existing source is not checked out at version ${version}: ${destination}." >&2
        exit 1
    fi
}

build_ucx() {
    echo "Building UCX ${UCX_VERSION}."
    clone_source "https://github.com/openucx/ucx.git" "${UCX_VERSION}" "${UCX_SOURCE_DIR}"
    cd "${UCX_SOURCE_DIR}"

    if git apply --check "${UCX_PATCH_FILE}"; then
        git apply "${UCX_PATCH_FILE}"
    elif ! git apply --reverse --check "${UCX_PATCH_FILE}"; then
        echo "UCX source does not match the expected ${UCX_VERSION} patch context." >&2
        exit 1
    fi

    local types_file="src/ucp/core/ucp_types.h"
    grep -Eq '^typedef uint16_t[[:space:]]+ucp_worker_cfg_index_t;$' "${types_file}"
    grep -Eq '^#define[[:space:]]+UCP_WORKER_MAX_EP_CONFIG[[:space:]]+UINT16_MAX$' "${types_file}"
    grep -Eq '^#define[[:space:]]+UCP_WORKER_MAX_RKEY_CONFIG[[:space:]]+UINT16_MAX$' "${types_file}"
    grep -Eq '^#define[[:space:]]+UCP_WORKER_CFG_INDEX_NULL[[:space:]]+UINT16_MAX$' "${types_file}"
    grep -q 'ucs_array_length(&worker->rkey_config)' src/ucp/core/ucp_worker.c

    ./autogen.sh
    rm -rf build
    mkdir build
    cd build
    ../configure \
        --prefix="${UCX_PREFIX}" \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --without-go \
        --with-cuda="${CUDA_PATH}" \
        --with-verbs \
        --with-dm \
        --enable-mt \
        --with-rdmacm \
        --with-mlx5 \
        --with-ib-hw-tm
    make -j "${MAX_JOBS}"
    make install-strip

    touch "${UCX_PREFIX}/.psrl-ucx-1.20-worker-config-uint16-complete"
    export PATH="${UCX_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${UCX_PREFIX}/lib/ucx:${LD_LIBRARY_PATH:-}"
    export PKG_CONFIG_PATH="${UCX_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${UCX_PREFIX}/lib/ucx:${LD_LIBRARY_PATH:-}" \
        "${UCX_PREFIX}/bin/ucx_info" -v
}

embed_nixl_python_runtime() {
    local site so shim
    local rpath="\$ORIGIN/../../../../lib64"

    for so in "${NIXL_PREFIX}"/lib/python*/site-packages/nixl_cu12/_bindings*.so; do
        [[ -f "${so}" ]] || continue
        patchelf --force-rpath --set-rpath "${rpath}" "${so}"
    done

    for site in "${NIXL_PREFIX}"/lib/python*/site-packages; do
        [[ -d "${site}/nixl_cu12" ]] || continue
        mkdir -p "${site}/nixl"
        shim="${site}/nixl/__init__.py"
        cat > "${shim}" <<'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the source-built NIXL 0.10.1 CUDA 12 bindings instead of pip nixl 1.5.0."""

import ctypes
import os
import sys
from importlib import import_module
from pathlib import Path

_PREFIX = Path(__file__).resolve().parents[4]
_LIB64 = _PREFIX / "lib64"
_PLUGIN_DIR = _LIB64 / "plugins"

if _PLUGIN_DIR.is_dir():
    os.environ.setdefault("NIXL_PLUGIN_DIR", str(_PLUGIN_DIR))

# Absolute CDLL pins SONAME resolution before pip 1.5.0 libnixl.so can win.
for _name in ("libserdes.so", "libnixl_common.so", "libstream.so", "libnixl_build.so", "libnixl.so"):
    _path = _LIB64 / _name
    if _path.is_file():
        ctypes.CDLL(str(_path), mode=ctypes.RTLD_GLOBAL)

_backend = import_module("nixl_cu12")
for _sub_name in ("_api", "_bindings", "_utils", "logging"):
    _module = import_module(f"{_backend.__name__}.{_sub_name}")
    sys.modules[f"nixl.{_sub_name}"] = _module
    setattr(sys.modules[__name__], _sub_name, _module)
    for _attr in dir(_module):
        if not _attr.startswith("_"):
            setattr(sys.modules[__name__], _attr, getattr(_module, _attr))
EOF
    done
}

apply_nixl_patch() {
    cd "${NIXL_SOURCE_DIR}"
    local asio_wrap="subprojects/asio.wrap"
    if [[ -f "${asio_wrap}" ]]; then
        sed -i \
            's#^source_url = https://sourceforge.net/projects/asio/.*#source_url = https://github.com/mesonbuild/wrapdb/releases/download/asio_1.30.2-2/asio-1.30.2.tar.gz#' \
            "${asio_wrap}"
        grep -Fxq \
            'source_url = https://github.com/mesonbuild/wrapdb/releases/download/asio_1.30.2-2/asio-1.30.2.tar.gz' \
            "${asio_wrap}"
    fi

    if git apply --check "${NIXL_PATCH_FILE}"; then
        git apply "${NIXL_PATCH_FILE}"
    elif ! git apply --reverse --check "${NIXL_PATCH_FILE}"; then
        echo "NIXL source does not match the expected ${NIXL_VERSION} patch context." >&2
        exit 1
    fi

    grep -q 'listen(socketFd, 4096)' src/utils/stream/metadata_stream.cpp
    grep -q 'err_handling_mode = UCP_ERR_HANDLING_MODE_NONE' src/plugins/ucx/ucx_backend.cpp
    grep -q 'ucx_err_mode_to_string(UCP_ERR_HANDLING_MODE_NONE)' src/plugins/ucx/ucx_utils.cpp
    grep -q 'NIXL_THREAD_SYNC_RW' src/bindings/python/nixl_bindings.cpp
    grep -q 'init\["num_workers"\] = str(nixl_conf.num_workers)' src/api/python/_api.py
}

build_nixl() {
    echo "Building NIXL ${NIXL_VERSION}."
    clone_source "https://github.com/ai-dynamo/nixl.git" "${NIXL_VERSION}" "${NIXL_SOURCE_DIR}"
    apply_nixl_patch
    cd "${NIXL_SOURCE_DIR}"

    local wrap_args=()
    if [[ -n "${NIXL_SUBPROJECT_CACHE_DIR}" ]]; then
        if [[ ! -d "${NIXL_SUBPROJECT_CACHE_DIR}" ]]; then
            echo "NIXL subproject cache is unavailable: ${NIXL_SUBPROJECT_CACHE_DIR}." >&2
            exit 1
        fi
        mkdir -p subprojects/packagecache
        cp -a "${NIXL_SUBPROJECT_CACHE_DIR}/." subprojects/packagecache/
        wrap_args+=(--wrap-mode=nodownload)
    fi

    rm -rf build
    meson setup build \
        --prefix="${NIXL_PREFIX}" \
        "${wrap_args[@]}" \
        -Dbuild_docs=false \
        -Denable_plugins=UCX,POSIX,GDS,GDS_MT \
        -Ducx_path="${UCX_PREFIX}" \
        -Dinstall_headers=true \
        -Ddisable_gds_backend=false
    meson compile -C build -j "${MAX_JOBS}"
    meson install -C build --no-rebuild
    embed_nixl_python_runtime

    if [[ "${NIXL_BUILD_WHEEL}" == "true" ]]; then
        if ! python -c 'import mesonpy' 2>/dev/null; then
            echo "NIXL_BUILD_WHEEL=true requires meson-python in the active environment." >&2
            exit 1
        fi
        mkdir -p "${NIXL_WHEEL_DIR}"
        python -m pip wheel \
            --no-deps \
            --no-build-isolation \
            --wheel-dir "${NIXL_WHEEL_DIR}" \
            --config-settings=setup-args="-Ducx_path=${UCX_PREFIX}" \
            --config-settings=setup-args="-Dbuild_docs=false" \
            --config-settings=setup-args="-Denable_plugins=UCX,POSIX,GDS,GDS_MT" \
            --config-settings=setup-args="-Ddisable_gds_backend=false" \
            .
    elif [[ "${NIXL_BUILD_WHEEL}" != "false" ]]; then
        echo "NIXL_BUILD_WHEEL must be true or false, got ${NIXL_BUILD_WHEEL}." >&2
        exit 1
    fi

    touch "${NIXL_PREFIX}/.psrl-nixl-${NIXL_VERSION}-complete"
}

main() {
    if [[ "${UCX_VERSION}" != "v1.20.0" ]]; then
        echo "The PSRL UCX patch is pinned to UCX v1.20.0." >&2
        exit 1
    fi
    if [[ "${NIXL_VERSION}" != "0.10.1" ]]; then
        echo "The PSRL compatibility patch is pinned to NIXL 0.10.1." >&2
        exit 1
    fi

    require_command git
    require_command make
    require_command meson
    require_command python
    require_command patchelf
    mkdir -p "${BUILD_ROOT}"

    build_ucx
    build_nixl

    echo "Installed UCX ${UCX_VERSION} at ${UCX_PREFIX}."
    echo "Installed NIXL ${NIXL_VERSION} at ${NIXL_PREFIX}."
    if [[ "${NIXL_BUILD_WHEEL}" == "true" ]]; then
        echo "Built the NIXL wheel in ${NIXL_WHEEL_DIR}."
    fi
}

main "$@"
