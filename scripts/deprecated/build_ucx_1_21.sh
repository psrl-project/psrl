#!/usr/bin/env bash
set -euo pipefail

# Build upstream UCX 1.21, which uses a uint16 worker-config index and dynamic
# rkey-config array with UCP_WORKER_MAX_RKEY_CONFIG=UINT16_MAX. Runtime uses
# LD_LIBRARY_PATH, so no root privileges or ldconfig are required.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UCX_VERSION="${UCX_VERSION:-v1.21.0}"
MAX_JOBS="${MAX_JOBS:-32}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
BUILD_ROOT="${UCX_BUILD_ROOT:-/tmp/psrl-ucx-1.21}"
PREFIX="${UCX_PREFIX:-${REPO_ROOT}/third_party/ucx_1_21}"
DESTDIR="${UCX_DESTDIR:-}"
SOURCE_DIR="${BUILD_ROOT}/ucx"

mkdir -p "${BUILD_ROOT}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git clone --depth 1 --branch "${UCX_VERSION}" https://github.com/openucx/ucx.git "${SOURCE_DIR}"
fi
cd "${SOURCE_DIR}"
git checkout --force "${UCX_VERSION}" >/dev/null

# Refuse to publish an older or structurally different UCX by accident.
types_file="src/ucp/core/ucp_types.h"
grep -Eq '^typedef uint16_t[[:space:]]+ucp_worker_cfg_index_t;$' "${types_file}"
grep -Eq '^#define[[:space:]]+UCP_WORKER_MAX_RKEY_CONFIG[[:space:]]+UINT16_MAX$' "${types_file}"
grep -Eq '^#define[[:space:]]+UCP_WORKER_CFG_INDEX_NULL[[:space:]]+UINT16_MAX$' "${types_file}"
grep -q 'ucs_array_length(&worker->rkey_config)' src/ucp/core/ucp_worker.c

./autogen.sh
rm -rf build
mkdir build
cd build
../configure \
    --prefix="${PREFIX}" --enable-shared --disable-static \
    --disable-doxygen-doc --enable-optimizations \
    --enable-cma --enable-devel-headers --without-go \
    --with-cuda="${CUDA_PATH}" --with-verbs --with-dm --enable-mt \
    --with-rdmacm --with-mlx5 --with-ib-hw-tm
make -j "${MAX_JOBS}"
make ${DESTDIR:+DESTDIR="${DESTDIR}"} install-strip
INSTALL_PREFIX="${DESTDIR}${PREFIX}"
touch "${INSTALL_PREFIX}/.psrl-ucx-1.21-complete"
echo "Built upstream UCX ${UCX_VERSION} with UCP_WORKER_MAX_RKEY_CONFIG=UINT16_MAX"
echo "prefix=${PREFIX}"
if [[ -z "${DESTDIR}" ]]; then
    LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib/ucx:${LD_LIBRARY_PATH:-}" \
        "${PREFIX}/bin/ucx_info" -v | head -8
else
    echo "staged_prefix=${INSTALL_PREFIX}"
fi
