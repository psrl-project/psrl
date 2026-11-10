#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

CUDA_PATH=${CUDA_PATH:-"/usr/local/cuda"}
MAX_JOBS=${MAX_JOBS:-32}
REQUIRED_UCX_VERSION="1.20.0"
UCX_GIT_REF="${UCX_GIT_REF:-v1.20.0}"
NIXL_VERSION="0.10.1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
UCX_PREFIX="${UCX_PREFIX:-$THIRD_PARTY_PATH/ucx_1_20_rkey65535}"
NIXL_PREFIX="${NIXL_PREFIX:-$THIRD_PARTY_PATH/nixl_ucx_1_20_rkey65535_backlog4096}"
UCX_SOURCE_DIR="${UCX_SOURCE_DIR:-$THIRD_PARTY_PATH/ucx_1_20_rkey65535_src}"
NIXL_SOURCE_DIR="${NIXL_SOURCE_DIR:-$THIRD_PARTY_PATH/nixl_0_10_1_ucx_1_20_rkey65535_backlog4096_src}"
NIXL_SUBPROJECT_CACHE_DIR="${NIXL_SUBPROJECT_CACHE_DIR:-}"
UCX_PATCH_FILE="$PSRL_PATH/patch/ucx/ucx-1.20.0-rkey-config-uint16-dynamic.patch"
NIXL_PATCH_FILE="$PSRL_PATH/patch/nixl/nixl.patch"
UCX_INSTALL_MARKER="$UCX_PREFIX/.psrl-ucx-1.20-worker-config-uint16-complete"
mkdir -p "$THIRD_PARTY_PATH"

if [[ -x "$UCX_PREFIX/bin/ucx_info" ]] && \
   [[ -f "$UCX_INSTALL_MARKER" ]] && \
   LD_LIBRARY_PATH="$UCX_PREFIX/lib:$UCX_PREFIX/lib/ucx" \
       "$UCX_PREFIX/bin/ucx_info" -v 2>/dev/null | grep -q "Library version: $REQUIRED_UCX_VERSION"; then
    echo "1. Patched UCX $REQUIRED_UCX_VERSION is already installed at $UCX_PREFIX"
else
    echo "UCX info was not found. Building UCX $REQUIRED_UCX_VERSION."
fi

    echo "Applying UCX dynamic worker-config patch..."
    git apply --check "$UCX_PATCH_FILE"
    git apply "$UCX_PATCH_FILE"
    grep -Eq '^typedef uint16_t[[:space:]]+ucp_worker_cfg_index_t;$' src/ucp/core/ucp_types.h
    grep -Eq '^#define[[:space:]]+UCP_WORKER_MAX_EP_CONFIG[[:space:]]+UINT16_MAX$' src/ucp/core/ucp_types.h
    grep -Eq '^#define[[:space:]]+UCP_WORKER_MAX_RKEY_CONFIG[[:space:]]+UINT16_MAX$' src/ucp/core/ucp_types.h
    grep -Eq '^#define[[:space:]]+UCP_WORKER_CFG_INDEX_NULL[[:space:]]+UINT16_MAX$' src/ucp/core/ucp_types.h
    grep -q 'ucs_array_length(&worker->rkey_config)' src/ucp/core/ucp_worker.c

    # Checking Mellanox NICs
    MLX_OPTS=""
    if lspci | grep -i mellanox > /dev/null || command -v ibstat > /dev/null; then
        echo "Mellanox NIC detected, adding Mellanox-specific options"
        MLX_OPTS="--with-rdmacm \
                  --with-mlx5   \
                  --with-ib-hw-tm"
    fi

    ./autogen.sh && ./configure     \
        --prefix=$UCX_PREFIX        \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --without-go                \
        --with-cuda=$CUDA_PATH      \
        --with-verbs                \
        --with-dm                   \
        --enable-mt                 \
        $MLX_OPTS &&                \
    make -j "$MAX_JOBS" &&         \
    make -j "$MAX_JOBS" install-strip
    popd
    rm -rf "$UCX_SOURCE_DIR"
fi

touch "$UCX_INSTALL_MARKER"
export PATH="$UCX_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$UCX_PREFIX/lib:$UCX_PREFIX/lib/ucx:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$UCX_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# Recommend to use gcc 11.x.x, gcc-toolset-13 may have error with nixl
echo "2. Install nixl"
mkdir -p $THIRD_PARTY_PATH/nixl_src
pushd $THIRD_PARTY_PATH/nixl_src
git clone -b v1.2.0 https://github.com/ai-dynamo/nixl.git
cd nixl
mkdir -p build
MESON_WRAP_ARGS=()
if [[ -n "$NIXL_SUBPROJECT_CACHE_DIR" ]]; then
    if [[ ! -d "$NIXL_SUBPROJECT_CACHE_DIR" ]]; then
        echo "NIXL subproject cache not found: $NIXL_SUBPROJECT_CACHE_DIR" >&2
        exit 1
    fi
    cp -a "$NIXL_SUBPROJECT_CACHE_DIR/." subprojects/
    MESON_WRAP_ARGS+=(--wrap-mode=nodownload)
fi
# Disable obj backend
sed -i "s/subdir('obj')/# subdir('obj')/" src/plugins/meson.build

# Fix metadata_stream: acceptClient() and acceptClientsAsync() both spam ERROR
# logs ("Cannot accept client connection: Bad file descriptor / Socket operation
# on non-socket") after agent teardown, because the background listener thread
# keeps calling accept() on an fd that has already been closed and potentially
# reused by Ray or another process.
# The underlying thread/fd lifecycle in nixl is complex to fix correctly without
# larger refactoring. The pragmatic fix is to demote both NIXL_PERROR log lines
# to NIXL_DEBUG: the noise disappears at the default log level (WARN), and the
# messages remain visible when NIXL_LOG_LEVEL=DEBUG is set for debugging.
STREAM_CPP="$NIXL_SOURCE_DIR/src/utils/stream/metadata_stream.cpp"
sed -i 's/NIXL_PERROR << "Cannot accept client connection"/NIXL_DEBUG << "Cannot accept client connection"/g' "$STREAM_CPP"
echo "metadata_stream.cpp patched: demoted 'Cannot accept' logs to DEBUG."

# Disable err handling for ucp (will make NIXL READ slower 10x!)
echo "Applying nixl patch..."
git apply --whitespace=nowarn "$NIXL_PATCH_FILE"
grep -q 'listen(socketFd, 4096)' "$STREAM_CPP"
grep -q 'NIXL_THREAD_SYNC_RW' src/bindings/python/nixl_bindings.cpp
grep -q 'init\["num_workers"\] = str(nixl_conf.num_workers)' src/api/python/_api.py
meson setup build \
    "${MESON_WRAP_ARGS[@]}" \
    --prefix="$NIXL_PREFIX" \
    -Dbuild_docs=false \
    -Ducx_path="$UCX_PREFIX" \
    -Dinstall_headers=true \
    -Ddisable_gds_backend=false
cd build
ninja -j "$MAX_JOBS"
ninja install -j "$MAX_JOBS"
cd ..
python -m pip install --force-reinstall --no-deps \
    build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
popd
rm -rf "$NIXL_SOURCE_DIR"

echo "Successfully installed UCX $REQUIRED_UCX_VERSION (ep/rkey configs=UINT16_MAX) and NIXL $NIXL_VERSION (backlog=4096)"
