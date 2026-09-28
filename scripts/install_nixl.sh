#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

CUDA_PATH=${CUDA_PATH:-"/usr/local/cuda"}
MAX_JOBS=${MAX_JOBS:-32}
REQUIRED_UCX_VERSION="1.20.0"
UCX_PREFIX="/usr"
INSTALL_UCX=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
mkdir -p $THIRD_PARTY_PATH

# Prefer the UCX installed by this repository when it exists.
if [ -x "$THIRD_PARTY_PATH/ucx/bin/ucx_info" ]; then
    PATH="$THIRD_PARTY_PATH/ucx/bin:$PATH"
    export PATH
fi

if command -v ucx_info >/dev/null 2>&1; then
    echo "Detected existing UCX installation via ucx_info"
    UCX_INFO_OUTPUT=$(ucx_info -v 2>/dev/null || true)
    DETECTED_VERSION=$(echo "$UCX_INFO_OUTPUT" | grep -Eo '([0-9]+\.){2}[0-9]+' | head -n1)
    DETECTED_PREFIX=$(dirname "$(dirname "$(readlink -f "$(command -v ucx_info)")")")

    if [ -n "$DETECTED_PREFIX" ]; then
        UCX_PREFIX="$DETECTED_PREFIX"
    fi

    if [ -n "$DETECTED_VERSION" ] && [ "$(printf '%s\n%s\n' "$DETECTED_VERSION" "$REQUIRED_UCX_VERSION" | sort -V | head -n1)" = "$REQUIRED_UCX_VERSION" ]; then
        echo "UCX version $DETECTED_VERSION found at $UCX_PREFIX (>= $REQUIRED_UCX_VERSION), skipping UCX build."
        INSTALL_UCX=false
    else
        echo "UCX version $DETECTED_VERSION found at $UCX_PREFIX (< $REQUIRED_UCX_VERSION), will build UCX $REQUIRED_UCX_VERSION."
    fi
else
    echo "UCX info was not found. Building UCX $REQUIRED_UCX_VERSION."
fi

if $INSTALL_UCX; then
    echo "1. Install ucx"
    UCX_PREFIX="$THIRD_PARTY_PATH/ucx"
    mkdir -p $THIRD_PARTY_PATH/ucx_src
    pushd $THIRD_PARTY_PATH/ucx_src
    if [ ! -d ucx/.git ]; then
        git clone -b "v$REQUIRED_UCX_VERSION" https://github.com/openucx/ucx.git
    fi
    cd ucx

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
    make -j $MAX_JOBS &&            \
    make -j $MAX_JOBS install-strip &&  \
    ldconfig
    popd
    rm -rf $THIRD_PARTY_PATH/ucx_src
else
    echo "1. Skip UCX installation"
fi

# Recommend to use gcc 11.x.x, gcc-toolset-13 may have error with nixl
echo "2. Install nixl"
mkdir -p $THIRD_PARTY_PATH/nixl_src
pushd $THIRD_PARTY_PATH/nixl_src
if [ ! -d nixl/.git ]; then
    git clone -b v1.2.0 https://github.com/ai-dynamo/nixl.git
fi
cd nixl
mkdir -p build
# Disable obj backend
sed -i "s/subdir('obj')/# subdir('obj')/" "$THIRD_PARTY_PATH/nixl_src/nixl/src/plugins/meson.build"
# Disable err handling for ucp (will make NIXL READ slower 10x!)
echo "Applying nixl patch..."
meson setup build \
    --prefix=$THIRD_PARTY_PATH/nixl \
    -Dbuild_docs=false \
    -Ducx_path=$UCX_PREFIX \
    -Dinstall_headers=true \
    -Ddisable_gds_backend=false
cd build
ninja -j $MAX_JOBS
ninja install -j $MAX_JOBS
cd ..
NIXL_PYTHON_SITE="$THIRD_PARTY_PATH/nixl/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
PYTHON_SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$NIXL_PYTHON_SITE" > "$PYTHON_SITE/nixl-local.pth"
python -m pip install --no-deps build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
popd
rm -rf $THIRD_PARTY_PATH/nixl_src

echo "Successfully installed all packages for nixl"
