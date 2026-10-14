# syntax=docker/dockerfile:1.7

# PSRL GPU development/runtime image.
# Build from the repository root:
#   docker build --progress=plain -f docker/Dockerfile -t psrl:latest .

ARG CUDA_VERSION=13.0.3
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG CUDA_VERSION
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.8.22
ARG TORCH_CUDA_WHEEL_TAG=cu130
ARG MAX_JOBS=32
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;10.0;12.0+PTX"
ARG FLASH_ATTN_CUDA_ARCHS="80;90;100;120"
ARG INSTALL_FLASH_ATTN_3=1

# Source revisions are build arguments so downstream releases can update one
# component without rewriting the installation graph. Patch compatibility must
# be revalidated whenever one of these defaults changes.
ARG UCX_REF=v1.20.0
ARG NIXL_REF=v1.2.0
ARG VLLM_REF=v0.22.0
ARG VERL_REF=e5ca4acb
ARG TRANSFER_QUEUE_REF=434f8c476b4be24bc087e6e95070e64efcc739f9
ARG LMCACHE_REF=v0.4.6
ARG MEGATRON_REF=c049020
ARG MEGATRON_BRIDGE_REF=94d1870
ARG TMS_REF=d64a6394d1e09c613fab90260054cecc2684586d
ARG SMG_REF=psrl-dev
ARG PSRL_COMMIT=main
ARG APEX_REF=10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
ARG TRANSFORMER_ENGINE_REF=v2.17
ARG DEEPEP_REF=73b6ea4
ARG GDRCOPY_REF=v2.5.1
ARG NVSHMEM_VERSION=3.3.24
ARG MOONCAKE_VERSION=0.3.10.post2
ARG FLASH_ATTN_VERSION=2.8.3.post1
# No tagged FlashAttention-3 release contains the upstream CUDA 13 toolchain
# path yet. Pin a `main` commit that uses CUDA 13's system `nvcc`, adds the
# relocated CCCL headers, and vendors a CUDA 13-compatible CUTLASS revision.
ARG FLASH_ATTN_3_REF=df61ab6c4a0fb1f94f1f43b2a23479a0ab92b8ab
ARG FLASHINFER_VERSION=0.6.11.post3

ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    CUDA_HOME=/usr/local/cuda \
    NVSHMEM_DIR=/root/deepep/nvshmem \
    PSRL_HOME=/home/psrl \
    PSRL_WORKSPACE=/home/workspace \
    PSRL_THIRD_PARTY=/root \
    PSRL_PATCH_DIR=/root/patch \
    UCX_HOME=/root/ucx \
    NIXL_HOME=/root/nixl-install \
    PROTOC=/usr/bin/protoc \
    PROTOC_INCLUDE=/usr/include \
    VIRTUAL_ENV=/root/venv \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PATH=/root/venv/bin:/root/.cargo/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/root/deepep/nvshmem/lib:/root/nixl-install/lib:/root/nixl-install/lib64:/root/ucx/lib:/root/ucx/lib/ucx:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PKG_CONFIG_PATH=/root/ucx/lib/pkgconfig:${PKG_CONFIG_PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /root

# Keep the devel toolchain in the published image: PSRL dependencies are
# editable source checkouts and several runtime paths JIT-compile CUDA kernels.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        ffmpeg \
        git \
        git-lfs \
        gnupg \
        ibverbs-providers \
        ibverbs-utils \
        jq \
        libaio-dev \
        libcap-dev \
        libcurl4-openssl-dev \
        libgl1 \
        libibverbs-dev \
        libnuma-dev \
        libprotobuf-dev \
        librdmacm-dev \
        libsm6 \
        libssl-dev \
        libtool \
        libtool-bin \
        libxext6 \
        meson \
        ninja-build \
        numactl \
        openssh-client \
        openssh-server \
        patchelf \
        pciutils \
        pkg-config \
        protobuf-compiler \
        pssh \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        rdma-core \
        rsync \
        unzip \
        vim \
        wget && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/bin/parallel-ssh /usr/bin/pssh && \
    python -m venv ${VIRTUAL_ENV} && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install "uv==${UV_VERSION}"

ARG DOCKER_APT_MIRROR=https://mirrors.tencentyun.com/docker-ce
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    codename="$(. /etc/os-release && printf '%s' "${VERSION_CODENAME}")" && \
    test -n "${codename}" && \
    install -m 0755 -d /usr/share/keyrings && \
    curl -fsSL "${DOCKER_APT_MIRROR}/linux/ubuntu/gpg" -o /tmp/docker.asc && \
    gpg --dearmor --yes -o /usr/share/keyrings/docker-archive-keyring.gpg /tmp/docker.asc && \
    chmod a+r /usr/share/keyrings/docker-archive-keyring.gpg && \
    rm -f /tmp/docker.asc && \
    printf 'deb [arch=%s signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] %s/linux/ubuntu %s stable\n' \
        "$(dpkg --print-architecture)" "${DOCKER_APT_MIRROR}" "${codename}" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        containerd.io \
        docker-buildx-plugin \
        docker-ce \
        docker-ce-cli \
        docker-ce-rootless-extras \
        docker-compose-plugin && \
    docker --version && \
    dockerd --version && \
    containerd --version

# Install the shared PyTorch/CUDA ABI first. Every native extension below is
# compiled against this exact environment.
RUN --mount=type=cache,target=/root/.cache/uv \
    CUDA_MAJOR="${CUDA_VERSION%%.*}" && \
    expected_cuda="$(echo "${CUDA_VERSION}" | cut -d. -f1,2)" && \
    uv pip install \
        "torch==2.11.0+${TORCH_CUDA_WHEEL_TAG}" \
        "torchvision==0.26.0+${TORCH_CUDA_WHEEL_TAG}" \
        "torchaudio==2.11.0+${TORCH_CUDA_WHEEL_TAG}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_WHEEL_TAG}" && \
    uv pip install torchdata && \
    uv pip install "tensordict==0.13.0" --no-deps && \
    uv pip install \
        triton \
        "nvidia-nccl-cu${CUDA_MAJOR}==2.30.7" && \
    python -c \
        "import torch; expected='${expected_cuda}'; assert torch.version.cuda == expected, \
        f'Expected PyTorch CUDA {expected}, got {torch.version.cuda}.'"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        accelerate \
        aiohttp \
        blobfile \
        cachetools \
        click==8.2.1 \
        codetiming \
        datasets \
        dill \
        einops \
        "fastapi[standard]>=0.115.0" \
        flask \
        "grpcio>=1.81.1" \
        "grpcio-tools>=1.81.1" \
        hf-transfer \
        hydra-core \
        liger-kernel \
        mathruler \
        matplotlib \
        maturin \
        "nvidia-cudnn-frontend>=1.13.0" \
        "nvidia-ml-py>=12.560.30" \
        numpy \
        "optree>=0.13.0" \
        pandas \
        paramiko \
        peft \
        "protobuf>=7.35.1,<8" \
        puccinialin \
        py-spy \
        "pyarrow>=19.0.1" \
        pybind11 \
        pylatexenc \
        qwen-vl-utils \
        ray[default] \
        requests \
        sortedcontainers \
        setuptools \
        tomlkit \
        transformers==5.10.1 \
        trl \
        wandb \
        xgrammar \
        pulp \
        scipy \
        ninja && \
    uv pip install nvidia-modelopt --no-deps && \
    expected_cuda="$(echo "${CUDA_VERSION}" | cut -d. -f1,2)" && \
    python -c \
        "import torch; expected='${expected_cuda}'; assert torch.version.cuda == expected, \
        f'A dependency replaced PyTorch CUDA {expected} with {torch.version.cuda}.'"

# Rust is needed for the SMG gateway and its Python binding. SMG may carry a
# rust-toolchain file; rustup will honor it when Cargo enters the checkout.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal --default-toolchain stable && \
    rustup component add rustfmt clippy

# Copy only immutable patch inputs before cloning large repositories. This is
# the same source-first patch pattern used by vime and fails during build when
# a revision drifts away from its PSRL patch.
COPY docker/patch/ ${PSRL_PATCH_DIR}/

# UCX and NIXL provide the PSRL model-transfer data plane. NIXL v1.2.0 follows
# the upstream CUDA 13 source-packaging flow without the legacy PSRL patch.
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/openucx/ucx.git /root/ucx-source && \
    git -C /root/ucx-source checkout ${UCX_REF} && \
    cd /root/ucx-source && \
    ./autogen.sh && \
    ./configure \
        --prefix=${UCX_HOME} \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --without-go \
        --with-cuda=${CUDA_HOME} \
        --with-verbs \
        --with-rdmacm \
        --with-mlx5 \
        --with-dm \
        --enable-mt && \
    make -j${MAX_JOBS} && \
    make -j${MAX_JOBS} install-strip && \
    rm -rf /root/ucx-source

RUN --mount=type=cache,target=/root/.cache/uv uv pip install meson-python meson tomlkit

RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/ai-dynamo/nixl.git ${PSRL_THIRD_PARTY}/nixl && \
    git -C ${PSRL_THIRD_PARTY}/nixl checkout ${NIXL_REF} && \
    cd ${PSRL_THIRD_PARTY}/nixl && \
    python -c \
        "import torch; assert torch.version.cuda == '13.0', \
        f'NIXL CUDA 13 requires PyTorch CUDA 13.0, got {torch.version.cuda}.'" && \
    ./contrib/tomlutil.py --wheel-name nixl-cu13 pyproject.toml && \
    uv pip install --no-deps . --no-build-isolation && \
    meson setup build \
        --prefix=${NIXL_HOME} \
        -Dbuild_docs=false \
        -Ducx_path=${UCX_HOME} \
        -Dinstall_headers=true \
        -Ddisable_gds_backend=false && \
    ninja -C build -j${MAX_JOBS} install && \
    uv pip install --no-deps build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl && \
    python -c \
        "from importlib.metadata import version; import nixl; version('nixl-cu13'); \
        nixl.nixl_agent('psrl-build-check')"

# vLLM is installed from a patched editable checkout so PSRL can access the
# patched scheduler, gRPC, memory-suspension, and weight-layout behavior.
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/vllm-project/vllm.git ${PSRL_THIRD_PARTY}/vllm && \
    git -C ${PSRL_THIRD_PARTY}/vllm checkout ${VLLM_REF} && \
    git -C ${PSRL_THIRD_PARTY}/vllm apply --check ${PSRL_PATCH_DIR}/vllm.patch && \
    git -C ${PSRL_THIRD_PARTY}/vllm apply ${PSRL_PATCH_DIR}/vllm.patch && \
    cd ${PSRL_THIRD_PARTY}/vllm && \
    python use_existing_torch.py && \
    uv pip install -r requirements/build/cuda.txt && \
    uv pip install --no-build-isolation -e .

# Match vLLM's DeepEP v2 installation path. The helper installs the pinned
# DeepEP revision and its legacy NVSHMEM backend for workloads that still use it.
RUN git clone --depth 1 --branch ${GDRCOPY_REF} \
        https://github.com/NVIDIA/gdrcopy.git /root/gdrcopy && \
    make -C /root/gdrcopy -j${MAX_JOBS} prefix=/usr/local lib_install && \
    ldconfig && \
    rm -rf /root/gdrcopy

RUN --mount=type=cache,target=/root/.cache/uv \
    cd ${PSRL_THIRD_PARTY}/vllm && \
    NVSHMEM_VER=${NVSHMEM_VERSION} \
    TORCH_CUDA_ARCH_LIST="9.0a 10.0a" \
        bash tools/ep_kernels/install_python_libraries.sh \
            --workspace /root/deepep \
            --mode install \
            --deepep-ref ${DEEPEP_REF} \
            --nvshmem-ver ${NVSHMEM_VERSION}

# Mooncake TransferEngine is shared by the vLLM connector and TransferQueue's
# MooncakeStore backend. CUDA 13 uses the dedicated CUDA 13 wheel.
RUN --mount=type=cache,target=/root/.cache/uv \
    CUDA_MAJOR="${CUDA_VERSION%%.*}" && \
    if [[ "${CUDA_MAJOR}" -ge 13 ]]; then \
        MOONCAKE_PACKAGE=mooncake-transfer-engine-cuda13; \
    else \
        MOONCAKE_PACKAGE=mooncake-transfer-engine; \
    fi && \
    uv pip install \
        "cupy-cuda${CUDA_MAJOR}x<14.1.0" \
        "${MOONCAKE_PACKAGE}==${MOONCAKE_VERSION}"

# Build the CUDA extension stack after torch and vLLM so every extension sees
# one torch ABI. FlashAttention 3 remains optional because it targets Hopper.
RUN --mount=type=cache,target=/root/.cache/uv \
    FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
        uv pip install --no-build-isolation --no-deps "flash-attn==${FLASH_ATTN_VERSION}" && \
    if [[ "${INSTALL_FLASH_ATTN_3}" == "1" ]]; then \
        test -f "${CUDA_HOME}/include/cccl/cuda/std/utility" && \
        git clone --filter=blob:none --no-checkout \
            https://github.com/Dao-AILab/flash-attention.git /root/flash-attention && \
        git -C /root/flash-attention checkout --detach "${FLASH_ATTN_3_REF}" && \
        test "$(git -C /root/flash-attention rev-parse HEAD)" = "${FLASH_ATTN_3_REF}" && \
        cd /root/flash-attention/hopper && \
        FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py install && \
        cd /root && \
        python -c \
            "import torch, flash_attn_3; from flash_attn_3 import flash_attn_interface; \
            module_path=flash_attn_3.__file__; \
            assert not module_path.startswith('/root/flash-attention/'), \
            f'FlashAttention-3 imported from its source tree: {module_path}.'; \
            assert torch.version.cuda == '13.0', \
            f'FlashAttention-3 expected PyTorch CUDA 13.0, got {torch.version.cuda}.'" && \
        rm -rf /root/flash-attention; \
    fi && \
    CUDA_WHEEL_TAG="cu$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr -d '.')" && \
    uv pip install --no-build-isolation \
        "flashinfer-python==${FLASHINFER_VERSION}" \
        --extra-index-url "https://flashinfer.ai/whl/${CUDA_WHEEL_TAG}" && \
    uv pip install \
        "flashinfer-cubin==${FLASHINFER_VERSION}" \
        --index-url "https://flashinfer.ai/whl" && \
    uv pip install "flash-linear-attention==0.4.2"

RUN --mount=type=cache,target=/root/.cache/uv \
    CUDA_MAJOR="${CUDA_VERSION%%.*}" && \
    uv pip install "nvidia-nccl-cu${CUDA_MAJOR}==2.30.7"

RUN --mount=type=cache,target=/root/.cache/uv \
    NVCC_APPEND_FLAGS="--threads 4" \
        pip install --no-build-isolation --no-cache-dir \
            --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
            "git+https://github.com/NVIDIA/apex.git@${APEX_REF}" && \
    CUDNN_PATH="$(python -c 'import site; print(site.getsitepackages()[0] + "/nvidia/cudnn")')" \
    NVTE_FRAMEWORK=pytorch \
    NVTE_WITH_NCCL_EP=0 \
    NVTE_BUILD_THREADS_PER_JOB=4 \
        uv pip install --no-build-isolation \
            "git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_REF}" && \
    python -c \
        "from importlib.metadata import version; import torch, transformer_engine.pytorch; \
        version('transformer-engine'); assert torch.version.cuda == '13.0', \
        f'TransformerEngine expected PyTorch CUDA 13.0, got {torch.version.cuda}.'"

# Megatron-LM and MegatronBridge are retained as editable source trees because
# PSRL's training backend imports internal provider and checkpoint utilities.
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/NVIDIA/Megatron-LM.git ${PSRL_THIRD_PARTY}/Megatron-LM && \
    git -C ${PSRL_THIRD_PARTY}/Megatron-LM checkout ${MEGATRON_REF} && \
    uv pip install --no-deps --no-build-isolation -e ${PSRL_THIRD_PARTY}/Megatron-LM && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git ${PSRL_THIRD_PARTY}/Megatron-Bridge && \
    git -C ${PSRL_THIRD_PARTY}/Megatron-Bridge checkout ${MEGATRON_BRIDGE_REF} && \
    sed -i \
        's|^megatron-core = { path = "3rdparty/Megatron-LM/", editable = true }|# megatron-core is supplied by PSRL.|' \
        ${PSRL_THIRD_PARTY}/Megatron-Bridge/pyproject.toml && \
    git -C ${PSRL_THIRD_PARTY}/Megatron-Bridge apply --check ${PSRL_PATCH_DIR}/megatron_bridge.patch && \
    git -C ${PSRL_THIRD_PARTY}/Megatron-Bridge apply ${PSRL_PATCH_DIR}/megatron_bridge.patch && \
    uv pip install --no-deps --no-build-isolation -e ${PSRL_THIRD_PARTY}/Megatron-Bridge

RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/fzyzcjy/torch_memory_saver.git ${PSRL_THIRD_PARTY}/torch_memory_saver && \
    git -C ${PSRL_THIRD_PARTY}/torch_memory_saver checkout ${TMS_REF} && \
    TMS_CUDA_MAJOR="${CUDA_VERSION%%.*}" \
        uv pip install --no-build-isolation -e ${PSRL_THIRD_PARTY}/torch_memory_saver

# Install the two patched data/control-plane dependencies before PSRL.
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/volcengine/verl.git ${PSRL_THIRD_PARTY}/verl && \
    git -C ${PSRL_THIRD_PARTY}/verl checkout ${VERL_REF} && \
    git -C ${PSRL_THIRD_PARTY}/verl apply --check ${PSRL_PATCH_DIR}/verl.patch && \
    git -C ${PSRL_THIRD_PARTY}/verl apply ${PSRL_PATCH_DIR}/verl.patch && \
    uv pip install -e ${PSRL_THIRD_PARTY}/verl && \
    git clone https://github.com/Ascend/TransferQueue.git ${PSRL_THIRD_PARTY}/TransferQueue && \
    git -C ${PSRL_THIRD_PARTY}/TransferQueue checkout ${TRANSFER_QUEUE_REF} && \
    git -C ${PSRL_THIRD_PARTY}/TransferQueue apply --check ${PSRL_PATCH_DIR}/transfer_queue.patch && \
    git -C ${PSRL_THIRD_PARTY}/TransferQueue apply ${PSRL_PATCH_DIR}/transfer_queue.patch && \
    uv pip install -e ${PSRL_THIRD_PARTY}/TransferQueue

# Install numpy to overwrite verl's numpy version.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "numpy==2.2.6"

# LMCache compiles against CUDA and the already installed torch/vLLM/NIXL stack.
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/LMCache/LMCache.git ${PSRL_THIRD_PARTY}/LMCache && \
    git -C ${PSRL_THIRD_PARTY}/LMCache checkout ${LMCACHE_REF} && \
    git -C ${PSRL_THIRD_PARTY}/LMCache apply --check ${PSRL_PATCH_DIR}/lm_cache.patch && \
    git -C ${PSRL_THIRD_PARTY}/LMCache apply ${PSRL_PATCH_DIR}/lm_cache.patch && \
    LMCACHE_CUDA_MAJOR="${CUDA_VERSION%%.*}" \
        uv pip install --no-build-isolation -e ${PSRL_THIRD_PARTY}/LMCache && \
    python -c "import lmcache"

# Build SMG with PSRL routing/state support and install all Python protocol
# packages plus the native binding.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    git clone https://github.com/psrl-project/smg.git ${PSRL_THIRD_PARTY}/smg && \
    git -C ${PSRL_THIRD_PARTY}/smg checkout ${SMG_REF} && \
    cd ${PSRL_THIRD_PARTY}/smg && \
    "${PROTOC}" --version && \
    test -f "${PROTOC_INCLUDE}/google/protobuf/timestamp.proto" && \
    test -f "${PROTOC_INCLUDE}/google/protobuf/struct.proto" && \
    cargo build --release && \
    uv pip install --no-build-isolation -e crates/grpc_client/python/ && \
    uv pip install --no-build-isolation -e crates/psrl_state/python/ && \
    uv pip install -e grpc_servicer/ && \
    uv pip install "protobuf>=7.35.1,<8" && \
    cd bindings/python && \
    maturin develop --features vendored-openssl --release

# Clone PSRL last so changing `PSRL_COMMIT` does not invalidate native
# dependency layers. The PSRL patch plugin remains reachable by pyproject.toml.
WORKDIR /home
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone https://github.com/psrl-project/psrl.git ${PSRL_HOME} && \
    git -C ${PSRL_HOME} checkout ${PSRL_COMMIT} && \
    cd ${PSRL_HOME} && \
    uv pip install --no-build-isolation -e . && \
    python -c \
        "from importlib.metadata import entry_points, version; import vllm_patches; \
        version('vllm-psrl-patches'); assert any( \
        ep.name == 'custom_patches' for ep in entry_points(group='vllm.general_plugins'))" && \
    expected_cuda="$(echo "${CUDA_VERSION}" | cut -d. -f1,2)" && \
    nvcc_cuda="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')" && \
    if [[ "${nvcc_cuda}" != "${expected_cuda}" ]]; then \
        echo "Expected NVCC CUDA ${expected_cuda}, got ${nvcc_cuda}." >&2; \
        exit 1; \
    fi && \
    python -c \
        "import torch; expected='${expected_cuda}'; assert torch.version.cuda == expected, \
        f'Expected PyTorch CUDA {expected}, got {torch.version.cuda}.'" && \
    python -c "import psrl, smg.router, torch, transfer_queue, verl, vllm, vllm_patches" && \
    python -c "import lmcache, nixl, torch_memory_saver"

RUN --mount=type=cache,target=/root/.cache/uv \
    CUDA_MAJOR="${CUDA_VERSION%%.*}" && \
    uv pip install "nvidia-nccl-cu${CUDA_MAJOR}==2.30.7"

RUN rm -rf /root/.cache/pip /root/.cache/uv

WORKDIR ${PSRL_HOME}

RUN mkdir -p ${PSRL_WORKSPACE}/env
COPY docker/env.sh ${PSRL_WORKSPACE}/env/psrl.sh

ENV VIRTUAL_ENV=/root/venv \
    PATH="/root/venv/bin:/root/.cargo/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

RUN echo 'export VIRTUAL_ENV=/root/venv' >> ~/.bashrc && \
    echo 'export PATH=/root/venv/bin:/root/.cargo/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' >> ~/.bashrc

LABEL org.opencontainers.image.title="PSRL CUDA" \
      org.opencontainers.image.description="PSRL post-training environment with CUDA, vLLM, veRL, SMG, NIXL, LMCache, Megatron, Mooncake, and DeepEP" \
      org.opencontainers.image.source="https://github.com/psrl-project/psrl"

# PSRL jobs are launched through Hydra/Ray and do not have a single fixed
# service entrypoint. Keeping an empty entrypoint also avoids inheriting a
# serving command when users override BASE_IMAGE downstream.
ENTRYPOINT []
CMD ["/bin/bash"]
