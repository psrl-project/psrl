#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

VLLM_PATH=${VLLM_PATH:-}
VERL_PATH=${VERL_PATH:-}
TQ_PATH=${TQ_PATH:-}
MAX_JOBS=${MAX_JOBS:-64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
mkdir -p $THIRD_PARTY_PATH

echo "0. Install uv to boost installation speed"
python -m pip install uv

echo "1. Install pytorch and tensordict"
python -m uv pip install --no-cache-dir "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --no-cache-dir "triton==3.6.0" "tensordict==0.12.4" torchdata

# For ncclCommSuspend/ncclCommResume support
# 2.28.9 is not enough
python -m uv pip install nvidia-nccl-cu12==2.29.7

echo "2. Install basic packages"
python -m uv pip install "transformers==5.10.1" accelerate datasets peft hf-transfer matplotlib flask click==8.2.1 \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas paramiko sortedcontainers \
    ray[default]==2.49.1 codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pre-commit ruff meson ninja pynvml requests einops trl maturin puccinialin protoc-wheel-0 nvidia-modelopt[torch]

python -m uv pip uninstall -y pynvml nvidia-ml-py
python -m uv pip install --no-cache-dir "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" "nvidia-cudnn-frontend>=1.13.0"
# Match the protobuf 7 runtime before no-build-isolation generates SMG stubs.
python -m uv pip install --no-cache-dir "grpcio-tools>=1.81.1" "protobuf>=7.35.1,<8"

echo "4. Install FlashAttention and FlashInfer"
# Auto-detect the GPU compute capability so we build for the right arch and
# only attempt FlashAttention 3 (Hopper+) where it is actually supported.
# Example: A800 -> 8.0 -> ARCH=80 ; H100 -> 9.0 -> ARCH=90.
if [ -n "${FLASH_ATTN_CUDA_ARCHS:-}" ]; then
    echo "  Using pre-set FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS}"
else
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')
    if [ -z "$CC" ]; then
        echo "[ERROR] Could not detect GPU compute capability via nvidia-smi." >&2
        echo "        Set FLASH_ATTN_CUDA_ARCHS manually (e.g. 80 for A100/A800, 90 for H100) and re-run." >&2
        exit 1
    fi
    FLASH_ATTN_CUDA_ARCHS="${CC%.*}${CC#*.}"   # "8.0" -> "80", "9.0" -> "90"
    echo "  Detected compute_cap=${CC} -> FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS}"
fi

# Install FlashAttention 2 for packages that import `flash_attn`.
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" FLASH_ATTENTION_FORCE_BUILD="TRUE" FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" python -m uv pip install -U "flash-attn==2.8.1" --no-build-isolation --no-deps

# FlashAttention 3 only supports Hopper+ (sm90 and newer). Build it only there.
if [ "${FLASH_ATTN_CUDA_ARCHS}" -ge 90 ] 2>/dev/null; then
    echo "  Arch ${FLASH_ATTN_CUDA_ARCHS} supports FlashAttention 3, installing FA3 beta..."
    git clone --depth 1 --branch v2.8.1 https://github.com/Dao-AILab/flash-attention.git flash_attn_src
    pushd flash_attn_src/hopper
    python setup.py install
    python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    mkdir -p $python_path/flash_attn_3
    wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/7b0bfcc3d1f69786f0c4277c582ad58acdfb297d/hopper/flash_attn_interface.py
    popd
    rm -rf flash_attn_src
else
    echo "  Arch ${FLASH_ATTN_CUDA_ARCHS} < 90 (not Hopper), skipping FlashAttention 3 install."
fi

echo "5. Install apex"
mkdir -p apex_src
pushd apex_src
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=$MAX_JOBS python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd
rm -rf apex_src

echo "6. Install SMG (with PSRL policies)"
# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo not found. Please install Rust first:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  source \$HOME/.cargo/env"
    exit 1
fi

echo "Building SMG from source..."
pushd $THIRD_PARTY_PATH
git clone https://github.com/psrl-project/smg.git -b psrl-dev
cd smg
# Comment out the smg-tui workspace dependency (not needed for PSRL build)
sed -i 's|^smg-tui = { version = "0.1.0", path = "tui" }|# smg-tui = { version = "0.1.0", path = "tui" }|' Cargo.toml
# Build release binary with PSRL policies
cargo build --release
# --no-build-isolation: generate stubs with the ambient grpcio-tools installed above.
python -m uv pip install --no-build-isolation -e crates/grpc_client/python/
python -m uv pip install --no-build-isolation -e crates/psrl_state/python/
python -m uv pip install -e grpc_servicer/
# Re-assert protobuf 7 after servicer deps (pip may leave an older runtime).
python -m uv pip install --no-cache-dir "protobuf>=7.35.1,<8"

echo "Build python binding of smg..."
cd bindings/python
maturin develop --features vendored-openssl --release
popd

# Verify the binding is importable
python -c "from smg.router import Router; print('  ✓ smg.router binding installed successfully')" || {
    echo "Error: smg.router binding failed to install"
    exit 1
}

echo "7. Install vllm and verl"
if [ -z "$VLLM_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone -b releases/v0.22.0 https://github.com/vllm-project/vllm.git
    VLLM_PATH=$THIRD_PARTY_PATH/vllm
    popd
fi
pushd $VLLM_PATH
python use_existing_torch.py
python -m uv pip install -r requirements/build/cuda.txt
python -m uv pip install --no-build-isolation -e .
popd

if [ -z "$VERL_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone https://github.com/volcengine/verl.git
    VERL_PATH=$THIRD_PARTY_PATH/verl
    cd $VERL_PATH
    git checkout e5ca4acb
    popd
fi
pushd $VERL_PATH
python -m uv pip install -e .
popd

if [ -z "$TQ_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone https://github.com/Ascend/TransferQueue.git
    TQ_PATH=$THIRD_PARTY_PATH/TransferQueue
    cd $TQ_PATH
    git checkout 434f8c4
    popd
fi
pushd $TQ_PATH
python -m uv pip install -e .
popd

echo "8. Apply patch for dependencies..."

bash "$PSRL_PATH/patch/apply_patch.sh" vllm
bash "$PSRL_PATH/patch/apply_patch.sh" verl
bash "$PSRL_PATH/patch/apply_patch.sh" transfer_queue

echo "9. Install torch_memory_saver"
python -m uv pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a6394d1e09c613fab90260054cecc2684586d --no-cache-dir --force-reinstall

echo "10. Install flashinfer"
mkdir -p flashinfer_src
pushd flashinfer_src
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive --branch v0.6.11.post3
cd flashinfer
python -m uv pip install -v .
cd flashinfer-cubin
python -m build --no-isolation --wheel
python -m uv pip install dist/*.whl
popd
rm -rf flashinfer_src

echo "Successfully installed all basic packages"
