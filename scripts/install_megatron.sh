#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"

# Function: Install cuDNN and set CUDNN_PATH if not already installed or version is too low
install_cudnn() {
    local required_version="9.8.0.87"
    echo "Checking cuDNN installation..."

    # Check if nvidia-cudnn-cu12 is installed and get its version
    local installed_version=$(python -m pip show nvidia-cudnn-cu12 2>/dev/null | grep '^Version:' | awk '{print $2}')
    
    if [ -n "$installed_version" ]; then
        echo "Installed cuDNN version: $installed_version"
        # Compare versions
        if [ "$(printf '%s\n' "$required_version" "$installed_version" | sort -V | head -n1)" = "$required_version" ]; then
            echo "cuDNN version $installed_version is sufficient (>= $required_version)."
            set_cudnn_path
            return 0
        else
            echo "cuDNN version $installed_version is lower than required ($required_version). Proceeding to install..."
        fi
    else
        echo "cuDNN is not installed. Proceeding to install..."
    fi

    # Install cuDNN
    echo "Installing cuDNN Python package..."
    python -m pip install nvidia-cudnn-cu12=="$required_version"
    
    # Verify installation and set path
    set_cudnn_path
}

# Function: Set CUDNN_PATH after installation
set_cudnn_path() {
    local site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
    local cudnn_path="$site_packages/nvidia/cudnn"
    
    if [ ! -d "$cudnn_path" ] || [ ! -f "$cudnn_path/include/cudnn_graph.h" ]; then
        echo "Error: cuDNN installation failed or path not found at $cudnn_path"
        exit 1
    fi
    
    echo "Setting CUDNN_PATH to: $cudnn_path"
    export CUDNN_PATH="$cudnn_path"
}

echo "1. Install cuDNN Python package"
install_cudnn

echo "2. Install TransformerEngine"
echo "Notice: TransformerEngine installation can take a long time, please be patient"
NVTE_FRAMEWORK=pytorch python -m pip install --no-cache-dir --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@v2.7

echo "3. Install Megatron"
python -m pip install git+https://github.com/NVIDIA/Megatron-LM.git@c049020 --no-deps --no-build-isolation

echo "4. Install Megatron-Bridge"
pushd $THIRD_PARTY_PATH
# git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
# git checkout 94d1870
# Comment out the local megatron-core source (we install Megatron-LM separately above)
sed -i 's|^megatron-core = { path = "3rdparty/Megatron-LM/", editable = true }|# megatron-core = { path = "3rdparty/Megatron-LM/", editable = true }|' pyproject.toml
python -m uv pip install -e . --no-build-isolation --no-cache-dir --no-deps
popd

bash "$PSRL_PATH/patch/apply_patch.sh" megatron_bridge

echo "5. Apply TransformerEngine patch (flash-attn-3 + context parallel compat)"
bash "$PSRL_PATH/patch/apply_patch.sh" transformer_engine --force

echo "Successfully installed all packages for Megatron"
