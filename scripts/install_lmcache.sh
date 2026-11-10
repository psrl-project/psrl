#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

LMCACHE_PATH=${LMCACHE_PATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
mkdir -p $THIRD_PARTY_PATH

export LMCACHE_CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')"

echo "1. Install LMCache"
if [ -z "$LMCACHE_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    git clone https://github.com/LMCache/LMCache.git
    LMCACHE_PATH=$THIRD_PARTY_PATH/LMCache
    popd
fi
pushd $LMCACHE_PATH
git checkout v0.4.6
git apply "${PSRL_PATH}/patch/lm_cache/a7f22be7.patch"
git apply "${PSRL_PATH}/patch/lm_cache/dependencies.patch"
rm -rf ./*.so ./build
python -m pip uninstall lmcache -y
python -m uv pip install --no-cache-dir -e . --no-build-isolation -v \
    "nixl==0.10.1" "nixl-cu12==0.10.1"
popd

echo "Check if LMCache is installed correctly"
python -c "import torch; import lmcache.c_ops as lmc_ops; print('OK')"


python -m uv pip install "nvidia-nccl-cu12==2.29.7"
python "${SCRIPT_DIR}/check_environment.py"
echo "Successfully installed LMCache"
