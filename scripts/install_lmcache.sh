#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

LMCACHE_PATH=${LMCACHE_PATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_PATH="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_PATH="$PSRL_PATH/third_party"
mkdir -p $THIRD_PARTY_PATH

echo "1. Install LMCache"
if [ -z "$LMCACHE_PATH" ]; then
    pushd $THIRD_PARTY_PATH
    # git clone https://github.com/LMCache/LMCache.git
    LMCACHE_PATH=$THIRD_PARTY_PATH/LMCache
    popd
fi
pushd $LMCACHE_PATH
# git checkout v0.4.6
rm -rf ./*.so ./build
python -m pip uninstall lmcache -y
python -m pip install --no-cache-dir -e . --no-build-isolation -v
popd

echo "Check if LMCache is installed correctly"
python -c "import torch; import lmcache.c_ops as lmc_ops; print('OK')"

echo "2. Apply patch for lmcache"
# bash "$PSRL_PATH/patch/apply_patch.sh" lm_cache

echo "Successfully installed LMCache"
