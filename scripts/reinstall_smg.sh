#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SMG_DIR="${SMG_DIR:-${PROJECT_ROOT}/third_party/smg}"

if [[ ! -f "${SMG_DIR}/Cargo.toml" ]]; then
    echo "Error: SMG source directory not found: ${SMG_DIR}" >&2
    exit 1
fi

SMG_DIR="${SMG_DIR}" bash "${SCRIPT_DIR}/apply_smg_patch.sh"

python -m uv pip uninstall -y smg smg-grpc-proto smg-grpc-servicer psrl-state-grpc-proto

# Match the protobuf 7 runtime before no-build-isolation generates SMG stubs.
python -m uv pip install --no-cache-dir "grpcio-tools>=1.81.1" "protobuf>=7.35.1,<8"

(
    cd -- "${SMG_DIR}"
    cargo build --release
    # --no-build-isolation: generate stubs with the ambient grpcio-tools above.
    python -m uv pip install --no-build-isolation -e crates/grpc_client/python/
    python -m uv pip install --no-build-isolation -e crates/psrl_state/python/
    python -m uv pip install -e grpc_servicer/

    # Re-assert protobuf 7 after servicer deps (pip may leave an older runtime).
    python -m uv pip install --no-cache-dir "protobuf>=7.35.1,<8"

    cd -- bindings/python
    maturin develop --features vendored-openssl --release
)
