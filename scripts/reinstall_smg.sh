#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SMG_DIR="${SMG_DIR:-${PROJECT_ROOT}/third_party/smg}"

if [[ ! -f "${SMG_DIR}/Cargo.toml" ]]; then
    echo "Error: SMG source directory not found: ${SMG_DIR}" >&2
    exit 1
fi

(
    cd -- "${SMG_DIR}"
    cargo build --release
    # Match the protobuf runtime used to generate the native service stubs.
    python -m uv pip install --no-cache-dir "grpcio-tools>=1.81.1" "protobuf>=7.35.1,<8"
    # Resolve local protocol packages and the servicer together so dependencies retain the editable sources.
    python -m uv pip install --reinstall --no-build-isolation \
        -e crates/grpc_client/python/ \
        -e crates/psrl_state/python/ \
        -e grpc_servicer/

    # Re-assert protobuf 7 after servicer deps (pip may leave an older runtime).
    python -m uv pip install --no-cache-dir "protobuf>=7.35.1,<8"

    cd -- bindings/python
    maturin develop --features vendored-openssl --release
)
