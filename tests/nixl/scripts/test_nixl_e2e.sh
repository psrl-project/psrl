#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env/psrl.sh
python -m pytest tests/nixl/test_nixl_e2e.py --run-gpu "$@"
