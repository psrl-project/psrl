#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env/psrl.sh
python -m pytest tests/fsdp/test_model_state.py --run-gpu "$@"
