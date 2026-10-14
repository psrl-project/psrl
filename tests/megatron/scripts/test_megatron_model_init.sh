#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env/psrl.sh
python -m pytest tests/megatron/test_megatron_model_init.py --run-gpu "$@"
