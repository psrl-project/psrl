#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env/psrl.sh
python -m pytest tests/torch_dist/test_broadcast.py "$@"
