#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANAYLSIS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Map TP -> jsonl path. Add more --tp-file entries as runs become available.
python "${ANAYLSIS_DIR}/plot_chunked_prefill_micro.py" \
  --tp-file "1:/apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl/examples/bench/rollout/exp/chunked_prefill/micro_TP1_N32768_TP1_20260804_142101.jsonl" \
  --tp-file "2:/apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl/examples/bench/rollout/exp/chunked_prefill/micro_TP2_N32768_TP2_20260804_142633.jsonl" \
  --out "${ANAYLSIS_DIR}/32b_chunked_prefill_multi_throughput.png"
