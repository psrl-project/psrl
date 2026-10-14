#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANAYLSIS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${ANAYLSIS_DIR}/plot_prefill_step.py" \
  /apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl/examples/mini_swe/megatron_psrl_log/sticky_thunder_agent_kv_aware_GRPO-SWE-agent-LM-7B-swe_gym-megatron-staleness_1/Prefill_I1.log \
  --out "${ANAYLSIS_DIR}/sticky_thunder_agent_kv_aware_prefill_step.png"
