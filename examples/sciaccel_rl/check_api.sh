#!/usr/bin/env bash
# Verify an OpenAI-compatible endpoint answers before spending a batch rollout on it.
# Usage: `OPENROUTER_API_KEY=sk-... check_api.sh [model]`
set -euo pipefail

usage() { sed -n '2,3p' "$0"; }
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

API_BASE=${API_BASE:-https://openrouter.ai/api/v1}
API_KEY=${OPENROUTER_API_KEY:-${API_KEY:-}}
MODEL=${1:-${MODEL:-nvidia/nemotron-3-super-120b-a12b:free}}

if [[ -z "${API_KEY}" ]]; then
    echo "ERROR: set OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)." >&2
    exit 1
fi

echo "endpoint: ${API_BASE}"
echo "model:    ${MODEL}"

# The corporate proxy is the only route out, so curl must use it. NO_PROXY still
# exempts in-cluster addresses.
body=$(curl -sS --max-time 120 "${API_BASE}/chat/completions" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":16}")

python3 - "$body" <<'PY'
import json, sys
try:
    d = json.loads(sys.argv[1])
except json.JSONDecodeError:
    print(f"FAIL: endpoint returned non-JSON:\n{sys.argv[1][:400]}"); sys.exit(1)
if "error" in d:
    print(f"FAIL: {json.dumps(d['error'])[:400]}"); sys.exit(1)
choice = (d.get("choices") or [{}])[0]
content = (choice.get("message") or {}).get("content", "")
usage = d.get("usage") or {}
print(f"OK: reply={content.strip()[:60]!r}")
print(f"    usage={usage}")
if not usage.get("completion_tokens"):
    print("    WARNING: no completion_tokens in usage. The session proxy reports")
    print("    turn token counts from this field, so they will read as zero.")
PY
