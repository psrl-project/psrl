#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PSRL_WORKSPACE=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)

N=${1:-256}
shift $(( $# < 1 ? $# : 1 )) || true

VERL_ROUTER_ADDRESS=${VERL_ROUTER_ADDRESS:-${2:-}}
VERL_WORKER_ADDRESS=${VERL_WORKER_ADDRESS:-}
PSRL_GATEWAY_ADDRESS=${PSRL_GATEWAY_ADDRESS:-}

if [[ -z "${VERL_ROUTER_ADDRESS}" ]]; then
  echo "Usage:" >&2
  echo "  VERL_ROUTER_ADDRESS=host:port [VERL_WORKER_ADDRESS=host:port] [PSRL_GATEWAY_ADDRESS=host:port] $0 [n=256] [extra hydra overrides...]" >&2
  echo "" >&2
  echo "Purpose:" >&2
  echo "  Compare the same RM chat-completion requests across:" >&2
  echo "    1. PSRL internal Ray/DataProto reward router" >&2
  echo "    2. verl reward router HTTP /v1/chat/completions" >&2
  echo "    3. optional direct verl worker HTTP /v1/chat/completions" >&2
  echo "    4. optional PSRL gateway HTTP /v1/chat/completions" >&2
  exit 2
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR=${OUT_DIR:-"${PSRL_WORKSPACE}/logs/rm_router_compare"}
RUN_DIR="${BASE_OUT_DIR}/${RUN_ID}_n${N}"
mkdir -p "${RUN_DIR}"

run_case() {
  local label=$1
  local mode=$2
  local address=${3:-}
  shift 3 || true

  local case_dir="${RUN_DIR}/${label}"
  mkdir -p "${case_dir}"

  echo "[compare_rm_router_paths] running ${label}"
  if [[ "${mode}" == "http" ]]; then
    OUT_DIR="${case_dir}" bash "${PSRL_WORKSPACE}/scripts/run_rm_path_benchmark.sh" http "${N}" "${address}" "$@" \
      "+bench.label=${label}"
  else
    OUT_DIR="${case_dir}" bash "${PSRL_WORKSPACE}/scripts/run_rm_path_benchmark.sh" "${mode}" "${N}" "$@" \
      "+bench.label=${label}"
  fi
}

run_case "psrl_router_internal" "psrl_router" "" "$@"
run_case "verl_router_http" "http" "${VERL_ROUTER_ADDRESS}" "$@"

if [[ -n "${VERL_WORKER_ADDRESS}" ]]; then
  run_case "verl_worker_direct_http" "http" "${VERL_WORKER_ADDRESS}" "$@"
else
  echo "[compare_rm_router_paths] skip verl_worker_direct_http: VERL_WORKER_ADDRESS not set"
fi

if [[ -n "${PSRL_GATEWAY_ADDRESS}" ]]; then
  run_case "psrl_gateway_http" "http" "${PSRL_GATEWAY_ADDRESS}" "$@"
else
  echo "[compare_rm_router_paths] skip psrl_gateway_http: PSRL_GATEWAY_ADDRESS not set"
fi

python - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for path in sorted(run_dir.glob("*/*.json")):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", payload)
    label = summary.get("label") or path.parent.name
    rows.append(
        {
            "label": label,
            "wall_s": summary.get("wall_s", 0.0),
            "rps": summary.get("requests_per_second", 0.0),
            "lat_p50": (summary.get("latency_s") or {}).get("p50", 0.0),
            "lat_p90": (summary.get("latency_s") or {}).get("p90", 0.0),
            "lat_p99": (summary.get("latency_s") or {}).get("p99", 0.0),
            "in_mean": (summary.get("rm_input_len") or {}).get("mean", 0.0),
            "out_mean": (summary.get("rm_output_len") or {}).get("mean", 0.0),
            "local_in_chat_mean": (summary.get("rm_input_len_local_chat") or {}).get("mean", 0.0),
            "local_out_plain_mean": (summary.get("rm_output_len_local_plain") or {}).get("mean", 0.0),
            "path": str(path),
        }
    )

report = run_dir / "summary.tsv"
with report.open("w", encoding="utf-8") as f:
    f.write(
        "label\twall_s\trps\tlat_p50\tlat_p90\tlat_p99\t"
        "rm_input_mean\trm_output_mean\tlocal_chat_input_mean\tlocal_plain_output_mean\tjson\n"
    )
    for row in rows:
        f.write(
            f"{row['label']}\t{row['wall_s']:.4f}\t{row['rps']:.4f}\t"
            f"{row['lat_p50']:.4f}\t{row['lat_p90']:.4f}\t{row['lat_p99']:.4f}\t"
            f"{row['in_mean']:.2f}\t{row['out_mean']:.2f}\t"
            f"{row['local_in_chat_mean']:.2f}\t{row['local_out_plain_mean']:.2f}\t"
            f"{row['path']}\n"
        )

print(f"[compare_rm_router_paths] summary={report}")
print(report.read_text(encoding="utf-8"))
PY
