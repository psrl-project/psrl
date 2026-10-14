#!/usr/bin/env bash
# Check or download the RULER-HQA eval_{length}.json files used by MemAgent.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DATA_DIR="${DATA_DIR:-${DATA_ROOT:-${SCRIPT_DIR}/data/hotpotqa}}"
HF_DATASET="${HF_DATASET:-BytedTsinghua-SIA/hotpotqa}"
LENGTHS="${LENGTHS:-${LENGTH:-50 100 200 400 800 1600 3200 6400}}"
DOWNLOAD=0

case "${1:-}" in
    "") ;;
    --download) DOWNLOAD=1 ;;
    --help|-h)
        echo "Usage: LENGTHS='50 200 800' DATA_DIR=/data/hotpotqa $0 [--download]"
        exit 0
        ;;
    *)
        echo "ERROR: unknown argument: $1" >&2
        exit 2
        ;;
esac

mkdir -p "${DATA_DIR}"

download_one() {
    local length="$1"
    local filename="eval_${length}.json"
    local destination="${DATA_DIR}/${filename}"
    if [[ -f "${destination}" ]]; then
        echo "[skip] ${destination}"
        return
    fi
    echo "[download] ${HF_DATASET}/${filename} -> ${destination}"
    python3 - "${HF_DATASET}" "${filename}" "${destination}" <<'PY'
import shutil
import sys

from huggingface_hub import hf_hub_download

repository, filename, destination = sys.argv[1:]
cached = hf_hub_download(repo_id=repository, filename=filename, repo_type="dataset")
shutil.copy2(cached, destination)
PY
}

missing=0
for length in ${LENGTHS}; do
    if [[ "${DOWNLOAD}" == "1" ]]; then
        download_one "${length}"
    elif [[ ! -f "${DATA_DIR}/eval_${length}.json" && ! -f "${DATA_DIR}/eval_${length}.jsonl" ]]; then
        echo "MISSING: ${DATA_DIR}/eval_${length}.{json,jsonl}" >&2
        missing=$((missing + 1))
    fi
done

if (( missing > 0 )); then
    echo "Download the missing files with:" >&2
    echo "  LENGTHS='${LENGTHS}' DATA_DIR='${DATA_DIR}' bash '${SCRIPT_DIR}/prepare-eval-data.sh' --download" >&2
    exit 1
fi

echo "RULER-HQA evaluation data is ready in ${DATA_DIR}"
