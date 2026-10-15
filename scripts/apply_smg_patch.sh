#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SMG_DIR="${SMG_DIR:-${PROJECT_ROOT}/third_party/smg}"
for patch_name in reward_itl elastic_runtime; do
    patch_file="${PROJECT_ROOT}/patch/smg/${patch_name}.patch"
    if git -C "${SMG_DIR}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
        continue
    fi
    git -C "${SMG_DIR}" apply --check "${patch_file}"
    git -C "${SMG_DIR}" apply "${patch_file}"
done
