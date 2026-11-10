#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script to activate the installed NIXL stack." >&2
    exit 1
fi

_psrl_activate_nixl() {
    local repo_root ucx_prefix nixl_prefix python_version python_site
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    ucx_prefix="${UCX_PREFIX:-${repo_root}/third_party/ucx_1_20_rkey65535}"
    nixl_prefix="${NIXL_PREFIX:-${repo_root}/third_party/nixl_ucx_1_20_rkey65535_backlog4096}"
    python_version="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" || return 1
    python_site="${nixl_prefix}/lib/python${python_version}/site-packages"

    if [[ ! -f "${ucx_prefix}/.psrl-ucx-1.20-worker-config-uint16-complete" ||
          ! -f "${nixl_prefix}/.psrl-nixl-0.10.1-complete" ||
          ! -f "${python_site}/nixl/__init__.py" ]]; then
        echo "Build the UCX 1.20.0 and NIXL 0.10.1 stack with scripts/install_nixl.sh first." >&2
        return 1
    fi

    export PATH="${ucx_prefix}/bin:${PATH}"
    export LD_LIBRARY_PATH="${nixl_prefix}/lib64:${ucx_prefix}/lib:${ucx_prefix}/lib/ucx:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${python_site}:${PYTHONPATH:-}"
    export NIXL_PLUGIN_DIR="${nixl_prefix}/lib64/plugins"
    export PKG_CONFIG_PATH="${ucx_prefix}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
}

if _psrl_activate_nixl; then
    unset -f _psrl_activate_nixl
else
    unset -f _psrl_activate_nixl
    return 1
fi
