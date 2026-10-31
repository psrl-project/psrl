#!/bin/bash
# Source this before running NIXL/nixl_ep code on any node of the cluster.
#
#   source scripts/nixl_env.sh
#
# Why this is needed rather than relying on the install: the nixl_ep extension lives on
# the shared conda filesystem, so every node sees the same .so -- but its RUNPATH was
# baked at build time and points at the builder's node-local /tmp build tree:
#
#   RUNPATH: ... :/tmp/nixl_check/build_ep/src/core: ...
#
# On any other node that path does not exist, so the import fails with
# "libnixl.so: cannot open shared object file". The runtime libraries and backend
# plugins are therefore staged on the shared filesystem and pointed at explicitly.
#
# Rebuilding NIXL per node would also work, but one shared copy keeps every node on
# byte-identical libraries -- a version skew between nodes in an RDMA path is the kind
# of fault that shows up as a wrong-bytes transfer rather than a clean error.

# Activate the cluster's canonical environment first. Every node must agree on the
# interpreter: Ray refuses to join a cluster whose head runs a different Python
# ("Version mismatch: ... Python: 3.12.13 / 3.12.2"), and an interactive shell picks up
# a different conda than a non-interactive `ssh host cmd` does. Sourcing one shared file
# on every node is what makes that impossible to get wrong.
#
# It also exports the proxy the cluster needs for outbound HTTP; callers that talk gRPC
# (Ray) must `unset http_proxy https_proxy` afterwards, as the file's own comment says.
_PSRL_CLUSTER_ENV=/apdcephfs_zwfy10_303541817/share_303541817/ls/env/psrl.sh
if [ -f "${_PSRL_CLUSTER_ENV}" ]; then
    # shellcheck disable=SC1090
    source "${_PSRL_CLUSTER_ENV}" > /dev/null 2>&1
fi

_PSRL_CONDA_LIB=/apdcephfs_zwfy10/share_303541817/lhy/anaconda3/lib
_PSRL_NIXL_PLUGINS=/apdcephfs_zwfy10/share_303541817/lhy/nixl_runtime/plugins

# NIXL and the planner identify a node by `get_local_ip()`, which falls back to
# `socket.gethostbyname(hostname)`. On these hosts the hostname resolves to several
# addresses, including the 29.161.x RoCE bonds that are not mutually reachable between
# every pair of nodes. The cluster mesh is the 29.162.x bond1 address. Pin it unless the
# caller already chose one.
if [ -z "${LOCAL_NODE_IP:-}" ]; then
    _PSRL_BOND1_IP=$(ip -o -4 addr show bond1 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
    if [ -n "${_PSRL_BOND1_IP}" ]; then
        export LOCAL_NODE_IP="${_PSRL_BOND1_IP}"
    fi
    unset _PSRL_BOND1_IP
fi

export LD_LIBRARY_PATH="${_PSRL_CONDA_LIB}:${LD_LIBRARY_PATH:-}"
# The nixl_cu12 wheel's bindings have a RUNPATH to an API-2 libnixl, while the
# staged runtime tree and this cluster's libnixl are the API-1 pair. Preload the
# cluster library so the bindings and plugins stay on the same API.
export LD_PRELOAD="${_PSRL_CONDA_LIB}/libnixl.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export NIXL_PLUGIN_DIR="${_PSRL_NIXL_PLUGINS}"

# Device list and RoCE reachability mode, taken from the cluster's own env
# (/jizhicfs/johnnyslin/env/psrl.sh).
#
# Getting this wrong does not fail loudly, it just runs ~24x slower:
#   * `bond1` alone names an IP bond with no RDMA device behind it, so UCX finds no
#     verbs transport and silently falls back to tcp. Measured 17.3 s for 4 GiB
#     (~240 MB/s) on a 200 Gbps fabric.
#   * Hand-picking the underlying `mlx5_8`/`mlx5_9` is worse: the RDMA path is selected,
#     QPs connect and GIDs exchange, then the data phase hangs -- each node's bond1 sits
#     in its own /26 and nothing routes RoCE between them.
#   * The full list below lets UCX pick a reachable pair itself, and
#     ROCE_REACHABILITY_MODE=all stops it from writing off peers that are not
#     subnet-local. Measured 0.70 s for the same 4 GiB (~6.1 GB/s), wait time
#     17.28 s -> 0.00028 s.
#
# `mlx5_bond_N` are the bonded 200 Gbps RoCE devices (GID index 3 is the RoCE v2 entry,
# matching the cluster's NCCL_IB_GID_INDEX=3); the plain `bondN` entries stay in the
# list as the fallback UCX uses when no RDMA pair is mutually reachable.
# Set unconditionally, NOT via ${VAR:-default}. The cluster profile already exports
# `UCX_NET_DEVICES=bond1`, so a :- default is a no-op and leaves the slow TCP path in
# place -- which is how an earlier measurement of this came out at 19 s instead of
# 0.70 s for the same node pair. Override with PSRL_UCX_NET_DEVICES when a specific
# device list is wanted.
export UCX_NET_DEVICES="${PSRL_UCX_NET_DEVICES:-bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,mlx5_bond_1:1,mlx5_bond_4:1,mlx5_bond_3:1,mlx5_bond_2:1,mlx5_bond_7:1,mlx5_bond_6:1,mlx5_bond_8:1,mlx5_bond_5:1}"
export UCX_IB_ROCE_REACHABILITY_MODE="${PSRL_UCX_ROCE_REACHABILITY:-all}"

# vLLM 0.30 binds `ncclCommSuspend`, which NCCL exports from 2.29.7 on. The shared conda
# ships torch's NCCL 2.28.9, so the binding fails and vLLM silently disables pynccl for
# every group. TP all-reduce then falls back to torch.distributed, and MoE with
# data_parallel_size > 1 asserts in `all_gatherv`. Point pynccl at a staged 2.29.7
# (`pip install --target <dir> nvidia-nccl-cu12==2.29.7`); torch keeps its own NCCL.
_PSRL_NCCL_SO="${PSRL_NCCL_SO:-/apdcephfs_zwfy10_303541817/share_303541817/ls/psrl-wip/.pydeps_nccl/nvidia/nccl/lib/libnccl.so.2}"
if [ -z "${VLLM_NCCL_SO_PATH:-}" ] && [ -f "${_PSRL_NCCL_SO}" ]; then
    export VLLM_NCCL_SO_PATH="${_PSRL_NCCL_SO}"
fi

# transformers 5.12 requires huggingface-hub <2.0, and the shared conda was moved to 2.2.0
# (2026-10-13), so every `import transformers` -- and with it verl, the converters and
# both workers -- raises at import time. A staged 1.33.0 is prepended rather than the
# shared environment being changed under everyone else
# (`pip install --no-deps --target <dir> "huggingface-hub>=1.5,<2"`).
_PSRL_HFHUB="${PSRL_HFHUB_DIR:-/apdcephfs_zwfy10_303541817/share_303541817/ls/psrl-wip/.pydeps_hfhub}"
if [ -d "${_PSRL_HFHUB}/huggingface_hub" ]; then
    case ":${PYTHONPATH:-}:" in
        *":${_PSRL_HFHUB}:"*) ;;
        *) export PYTHONPATH="${_PSRL_HFHUB}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
fi
unset _PSRL_CONDA_LIB _PSRL_NIXL_PLUGINS _PSRL_CLUSTER_ENV _PSRL_NCCL_SO _PSRL_HFHUB
