#!/bin/bash
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=60

export TORCH_NCCL_TRACE_BUFFER_SIZE=10
export PYTORCH_NVML_BASED_CUDA_CHECK=1

export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy

export RAY_worker_register_timeout_seconds=360

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

export UCX_NET_DEVICES="bond1,bond2,bond3,bond4,bond5,bond6,bond7,bond8,mlx5_bond_1:1,mlx5_bond_4:1,mlx5_bond_3:1,mlx5_bond_2:1,mlx5_bond_7:1,mlx5_bond_6:1,mlx5_bond_8:1,mlx5_bond_5:1"
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_HCA="mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"
export NCCL_CUMEM_ENABLE=0
export NCCL_ALGO="allgather:^tree"
export NCCL_PRIMS_PROFILE_ENABLE=0
export NCCL_CHECK_DISABLE=1
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_DISABLE=0
export NCCL_IB_SL=3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=VERSION
export NCCL_NET_GDR_READ=1
export NCCL_SOCKET_NTHREADS=8
export NCCL_COLLNET_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0

export SHARP_COLL_ENABLE_SAT=0

export CUDA_DEVICE_MAX_CONNECTIONS=1

export no_proxy="127.0.0.1,localhost"
NODE_IPS=$(echo $NODE_IP_LIST | sed 's/:8//g')
EXISTING_NO_PROXY=$no_proxy
if [ -n "$EXISTING_NO_PROXY" ]; then
    NEW_NO_PROXY="$EXISTING_NO_PROXY,$NODE_IPS"
else
    NEW_NO_PROXY="$NODE_IPS"
fi
export no_proxy="$NEW_NO_PROXY"