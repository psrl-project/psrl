# Installation

PSRL supports two installation paths:

- **Docker:** a reproducible image with the pinned runtime and native
  dependencies already installed.
- **From source:** a conda-based setup for development or custom dependency versions.

## Install with Docker

The Docker build context is the repository root. Build the image that matches your
workload, then run PSRL inside it.

### Prerequisites

- Docker with BuildKit enabled.
- For GPU training, an NVIDIA driver (R580 or newer) and the NVIDIA Container Toolkit.
- A persistent host directory for models, datasets, and checkpoints.

### Build the image

GPU image:

```bash
docker build --progress=plain \
  -f docker/Dockerfile \
  -t psrl:latest .
```

CPU image (for imports, tests, and gateway/data-plane development):

```bash
docker buildx build --load --progress=plain \
  --platform=linux/amd64 \
  -f docker/Dockerfile.cpu \
  -t psrl:cpu .
```

The GPU image contains the full PSRL stack, including CUDA, PyTorch, vLLM, veRL,
NIXL/UCX, LMCache, Megatron, SMG, and PSRL. The CPU image intentionally omits
CUDA-only libraries and cannot run GPU training.

### Start the container

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/home/psrl" \
  -it psrl:latest
```

The image already installs PSRL at `/home/psrl`, so no additional `pip install`
command is needed. For a CPU shell, omit `--gpus all` and replace `psrl:latest` with
`psrl:cpu`.

For multi-node NIXL/UCX/Mooncake traffic, start the same image on every node and use
your cluster's standard RDMA device configuration (for example,
`--device=/dev/infiniband`).

## Install from source

PSRL's source installation combines pinned PyTorch, vLLM, veRL, SMG, NIXL, and
optional KV-cache components.

### Prerequisites

| Requirement | Notes |
|---|---|
| OS | Ubuntu 22.04+ |
| GPU | NVIDIA GPU supported by the pinned PyTorch/CUDA stack |
| CUDA | CUDA 12.8-compatible driver/toolchain |
| Python | 3.12 recommended |
| Rust/Cargo | Required to build SMG's Rust gateway and Python binding |
| Build tools | Git, GCC 11+ compatible with CUDA, CMake/Ninja, Autotools, and sufficient disk space |
| Multi-node network | InfiniBand/RoCE recommended for `nixl_cpu` and LMCache P2P |

Install Rust before running the core installer:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

### Create the Environment

```bash
conda create -n psrl python=3.12
conda activate psrl
```

All nodes in a Ray cluster must use the same environment and see the same PSRL
checkout or installed package.

### Core Installation

```bash
bash scripts/install_basic.sh
python -m pip install -e '.[test]'
```

`install_basic.sh` currently installs:

- PyTorch 2.11.0 for CUDA 12.8, Triton, TensorDict, FlashAttention, FlashInfer,
  Apex, and common Python dependencies.
- SMG, including the release Rust gateway, Python
  binding, gRPC protocol/client package, PSRL state protocol, and gRPC servicer.
- vLLM `v0.22.0` and the PSRL vLLM patches.
- A pinned veRL checkout and the PSRL veRL patches. The pinned veRL requirements install `TransferQueue==0.1.8` during the core installation. `SimpleStorage` is the default TransferQueue backend.
- `torch_memory_saver`.

The installer clones dependencies under `third_party/` and applies the vLLM and
veRL patches before building. Set `VLLM_PATH` or `VERL_PATH` to use an existing
checkout at the baseline listed in [the patch guide](../../patch/README.md).

### SMG Requirements

SMG is the mandatory online request path:

- A `RolloutGateway` Ray actor starts the SMG Router process automatically.
- Rollout instances register as gRPC workers, so SMG's gRPC client/proto and
  servicer packages must be installed.
- The PSRL worker selector uses the SMG `psrl-state` gRPC protocol to contact
  PSManager.
- SessionRouter/TITO-based agent loops require an SMG build with TITO enabled.

The core installer performs the equivalent of:

```bash
cd third_party/smg
cargo build --release
python -m pip install -e crates/grpc_client/python/
python -m pip install -e crates/psrl_state/python/
python -m pip install -e grpc_servicer/
cd bindings/python
maturin develop --features vendored-openssl
```

For SMG development, rebuild from the exact SMG checkout used by PSRL, then restart
the training job so the gateway subprocess loads the new binding.

### Optional Performance Components

Run these after the core installer:

::::{tab-set}

:::{tab-item} NIXL / RDMA

```bash
bash scripts/install_nixl.sh
source scripts/activate_nixl.sh
```

Required for `psrl.ps_mode=nixl_cpu`. NIXL/UCX selects local shared-memory/IPC paths
where possible and RDMA-capable transports across nodes.

For RDMA jobs, set `UCX_NET_DEVICES` to the ports allocated to the job.
Use `ucx_info -d` to list available devices. Selecting every interface on hosts
with many HCAs can exceed UCX's transport-resource limit.
:::

:::{tab-item} Megatron
:sync: megatron

Installs Megatron as an alternative training backend for large models.

```bash
bash scripts/install_megatron.sh
```

Required for Megatron actor/critic training with TP, PP, CP, or EP.
:::

:::{tab-item} LMCache
:sync: lmcache

Installs LMCache for KV cache offloading to CPU/disk and cross-instance P2P KV
transfer. Dramatically reduces re-prefill cost in multi-turn agentic RL workloads.

```bash
bash scripts/install_lmcache.sh
```

Required for `psrl.lmcache.enable=True` and cross-instance KV transfer. P2P transfer
with `p2p_transfer_channel=nixl` also requires NIXL/UCX.
:::

::::

### Verification

Run the checks below in sequence. Commands marked **(optional)** apply only if you
installed the corresponding component.

**Core**

```bash
# Python and CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# PSRL package
python -c "import psrl; print('PSRL:', psrl.__file__)"

# vLLM (patched third-party copy)
python -c "import vllm; print('vLLM:', vllm.__version__)"

# Ray
python -c "import ray; ray.init(); print('Ray resources:', ray.cluster_resources()); ray.shutdown()"

# veRL (patched third-party copy)
python -c "import verl; print('veRL:', verl.__file__)"

# FlashAttention
python -c "import flash_attn; print('FlashAttention:', flash_attn.__version__)"

# TMS (torch_memory_saver)
python -c "import torch_memory_saver; print('TMS:', torch_memory_saver.__file__)"
```

**NIXL** *(optional, installed via `install_nixl.sh`)*

```bash
python -c "import nixl; print('NIXL OK')"
```

**Megatron** *(optional, installed via `install_megatron.sh`)*

```bash
python -c "import megatron; print('Megatron:', megatron.__file__)"
```

**LMCache** *(optional, installed via `install_lmcache.sh`)*

```bash
python -c "import lmcache; print('LMCache:', lmcache.__file__)"
```

Run dependency and unit checks in the same environment:

```bash
python scripts/check_environment.py
python -m pytest tests
```

GPU tests are skipped by default. On a GPU node, add `--run-gpu` and provide
`--gpu-model-path` for tests that need a local checkpoint.
Proceed to the {doc}`quickstart` to run your first training job.
