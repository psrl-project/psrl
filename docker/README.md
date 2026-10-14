# PSRL Docker images

PSRL provides reproducible GPU and CPU images. The Docker build context is the
repository root, and the Dockerfiles install the pinned third-party components
and PSRL itself. No extra `pip install` step is required after the image is built.

## Prerequisites

- Docker with BuildKit enabled.
- For the GPU image: an NVIDIA driver (R580 or newer) and the NVIDIA Container
  Toolkit.
- Sufficient disk space and build time for the native CUDA dependencies.

## Build an image

Run these commands from the repository root.

### GPU image

```bash
docker build --progress=plain \
  -f docker/Dockerfile \
  -t psrl:latest .
```

The GPU image includes CUDA 13.0.3, PyTorch 2.11, vLLM, veRL, NIXL/UCX,
LMCache, Megatron, SMG, and the PSRL runtime. It is the image to use for
training and multi-node rollout.

For a release or a reproducible build, pin the PSRL revision explicitly:

```bash
docker build --progress=plain \
  --build-arg PSRL_COMMIT=<commit-sha> \
  -f docker/Dockerfile \
  -t psrl:latest .
```

### CPU image

```bash
docker buildx build --load --progress=plain \
  --platform=linux/amd64 \
  -f docker/Dockerfile.cpu \
  -t psrl:cpu .
```

The CPU image is intended for imports, configuration checks, unit tests, and
gateway/data-plane development. It does not support PSRL GPU training.

## Start a container

The image already contains PSRL at `/home/psrl`.

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/home/psrl" \
  -it psrl:latest
```

Inside the container, run the usual PSRL commands from `/home/psrl`:

```bash
bash examples/ray/ray_start.sh ${PSRL_WORKSPACE}/hosts/16GPUs
bash examples/dapo_trainer/qwen2.5_3b_fsdp.sh
```

For a CPU shell, omit `--gpus all` and use the CPU tag:

```bash
docker run --rm -it \
  --ipc=host \
  --network=host \
  -v "$PWD:/home/psrl" \
  psrl:cpu
```

For multi-node NIXL/UCX/Mooncake traffic, start the same image on every node, and expose the host's RDMA devices using your cluster's standard Docker configuration (for example, `--device=/dev/infiniband`).

## Notes

- The image starts a shell and does not launch Ray or a training job
  automatically.
- The GPU image uses a CUDA 13.0 / PyTorch `cu130` stack. If you override
  `CUDA_VERSION` or `BASE_IMAGE`, revalidate the matching PyTorch wheel and
  constraint settings in `docker/`.
- When changing a pinned third-party revision, rebase the matching patch in
  `docker/patch/` and rebuild the image from a clean cache.
