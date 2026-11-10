# Third-party patches

Each core dependency has one complete patch for the upstream baseline below.

| Dependency | Upstream baseline | Patch |
|---|---|---|
| UCX | v1.20.0 (`4b7a6ca8410f`) | `ucx/v1.20.0.patch` |
| NIXL | 0.10.1 (`d5c127e558fd`) | `nixl/0.10.1.patch` |
| veRL | `e5ca4acb1534` | `verl/e5ca4acb.patch` |
| vLLM | v0.22.0 (`0b3ba88f1659`) | `vllm/v0.22.0.patch` |

`deprecated/` is an archive and is excluded from patch selection.
The vLLM Python plugin in `vllm/vllm_plugin/` is installed separately by
`scripts/install_basic.sh` and remains required.

## Install

From the repository root, in a Python 3.12 conda environment:

```bash
bash scripts/install_basic.sh
bash scripts/install_nixl.sh
source scripts/activate_nixl.sh
bash scripts/install_megatron.sh
python -m pip install -e '.[test]'
python scripts/check_environment.py
python -m pytest tests
```

The full test suite imports the Megatron backend during collection.
See [the installation guide](../docs/tutorial/installation.md) for build prerequisites
and optional LMCache setup. GPU tests require an available NVIDIA driver
and the explicit `--run-gpu` option.

The installer pins NCCL 2.29.7 for communicator suspend and resume.
`check_environment.py` accepts only its known PyTorch wheel metadata mismatch,
checks both native symbols, and rejects all other dependency conflicts.

## Apply a patch

For other installed dependencies, the shared patch helper remains available:

```bash
bash patch/apply_patch.sh lm_cache --list
```

## Regenerate a patch

Use a clean index or a disposable checkout. Include added source files and tests,
and dependency declarations. Exclude build outputs and local editor settings.

```bash
git -C third_party/vllm add vllm requirements/cuda.txt tests/v1/streaming_input/test_async_llm_streaming.py
git -C third_party/vllm diff --cached --binary --full-index \
  0b3ba88f165976e77ca5e6a7a3f5bba4562b80af > patch/vllm/v0.22.0.patch
```

Update the baseline table and the installer checkout when changing the upstream
baseline. Verify on a newly fetched checkout and run the repository tests before
replacing an active patch.
