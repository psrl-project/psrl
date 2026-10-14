# Contributing to PSRL

Thank you for considering a contribution to PSRL! We welcome contributions of all kinds — bug fixes, new features, documentation improvements, or even just feedback. Whether you're experienced with distributed training frameworks or just getting started, your help is appreciated.

Your support can take many forms:
- Report bugs or unexpected behaviors.
- Propose or implement new features.
- Improve documentation or add examples.
- Review pull requests and help other contributors.

## Code Linting and Formatting

PSRL uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, enforced via [pre-commit](https://pre-commit.com/). Code style settings:

- **Line length:** 119 characters
- **Rules:** pycodestyle (E), Pyflakes (F), pyupgrade (UP), flake8-bugbear (B), isort (I)

### Setup

```bash
pip install pre-commit
pre-commit install
```

### Running

```bash
# Check and auto-fix staged changes before committing
pre-commit run

# Check and auto-fix all files in the repository
pre-commit run --all-files

# Run only the ruff linter
pre-commit run --all-files ruff-check

# Run only the ruff formatter
pre-commit run --all-files ruff-format
```

The CI `pre-commit` workflow runs pre-commit on changed files on every pull request and push to `main`. All checks must pass before merging.

## Testing

PSRL's test suite is organized under `tests/` by feature area. CI workflows target different hardware tiers.

### Test structure

```
tests/
├── agent_loop/         # Multi-turn agent loop and agent data
├── checkpoint/         # Checkpoint save/load
├── config/             # Trainer and Hydra config
├── converter/          # vLLM weight converter — SupportsWeightLayoutSpec, ParameterMapping
├── dataset/            # Data processor and dataset utilities
├── e2e/                # End-to-end training smoke paths
├── elastic_rm/         # Elastic resource manager / scaling policy
├── environments/       # Environment plugin registry
├── fsdp/               # FSDP1/FSDP2 model loading (GPU, torchrun)
├── gen_dplb/           # StatsRecorder and rollout generation / DP load-balancing stats
├── megatron/           # Megatron model initialization (GPU)
├── mem_agent/          # Memory-agent helpers
├── nixl/               # NIXL communication planner, sharding, e2e (multi-node GPU)
├── parameter_server/   # RequestStatusTracker and PSManager
├── ray_utils/          # Ray actor utilities (real Ray cluster, GPU runner)
├── staleness/          # StalenessBuffer and StalenessInventory
├── state_dict/         # State dict conversion scripts (GPU, torchrun)
├── tito/               # SessionRouter / TITO integration
├── tools/              # MCP tool integration
├── torch_dist/         # torch.distributed broadcast (GPU, torchrun)
├── trainer/            # PSRL_Role enum and trainer config dataclasses
├── unit/               # Small isolated unit tests
└── workers/            # Worker-level tests
```

### Running tests locally

**CPU-only tests** (no GPU, no Ray required):

```bash
pip install -e ".[test]"
pytest -m cpu_test --ignore=tests/nixl --ignore=tests/ray_utils tests/
```

**GPU + Ray integration tests** (requires a GPU machine with the PSRL conda env activated):

```bash
conda activate psrl
pytest -m "not cpu_test" --ignore=tests/nixl -x tests/
```

**NIXL CPU tests** (communication planner logic, no GPU):

```bash
pytest -m cpu_test tests/nixl/ -v
```

**NIXL distributed e2e test** (multi-GPU, multi-node):

```bash
bash tests/nixl/scripts/test_nixl_e2e.sh
```

**torchrun-based tests** (GPU, 2–5 processes):

```bash
bash tests/fsdp/scripts/test_fsdp1_load_model.sh
bash tests/fsdp/scripts/test_fsdp2_load_model.sh
bash tests/torch_dist/scripts/test_broadcast.sh
bash tests/state_dict/scripts/test_vllm_converter_new_api.sh
```

**NIXL send/recv test** (two nodes required — run on each node separately):

```bash
# On target node first:
recv_IP=<target_ip> bash tests/nixl/scripts/test_send_recv_target.sh

# Then on initiator node:
SEND_IP=<initiator_ip> RECV_IP=<target_ip> bash tests/nixl/scripts/test_send_recv_initiator.sh
```

### Test markers

| Marker | Meaning |
|---|---|
| `cpu_test` | Runs on CPU only — no GPU, no Ray cluster required. Selected by `cpu_unit_tests` CI. |
| `unit` | Tests a single, well-isolated functionality. |
| `integration` | Tests elements when integrated into subsystems. |
| `system` | Tests at the highest integration level. |
| `skipduringci` | Skipped in CI but useful for validating local setups. |
| `pleasefixme` | Test is currently broken and needs fixing. |

Mark new CPU-only tests at the module level:

```python
import pytest
pytestmark = pytest.mark.cpu_test
```

### Large-array tests

`tests/ray_utils/test_lazy_primitives.py` and `tests/ray_utils/test_get_ref.py` default to small arrays for CI. To run the original large benchmark sizes (40 GB):

```bash
PSRL_TEST_LARGE_ARRAYS=1 pytest tests/ray_utils/test_lazy_primitives.py tests/ray_utils/test_get_ref.py -v
```

### CI/CD workflows

| Workflow | Trigger | Runner | What it runs |
|---|---|---|---|
| `pre-commit.yml` | PR + push to `main` | `ubuntu-latest` | Ruff lint and format on changed files |
| `smoke_cpu_tests.yml` | PR + push to `main` | `ubuntu-latest` | Minimal CPU smoke pytest (no full PSRL install) |
| `cpu_unit_tests.yml` | PR + push to `main` (`.py` changes) | `ubuntu-latest` | `pytest -m cpu_test` — fast feedback on every PR |
| `gpu_unit_tests.yml` | Push to `main` + dispatch | `self-hosted [gpu]` | All non-`cpu_test` pytest tests (Ray, vLLM, gRPC) |
| `gpu_distributed.yml` | Push to `main` + dispatch | `self-hosted [gpu]` | torchrun tests: FSDP, torch_dist, state_dict converter |
| `nixl_distributed.yml` | NIXL/nixl source changes + dispatch | `self-hosted [gpu, multi-node]` | NIXL pytest tests + nixl_e2e script |

### Adding new tests

1. **Place tests in the appropriate feature directory** under `tests/`. If none fits, create a new one following the naming convention.
2. **Mark CPU-only tests** with `pytestmark = pytest.mark.cpu_test` at the module top. These run on every PR.
3. **Tests requiring GPU or Ray** should have no `cpu_test` marker; they run in `gpu_unit_tests`.
4. **Tests requiring torchrun** belong in a `scripts/` subdirectory as shell scripts, invoked by `gpu_distributed.yml`.
5. **Update the relevant CI workflow** if your new test requires a different runner or path trigger.

## Pull Requests

- Follow the pre-commit lint rules — all checks must pass.
- Add or update tests for any code change; explain in the PR if tests are not applicable.
- Update docstrings and type hints for any public API changes.
- Keep commits focused; prefer smaller PRs over large ones.

## License

See the [LICENSE](LICENSE) file for full details.
