"""Read-only harness runtime trees mounted into task sandboxes.

Each harness ships as a self-contained runtime tree on the worker's shared
filesystem, laid out by convention as ``<root>/<kind>/bin/<executable>``. A task
sandbox binds that one tree read-only, so the harness never installs into (or
mutates) the task image's global toolchain: no Node, no npm, no global prefix.

The container mount point is configuration. The host root is the single
environment input. Every sandbox path is derived from those two values.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath

from psrl.sandbox import MountSpec

# Host directory holding one runtime tree per harness kind.
RUNTIME_ROOT_ENV = "PSRL_HARNESS_RUNTIME_ROOT"

# Convention inside a runtime tree: executables live under ``bin/``.
_EXECUTABLE_SUBDIR = "bin"


def host_runtime_dir(kind: str) -> str:
    """Return the host runtime-tree directory for one harness kind."""
    root = os.environ.get(RUNTIME_ROOT_ENV, "").strip()
    if not root:
        raise RuntimeError(
            f"{RUNTIME_ROOT_ENV} is not set; point it at the directory that holds the "
            f"per-harness runtime trees (expected layout: <root>/{kind}/{_EXECUTABLE_SUBDIR}/..."
            "). Run examples/mini_swe/prepare/docker_scripts/build_harness_runtimes.sh."
        )
    path = os.path.join(os.path.expanduser(root), kind)
    if not os.path.isdir(path):
        raise RuntimeError(
            f"Harness runtime tree for {kind!r} is missing on this worker: {path!r}. "
            "Run examples/mini_swe/prepare/docker_scripts/build_harness_runtimes.sh."
        )
    return path


def executable_path(runtime_mount: str, executable: str) -> str:
    """Return the in-sandbox path of one harness executable."""
    return str(PurePosixPath(runtime_mount) / _EXECUTABLE_SUBDIR / executable)


def runtime_mount_spec(kind: str, runtime_mount: str) -> MountSpec:
    """Bind one harness runtime tree read-only into the sandbox."""
    return MountSpec(host_runtime_dir(kind), runtime_mount, read_only=True)
