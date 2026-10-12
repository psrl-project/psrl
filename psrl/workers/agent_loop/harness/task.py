"""Task contract consumed by the generic harness agent loop."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from psrl.sandbox import SandboxSpec, SnapshotKind

TaskStateT = TypeVar("TaskStateT")


@dataclass(frozen=True)
class HarnessTaskContext(Generic[TaskStateT]):
    """
    Store an immutable task description with opaque task-specific state.

    The context does not own generic runtime resources. `HarnessAgentLoop` owns
    the session, sandbox lease, harness process, and snapshot lifecycles. The
    opaque `state` may carry task resources released by `close_harness_task`.
    """

    state: TaskStateT
    prompt: str
    sandbox_spec: SandboxSpec
    backend: str | None = None
    clean_sandbox_spec: SandboxSpec | None = None
    collect_resource_metrics: bool = False
    runtime_mount_target: str | None = None


def clean_snapshot_compatible(task: HarnessTaskContext, kind: SnapshotKind | None = None) -> bool:
    """
    Return whether the rollout snapshot can seed the task's clean sandbox.

    A ``FILESYSTEM`` snapshot (e.g. a docker commit) is self-contained: restore
    recreates the container from the committed image with the *caller's* source
    reference and resource flags, so the rollout and clean specs may differ in
    both. (This is what lets a lightweight rollout sandbox seed a heavier
    grader sandbox.) A ``FULL_STATE`` snapshot (microVM) must instead be rebuilt
    with the exact same source and resources, so those are compared.

    Mounts: docker commit does not capture host bind mounts, so the committed
    image can only seed the grader when the rollout carries no content-bearing
    mounts the grader relies on. The read-only harness runtime mount is
    excluded. The grader never needs it.
    """
    clean_spec = task.clean_sandbox_spec
    if not (task.sandbox_spec.state_policy.enabled and clean_spec is not None and clean_spec.state_policy.enabled):
        return False
    if not _mounts_compatible(task.sandbox_spec.mounts, clean_spec.mounts, task.runtime_mount_target):
        return False
    if kind == SnapshotKind.FILESYSTEM:
        return True
    if task.sandbox_spec.resources != clean_spec.resources:
        return False
    if task.sandbox_spec.source != clean_spec.source:
        return False
    return True


def _mounts_compatible(rollout_mounts, clean_mounts, runtime_mount_target: str | None) -> bool:
    """Whether the committed image can satisfy the grader's mount expectations."""
    effective_rollout = [m for m in rollout_mounts if m.target != runtime_mount_target]
    return list(effective_rollout) == list(clean_mounts)
