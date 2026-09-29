"""Backend-neutral sandbox contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SandboxFeature(str, Enum):
    """Optional semantic features exposed by a sandbox backend."""

    # Suspend processes while retaining the sandbox's allocated host resources.
    FREEZE = "freeze"
    # Persist a sandbox so its compute resources can be released and later reacquired.
    HIBERNATE = "hibernate"
    # Capture durable filesystem state without promising process or memory state.
    FILESYSTEM_SNAPSHOT = "filesystem_snapshot"
    # Capture filesystem, process, and memory state as one restorable checkpoint.
    FULL_STATE_SNAPSHOT = "full_state_snapshot"
    # Create a new sandbox from a previously captured snapshot.
    RESTORE = "restore"
    # Clone a live sandbox through a backend-native copy-on-write operation.
    NATIVE_FORK = "native_fork"
    # Bind a path on the local host into the sandbox.
    HOST_MOUNT = "host_mount"


class SandboxSourceKind(str, Enum):
    """Portable sandbox source kinds."""

    # OCI-style image containing the sandbox filesystem and startup metadata.
    IMAGE = "image"
    # Provider-managed template that may include prebuilt runtime configuration.
    TEMPLATE = "template"


class SnapshotKind(str, Enum):
    """State included in a snapshot."""

    # Filesystem contents only. Running process and memory state are excluded.
    FILESYSTEM = "filesystem"
    # Filesystem, running processes, and memory state when supported by the backend.
    FULL_STATE = "full_state"


class PauseMode(str, Enum):
    """Pause semantics requested by the caller."""

    # Stop process scheduling but keep the sandbox resident on the current host.
    FREEZE = "freeze"
    # Release compute resources while retaining enough state for a later resume.
    HIBERNATE = "hibernate"


class SandboxStatus(str, Enum):
    """Backend-neutral session state."""

    # The sandbox accepts commands.
    RUNNING = "running"
    # The sandbox exists but cannot execute commands until resumed.
    PAUSED = "paused"
    # The sandbox workload exited but its runtime object still exists.
    EXITED = "exited"
    # The sandbox runtime object has been destroyed.
    TERMINATED = "terminated"
    # The provider cannot map its current state to a portable state.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SandboxStatePolicy:
    """Safety requirements for full-state snapshots and branches."""

    enabled: bool = False
    allow_secret_capture: bool = False
    allow_external_side_effects: bool = False
    reseed_after_restore: bool = True


@dataclass(frozen=True)
class SandboxRef:
    """Stable reference to a backend-owned sandbox."""

    backend: str
    sandbox_id: str


@dataclass(frozen=True)
class SnapshotRef:
    """Opaque reference to backend-owned snapshot state."""

    backend: str
    snapshot_id: str
    kind: SnapshotKind
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxSource:
    """Source from which a sandbox is provisioned."""

    kind: SandboxSourceKind
    reference: str

    def __post_init__(self) -> None:
        if not self.reference.strip():
            raise ValueError("Sandbox source reference cannot be empty.")

    @classmethod
    def image(cls, image: str) -> SandboxSource:
        """Create an image-backed source."""
        return cls(SandboxSourceKind.IMAGE, image)

    @classmethod
    def template(cls, template: str) -> SandboxSource:
        """Create a provider-template-backed source."""
        return cls(SandboxSourceKind.TEMPLATE, template)


@dataclass(frozen=True)
class ResourceSpec:
    """Optional portable resource requests."""

    cpu_count: float | None = None
    memory_mb: int | None = None
    disk_mb: int | None = None

    def __post_init__(self) -> None:
        if self.cpu_count is not None and self.cpu_count <= 0:
            raise ValueError("Sandbox cpu_count must be greater than zero.")
        if self.memory_mb is not None and self.memory_mb <= 0:
            raise ValueError("Sandbox memory_mb must be greater than zero.")
        if self.disk_mb is not None and self.disk_mb <= 0:
            raise ValueError("Sandbox disk_mb must be greater than zero.")


@dataclass(frozen=True)
class MountSpec:
    """Local host bind mount requested by a sandbox workload."""

    source: str
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class SandboxSpec:
    """Portable sandbox creation request."""

    source: SandboxSource
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    workdir: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[MountSpec, ...] = ()
    policy_profile: str | None = None
    idle_timeout_s: float | None = None
    idempotency_key: str | None = None
    state_policy: SandboxStatePolicy = field(default_factory=SandboxStatePolicy)
    required_features: frozenset[SandboxFeature] = frozenset()

    def __post_init__(self) -> None:
        if self.idle_timeout_s is not None and self.idle_timeout_s <= 0:
            raise ValueError("Sandbox timeout must be greater than zero.")
        if self.idempotency_key is not None and not self.idempotency_key.strip():
            raise ValueError("Sandbox idempotency_key cannot be empty.")


@dataclass(frozen=True)
class ExecResult:
    """Completed command result."""

    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ResourceUsage:
    """Point-in-time resource usage reported by a backend."""

    memory_bytes: int = 0
    peak_memory_bytes: int = 0
    cpu_total_ns: int = 0


@dataclass(frozen=True)
class SandboxCapabilities:
    """Semantic capabilities implemented by one backend or session."""

    features: frozenset[SandboxFeature] = frozenset()

    def supports(self, feature: SandboxFeature) -> bool:
        """Return whether the feature is implemented with its declared semantics."""
        return feature in self.features

    def require(self, *features: SandboxFeature) -> None:
        """Raise when one or more required features are unavailable."""
        missing = [feature.value for feature in features if feature not in self.features]
        if missing:
            raise RuntimeError(f"Sandbox does not support: {', '.join(sorted(missing))}.")


class SandboxSession(ABC):
    """One live execution environment.

    Command and file operations form the required data plane. State operations
    have default unsupported implementations and are enabled by capabilities,
    which keeps simple backends free of state-operation boilerplate.
    """

    @property
    @abstractmethod
    def ref(self) -> SandboxRef:
        """Return the stable backend reference."""

    @property
    @abstractmethod
    def capabilities(self) -> SandboxCapabilities:
        """Return features implemented by this session."""

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        """Execute a command and collect its result."""

    @abstractmethod
    async def read_bytes(self, path: str) -> bytes:
        """Read a file without assuming text encoding."""

    @abstractmethod
    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write a complete file."""

    @abstractmethod
    async def status(self) -> SandboxStatus:
        """Return the current session status."""

    @abstractmethod
    async def terminate(self) -> None:
        """Idempotently destroy the session."""

    async def stats(self) -> ResourceUsage:
        """Return resource usage when supported."""
        return ResourceUsage()

    def resolve_callback_url(self, url: str) -> str:
        """Translate a worker URL into an equivalent URL reachable from this sandbox."""
        return url

    @property
    def spec(self) -> SandboxSpec | None:
        """Return the creation spec when this process created the session."""
        return None

    @property
    def command_count(self) -> int:
        """Return the number of user commands executed by this session object."""
        return 0

    async def refresh_transport(self) -> None:
        """Drop stale provider connections after restoring VM state."""
        return None

    async def pause(self, mode: PauseMode) -> None:
        """Pause the session with explicit semantics."""
        raise NotImplementedError(f"Sandbox {self.ref} does not support {mode.value}.")

    async def resume(self) -> None:
        """Resume a paused session."""
        raise NotImplementedError(f"Sandbox {self.ref} does not support resume.")

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        """Capture session state."""
        raise NotImplementedError(f"Sandbox {self.ref} does not support {kind.value} snapshots.")

    async def fork(self) -> SandboxSession:
        """Create a backend-native branch of the session."""
        raise NotImplementedError(f"Sandbox {self.ref} does not support native fork.")


class SandboxBackend(ABC):
    """Provisioning and reconnection boundary for one runtime backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique manager registration name."""

    @property
    @abstractmethod
    def capabilities(self) -> SandboxCapabilities:
        """Return capabilities available to newly created sessions."""

    @property
    def uses_node_capacity(self) -> bool:
        """Return whether sessions consume resources on the worker's node."""
        return False

    @abstractmethod
    async def create(self, spec: SandboxSpec) -> SandboxSession:
        """Create a session from a portable specification."""

    @abstractmethod
    async def connect(self, sandbox_id: str) -> SandboxSession:
        """Connect to and, when necessary, resume a session."""

    async def prepare(self, spec: SandboxSpec) -> None:
        """
        Warm reusable artifacts without allocating a sandbox or capacity lease.

        Backends without artifact preparation may leave this as a no-op.
        """
        return None

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        """Create a session from a snapshot when supported."""
        raise NotImplementedError(f"Backend {self.name!r} does not support restore.")

    async def shutdown(self) -> None:
        """Release backend-level resources."""
        return None

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """Delete a provider-owned snapshot when supported."""
        raise NotImplementedError(f"Backend {self.name!r} does not support snapshot deletion.")

    def metrics_snapshot(self):
        """Return backend-local metrics without imposing an exporter."""
        from psrl.sandbox.metrics import SandboxMetricsSnapshot

        return SandboxMetricsSnapshot()
