"""Node-local shared-memory cache for final reward-model weight arenas."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import mmap
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from psrl.utils.weight_arena import WeightArenaHandle, gb_to_bytes

_NETWORK_FILESYSTEMS = {
    "9p",
    "afs",
    "ceph",
    "cifs",
    "fuse.ceph",
    "fuse.glusterfs",
    "fuse.sshfs",
    "gfs",
    "gfs2",
    "glusterfs",
    "gpfs",
    "lustre",
    "nfs",
    "nfs4",
    "ocfs2",
    "smb3",
}
_MEMORY_FILESYSTEMS = {"devtmpfs", "hugetlbfs", "ramfs", "tmpfs"}
_MANIFEST_VERSION = 2
_DEFAULT_SHM_RESERVE_GB = 256
_DEFAULT_MEMFD_RESERVE_GB = 256
_DEFAULT_SHM_RESERVE_BYTES = gb_to_bytes(
    _DEFAULT_SHM_RESERVE_GB,
    field_name="reward_node_cache_shm_reserve_gb",
)
_DEFAULT_MEMFD_RESERVE_BYTES = gb_to_bytes(
    _DEFAULT_MEMFD_RESERVE_GB,
    field_name="reward_node_cache_memfd_reserve_gb",
)


@dataclass(frozen=True)
class WeightCacheTopology:
    """Model-parallel topology identifying one final model shard."""

    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    ep_size: int
    ep_rank: int


@dataclass
class NodeSharedArenaCache:
    """Mapped shared-memory arena views and their namespace lease."""

    arenas: tuple[torch.Tensor, ...]
    cache_key: str
    data_path: Path | None
    inode: int
    mapped_bytes: int
    unique_cache_bytes: int
    builder: bool
    backend: str
    topology: WeightCacheTopology
    job_id: str
    node_id: str
    _mapping: mmap.mmap
    _lease: CacheNamespaceLease
    _backend_release: Callable[[], None] | None = None
    _closed: bool = False

    def close(self) -> None:
        """Close mappings and remove the namespace when this is its last user."""
        if self._closed:
            return
        self.arenas = ()
        self._mapping.close()
        if self._backend_release is not None:
            self._backend_release()
        self._lease.close_and_cleanup()
        self._closed = True

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe sharing and process-memory diagnostics."""
        memory = _process_memory_rollup()
        return {
            "cache_key": self.cache_key,
            "cache_path": str(self.data_path) if self.data_path is not None else f"memfd:{self.cache_key}",
            "backend": self.backend,
            "inode": self.inode,
            "mapped_bytes": self.mapped_bytes,
            "unique_cache_bytes": self.unique_cache_bytes,
            "builder": self.builder,
            "role": "builder" if self.builder else "consumer",
            "topology": asdict(self.topology),
            "job_id": self.job_id,
            "node_id": self.node_id,
            **memory,
        }

    def drop_resident_pages(self) -> bool:
        """Drop this process's clean mappings while retaining shared backing pages."""
        madvise = getattr(self._mapping, "madvise", None)
        advice = getattr(mmap, "MADV_DONTNEED", None)
        if madvise is not None and advice is not None:
            try:
                madvise(advice)
                return True
            except OSError:
                pass
        return False


class CacheNamespaceLease:
    """Shared process lease for one Ray job's node-local cache namespace."""

    def __init__(self, root: Path, job_id: str, timeout_s: float):
        self.root = root
        self.job_id = _safe_component(job_id)
        self.namespace = root / self.job_id
        self._lease_path = root / ".leases" / f"{self.job_id}.lock"
        self._timeout_s = timeout_s
        self._closed = False
        self._lease_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _cleanup_stale_namespaces(root, timeout_s)
        self._fd = os.open(self._lease_path, os.O_RDWR | os.O_CREAT, 0o600)

        first_user = _try_lock(self._fd, fcntl.LOCK_EX)
        if first_user:
            shutil.rmtree(self.namespace, ignore_errors=True)
            self.namespace.mkdir(parents=True, exist_ok=True, mode=0o700)
            fcntl.flock(self._fd, fcntl.LOCK_SH)
        else:
            _lock_with_timeout(self._fd, fcntl.LOCK_SH, timeout_s, self._lease_path)
            self.namespace.mkdir(parents=True, exist_ok=True, mode=0o700)

    def close_and_cleanup(self) -> None:
        """Release this process and delete cache data when no users remain."""
        if self._closed:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        if _try_lock(self._fd, fcntl.LOCK_EX):
            shutil.rmtree(self.namespace, ignore_errors=True)
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._closed = True


def validate_reward_cache_config(config: dict[str, Any]) -> str:
    """Validate reward CPU cache mode and node-shared prerequisites."""
    cache_mode = config.get("reward_cpu_cache_mode", "node_shared")
    if cache_mode not in ("node_shared", "process_local"):
        raise ValueError(
            "psrl.nixl.weight_arena.reward_cpu_cache_mode must be 'node_shared' or "
            f"'process_local', got {cache_mode!r}."
        )
    if cache_mode == "process_local":
        return cache_mode
    if not config.get("reward_enabled", False):
        raise RuntimeError("Node-shared reward CPU cache requires reward_enabled=true.")
    if config.get("reward_cpu_cache_pin_memory", False):
        raise RuntimeError("Node-shared reward CPU cache requires reward_cpu_cache_pin_memory=false.")
    wait_timeout_s = float(config.get("reward_node_cache_wait_timeout_s", 1800))
    if wait_timeout_s <= 0:
        raise RuntimeError("reward_node_cache_wait_timeout_s must be positive in node-shared reward cache mode.")
    validate_node_cache_directory(config.get("reward_node_cache_dir", "/dev/shm/psrl-rm-weight-cache"))
    backend = config.get("reward_node_cache_backend", "auto")
    if backend not in ("auto", "shm", "memfd"):
        raise RuntimeError(f"reward_node_cache_backend must be 'auto', 'shm', or 'memfd', got {backend!r}.")
    for key, default in (
        ("reward_node_cache_shm_reserve_gb", _DEFAULT_SHM_RESERVE_GB),
        ("reward_node_cache_memfd_reserve_gb", _DEFAULT_MEMFD_RESERVE_GB),
    ):
        try:
            gb_to_bytes(config.get(key, default), field_name=key, allow_zero=True)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    if backend in ("auto", "memfd") and not hasattr(os, "memfd_create"):
        raise RuntimeError("memfd reward cache backend requires Linux os.memfd_create support.")
    return cache_mode


def validate_node_cache_directory(cache_root: str | os.PathLike[str]) -> Path:
    """Create and validate a node-local memory-backed cache root.

    Args:
        cache_root (str | os.PathLike[str]): Configured cache root.

    Returns:
        Path: Resolved cache root.

    Raises:
        RuntimeError: If the path is not backed by a local memory filesystem.
    """
    root = Path(cache_root).expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = root.resolve()
    if root.stat().st_uid != os.geteuid():
        raise RuntimeError(f"Reward node cache root must be owned by the current user: {root}.")
    fs_type, mount_point = _filesystem_for_path(root)
    if fs_type not in _MEMORY_FILESYSTEMS:
        storage_kind = (
            "network" if (fs_type in _NETWORK_FILESYSTEMS or fs_type.startswith(("fuse.s3", "fuse.gcs"))) else "disk"
        )
        raise RuntimeError(
            f"Reward node cache must use node-local shared memory, but {root} is on {storage_kind} "
            f"filesystem {fs_type!r} mounted at {mount_point}."
        )
    return root


def fingerprint_safetensors_checkpoint(checkpoint_path: str | os.PathLike[str]) -> str:
    """Fingerprint a safetensors checkpoint without reading tensor payloads."""
    root = Path(checkpoint_path).resolve()
    if root.is_file():
        files = [root] if root.suffix == ".safetensors" else []
        relative_root = root.parent
    else:
        files = sorted(path for path in root.rglob("*.safetensors") if path.is_file())
        relative_root = root
    if not files:
        raise RuntimeError(f"Node-shared reward CPU cache requires a safetensors checkpoint, got {root}.")

    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(str(path.relative_to(relative_root)).encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    for index_path in sorted(root.glob("*.safetensors.index.json")) if root.is_dir() else ():
        digest.update(index_path.name.encode())
        digest.update(index_path.read_bytes())
    config_path = root / "config.json" if root.is_dir() else None
    if config_path is not None and config_path.is_file():
        digest.update(config_path.name.encode())
        digest.update(config_path.read_bytes())
    return digest.hexdigest()


def arena_layout_fingerprint(handle: WeightArenaHandle) -> str:
    """Fingerprint final arena sizes and storage placements."""
    payload = {
        "arenas": [
            {"shape": list(arena.shape), "dtype": str(arena.dtype), "nbytes": _tensor_nbytes(arena)}
            for arena in handle.arenas
        ],
        "placements": [
            {
                "arena_index": placement.arena_index,
                "offset_bytes": placement.offset_bytes,
                "nbytes": placement.nbytes,
                "binding_names": placement.binding_names,
            }
            for placement in handle.placements
        ],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@torch.no_grad()
def publish_or_map_weight_arena(
    handle: WeightArenaHandle,
    *,
    cache_root: str | os.PathLike[str],
    job_id: str,
    node_id: str,
    model_id: str,
    checkpoint_fingerprint: str,
    model_dtype: str,
    quantization: str | None,
    topology: WeightCacheTopology,
    timeout_s: float,
    populate_handle: Callable[[], None] | None = None,
    backend: str = "auto",
    shm_reserve_bytes: int = _DEFAULT_SHM_RESERVE_BYTES,
    memfd_reserve_bytes: int = _DEFAULT_MEMFD_RESERVE_BYTES,
) -> NodeSharedArenaCache:
    """Publish one final GPU arena shard or map an existing identical shard.

    When ``populate_handle`` is provided, it runs under the per-shard build lock
    only when the shared arena does not exist. Consumers of the same topology
    shard wait for that builder, while distinct topology shards build in
    parallel.
    """
    if timeout_s <= 0:
        raise ValueError(f"Reward node cache timeout must be positive, got {timeout_s}.")
    if backend not in ("auto", "shm", "memfd"):
        raise ValueError(f"Unsupported reward node cache backend: {backend!r}.")
    if shm_reserve_bytes < 0 or memfd_reserve_bytes < 0:
        raise ValueError("Reward node cache reserve values must be non-negative.")
    handle.assert_virtual_addresses_unchanged()
    root = validate_node_cache_directory(cache_root)
    lease = CacheNamespaceLease(root, job_id, timeout_s)
    layout_fingerprint = arena_layout_fingerprint(handle)
    key_payload = {
        "version": _MANIFEST_VERSION,
        "job_id": job_id,
        "model_id": model_id,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "dtype": model_dtype,
        "quantization": quantization,
        "topology": asdict(topology),
        "arena_layout": layout_fingerprint,
    }
    cache_key = hashlib.sha256(_canonical_json(key_payload)).hexdigest()
    shard_dir = lease.namespace / cache_key[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)
    data_path = shard_dir / f"{cache_key}.arena"
    manifest_path = shard_dir / f"{cache_key}.json"
    lock_path = shard_dir / f"{cache_key}.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    builder = False
    selected_backend = "shm"
    mapping: mmap.mmap | None = None
    arenas: tuple[torch.Tensor, ...] | None = None
    data_path_result: Path | None = None
    inode = 0
    unique_cache_bytes = 0
    backend_release: Callable[[], None] | None = None
    try:
        _lock_with_timeout(lock_fd, fcntl.LOCK_EX, timeout_s, lock_path)
        if backend == "memfd":
            manifest = None
        else:
            try:
                manifest = _read_manifest(manifest_path)
            except RuntimeError:
                manifest_path.unlink(missing_ok=True)
                data_path.unlink(missing_ok=True)
                manifest = None
        if manifest is not None:
            try:
                _validate_manifest(manifest, key_payload, data_path, handle)
            except RuntimeError as exc:
                if "fingerprint does not match" in str(exc):
                    raise
                manifest_path.unlink(missing_ok=True)
                data_path.unlink(missing_ok=True)
                manifest = None
        if manifest is None and backend != "shm":
            memfd_result = _try_map_memfd(
                lease,
                cache_key=cache_key,
                expected_key=key_payload,
                handle=handle,
                node_id=node_id,
                timeout_s=timeout_s,
                start_broker=backend == "memfd",
            )
            if memfd_result is not None:
                manifest, mapping, arenas, inode, unique_cache_bytes, backend_release = memfd_result
                selected_backend = "memfd"
        if manifest is None:
            builder = True
            _remove_partial_files(shard_dir, cache_key)
            selected_backend = _select_backend(
                backend,
                handle=handle,
                topology=topology,
                namespace=lease.namespace,
                shm_reserve_bytes=shm_reserve_bytes,
                memfd_reserve_bytes=memfd_reserve_bytes,
                broker_token=_broker_token(lease, node_id),
            )
            if populate_handle is not None:
                populate_handle()
                handle.assert_virtual_addresses_unchanged()
            if selected_backend == "shm":
                try:
                    manifest = _publish_arena(
                        handle,
                        data_path=data_path,
                        manifest_path=manifest_path,
                        cache_key=cache_key,
                        key_payload=key_payload,
                    )
                except (SharedMemoryCapacityError, OSError) as exc:
                    if isinstance(exc, OSError) and exc.errno not in (errno.ENOSPC, errno.ENOMEM):
                        raise
                    if backend == "shm":
                        if isinstance(exc, SharedMemoryCapacityError):
                            raise
                        raise SharedMemoryCapacityError(
                            f"Insufficient shared memory for reward node cache: {exc}."
                        ) from exc
                    _mark_prefer_memfd(lease.namespace)
                    selected_backend = "memfd"
            if selected_backend == "memfd":
                manifest, mapping, arenas, inode, unique_cache_bytes, backend_release = _publish_memfd(
                    handle,
                    lease=lease,
                    node_id=node_id,
                    cache_key=cache_key,
                    key_payload=key_payload,
                    timeout_s=timeout_s,
                    memfd_reserve_bytes=memfd_reserve_bytes,
                    topology=topology,
                )
    except Exception:
        if mapping is not None:
            mapping.close()
        if backend_release is not None:
            backend_release()
        lease.close_and_cleanup()
        raise
    finally:
        os.close(lock_fd)

    if selected_backend == "shm":
        mapping, arenas = _map_arena_file(data_path, manifest["arenas"])
        data_path_result = data_path
        inode = data_path.stat().st_ino
        unique_cache_bytes = _namespace_cache_bytes(lease.namespace)
    assert mapping is not None and arenas is not None
    return NodeSharedArenaCache(
        arenas=arenas,
        cache_key=cache_key,
        data_path=data_path_result,
        inode=inode,
        mapped_bytes=sum(item["nbytes"] for item in manifest["arenas"]),
        unique_cache_bytes=unique_cache_bytes,
        builder=builder,
        backend=selected_backend,
        topology=topology,
        job_id=job_id,
        node_id=node_id,
        _mapping=mapping,
        _lease=lease,
        _backend_release=backend_release,
    )


class SharedMemoryCapacityError(RuntimeError):
    """Raised before writing when the /dev/shm backend cannot reserve its arena."""


def _select_backend(
    requested: str,
    *,
    handle: WeightArenaHandle,
    topology: WeightCacheTopology,
    namespace: Path,
    shm_reserve_bytes: int,
    memfd_reserve_bytes: int,
    broker_token: str,
) -> str:
    if requested == "shm":
        return "shm"
    if requested == "memfd" or (namespace / ".prefer-memfd").exists():
        _validate_memfd_capacity(
            handle,
            topology,
            namespace,
            broker_token,
            memfd_reserve_bytes,
        )
        return "memfd"
    shard_bytes = sum(_tensor_nbytes(arena) for arena in handle.arenas)
    projected_total = shard_bytes * max(topology.tp_size, topology.ep_size) * topology.pp_size
    existing_shm = _namespace_cache_bytes(namespace)
    remaining = max(shard_bytes, projected_total - existing_shm)
    free_bytes = shutil.disk_usage(namespace).free
    if free_bytes >= remaining + shm_reserve_bytes:
        return "shm"
    _mark_prefer_memfd(namespace)
    _validate_memfd_capacity(
        handle,
        topology,
        namespace,
        broker_token,
        memfd_reserve_bytes,
    )
    return "memfd"


def _mark_prefer_memfd(namespace: Path) -> None:
    """Best-effort node-wide hint; low shm space must not block fallback."""
    try:
        (namespace / ".prefer-memfd").touch(mode=0o600, exist_ok=True)
    except OSError as exc:
        if exc.errno not in (errno.ENOSPC, errno.ENOMEM):
            raise


def _validate_memfd_capacity(
    handle: WeightArenaHandle,
    topology: WeightCacheTopology,
    namespace: Path,
    broker_token: str,
    reserve_bytes: int,
) -> None:
    shard_bytes = sum(_tensor_nbytes(arena) for arena in handle.arenas)
    projected_total = shard_bytes * max(topology.tp_size, topology.ep_size) * topology.pp_size
    existing_shm = _namespace_cache_bytes(namespace)
    existing_memfd = _broker_total_bytes(broker_token)
    remaining = max(shard_bytes, projected_total - existing_shm - existing_memfd)
    available = _mem_available_bytes()
    if available < remaining + reserve_bytes:
        raise RuntimeError(
            "Insufficient DRAM for reward memfd cache: "
            f"required={remaining} reserve={reserve_bytes} available={available}."
        )


def _try_map_memfd(
    lease: CacheNamespaceLease,
    *,
    cache_key: str,
    expected_key: dict[str, Any],
    handle: WeightArenaHandle,
    node_id: str,
    timeout_s: float,
    start_broker: bool,
) -> (
    tuple[
        dict[str, Any],
        mmap.mmap,
        tuple[torch.Tensor, ...],
        int,
        int,
        Callable[[], None],
    ]
    | None
):
    token = _broker_token(lease, node_id)
    if start_broker:
        _ensure_broker(token, timeout_s)
    else:
        # Do not start an idle broker merely to probe an auto backend.
        from psrl.utils.node_shared_memfd_broker import request

        try:
            request(token, {"command": "ping"}, timeout_s=min(timeout_s, 0.5))
        except (ConnectionError, FileNotFoundError, OSError, RuntimeError, TimeoutError):
            return None
    identity = _broker_client_identity()
    from psrl.utils.node_shared_memfd_broker import request

    response, fd = request(
        token,
        {"command": "get", "key": cache_key, **identity},
        timeout_s=timeout_s,
    )
    if not response.get("found", False):
        return None
    if fd is None:
        raise RuntimeError("memfd broker reported a cache hit without transferring a descriptor")
    manifest = response["manifest"]
    release = _memfd_release_callback(token, cache_key, identity, timeout_s)
    try:
        _validate_memfd_manifest(manifest, expected_key, fd, handle)
        inode = os.fstat(fd).st_ino
        mapping, arenas = _map_arena_fd(fd, manifest["arenas"])
    except Exception:
        release()
        raise
    finally:
        os.close(fd)
    return (
        manifest,
        mapping,
        arenas,
        inode,
        int(response.get("unique_cache_bytes", manifest["total_bytes"])),
        release,
    )


def _publish_memfd(
    handle: WeightArenaHandle,
    *,
    lease: CacheNamespaceLease,
    node_id: str,
    cache_key: str,
    key_payload: dict[str, Any],
    timeout_s: float,
    memfd_reserve_bytes: int,
    topology: WeightCacheTopology,
) -> tuple[
    dict[str, Any],
    mmap.mmap,
    tuple[torch.Tensor, ...],
    int,
    int,
    Callable[[], None],
]:
    token = _broker_token(lease, node_id)
    _ensure_broker(token, timeout_s)
    _validate_memfd_capacity(
        handle,
        topology,
        lease.namespace,
        token,
        memfd_reserve_bytes,
    )
    arena_layout, total_bytes = _arena_layout(handle)
    flags = getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    fd = os.memfd_create(f"psrl-rm-{cache_key[:16]}", flags)
    release: Callable[[], None] | None = None
    success = False
    try:
        try:
            os.posix_fallocate(fd, 0, total_bytes)
        except AttributeError:
            os.ftruncate(fd, total_bytes)
        except OSError as exc:
            raise RuntimeError(f"Failed to reserve {total_bytes} DRAM bytes for reward memfd cache: {exc}.") from exc
        mapping = mmap.mmap(fd, total_bytes, access=mmap.ACCESS_WRITE)
        try:
            _copy_handle_to_mapping(handle, mapping, arena_layout, total_bytes)
            mapping.flush()
        finally:
            mapping.close()
        _seal_memfd(fd)
        manifest = _arena_manifest(cache_key, key_payload, arena_layout, total_bytes, backend="memfd")
        identity = _broker_client_identity()
        from psrl.utils.node_shared_memfd_broker import request

        response, mapped_fd = request(
            token,
            {"command": "register", "key": cache_key, "manifest": manifest, **identity},
            send_fd=fd,
            timeout_s=timeout_s,
        )
        if mapped_fd is None:
            raise RuntimeError("memfd broker registered an arena without returning a descriptor")
        release = _memfd_release_callback(token, cache_key, identity, timeout_s)
        try:
            inode = os.fstat(mapped_fd).st_ino
            mapped, arenas = _map_arena_fd(mapped_fd, arena_layout)
        finally:
            os.close(mapped_fd)
        success = True
    finally:
        if not success and release is not None:
            # Registration may have succeeded before mapping failed.  Release
            # the broker client reference so the arena can be reclaimed.
            release()
        os.close(fd)
    return (
        manifest,
        mapped,
        arenas,
        inode,
        int(response.get("unique_cache_bytes", total_bytes)),
        release,
    )


def _broker_token(lease: CacheNamespaceLease, node_id: str) -> str:
    value = f"{os.geteuid()}:{lease.job_id}:{node_id}"
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _ensure_broker(token: str, timeout_s: float) -> None:
    from psrl.utils.node_shared_memfd_broker import request

    try:
        request(token, {"command": "ping"}, timeout_s=min(timeout_s, 1.0))
        return
    except (ConnectionError, FileNotFoundError, OSError, RuntimeError):
        pass
    subprocess.Popen(
        [sys.executable, "-m", "psrl.utils.node_shared_memfd_broker", token],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + min(timeout_s, 10.0)
    while time.monotonic() < deadline:
        try:
            request(token, {"command": "ping"}, timeout_s=0.5)
            return
        except (ConnectionError, FileNotFoundError, OSError, RuntimeError):
            time.sleep(0.05)
    raise TimeoutError(f"Timed out starting reward memfd broker {token}.")


def _broker_client_identity() -> dict[str, Any]:
    from psrl.utils.node_shared_memfd_broker import process_start_time

    return {
        "client_id": uuid.uuid4().hex,
        "pid": os.getpid(),
        "start_time": process_start_time(),
    }


def _broker_total_bytes(token: str) -> int:
    try:
        _ensure_broker(token, 10.0)
        from psrl.utils.node_shared_memfd_broker import request

        response, fd = request(token, {"command": "ping"}, timeout_s=1.0)
        if fd is not None:
            os.close(fd)
        return int(response.get("unique_cache_bytes", 0))
    except (ConnectionError, FileNotFoundError, OSError, RuntimeError, TimeoutError):
        return 0


def _memfd_release_callback(
    token: str,
    cache_key: str,
    identity: dict[str, Any],
    timeout_s: float,
) -> Callable[[], None]:
    def release() -> None:
        from psrl.utils.node_shared_memfd_broker import request

        try:
            _, fd = request(
                token,
                {"command": "release", "key": cache_key, **identity},
                timeout_s=min(timeout_s, 5.0),
            )
            if fd is not None:
                os.close(fd)
        except (ConnectionError, FileNotFoundError, OSError, RuntimeError):
            pass

    return release


def _mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read MemAvailable from /proc/meminfo.")


def _seal_memfd(fd: int) -> None:
    required = ("F_ADD_SEALS", "F_SEAL_SEAL", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE")
    if any(not hasattr(fcntl, name) for name in required):
        raise RuntimeError("Kernel/Python does not expose required memfd sealing constants.")
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)


def _publish_arena(
    handle: WeightArenaHandle,
    *,
    data_path: Path,
    manifest_path: Path,
    cache_key: str,
    key_payload: dict[str, Any],
) -> dict[str, Any]:
    arena_layout, offset = _arena_layout(handle)
    free_bytes = shutil.disk_usage(data_path.parent).free
    if free_bytes < offset:
        raise SharedMemoryCapacityError(
            f"Insufficient shared memory for reward node cache: required={offset} free={free_bytes} "
            f"path={data_path.parent}."
        )

    temp_data = data_path.with_name(f".{cache_key}.{os.getpid()}.arena.tmp")
    temp_manifest = manifest_path.with_name(f".{cache_key}.{os.getpid()}.json.tmp")
    fd = os.open(temp_data, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            os.posix_fallocate(fd, 0, offset)
        except AttributeError:
            os.ftruncate(fd, offset)
        except OSError as exc:
            error_type = SharedMemoryCapacityError if exc.errno in (errno.ENOSPC, errno.ENOMEM) else RuntimeError
            raise error_type(f"Failed to reserve {offset} bytes for reward node cache at {temp_data}: {exc}.") from exc
        mapping = mmap.mmap(fd, offset, access=mmap.ACCESS_WRITE)
        try:
            _copy_handle_to_mapping(handle, mapping, arena_layout, offset)
            mapping.flush()
        finally:
            mapping.close()
        os.fsync(fd)
    except Exception:
        temp_data.unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)

    manifest = _arena_manifest(cache_key, key_payload, arena_layout, offset, backend="shm")
    try:
        temp_manifest.write_bytes(_canonical_json(manifest))
        with temp_manifest.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_data, data_path)
        os.replace(temp_manifest, manifest_path)
        _fsync_directory(data_path.parent)
    except Exception:
        temp_data.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
        raise
    return manifest


def _map_arena_file(
    data_path: Path,
    arena_layout: list[dict[str, int]],
) -> tuple[mmap.mmap, tuple[torch.Tensor, ...]]:
    fd = os.open(data_path, os.O_RDONLY)
    try:
        mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)
    finally:
        os.close(fd)
    arenas = tuple(
        torch.frombuffer(mapping, dtype=torch.uint8, count=item["nbytes"], offset=item["offset"])
        for item in arena_layout
    )
    return mapping, arenas


def _map_arena_fd(
    fd: int,
    arena_layout: list[dict[str, int]],
) -> tuple[mmap.mmap, tuple[torch.Tensor, ...]]:
    mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)
    arenas = tuple(
        torch.frombuffer(mapping, dtype=torch.uint8, count=item["nbytes"], offset=item["offset"])
        for item in arena_layout
    )
    return mapping, arenas


def _arena_layout(handle: WeightArenaHandle) -> tuple[list[dict[str, int]], int]:
    layout: list[dict[str, int]] = []
    offset = 0
    for arena in handle.arenas:
        nbytes = _tensor_nbytes(arena)
        layout.append({"offset": offset, "nbytes": nbytes})
        offset += nbytes
    return layout, offset


def _copy_handle_to_mapping(
    handle: WeightArenaHandle,
    mapping: mmap.mmap,
    arena_layout: list[dict[str, int]],
    total_bytes: int,
) -> None:
    mapped = torch.frombuffer(mapping, dtype=torch.uint8, count=total_bytes)
    try:
        for arena, item in zip(handle.arenas, arena_layout, strict=True):
            destination = mapped.narrow(0, item["offset"], item["nbytes"])
            source = arena.reshape(-1).view(torch.uint8)
            destination.copy_(source)
            if arena.device.type == "cuda":
                torch.cuda.synchronize(arena.device)
    finally:
        del mapped


def _arena_manifest(
    cache_key: str,
    key_payload: dict[str, Any],
    arena_layout: list[dict[str, int]],
    total_bytes: int,
    *,
    backend: str,
) -> dict[str, Any]:
    return {
        "version": _MANIFEST_VERSION,
        "cache_key": cache_key,
        "key": key_payload,
        "backend": backend,
        "arenas": arena_layout,
        "total_bytes": total_bytes,
        "builder_pid": os.getpid(),
        "builder_host": socket.gethostname(),
        "created_at_ns": time.time_ns(),
    }


def _validate_manifest(
    manifest: dict[str, Any],
    expected_key: dict[str, Any],
    data_path: Path,
    handle: WeightArenaHandle,
) -> None:
    if manifest.get("version") != _MANIFEST_VERSION:
        raise RuntimeError(f"Unsupported reward node cache manifest version: {manifest.get('version')!r}.")
    if manifest.get("key") != expected_key:
        raise RuntimeError("Reward node cache manifest fingerprint does not match the requested model shard.")
    if not data_path.is_file():
        raise RuntimeError(f"Reward node cache manifest exists without arena data: {data_path}.")
    expected_sizes = [_tensor_nbytes(arena) for arena in handle.arenas]
    actual_sizes = [int(item["nbytes"]) for item in manifest.get("arenas", ())]
    if actual_sizes != expected_sizes:
        raise RuntimeError(f"Reward node cache arena layout changed: expected={expected_sizes} actual={actual_sizes}.")
    if data_path.stat().st_size != sum(expected_sizes):
        raise RuntimeError(
            f"Reward node cache data is truncated: expected={sum(expected_sizes)} "
            f"actual={data_path.stat().st_size} path={data_path}."
        )


def _validate_memfd_manifest(
    manifest: dict[str, Any],
    expected_key: dict[str, Any],
    fd: int,
    handle: WeightArenaHandle,
) -> None:
    if manifest.get("version") != _MANIFEST_VERSION:
        raise RuntimeError(f"Unsupported reward memfd manifest version: {manifest.get('version')!r}.")
    if manifest.get("key") != expected_key or manifest.get("backend") != "memfd":
        raise RuntimeError("Reward memfd manifest fingerprint does not match the requested model shard.")
    expected_sizes = [_tensor_nbytes(arena) for arena in handle.arenas]
    actual_sizes = [int(item["nbytes"]) for item in manifest.get("arenas", ())]
    if actual_sizes != expected_sizes:
        raise RuntimeError(f"Reward memfd arena layout changed: expected={expected_sizes} actual={actual_sizes}.")
    actual_bytes = os.fstat(fd).st_size
    if actual_bytes != sum(expected_sizes):
        raise RuntimeError(f"Reward memfd is truncated: expected={sum(expected_sizes)} actual={actual_bytes}.")


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Reward node cache manifest is incomplete or invalid: {path}.") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Reward node cache manifest must be a JSON object: {path}.")
    return value


def _remove_partial_files(directory: Path, cache_key: str) -> None:
    for path in directory.glob(f".{cache_key}.*.tmp"):
        path.unlink(missing_ok=True)


def _cleanup_stale_namespaces(root: Path, timeout_s: float) -> None:
    leases_dir = root / ".leases"
    cleanup_path = leases_dir / ".cleanup.lock"
    cleanup_fd = os.open(cleanup_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _lock_with_timeout(cleanup_fd, fcntl.LOCK_EX, timeout_s, cleanup_path)
        leased_namespaces: set[str] = set()
        for lease_path in leases_dir.glob("*.lock"):
            if lease_path == cleanup_path:
                continue
            job_id = lease_path.stem
            leased_namespaces.add(job_id)
            lease_fd = os.open(lease_path, os.O_RDWR)
            try:
                if _try_lock(lease_fd, fcntl.LOCK_EX):
                    shutil.rmtree(root / job_id, ignore_errors=True)
                    fcntl.flock(lease_fd, fcntl.LOCK_UN)
            finally:
                os.close(lease_fd)
        for path in root.iterdir():
            if path.is_dir() and path.name != ".leases" and path.name not in leased_namespaces:
                shutil.rmtree(path, ignore_errors=True)
    finally:
        os.close(cleanup_fd)


def _lock_with_timeout(fd: int, operation: int, timeout_s: float, path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    while not _try_lock(fd, operation):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout_s}s waiting for reward node cache lock {path}.")
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _try_lock(fd: int, operation: int) -> bool:
    try:
        fcntl.flock(fd, operation | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _filesystem_for_path(path: Path) -> tuple[str, str]:
    best_mount = Path("/")
    best_fs = ""
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError as exc:
        raise RuntimeError("Cannot inspect filesystem type for reward node cache.") from exc
    for line in lines:
        fields = line.split()
        separator = fields.index("-")
        mount = Path(fields[4].replace("\\040", " "))
        fs_type = fields[separator + 1]
        try:
            path.relative_to(mount)
        except ValueError:
            continue
        if len(mount.parts) >= len(best_mount.parts):
            best_mount = mount
            best_fs = fs_type
    if not best_fs:
        raise RuntimeError(f"Cannot determine filesystem type for reward node cache path {path}.")
    return best_fs, str(best_mount)


def _namespace_cache_bytes(namespace: Path) -> int:
    return sum(path.stat().st_size for path in namespace.rglob("*.arena") if path.is_file())


def _process_memory_rollup() -> dict[str, int | None]:
    values: dict[str, int | None] = {"private_bytes": None, "pss_bytes": None}
    try:
        lines = Path("/proc/self/smaps_rollup").read_text().splitlines()
    except OSError:
        return values
    parsed: dict[str, int] = {}
    for line in lines:
        match = re.match(r"^(Pss|Private_Clean|Private_Dirty):\s+(\d+)\s+kB$", line)
        if match:
            parsed[match.group(1)] = int(match.group(2)) * 1024
    values["pss_bytes"] = parsed.get("Pss")
    if "Private_Clean" in parsed or "Private_Dirty" in parsed:
        values["private_bytes"] = parsed.get("Private_Clean", 0) + parsed.get("Private_Dirty", 0)
    return values


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Invalid reward node cache job id: {value!r}.")
    return safe


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
