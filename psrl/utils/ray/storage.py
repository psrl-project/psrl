"""Ray Plasma backing-store diagnostics used before large model startup."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def inspect_local_plasma_backing(node_id: str | None = None) -> dict[str, Any]:
    """Inspect the local raylet command and its allocated Plasma file blocks."""
    raylet_pid = None
    options: dict[str, str] = {}
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        try:
            parts = (proc_path / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [part.decode(errors="replace") for part in parts if part]
        if not args or "/raylet" not in args[0]:
            continue
        parsed = {
            key: value
            for arg in args[1:]
            if arg.startswith("--") and "=" in arg
            for key, value in [arg[2:].split("=", 1)]
        }
        if node_id is None or parsed.get("node_id") == node_id:
            raylet_pid = int(proc_path.name)
            options = parsed
            node_id = parsed.get("node_id", node_id)
            break

    if raylet_pid is None:
        return {"node_id": node_id, "error": "local raylet process was not found"}

    plasma_directory = options.get("plasma_directory")
    object_store_bytes = int(options.get("object_store_memory", "0"))
    if not plasma_directory or object_store_bytes <= 0:
        return {
            "node_id": node_id,
            "raylet_pid": raylet_pid,
            "error": "raylet did not expose plasma_directory/object_store_memory",
        }

    statvfs = os.statvfs(plasma_directory)
    free_bytes = statvfs.f_bavail * statvfs.f_frsize
    shm_statvfs = os.statvfs("/dev/shm")
    shm_free_bytes = shm_statvfs.f_bavail * shm_statvfs.f_frsize
    allocated_bytes = 0
    fd_root = Path(f"/proc/{raylet_pid}/fd")
    try:
        fd_paths = list(fd_root.iterdir())
    except OSError:
        fd_paths = []
    for fd_path in fd_paths:
        try:
            target = os.readlink(fd_path)
            if "plasma" not in os.path.basename(target):
                continue
            allocated_bytes += os.stat(fd_path).st_blocks * 512
        except OSError:
            continue

    return {
        "node_id": node_id,
        "raylet_pid": raylet_pid,
        "plasma_directory": plasma_directory,
        "object_store_bytes": object_store_bytes,
        "free_bytes": free_bytes,
        "shm_free_bytes": shm_free_bytes,
        "allocated_plasma_bytes": allocated_bytes,
        "backing_capacity_bytes": free_bytes + allocated_bytes,
    }


def plasma_backing_error(snapshot: dict[str, Any], reserve_bytes: int) -> str | None:
    """Return a remediation message when Plasma cannot be fully backed safely."""
    if snapshot.get("error"):
        return f"node_id={snapshot.get('node_id')} inspection_error={snapshot['error']}"
    capacity = int(snapshot["backing_capacity_bytes"])
    required = int(snapshot["object_store_bytes"]) + reserve_bytes
    if capacity < required:
        return (
            f"node_id={snapshot['node_id']} plasma_directory={snapshot['plasma_directory']} "
            f"object_store_bytes={snapshot['object_store_bytes']} backing_capacity_bytes={capacity} "
            f"reserve_bytes={reserve_bytes}"
        )
    if int(snapshot["shm_free_bytes"]) < reserve_bytes:
        return (
            f"node_id={snapshot['node_id']} plasma_directory={snapshot['plasma_directory']} "
            f"shm_free_bytes={snapshot['shm_free_bytes']} reserve_bytes={reserve_bytes}"
        )
    return None


if __name__ == "__main__":
    snapshot = inspect_local_plasma_backing()
    error = plasma_backing_error(snapshot, reserve_bytes=16 * 1024**3)
    print(json.dumps({"snapshot": snapshot, "error": error}, sort_keys=True))
    raise SystemExit(1 if error else 0)
