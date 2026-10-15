"""Node-local broker for sharing sealed memfd weight arenas across processes."""

from __future__ import annotations

import argparse
import array
import json
import os
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any

_MAX_MESSAGE_BYTES = 1024 * 1024
_SO_PEERCRED = getattr(socket, "SO_PEERCRED", 17)


@dataclass
class _Entry:
    fd: int
    manifest: dict[str, Any]
    clients: dict[str, tuple[int, int]] = field(default_factory=dict)


def broker_address(token: str) -> str:
    """Return a Linux abstract Unix-socket address for a broker token."""
    return f"\0psrl-rm-memfd-{token}"


def request(
    token: str,
    payload: dict[str, Any],
    *,
    send_fd: int | None = None,
    timeout_s: float = 10.0,
) -> tuple[dict[str, Any], int | None]:
    """Send one broker request and optionally transfer one file descriptor."""
    ancillary = []
    if send_fd is not None:
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [send_fd]))]
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
        client.settimeout(timeout_s)
        client.connect(broker_address(token))
        client.sendmsg([_encode(payload)], ancillary)
        raw, received_ancillary, _, _ = client.recvmsg(
            _MAX_MESSAGE_BYTES,
            socket.CMSG_SPACE(array.array("i").itemsize),
        )
    response = json.loads(raw)
    received_fd = _extract_fd(received_ancillary)
    if not response.get("ok", False):
        if received_fd is not None:
            os.close(received_fd)
        raise RuntimeError(f"memfd broker request failed: {response.get('error', response)!r}")
    return response, received_fd


def serve(token: str, *, idle_timeout_s: float = 30.0) -> None:
    """Serve memfd register/get/release requests until no live clients remain."""
    entries: dict[str, _Entry] = {}
    ever_registered = False
    empty_since = time.monotonic()
    address = broker_address(token)
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
        server.bind(address)
        server.listen(128)
        server.settimeout(1.0)
        while True:
            _reap_dead_clients(entries)
            if entries:
                empty_since = time.monotonic()
            elif (ever_registered or time.monotonic() - empty_since >= idle_timeout_s) and (
                time.monotonic() - empty_since >= idle_timeout_s
            ):
                return
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                try:
                    peer_pid, peer_uid, _ = struct.unpack(
                        "3i", connection.getsockopt(socket.SOL_SOCKET, _SO_PEERCRED, struct.calcsize("3i"))
                    )
                    if peer_uid != os.geteuid():
                        raise PermissionError(f"peer uid {peer_uid} does not match broker uid {os.geteuid()}")
                    raw, ancillary, _, _ = connection.recvmsg(
                        _MAX_MESSAGE_BYTES,
                        socket.CMSG_SPACE(array.array("i").itemsize),
                    )
                    payload = json.loads(raw)
                    transferred_fd = _extract_fd(ancillary)
                    response, response_fd = _handle_request(
                        entries,
                        payload,
                        transferred_fd=transferred_fd,
                        peer_pid=peer_pid,
                    )
                    if payload.get("command") == "register" and response.get("ok"):
                        ever_registered = True
                    response_ancillary = []
                    if response_fd is not None:
                        response_ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [response_fd]))]
                    connection.sendmsg([_encode(response)], response_ancillary)
                except Exception as exc:
                    connection.send(_encode({"ok": False, "error": str(exc)}))


def _handle_request(
    entries: dict[str, _Entry],
    payload: dict[str, Any],
    *,
    transferred_fd: int | None,
    peer_pid: int,
) -> tuple[dict[str, Any], int | None]:
    command = payload.get("command")
    if command == "ping":
        if transferred_fd is not None:
            os.close(transferred_fd)
        return {"ok": True, "unique_cache_bytes": _total_bytes(entries)}, None

    key = str(payload.get("key", ""))
    client_id = str(payload.get("client_id", ""))
    pid = int(payload.get("pid", peer_pid))
    start_time = int(payload.get("start_time", 0))
    if not key or not client_id or pid != peer_pid or start_time <= 0:
        if transferred_fd is not None:
            os.close(transferred_fd)
        raise ValueError("invalid memfd broker client identity")

    if command == "get":
        if transferred_fd is not None:
            os.close(transferred_fd)
        entry = entries.get(key)
        if entry is None:
            return {"ok": True, "found": False, "unique_cache_bytes": _total_bytes(entries)}, None
        entry.clients[client_id] = (pid, start_time)
        return {
            "ok": True,
            "found": True,
            "manifest": entry.manifest,
            "unique_cache_bytes": _total_bytes(entries),
        }, entry.fd

    if command == "register":
        if transferred_fd is None:
            raise ValueError("register requires an SCM_RIGHTS file descriptor")
        manifest = payload.get("manifest")
        if not isinstance(manifest, dict):
            os.close(transferred_fd)
            raise ValueError("register requires a manifest object")
        expected_size = int(manifest.get("total_bytes", -1))
        actual_size = os.fstat(transferred_fd).st_size
        if expected_size <= 0 or actual_size != expected_size:
            os.close(transferred_fd)
            raise ValueError(f"memfd size mismatch: expected={expected_size} actual={actual_size}")
        entry = entries.get(key)
        if entry is None:
            entry = _Entry(fd=transferred_fd, manifest=manifest)
            entries[key] = entry
        else:
            os.close(transferred_fd)
            if entry.manifest != manifest:
                raise ValueError("registered memfd manifest does not match existing entry")
        entry.clients[client_id] = (pid, start_time)
        return {
            "ok": True,
            "found": True,
            "manifest": entry.manifest,
            "unique_cache_bytes": _total_bytes(entries),
        }, entry.fd

    if command == "release":
        if transferred_fd is not None:
            os.close(transferred_fd)
        entry = entries.get(key)
        if entry is not None:
            entry.clients.pop(client_id, None)
            if not entry.clients:
                os.close(entry.fd)
                del entries[key]
        return {"ok": True, "unique_cache_bytes": _total_bytes(entries)}, None

    if transferred_fd is not None:
        os.close(transferred_fd)
    raise ValueError(f"unsupported memfd broker command: {command!r}")


def process_start_time(pid: int | None = None) -> int:
    """Return Linux /proc start ticks, which disambiguate PID reuse."""
    pid = os.getpid() if pid is None else pid
    with open(f"/proc/{pid}/stat") as stream:
        raw = stream.read()
    fields_after_comm = raw[raw.rfind(")") + 2 :].split()
    return int(fields_after_comm[19])


def _reap_dead_clients(entries: dict[str, _Entry]) -> None:
    for key, entry in list(entries.items()):
        for client_id, (pid, expected_start) in list(entry.clients.items()):
            try:
                alive = process_start_time(pid) == expected_start
            except (FileNotFoundError, ProcessLookupError, ValueError):
                alive = False
            if not alive:
                entry.clients.pop(client_id, None)
        if not entry.clients:
            os.close(entry.fd)
            del entries[key]


def _total_bytes(entries: dict[str, _Entry]) -> int:
    return sum(int(entry.manifest["total_bytes"]) for entry in entries.values())


def _extract_fd(ancillary: list[tuple[int, int, bytes]]) -> int | None:
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            values = array.array("i")
            values.frombytes(data[: values.itemsize])
            return int(values[0])
    return None


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("token")
    parser.add_argument("--idle-timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    serve(args.token, idle_timeout_s=args.idle_timeout_s)


if __name__ == "__main__":
    main()
