"""Persistent asynchronous Docker Engine API transport."""

import io
import json
import os
import tarfile
from base64 import urlsafe_b64encode
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.parse import quote

import aiohttp


class DockerEngineError(RuntimeError):
    """Docker Engine returned an unexpected response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Docker Engine returned HTTP {status}: {message}.")
        self.status = status
        self.message = message


class DockerExecOutputLimitError(RuntimeError):
    """A Docker exec stream exceeded the configured output budget.

    This is a transport guard rather than an Engine API failure: the response
    is HTTP 200, but the client stopped reading it. It is deliberately not a
    `DockerEngineError`, which carries an HTTP status that callers branch on.
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"Docker command output limit exceeded: {message}.")
        self.message = message


class DockerEngine(Protocol):
    """Operations consumed by `DockerBackend`."""

    async def info(self) -> Mapping[str, Any]: ...

    async def pull_image(self, reference: str, auth: Mapping[str, str] | None = None) -> None: ...

    async def image_exists(self, reference: str) -> bool: ...

    async def create_container(self, name: str, config: Mapping[str, Any]) -> str: ...

    async def start_container(self, container_id: str) -> None: ...

    async def commit_container(self, container_id: str, repository: str, tag: str) -> str: ...

    async def inspect_container(self, container_id: str) -> Mapping[str, Any] | None: ...

    async def remove_container(self, container_id: str) -> None: ...

    async def remove_image(self, reference: str) -> None: ...

    async def pause_container(self, container_id: str) -> None: ...

    async def unpause_container(self, container_id: str) -> None: ...

    async def exec(
        self,
        container_id: str,
        command: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, bytes, bytes]: ...

    async def read_file(self, container_id: str, path: str) -> bytes: ...

    async def write_file(self, container_id: str, path: str, data: bytes) -> None: ...

    async def stats(self, container_id: str) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class DockerEngineClient:
    """Connection-pooled client for a local or remote Docker daemon."""

    def __init__(
        self,
        docker_host: str | None = None,
        *,
        request_timeout_s: float = 180.0,
        connection_limit: int = 128,
        max_exec_output_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self.docker_host = docker_host or os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
        self.request_timeout_s = request_timeout_s
        self.connection_limit = connection_limit
        if connection_limit < 1 or request_timeout_s <= 0 or max_exec_output_bytes < 1:
            raise ValueError("Docker connection, timeout, and output limits must be positive.")
        self.max_exec_output_bytes = max_exec_output_bytes
        self._session: aiohttp.ClientSession | None = None
        self._base_url = "http://docker"
        self._closed = False

    def _connector(self) -> aiohttp.BaseConnector:
        if self.docker_host.startswith("unix://"):
            return aiohttp.UnixConnector(path=self.docker_host.removeprefix("unix://"), limit=self.connection_limit)
        if self.docker_host.startswith("tcp://"):
            self._base_url = "http://" + self.docker_host.removeprefix("tcp://")
            return aiohttp.TCPConnector(limit=self.connection_limit)
        if self.docker_host.startswith(("http://", "https://")):
            self._base_url = self.docker_host.rstrip("/")
            return aiohttp.TCPConnector(limit=self.connection_limit)
        raise ValueError(f"Unsupported Docker host {self.docker_host!r}.")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("Docker Engine client is closed.")
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
            self._session = aiohttp.ClientSession(connector=self._connector(), timeout=timeout)
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        expected: tuple[int, ...],
        timeout_s: float | None = None,
        max_response_bytes: int | None = None,
        **kwargs: Any,
    ) -> tuple[aiohttp.typedefs.LooseHeaders, bytes, int]:
        session = await self._get_session()
        if timeout_s is not None:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout_s)
        async with session.request(method, f"{self._base_url}{path}", **kwargs) as response:
            if max_response_bytes is None:
                body = await response.read()
            else:
                buffer = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if len(buffer) + len(chunk) > max_response_bytes:
                        raise DockerExecOutputLimitError(f"{max_response_bytes} bytes")
                    buffer.extend(chunk)
                body = bytes(buffer)
            if response.status not in expected:
                message = body.decode(errors="replace")
                try:
                    message = str(json.loads(message).get("message", message))
                except (json.JSONDecodeError, AttributeError):
                    pass
                raise DockerEngineError(response.status, message.strip())
            return response.headers, body, response.status

    async def info(self) -> Mapping[str, Any]:
        _, body, _ = await self._request("GET", "/info", expected=(200,))
        return json.loads(body)

    async def pull_image(self, reference: str, auth: Mapping[str, str] | None = None) -> None:
        image = quote(reference, safe="")
        headers = None
        if auth:
            headers = {
                "X-Registry-Auth": urlsafe_b64encode(json.dumps(dict(auth), separators=(",", ":")).encode()).decode()
            }
        session = await self._get_session()
        async with session.post(f"{self._base_url}/images/create?fromImage={image}", headers=headers) as response:
            if response.status != 200:
                raise DockerEngineError(response.status, (await response.text()).strip())
            # Docker can report pull failures inside an HTTP 200 JSON stream.
            # Consume progress incrementally instead of retaining every layer update.
            async for line in response.content:
                if not line.strip():
                    continue
                progress = json.loads(line)
                error = progress.get("error") or (progress.get("errorDetail") or {}).get("message")
                if error:
                    raise DockerEngineError(response.status, str(error))

    async def image_exists(self, reference: str) -> bool:
        """
        Check the daemon cache without downloading or refreshing a mutable tag.
        """
        _, _, status = await self._request("GET", f"/images/{quote(reference, safe='')}/json", expected=(200, 404))
        return status == 200

    async def create_container(self, name: str, config: Mapping[str, Any]) -> str:
        path = f"/containers/create?name={quote(name, safe='')}"
        _, body, _ = await self._request("POST", path, expected=(201,), json=dict(config))
        return str(json.loads(body)["Id"])

    async def start_container(self, container_id: str) -> None:
        await self._request("POST", f"/containers/{container_id}/start", expected=(204, 304))

    async def commit_container(self, container_id: str, repository: str, tag: str) -> str:
        """Commit a running container's writable layer into a new image.

        Used by the per-image bake (git-purge derivative) and by the clean
        verifier snapshot, so a later sandbox starts from the committed state
        instead of repeating the one-time work. Returns the committed image ID.
        """
        path = (
            f"/commit?container={quote(container_id, safe='')}"
            f"&repo={quote(repository, safe='')}&tag={quote(tag, safe='')}"
        )
        # The Engine API rejects /commit unless Content-Type is application/json.
        _, body, _ = await self._request("POST", path, expected=(201,), json={})
        return str(json.loads(body)["Id"])

    async def inspect_container(self, container_id: str) -> Mapping[str, Any] | None:
        try:
            _, body, _ = await self._request("GET", f"/containers/{container_id}/json", expected=(200,))
        except DockerEngineError as exc:
            if exc.status == 404:
                return None
            raise
        return json.loads(body)

    async def remove_image(self, reference: str) -> None:
        """Remove an image by reference (used to clean up committed snapshots)."""
        try:
            await self._request("DELETE", f"/images/{quote(reference, safe='')}", expected=(200,))
        except DockerEngineError as exc:
            if exc.status != 404:
                raise

    async def remove_container(self, container_id: str) -> None:
        try:
            await self._request("DELETE", f"/containers/{container_id}?force=1&v=1", expected=(204,))
        except DockerEngineError as exc:
            if exc.status != 404:
                raise

    async def pause_container(self, container_id: str) -> None:
        await self._request("POST", f"/containers/{container_id}/pause", expected=(204,))

    async def unpause_container(self, container_id: str) -> None:
        await self._request("POST", f"/containers/{container_id}/unpause", expected=(204,))

    @staticmethod
    def _demultiplex_exec(body: bytes) -> tuple[bytes, bytes]:
        stdout = bytearray()
        stderr = bytearray()
        offset = 0
        while offset + 8 <= len(body):
            stream = body[offset]
            frame_size = int.from_bytes(body[offset + 4 : offset + 8], "big")
            frame_end = offset + 8 + frame_size
            if stream not in (1, 2) or frame_end > len(body):
                return body, b""
            target = stdout if stream == 1 else stderr
            target.extend(body[offset + 8 : frame_end])
            offset = frame_end
        if offset != len(body):
            return body, b""
        return bytes(stdout), bytes(stderr)

    async def exec(
        self,
        container_id: str,
        command: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, bytes, bytes]:
        payload: dict[str, Any] = {
            "AttachStdout": True,
            "AttachStderr": True,
            "Tty": False,
            "Cmd": command,
        }
        if cwd:
            payload["WorkingDir"] = cwd
        if env:
            payload["Env"] = [f"{key}={value}" for key, value in env.items()]
        _, body, _ = await self._request(
            "POST",
            f"/containers/{container_id}/exec",
            expected=(201,),
            json=payload,
        )
        exec_id = str(json.loads(body)["Id"])
        _, output, _ = await self._request(
            "POST",
            f"/exec/{exec_id}/start",
            expected=(200,),
            timeout_s=timeout_s,
            max_response_bytes=self.max_exec_output_bytes,
            json={"Detach": False, "Tty": False},
        )
        _, inspect_body, _ = await self._request("GET", f"/exec/{exec_id}/json", expected=(200,))
        inspection = json.loads(inspect_body)
        if inspection.get("Running") or inspection.get("ExitCode") is None:
            raise DockerEngineError(200, "Docker exec stream ended before a final exit status was available")
        exit_code = int(inspection["ExitCode"])
        stdout, stderr = self._demultiplex_exec(output)
        return exit_code, stdout, stderr

    async def read_file(self, container_id: str, path: str) -> bytes:
        archive_path = quote(path, safe="")
        _, body, _ = await self._request(
            "GET",
            f"/containers/{container_id}/archive?path={archive_path}",
            expected=(200,),
        )
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            if len(members) != 1:
                raise RuntimeError(f"Docker archive for {path!r} did not contain exactly one file.")
            extracted = archive.extractfile(members[0])
            if extracted is None:
                raise RuntimeError(f"Docker archive for {path!r} could not be read.")
            return extracted.read()

    async def write_file(self, container_id: str, path: str, data: bytes) -> None:
        directory, _, filename = path.rpartition("/")
        directory = directory or "/"
        if not filename:
            raise ValueError("Sandbox file path must name a file.")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(filename)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
        archive_path = quote(directory, safe="")
        await self._request(
            "PUT",
            f"/containers/{container_id}/archive?path={archive_path}",
            expected=(200,),
            data=buffer.getvalue(),
            headers={"Content-Type": "application/x-tar"},
        )

    async def stats(self, container_id: str) -> Mapping[str, Any]:
        _, body, _ = await self._request(
            "GET",
            f"/containers/{container_id}/stats?stream=false&one-shot=true",
            expected=(200,),
        )
        return json.loads(body)

    async def close(self) -> None:
        self._closed = True
        if self._session is not None:
            await self._session.close()
            self._session = None
