"""Own sandbox containers and their persistent shells."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import logging
import os
import shlex
import subprocess
import time
import uuid

from psrl.workers.env_worker.sandbox import ExecResult, SandboxSpec, build_docker_run_argv
from psrl.workers.env_worker.shell import parse_sentinel, truncate_observation, wrap_command

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

SANDBOX_LABEL_KEY = "psrl.env_sandbox_id"
ACTOR_LABEL_KEY = "psrl.actor_id"

_SHELL_READ_CHUNK = 65536
_SHELL_POLL_INTERVAL_S = 0.05


class ContainerShell:
    """
    Long-lived interactive shell attached to one container.

    The container is started with `docker run -i ... /bin/bash -l`, so writes to
    stdin execute in a shell whose state persists across calls.
    """

    def __init__(self, process: subprocess.Popen, container_name: str):
        self.process = process
        self.container_name = container_name
        self._buffer = ""
        # NOTE(claude): Reads must never block the event loop thread, and the shell
        # never closes, so there is no EOF to wait on. A non-blocking fd plus
        # `os.read` gives us "return whatever is available right now" semantics.
        assert self.process.stdout is not None, "Shell stdout is not available."
        self._fd = self.process.stdout.fileno()
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def write(self, text: str) -> None:
        """Write command text to the shell's stdin."""
        assert self.process.stdin is not None, "Shell stdin is not available."
        self.process.stdin.write(text.encode())
        self.process.stdin.flush()

    def _read_available(self) -> bytes:
        """Read whatever bytes are buffered right now, without blocking."""
        try:
            return os.read(self._fd, _SHELL_READ_CHUNK)
        except BlockingIOError:
            return b""
        except OSError as error:
            psrl_logger.warning(f"Sandbox shell read failed: {error}.")
            return b""

    def drain(self) -> str:
        """
        Discard any output left over from a previously abandoned command.

        After a timeout the abandoned command keeps running inside the shell and will
        eventually emit both its output and its sentinel. Without this drain, that
        late output leaks into the next command's observation and its stale sentinel
        is misread as the next command's exit code. MLGym issues commands that
        routinely reach the 3600 second limit, so this path is common, not exotic.

        Returns:
            str: The discarded text, for logging.
        """
        stale = self._buffer
        self._buffer = ""
        while True:
            chunk = self._read_available()
            if not chunk:
                break
            stale += chunk.decode(errors="replace")
        if stale.strip():
            psrl_logger.warning(f"Discarded stale sandbox output. Characters={len(stale)}.")
        return stale

    async def read_until(self, deadline: float, no_output_timeout_s: float) -> str:
        """
        Read shell output until the sentinel appears or a deadline elapses.

        Args:
            deadline (float): Monotonic loop time after which reading stops.
            no_output_timeout_s (float): Seconds of total silence after which reading
                stops, even when the overall deadline has not been reached.

        Returns:
            str: Accumulated output, which the caller parses for the sentinel.
        """
        loop = asyncio.get_running_loop()
        last_output_at = loop.time()

        while True:
            chunk = self._read_available()
            if chunk:
                self._buffer += chunk.decode(errors="replace")
                last_output_at = loop.time()
                if parse_sentinel(self._buffer) is not None:
                    break
                continue

            now = loop.time()
            if now >= deadline:
                break
            if now - last_output_at >= no_output_timeout_s:
                psrl_logger.warning(f"Sandbox shell produced no output for {no_output_timeout_s}s.")
                break
            await asyncio.sleep(_SHELL_POLL_INTERVAL_S)

        buffer, self._buffer = self._buffer, ""
        return buffer

    def close(self) -> None:
        """Terminate the shell process."""
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except OSError:
            pass
        self.process.terminate()


class EnvWorker:
    """
    Own the sandbox containers scheduled onto one placement unit.

    The class is plain Python. `EnvWorkerManager` wraps it with `ray.remote` so the
    resource request can be built at runtime from configuration.
    """

    def __init__(
        self,
        worker_id: int,
        cpu_slots: int,
        gpu_slots: int,
        exec_default_timeout_s: float,
        max_observation_chars: int = 0,
        idle_sandbox_timeout_s: float = 7200.0,
    ):
        self.worker_id = worker_id
        self.cpu_slots = cpu_slots
        self.gpu_slots = gpu_slots
        self.exec_default_timeout_s = exec_default_timeout_s
        self.max_observation_chars = max_observation_chars
        self.idle_sandbox_timeout_s = idle_sandbox_timeout_s

        # NOTE(claude): Ray sets `CUDA_VISIBLE_DEVICES` to exactly the GPUs it granted
        # this actor. Deriving sandbox devices from it makes it impossible for a
        # sandbox to touch a training GPU.
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        self.available_gpu_indices: tuple[int, ...] = (
            tuple(int(part) for part in visible.split(",") if part.strip() != "") if visible else ()
        )

        self._shells: dict[str, ContainerShell] = {}
        self._container_names: dict[str, str] = {}
        self._sandbox_gpus: dict[str, tuple[int, ...]] = {}
        self._last_used_at: dict[str, float] = {}
        self._busy_gpu_indices: set[int] = set()

    def _allocate_gpu_indices(self, count: int) -> tuple[int, ...]:
        """
        Reserve `count` physical GPU indices for one sandbox.

        Raises:
            RuntimeError: If not enough free GPUs remain on this worker.
        """
        if count == 0:
            return ()
        free = [index for index in self.available_gpu_indices if index not in self._busy_gpu_indices]
        if len(free) < count:
            raise RuntimeError(
                f"Worker {self.worker_id} has no free GPU capacity for {count} device(s), free={free!r}."
            )
        chosen = tuple(free[:count])
        self._busy_gpu_indices.update(chosen)
        return chosen

    def _build_labels(self, spec: SandboxSpec, sandbox_id: str) -> dict[str, str]:
        """Merge caller labels with the labels the reaper needs for cleanup."""
        labels = dict(spec.labels)
        labels[SANDBOX_LABEL_KEY] = sandbox_id
        actor_id = os.getenv("PSRL_ACTOR_ID", "")
        if actor_id:
            labels[ACTOR_LABEL_KEY] = actor_id
        return labels

    async def create_sandbox(self, spec: SandboxSpec) -> str:
        """
        Start one container and its shell.

        Args:
            spec (SandboxSpec): Sandbox description.

        Returns:
            str: Sandbox id used by all later calls.

        Raises:
            RuntimeError: If the shell does not become responsive before the timeout.
        """
        sandbox_id = f"s-{uuid.uuid4().hex[:12]}"
        container_name = f"psrl-env-{sandbox_id}"
        gpu_indices = self._allocate_gpu_indices(spec.gpus)

        labelled_spec = SandboxSpec(
            image=spec.image,
            cpus=spec.cpus,
            memory=spec.memory,
            gpus=spec.gpus,
            mounts=spec.mounts,
            env=spec.env,
            network=spec.network,
            labels=self._build_labels(spec, sandbox_id),
            startup_timeout_s=spec.startup_timeout_s,
            idle_timeout_s=spec.idle_timeout_s,
        )
        argv = build_docker_run_argv(labelled_spec, container_name, gpu_indices)
        psrl_logger.info(f"Starting sandbox={sandbox_id!r} with image={spec.image!r}.")

        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
        except Exception:
            for index in gpu_indices:
                self._busy_gpu_indices.discard(index)
            raise
        shell = ContainerShell(process, container_name)
        self._shells[sandbox_id] = shell
        self._container_names[sandbox_id] = container_name
        self._sandbox_gpus[sandbox_id] = gpu_indices
        self._last_used_at[sandbox_id] = time.monotonic()

        probe = await self.exec(sandbox_id, "echo ready", timeout_s=spec.startup_timeout_s)
        if probe.timed_out or "ready" not in probe.stdout:
            await self.destroy_sandbox(sandbox_id)
            raise RuntimeError(
                f"Sandbox {sandbox_id!r} shell did not become ready within "
                f"{spec.startup_timeout_s}s, output={probe.stdout[:200]!r}."
            )
        return sandbox_id

    async def exec(
        self,
        sandbox_id: str,
        command: str,
        timeout_s: float,
        no_output_timeout_s: float | None = None,
    ) -> ExecResult:
        """
        Run one command in a sandbox's persistent shell.

        Timeouts set `ExecResult.timed_out` while the sandbox stays alive for later commands.

        Args:
            sandbox_id (str): Sandbox to run in.
            command (str): Shell command.
            timeout_s (float): Maximum seconds to wait for the sentinel.
            no_output_timeout_s (float | None): Maximum silent seconds. Defaults to
                `timeout_s`.

        Returns:
            ExecResult: Output, exit code, timeout flag, and duration.

        Raises:
            KeyError: If the sandbox is unknown.
        """
        if sandbox_id not in self._shells:
            raise KeyError(f"Unknown sandbox id {sandbox_id!r}.")

        shell = self._shells[sandbox_id]
        started_at = time.monotonic()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s

        # A previously timed-out command may still be emitting output and will emit a
        # stale sentinel. Drop all of it before issuing a new command.
        shell.drain()
        shell.write(wrap_command(command))
        buffer = await shell.read_until(
            deadline, float(timeout_s if no_output_timeout_s is None else no_output_timeout_s)
        )
        parsed = parse_sentinel(buffer)
        duration_s = time.monotonic() - started_at
        self._last_used_at[sandbox_id] = time.monotonic()

        if parsed is None:
            psrl_logger.warning(f"Sandbox command timed out. Sandbox={sandbox_id!r}, duration={duration_s:.1f}s.")
            return ExecResult(
                stdout=truncate_observation(buffer, self.max_observation_chars),
                exit_code=None,
                timed_out=True,
                duration_s=duration_s,
            )

        body, exit_code = parsed
        return ExecResult(
            stdout=truncate_observation(body, self.max_observation_chars),
            exit_code=exit_code,
            timed_out=False,
            duration_s=duration_s,
        )

    async def read_file(self, sandbox_id: str, path: str) -> bytes:
        """Copy one file out of a sandbox via docker exec."""
        container_name = self._container_names[sandbox_id]
        completed = await asyncio.to_thread(
            subprocess.run,
            ["docker", "exec", container_name, "cat", path],
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to read {path!r} from sandbox {sandbox_id!r}: "
                f"{completed.stderr.decode(errors='replace')[:200]}."
            )
        return completed.stdout

    async def write_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        """Write bytes into a sandbox file using a base64 heredoc."""
        encoded = base64.b64encode(data).decode()
        quoted_path = shlex.quote(path)
        command = f"printf %s {shlex.quote(encoded)} | base64 -d > {quoted_path}"
        result = await self.exec(sandbox_id, command, timeout_s=120.0)
        if result.timed_out or (result.exit_code not in (0, None)):
            raise RuntimeError(
                f"Failed to write {path!r} into sandbox {sandbox_id!r}, exit_code={result.exit_code!r}."
            )

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """Close a sandbox's shell and force-remove its container."""
        shell = self._shells.pop(sandbox_id, None)
        container_name = self._container_names.pop(sandbox_id, None)
        for index in self._sandbox_gpus.pop(sandbox_id, ()):
            self._busy_gpu_indices.discard(index)
        self._last_used_at.pop(sandbox_id, None)

        if shell is not None:
            shell.close()
        if container_name is not None:
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container_name],
                capture_output=True,
            )
            psrl_logger.info(f"Destroyed sandbox {sandbox_id!r}.")

    async def reap_idle_sandboxes(self) -> list[str]:
        """
        Destroy sandboxes that have been idle past the timeout.

        Returns:
            list[str]: Sandbox ids that were reaped.
        """
        now = time.monotonic()
        stale = [
            sandbox_id
            for sandbox_id, last_used in self._last_used_at.items()
            if now - last_used > self.idle_sandbox_timeout_s
        ]
        for sandbox_id in stale:
            psrl_logger.warning(f"Reaping idle sandbox {sandbox_id!r}.")
            await self.destroy_sandbox(sandbox_id)
        return stale

    async def live_sandbox_count(self) -> int:
        """Return how many sandboxes this worker currently owns."""
        return len(self._shells)

    async def node_ip(self) -> str:
        """Return the IP of the node this worker runs on."""
        import ray

        return ray.util.get_node_ip_address()
