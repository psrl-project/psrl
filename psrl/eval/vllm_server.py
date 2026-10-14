"""Launch and health-check a single vLLM OpenAI-compatible server process.

This is the primitive that `vllm_fleet` and `vllm_multinode` build on. It is
deliberately free of Hydra and OmegaConf: `psrl.eval.serve` owns config
composition and hands down a plain `ServerSpec`. Keeping this layer config-free
is what makes `build_command` a pure function that can be exercised without a
GPU.

Usage::

    from psrl.eval.vllm_server import ServerSpec, launch, wait_ready

    spec = ServerSpec(
        checkpoint="/models/Qwen3.5-9B",
        served_model_name="qwen35-9b",
        port=8000,
        tp=2,
        gpu_ids=(0, 1),
        max_model_len=131072,
    )
    handle = launch(spec, env_script="/env/psrl.sh", log_file="/tmp/vllm_8000.log")
    endpoints = wait_ready([handle], timeout_sec=1800.0)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

psrl_logger = logging.getLogger(__file__)

# vLLM binds 0.0.0.0 to listen on every interface, but curling 0.0.0.0 fails on
# some kernels, so health checks and endpoint URLs use this instead.
_LOOPBACK = "127.0.0.1"


@dataclass(frozen=True)
class ServerSpec:
    """Everything needed to launch one vLLM server process.

    Frozen so a spec can be logged, hashed, and reused across a fleet without
    any chance of a caller mutating it between launch and health check.

    Attributes:
        checkpoint: Absolute path to an HF-compatible checkpoint directory.
        served_model_name: Name advertised via /v1/models. Clients send this as
            the `model` field. Every replica in a fleet shares one name so
            clients stay agnostic about which replica they reach.
        host: Bind address.
        port: Bind port.
        tp: Tensor-parallel size.
        pp: Pipeline-parallel size.
        dp: Data-parallel size, i.e. replicas behind this one port and one
            process. Use it when a client can only be given a single URL.
            Prefer a `vllm_fleet` of independent servers otherwise: separate
            processes fail separately, and `--data-parallel-size` is broken in
            this repo's patched vLLM (the DP coordinator never reports its ZMQ
            addresses).
        gpu_ids: GPUs to expose as CUDA_VISIBLE_DEVICES. Empty inherits the
            caller's environment. Length must equal `tp * pp * dp` when non-empty.
        max_model_len: Context window. None lets vLLM read it from the config.
        gpu_memory_utilization: Fraction of VRAM vLLM may claim.
        tool_call_parser: Chat-template tool parser, e.g. "hermes". Empty
            disables OpenAI tool-call extraction, which is correct for agents
            that parse their own text protocol out of the assistant message.
        chat_template: Path to a jinja template. Empty uses the tokenizer's own.
        extra: Additional vLLM CLI args forwarded verbatim.
    """

    checkpoint: str
    served_model_name: str
    host: str = "0.0.0.0"
    port: int = 8000
    tp: int = 1
    pp: int = 1
    dp: int = 1
    gpu_ids: tuple[int, ...] = ()
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    tool_call_parser: str = ""
    chat_template: str = ""
    extra: tuple[str, ...] = ()

    @property
    def n_gpus(self) -> int:
        """Number of GPUs this server occupies."""
        return self.tp * self.pp * self.dp

    @property
    def url(self) -> str:
        """OpenAI-compatible base URL a client should use to reach this server."""
        host = _LOOPBACK if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}/v1"

    def validate(self) -> None:
        """Check the spec is internally consistent and the checkpoint is readable.

        Raises:
            ValueError: A field is out of range or `gpu_ids` disagrees with
                `tp * pp * dp`.
            FileNotFoundError: The checkpoint directory does not exist.
        """
        if not Path(self.checkpoint).is_dir():
            raise FileNotFoundError(f"Checkpoint is not a readable directory: {self.checkpoint!r}.")
        if not self.served_model_name:
            raise ValueError("served_model_name must not be empty.")
        for name in ("tp", "pp", "dp", "port"):
            if getattr(self, name) < 1:
                raise ValueError(f"Expected {name} >= 1, got {getattr(self, name)!r}.")
        if self.gpu_ids and len(self.gpu_ids) != self.n_gpus:
            raise ValueError(
                f"Expected {self.n_gpus} GPU(s) for tp={self.tp} pp={self.pp} dp={self.dp}, "
                f"got {len(self.gpu_ids)}: {list(self.gpu_ids)!r}."
            )
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(f"Expected gpu_memory_utilization in (0, 1], got {self.gpu_memory_utilization!r}.")


@dataclass
class ServerHandle:
    """A launched server process and where to find its output.

    Attributes:
        spec: The spec this process was launched from.
        process: The child process. Its PID is vLLM's own, not a wrapping
            shell's, because the launch command ends in `exec`.
        log_file: Where stdout and stderr were redirected.
    """

    spec: ServerSpec
    process: subprocess.Popen
    log_file: Path

    @property
    def pid(self) -> int:
        """PID of the vLLM process."""
        return self.process.pid

    def is_alive(self) -> bool:
        """Whether the process is still running."""
        return self.process.poll() is None

    def terminate(self, timeout_sec: float = 30.0) -> None:
        """Shut the server down, escalating to SIGKILL if it ignores SIGTERM.

        Signals the whole **process group**, not just the launched process. `launch`
        passes `start_new_session=True`, which makes the server a process-group
        leader, and vLLM's tensor-parallel workers (`VLLM::Worker_TPn`) plus its
        `VLLM::EngineCore` join that group. Signalling only the leader leaves those
        children alive still holding every GPU -- observed as ~90 GiB per GPU pinned
        by PIDs whose parent had already exited.

        Args:
            timeout_sec: How long to wait for a graceful exit before SIGKILL.
        """
        if not self.is_alive():
            # The leader is gone, but workers may have outlived it. Sweep the group.
            self._signal_group(signal.SIGKILL)
            return
        psrl_logger.info(f"Stopping vLLM on port {self.spec.port} (pid={self.pid})...")
        self._signal_group(signal.SIGTERM)
        try:
            self.process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            psrl_logger.warning(f"vLLM pid {self.pid} ignored SIGTERM after {timeout_sec:.0f}s, sending SIGKILL.")
            self._signal_group(signal.SIGKILL)
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                psrl_logger.error(f"vLLM pid {self.pid} survived SIGKILL. Its GPUs may stay occupied.")

    def _signal_group(self, sig: int) -> None:
        """Send `sig` to the server's process group, tolerating a group already gone."""
        try:
            os.killpg(os.getpgid(self.pid), sig)
        except (ProcessLookupError, PermissionError):
            # The process group is already gone or cannot be signalled.
            pass

    def tail_log(self, n_lines: int = 40) -> str:
        """Return the last `n_lines` of the log file, for error reporting."""
        try:
            lines = self.log_file.read_text(errors="replace").splitlines()
        except OSError as e:
            return f"(could not read {self.log_file}: {e})"
        return "\n".join(lines[-n_lines:])


@dataclass(frozen=True)
class Endpoint:
    """A server that answered a health check.

    Attributes:
        url: OpenAI-compatible base URL.
        host: Host the server runs on, as a client would address it.
        gpu_ids: GPUs the server occupies.
        pid: Process ID, for teardown.
        healthy: Whether /v1/models responded before the timeout.
    """

    url: str
    host: str
    gpu_ids: tuple[int, ...] = ()
    pid: int | None = None
    healthy: bool = True

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict for `endpoints.json`."""
        payload = dataclasses.asdict(self)
        payload["gpu_ids"] = list(self.gpu_ids)
        return payload


def is_moe_checkpoint(checkpoint: str | Path) -> bool:
    """Guess whether a checkpoint is a mixture-of-experts model from its config.

    Not pure -- it reads config.json -- which is why it is separate from
    `build_command`. Only needed to decide the DP async-scheduling workaround.

    Args:
        checkpoint: Path to an HF checkpoint directory.

    Returns:
        Whether the config looks like an MoE model. False if unreadable.
    """
    config_path = Path(checkpoint) / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return False
    architectures = config.get("architectures") or [""]
    if "moe" in str(architectures[0]).lower():
        return True
    return bool(config.get("num_experts") or config.get("num_local_experts"))


def build_command(spec: ServerSpec, disable_async_scheduling: bool = False) -> list[str]:
    """Assemble the vLLM CLI invocation for a spec.

    Pure: no environment reads, no filesystem access, no side effects. This is
    what makes `--dry-run` trustworthy and lets the argument logic be tested
    without a GPU. Anything needing to inspect the checkpoint is decided by the
    caller and passed in.

    Args:
        spec: The server to build a command for.
        disable_async_scheduling: Pass `--async-scheduling false`. Needed for
            MoE models under data parallelism, where vLLM v1 switches DP
            synchronization from NCCL to gloo TCP, which is fragile.

    Returns:
        argv for the vLLM OpenAI API server.
    """
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        spec.checkpoint,
        "--served-model-name",
        spec.served_model_name,
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--tensor-parallel-size",
        str(spec.tp),
        "--pipeline-parallel-size",
        str(spec.pp),
        "--gpu-memory-utilization",
        str(spec.gpu_memory_utilization),
    ]
    if spec.dp > 1:
        cmd += ["--data-parallel-size", str(spec.dp)]
        if disable_async_scheduling:
            cmd += ["--async-scheduling", "false"]
    if spec.max_model_len is not None:
        cmd += ["--max-model-len", str(spec.max_model_len)]
    if spec.tool_call_parser:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", spec.tool_call_parser]
    if spec.chat_template:
        cmd += ["--chat-template", spec.chat_template]
    cmd += list(spec.extra)
    return cmd


def build_shell_command(spec: ServerSpec, env_script: str = "") -> list[str]:
    """Wrap the vLLM command in a login shell that sources the env script.

    A Python process cannot source a bash script, and `psrl.sh` does far more
    than set PATH: it activates conda and sets the NCCL / UCX / cudnn
    LD_LIBRARY_PATH knobs vLLM needs. So the server is launched through
    `bash -lc`.

    The command ends in `exec`, which replaces the shell with vLLM instead of
    forking it. Without `exec` the recorded PID would be the shell's, and both
    `terminate()` and `is_alive()` would report on a process that is merely
    vLLM's parent -- SIGTERM would kill the shell and orphan the server.

    Args:
        spec: The server to launch.
        env_script: Script to source first. Empty skips sourcing.

    Returns:
        argv of the form `["bash", "-lc", "..."]`.
    """
    # Resolve the MoE async override here to keep the pure command builder free of checkpoint reads.
    disable_async = spec.dp > 1 and is_moe_checkpoint(spec.checkpoint)
    inner = f"exec {shlex.join(build_command(spec, disable_async_scheduling=disable_async))}"
    if env_script:
        # psrl.sh references unset variables, so nounset must stay off here.
        inner = f"source {shlex.quote(env_script)} && {inner}"
    return ["bash", "-lc", inner]


def launch(spec: ServerSpec, env_script: str = "", log_file: str | Path | None = None) -> ServerHandle:
    """Start one vLLM server in the background and return its handle.

    The child is detached with `start_new_session=True` (bash's `nohup setsid`)
    so that an ssh disconnect or a parent exit does not take the server down.
    stdin is closed to keep it from ever blocking on a read.

    Args:
        spec: The server to launch. Validated first.
        env_script: Env script to source before exec'ing vLLM.
        log_file: Where to write stdout and stderr. Defaults to
            `/tmp/vllm_<port>.log`.

    Returns:
        A handle wrapping the live process.
    """
    spec.validate()
    log_path = Path(log_file) if log_file else Path(f"/tmp/vllm_{spec.port}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = {}
    if spec.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in spec.gpu_ids)

    argv = build_shell_command(spec, env_script)
    psrl_logger.info(
        f"Launching vLLM on port {spec.port} (tp={spec.tp} pp={spec.pp} "
        f"gpus={list(spec.gpu_ids) or 'inherit'}), log={log_path}."
    )

    # Inherit the parent environment so proxy and cache settings survive, then
    # overlay the GPU pinning.
    child_env = {**os.environ, **env}
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=child_env,
        )
    return ServerHandle(spec=spec, process=process, log_file=log_path)


def _probe(url: str, timeout_sec: float = 5.0) -> bool:
    """Return whether /v1/models answers with a 2xx.

    Bypasses proxy environment variables. A corporate `http_proxy` would
    otherwise swallow requests to localhost and make every health check fail.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url}/models", timeout=timeout_sec) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_ready(
    handles: list[ServerHandle],
    timeout_sec: float = 1800.0,
    poll_interval_sec: float = 3.0,
) -> list[Endpoint]:
    """Poll every handle until it serves /v1/models, it dies, or time runs out.

    Handles are polled concurrently: one slow replica does not delay detecting
    that another is ready. A handle whose process exits is failed immediately
    rather than waited out, since a crashed load will never become ready and
    `timeout_sec` is generous enough (30 min, for first-time torch.compile) that
    waiting would waste most of it.

    Args:
        handles: Servers to wait for.
        timeout_sec: Overall budget, shared across all handles.
        poll_interval_sec: Delay between polling rounds.

    Returns:
        One `Endpoint` per handle, in the input order, each flagged healthy or not.
    """
    pending = {id(h): h for h in handles}
    ready: dict[int, bool] = {}
    deadline = time.monotonic() + timeout_sec

    psrl_logger.info(f"Waiting up to {timeout_sec:.0f}s for {len(handles)} server(s) to report ready...")
    while pending and time.monotonic() < deadline:
        for key, handle in list(pending.items()):
            if not handle.is_alive():
                rc = handle.process.returncode
                psrl_logger.error(
                    f"vLLM on port {handle.spec.port} exited with code {rc} before becoming ready. "
                    f"Last log lines:\n{handle.tail_log()}"
                )
                ready[key] = False
                del pending[key]
                continue
            if _probe(handle.spec.url):
                elapsed = timeout_sec - (deadline - time.monotonic())
                psrl_logger.info(f"Server ready at {handle.spec.url} (pid={handle.pid}) after {elapsed:.0f}s.")
                ready[key] = True
                del pending[key]
        if pending:
            time.sleep(poll_interval_sec)

    for key, handle in pending.items():
        psrl_logger.error(
            f"vLLM on port {handle.spec.port} did not become ready within {timeout_sec:.0f}s. "
            f"Last log lines:\n{handle.tail_log()}"
        )
        ready[key] = False

    return [
        Endpoint(
            url=h.spec.url,
            host=h.spec.host,
            gpu_ids=h.spec.gpu_ids,
            pid=h.pid,
            healthy=ready.get(id(h), False),
        )
        for h in handles
    ]
