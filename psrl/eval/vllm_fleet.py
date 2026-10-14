"""Launch N independent vLLM replicas on one host.

Each replica is a separate server process on its own port with its own slice of
the local GPUs. This exists because vLLM's `--data-parallel-size` is broken in
this repo's patched build (the DP coordinator fails to report its ZMQ addresses
during startup), and because independent replicas fail independently: one wedged
replica costs its share of capacity instead of hanging the whole server.

Usage::

    from psrl.eval.vllm_fleet import FleetSpec, launch_fleet

    fleet = FleetSpec(
        checkpoint="/models/Qwen3.5-9B", served_model_name="qwen35-9b", replicas=4, tp=2, max_model_len=131072
    )
    result = launch_fleet(fleet, env_script="/env/psrl.sh", output_dir="out/serve")
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from psrl.eval.vllm_server import Endpoint, ServerHandle, ServerSpec, build_shell_command, launch, wait_ready

psrl_logger = logging.getLogger(__file__)


@dataclass
class FleetSpec:
    """A group of identical vLLM replicas sharing one host.

    Attributes:
        checkpoint: Absolute path to an HF-compatible checkpoint directory.
        served_model_name: Name every replica advertises. Shared on purpose so
            clients need not know which replica they reached.
        replicas: How many server processes to start.
        tp: Tensor-parallel size per replica.
        pp: Pipeline-parallel size per replica.
        dp: Data-parallel size *inside* each replica, i.e. sub-replicas sharing
            one port and one process. Leave at 1 and raise `replicas` instead
            unless a consumer can only accept a single URL. Separate processes
            fail separately, and this repo's patched vLLM has a broken DP
            coordinator.
        base_port: Port of the first replica. Subsequent ones increment by 1.
        gpu_ids: GPUs to distribute across replicas. Empty means "discover the
            local GPUs and use them all".
        host: Bind address for every replica.
        max_model_len: Context window. None lets vLLM read it from the config.
        gpu_memory_utilization: Fraction of each GPU's VRAM vLLM may claim.
        tool_call_parser: Chat-template tool parser. Empty disables extraction.
        chat_template: Path to a jinja template. Empty uses the tokenizer's own.
        extra: Additional vLLM CLI args forwarded verbatim to every replica.
        wait_ready_sec: Budget for all replicas to report ready.
        min_healthy_frac: Fraction of replicas that must come up for the fleet to
            count as usable. Defaults to requiring every replica, since on a
            single host a missing replica usually means a real misconfiguration
            rather than the flakiness expected at multi-node scale.
    """

    checkpoint: str
    served_model_name: str
    replicas: int = 1
    tp: int = 1
    pp: int = 1
    dp: int = 1
    base_port: int = 8000
    gpu_ids: tuple[int, ...] = ()
    host: str = "0.0.0.0"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    tool_call_parser: str = ""
    chat_template: str = ""
    extra: tuple[str, ...] = ()
    wait_ready_sec: float = 1800.0
    min_healthy_frac: float = 1.0

    @property
    def gpus_per_replica(self) -> int:
        """GPUs occupied by each replica."""
        return self.tp * self.pp * self.dp


@dataclass
class FleetResult:
    """Outcome of a fleet launch.

    Attributes:
        served_model_name: Name every replica advertises.
        endpoints: One entry per replica, healthy or not, in launch order.
        handles: Live process handles, for teardown.
        endpoints_file: Where the machine-readable endpoint list was written.
    """

    served_model_name: str
    endpoints: list[Endpoint] = field(default_factory=list)
    handles: list[ServerHandle] = field(default_factory=list)
    endpoints_file: Path | None = None

    @property
    def healthy(self) -> list[Endpoint]:
        """Endpoints that answered a health check."""
        return [e for e in self.endpoints if e.healthy]

    @property
    def api_base(self) -> str:
        """Comma-separated healthy URLs, as eval harnesses expect."""
        return ",".join(e.url for e in self.healthy)

    def terminate(self) -> None:
        """Stop every replica in the fleet."""
        for handle in self.handles:
            handle.terminate()


def discover_local_gpus() -> list[int]:
    """List the GPU indices visible on this host.

    Honors CUDA_VISIBLE_DEVICES when set, so a caller already pinned to a subset
    does not get handed GPUs it cannot use.

    Returns:
        GPU indices, or an empty list if none could be detected.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [int(token) for token in visible.split(",") if token.strip()]
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        psrl_logger.warning(f"Could not query local GPUs with nvidia-smi: {e}.")
        return []
    return [int(line) for line in completed.stdout.split() if line.strip().isdigit()]


def partition_gpus(gpu_ids: list[int], replicas: int, gpus_each: int) -> list[tuple[int, ...]]:
    """Split GPUs into one contiguous block per replica.

    Contiguous blocks keep each replica's tensor-parallel group on GPUs that are
    most likely to share an NVLink domain, which matters for TP collectives.

    Args:
        gpu_ids: GPUs available to the fleet.
        replicas: Number of replicas to serve.
        gpus_each: GPUs each replica needs, i.e. `tp * pp`.

    Returns:
        One tuple of GPU indices per replica, in order.

    Raises:
        ValueError: The GPU count does not match `replicas * gpus_each`, or any
            argument is non-positive.
    """
    if replicas < 1:
        raise ValueError(f"Expected replicas >= 1, got {replicas!r}.")
    if gpus_each < 1:
        raise ValueError(f"Expected gpus_each >= 1, got {gpus_each!r}.")
    needed = replicas * gpus_each
    if len(gpu_ids) != needed:
        raise ValueError(
            f"Expected {needed} GPU(s) for {replicas} replica(s) x {gpus_each} GPU(s) each, "
            f"got {len(gpu_ids)}: {gpu_ids!r}. Adjust replicas, tp, or gpu_ids so they agree."
        )
    return [tuple(gpu_ids[i * gpus_each : (i + 1) * gpus_each]) for i in range(replicas)]


def build_specs(fleet: FleetSpec) -> list[ServerSpec]:
    """Expand a fleet into one `ServerSpec` per replica.

    Pure apart from GPU discovery when `fleet.gpu_ids` is empty, so `--dry-run`
    can print exactly what would launch.

    When `gpu_ids` is set, the count must match the fleet exactly: an explicit
    list is a statement about which GPUs to use, and quietly ignoring some of it
    would hide a mistake. Discovered GPUs are treated as an upper bound instead,
    so a 1-replica probe on an 8-GPU host takes the first GPUs it needs and
    reports what it left idle.

    Args:
        fleet: The fleet to expand.

    Returns:
        One spec per replica, in port order.

    Raises:
        ValueError: No GPUs are available, an explicit `gpu_ids` does not match
            `replicas * tp * pp`, or too few GPUs exist to satisfy the fleet.
    """
    needed = fleet.replicas * fleet.gpus_per_replica
    if fleet.gpu_ids:
        gpu_ids = list(fleet.gpu_ids)
    else:
        gpu_ids = discover_local_gpus()
        if not gpu_ids:
            raise ValueError("No GPUs available: nvidia-smi found none and gpu_ids was empty.")
        if len(gpu_ids) < needed:
            raise ValueError(
                f"Fleet needs {needed} GPU(s) for {fleet.replicas} replica(s) x "
                f"{fleet.gpus_per_replica} each, but only {len(gpu_ids)} are visible: {gpu_ids!r}."
            )
        if len(gpu_ids) > needed:
            psrl_logger.info(f"Using GPUs {gpu_ids[:needed]} and leaving {gpu_ids[needed:]} idle.")
            gpu_ids = gpu_ids[:needed]
    blocks = partition_gpus(gpu_ids, fleet.replicas, fleet.gpus_per_replica)
    return [
        ServerSpec(
            checkpoint=fleet.checkpoint,
            served_model_name=fleet.served_model_name,
            host=fleet.host,
            port=fleet.base_port + index,
            tp=fleet.tp,
            pp=fleet.pp,
            dp=fleet.dp,
            gpu_ids=block,
            max_model_len=fleet.max_model_len,
            gpu_memory_utilization=fleet.gpu_memory_utilization,
            tool_call_parser=fleet.tool_call_parser,
            chat_template=fleet.chat_template,
            extra=tuple(fleet.extra),
        )
        for index, block in enumerate(blocks)
    ]


def write_endpoints(
    path: str | Path,
    served_model_name: str,
    endpoints: list[Endpoint],
    healthy_only: bool = True,
) -> Path:
    """Write endpoints as JSON so downstream tools need not be told the topology.

    Recording only healthy endpoints by default is what keeps an eval from
    dispatching work to a replica that never came up.

    Args:
        path: Destination file.
        served_model_name: Name every endpoint advertises.
        endpoints: Endpoints to record.
        healthy_only: Skip endpoints that failed their health check.

    Returns:
        The path written.
    """
    selected = [e for e in endpoints if e.healthy or not healthy_only]
    payload = {
        "served_model_name": served_model_name,
        "n_endpoints": len(selected),
        "endpoints": [e.to_dict() for e in selected],
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def read_endpoints(path: str | Path) -> tuple[str, list[str]]:
    """Read back what `write_endpoints` wrote.

    Args:
        path: An `endpoints.json` file.

    Returns:
        The served model name and the endpoint URLs.
    """
    payload = json.loads(Path(path).read_text())
    return payload["served_model_name"], [entry["url"] for entry in payload["endpoints"]]


def launch_fleet(
    fleet: FleetSpec,
    env_script: str = "",
    output_dir: str | Path = "",
    dry_run: bool = False,
) -> FleetResult:
    """Start every replica, wait for readiness, and record the healthy endpoints.

    Replicas are started before any is polled, so their model loads overlap
    rather than serialize.

    Args:
        fleet: The fleet to launch.
        env_script: Env script each replica sources before exec'ing vLLM.
        output_dir: Where to write `endpoints.json` and per-replica logs.
            Defaults to /tmp.
        dry_run: Build and log the specs without launching anything.

    Returns:
        The launch outcome. On a dry run, `endpoints` and `handles` are empty.

    Raises:
        RuntimeError: Fewer than `min_healthy_frac` of replicas came up. Any
            replica that did start is torn down first, so a failed launch does
            not leave GPUs occupied.
    """
    specs = build_specs(fleet)
    out_dir = Path(output_dir) if output_dir else Path("/tmp")

    psrl_logger.info(
        f"Fleet: {fleet.replicas} replica(s) of {fleet.served_model_name} "
        f"(tp={fleet.tp} pp={fleet.pp}, max_model_len={fleet.max_model_len}) "
        f"on ports {specs[0].port}-{specs[-1].port}."
    )
    for spec in specs:
        psrl_logger.info(f"  port {spec.port}: GPUs {list(spec.gpu_ids)}")

    if dry_run:
        for spec in specs:
            psrl_logger.info(f"  would run: {build_shell_command(spec, env_script)[-1]}")
        return FleetResult(served_model_name=fleet.served_model_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    handles = [launch(spec, env_script=env_script, log_file=out_dir / f"vllm_{spec.port}.log") for spec in specs]
    endpoints = wait_ready(handles, timeout_sec=fleet.wait_ready_sec)

    result = FleetResult(
        served_model_name=fleet.served_model_name,
        endpoints=endpoints,
        handles=handles,
        endpoints_file=write_endpoints(out_dir / "endpoints.json", fleet.served_model_name, endpoints),
    )

    n_healthy = len(result.healthy)
    psrl_logger.info(f"Fleet ready: {n_healthy}/{len(endpoints)} replica(s) healthy.")
    if n_healthy < len(endpoints):
        for endpoint in endpoints:
            if not endpoint.healthy:
                psrl_logger.error(f"Replica at {endpoint.url} is unhealthy, see logs in {out_dir}.")

    required = fleet.min_healthy_frac * len(endpoints)
    if n_healthy < required:
        result.terminate()
        raise RuntimeError(
            f"Only {n_healthy}/{len(endpoints)} replica(s) became healthy, "
            f"below the required fraction {fleet.min_healthy_frac:.2f}. Stopped the fleet."
        )
    return result


def write_spec_json(path: str | Path, fleet: FleetSpec) -> Path:
    """Serialize a fleet spec to storage that another process can read.

    This is what keeps the multi-host ssh command simple. The alternative --
    interpolating every serving flag into a remote command string -- needs quoting
    once for the local shell and again for the remote, which is the most fragile
    part of a shell implementation. Passing one path means the remote command has
    exactly one variable argument.

    Args:
        path: Destination. For multi-host use it must be on shared storage.
        fleet: The spec to serialize.

    Returns:
        The path written.
    """
    payload = dataclasses.asdict(fleet)
    payload["gpu_ids"] = list(fleet.gpu_ids)
    payload["extra"] = list(fleet.extra)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def load_spec_json(path: str | Path) -> FleetSpec:
    """Rebuild a fleet spec written by `write_spec_json`.

    Args:
        path: A spec JSON file.

    Returns:
        The reconstructed spec, with sequence fields restored to tuples.
    """
    payload = json.loads(Path(path).read_text())
    payload["gpu_ids"] = tuple(payload.get("gpu_ids") or ())
    payload["extra"] = tuple(payload.get("extra") or ())
    return FleetSpec(**payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m psrl.eval.vllm_fleet",
        description="Launch N independent vLLM replicas on this host.",
    )
    parser.add_argument("--spec-json", required=True, help="Fleet spec JSON written by write_spec_json.")
    parser.add_argument("--output-dir", required=True, help="Where to write endpoints.json and replica logs.")
    parser.add_argument("--env-script", default="", help="Env script each replica sources before launching.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would launch and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m psrl.eval.vllm_fleet`.

    This exists so `vllm_multinode` has a single stable command to invoke on every
    remote host. Interactive use should prefer `python -m psrl.eval.serve`, which
    composes the spec from Hydra config instead of requiring a JSON file.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        launch_fleet(
            load_spec_json(args.spec_json),
            env_script=args.env_script,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        psrl_logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
