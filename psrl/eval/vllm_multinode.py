"""Fan a vLLM fleet out to every host in a hosts file.

One host runs one fleet of R replicas, so total capacity is `hosts x replicas`
endpoints. No replica spans a host boundary: cross-node tensor parallelism needs
a managed Ray cluster and makes a single node failure kill a whole replica. With
node-local replicas a wedged node costs its own share of capacity and nothing
else, which is the behaviour that matters once the host count grows.

Hosts files follow the convention already used across this repo
(`${PSRL_WORKSPACE}/hosts/<N>GPUs`): one address per line, `#` comments and blank
lines ignored. Scaling from 2 hosts to 16 is a different `--hosts-file`, nothing
more.

Usage::

    python -m psrl.eval.vllm_multinode \\
        --hosts-file ${PSRL_WORKSPACE}/hosts/32GPUs \\
        --spec-json /shared/run/fleet.json \\
        --output-dir /shared/run
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from psrl.eval.vllm_fleet import FleetSpec, load_spec_json, write_endpoints, write_spec_json
from psrl.eval.vllm_server import Endpoint

psrl_logger = logging.getLogger(__file__)

_SSH_OPTS = (
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=240",
)


@dataclass
class MultinodeSpec:
    """How to spread one fleet definition across many hosts.

    Attributes:
        hosts: Addresses to launch on, already stripped of comments and blanks.
        fleet: The per-host fleet. Every host launches this same definition, so
            capacity is `len(hosts) * fleet.replicas`.
        env_script: Env script each remote sources before launching.
        ssh_user: Optional ssh username.
        min_healthy_frac: Fraction of expected endpoints that must come up for the
            launch to count as usable. Defaults to half, because at scale a
            wedged node is routine and a multi-hour eval should degrade rather
            than abort.
    """

    hosts: list[str]
    fleet: FleetSpec
    env_script: str = ""
    ssh_user: str = ""
    min_healthy_frac: float = 0.5

    @property
    def n_expected(self) -> int:
        """Endpoints expected if every host comes up fully."""
        return len(self.hosts) * self.fleet.replicas


@dataclass
class MultinodeResult:
    """Outcome of a multi-host launch.

    Attributes:
        served_model_name: Name every endpoint advertises.
        endpoints: Healthy endpoints across all hosts.
        failed_hosts: Hosts that returned no healthy endpoint at all.
        endpoints_file: Where the aggregated endpoint list was written.
    """

    served_model_name: str
    endpoints: list[Endpoint] = field(default_factory=list)
    failed_hosts: list[str] = field(default_factory=list)
    endpoints_file: Path | None = None

    @property
    def api_base(self) -> str:
        """Comma-separated healthy URLs, as eval harnesses expect."""
        return ",".join(e.url for e in self.endpoints)


def read_hosts_file(path: str | Path) -> list[str]:
    """Parse a hosts file into addresses.

    Args:
        path: File with one host per line. `#` comments and blank lines ignored.

    Returns:
        Host addresses in file order.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The file contains no usable host.
    """
    text = Path(path).read_text()
    hosts = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not hosts:
        raise ValueError(f"No hosts found in {path!r}.")
    return hosts


def build_remote_command(spec_json: str, output_dir: str, env_script: str = "") -> str:
    """Build the shell command a remote host runs to launch its fleet.

    Only paths are interpolated, so `shlex.quote` on each is enough -- passing a spec
    file instead of ~17 serving flags is what keeps this short.

    The result still has to survive one more round of shell parsing: `ssh host bash
    -lc CMD` concatenates its arguments and the *remote* shell re-splits them, so the
    caller must quote this string again before handing it to ssh (see
    `_launch_one_host`). Skipping that made the remote run `bash -lc source <path> &&
    ...`, where `source` became the whole `-c` string with the path as `$0`, failing
    with "line 0: source: filename argument required".

    Args:
        spec_json: Shared-FS path to the fleet spec.
        output_dir: Shared-FS directory for this host's logs and endpoints.
        env_script: Env script to source first. Empty skips sourcing.

    Returns:
        A single shell command string, not yet quoted for ssh.
    """
    inner = (
        f"python -m psrl.eval.vllm_fleet --spec-json {shlex.quote(spec_json)} --output-dir {shlex.quote(output_dir)}"
    )
    if env_script:
        inner = f"source {shlex.quote(env_script)} && {inner}"
    return inner


def _rehost(url: str, host: str) -> str:
    """Rewrite a URL's host, keeping scheme, port, and path.

    A remote fleet binds 0.0.0.0 and therefore reports loopback URLs, which are
    useless to a coordinator on another machine. The endpoint has to be addressed
    by the host it actually runs on.
    """
    parsed = urlparse(url)
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunparse(parsed._replace(netloc=netloc))


def _launch_one_host(
    host: str,
    spec_json: str,
    output_dir: Path,
    env_script: str,
    ssh_user: str,
    timeout_sec: float,
) -> tuple[str, list[Endpoint], str]:
    """Launch one host's fleet over ssh and collect its endpoints.

    Returns:
        The host, its healthy endpoints re-addressed to that host, and an error
        string that is empty on success.
    """
    host_dir = output_dir / "hosts" / host.replace(":", "_").replace("/", "_")
    host_dir.mkdir(parents=True, exist_ok=True)
    launch_log = host_dir / "launch.log"

    command = build_remote_command(spec_json, str(host_dir), env_script)
    target = f"{ssh_user}@{host}" if ssh_user else host
    # SSH re-splits trailing arguments, so quote the command as one `-c` argument.
    argv = ["ssh", *_SSH_OPTS, target, "bash", "-lc", shlex.quote(command)]

    psrl_logger.info(f"Launching fleet on {host}...")
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        launch_log.write_text(f"ssh to {host} timed out after {timeout_sec:.0f}s.\n")
        return host, [], f"timed out after {timeout_sec:.0f}s"
    except OSError as e:
        return host, [], f"ssh failed: {e}"

    launch_log.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        return host, [], f"remote exited {completed.returncode}, see {launch_log}"

    # The remote writes endpoints.json into its own host_dir on shared storage.
    remote_endpoints = host_dir / "endpoints.json"
    if not remote_endpoints.is_file():
        return host, [], f"remote wrote no endpoints.json, see {launch_log}"

    payload = json.loads(remote_endpoints.read_text())
    endpoints = [
        Endpoint(
            url=_rehost(entry["url"], host),
            host=host,
            gpu_ids=tuple(entry.get("gpu_ids") or ()),
            pid=entry.get("pid"),
            healthy=entry.get("healthy", True),
        )
        for entry in payload.get("endpoints", [])
    ]
    healthy = [e for e in endpoints if e.healthy]
    if not healthy:
        return host, [], f"no healthy replica, see {launch_log}"
    psrl_logger.info(f"Host {host}: {len(healthy)}/{len(endpoints)} replica(s) healthy.")
    return host, healthy, ""


def launch_multinode(
    spec: MultinodeSpec,
    output_dir: str | Path,
    dry_run: bool = False,
) -> MultinodeResult:
    """Launch a fleet on every host concurrently and aggregate the endpoints.

    Hosts are launched in parallel so total wall time is the slowest host's model
    load, not the sum. Only healthy endpoints are recorded, so a downstream eval
    cannot dispatch work to a replica that never came up.

    Args:
        spec: Hosts, per-host fleet, and quorum policy.
        output_dir: Shared-FS directory for the spec, aggregated endpoints, and
            per-host logs. Must be readable from every host.
        dry_run: Print the per-host command without connecting.

    Returns:
        Aggregated healthy endpoints and the list of hosts that failed.

    Raises:
        RuntimeError: Healthy endpoints fell below `min_healthy_frac`. Servers on
            hosts that did come up are left running, since tearing down remote
            processes on a partial failure risks killing a usable fleet over a
            transient ssh error. The message says how to stop them.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_json = write_spec_json(out_dir / "fleet.json", spec.fleet)

    psrl_logger.info(
        f"Multinode launch: {len(spec.hosts)} host(s) x {spec.fleet.replicas} replica(s) "
        f"= {spec.n_expected} endpoint(s) of {spec.fleet.served_model_name}."
    )
    if dry_run:
        command = build_remote_command(str(spec_json), str(out_dir / "hosts" / "<host>"), spec.env_script)
        psrl_logger.info(f"  hosts: {spec.hosts}")
        psrl_logger.info(f"  would run on each: {command}")
        return MultinodeResult(served_model_name=spec.fleet.served_model_name)

    # Keep SSH alive beyond the remote readiness wait so a slow model load is not reported as a timeout.
    timeout_sec = spec.fleet.wait_ready_sec + 600.0
    with ThreadPoolExecutor(max_workers=max(len(spec.hosts), 1)) as pool:
        outcomes = list(
            pool.map(
                lambda host: _launch_one_host(
                    host, str(spec_json), out_dir, spec.env_script, spec.ssh_user, timeout_sec
                ),
                spec.hosts,
            )
        )

    endpoints: list[Endpoint] = []
    failed_hosts: list[str] = []
    for host, host_endpoints, error in outcomes:
        if error:
            psrl_logger.error(f"Host {host} failed: {error}.")
            failed_hosts.append(host)
        endpoints.extend(host_endpoints)

    result = MultinodeResult(
        served_model_name=spec.fleet.served_model_name,
        endpoints=endpoints,
        failed_hosts=failed_hosts,
        endpoints_file=write_endpoints(out_dir / "endpoints.json", spec.fleet.served_model_name, endpoints),
    )

    psrl_logger.info(
        f"Multinode ready: {len(endpoints)}/{spec.n_expected} endpoint(s) healthy "
        f"across {len(spec.hosts) - len(failed_hosts)}/{len(spec.hosts)} host(s)."
    )
    if failed_hosts:
        psrl_logger.warning(f"Degraded: {len(failed_hosts)} host(s) contributed nothing: {failed_hosts}.")

    required = spec.min_healthy_frac * spec.n_expected
    if len(endpoints) < required:
        raise RuntimeError(
            f"Only {len(endpoints)}/{spec.n_expected} endpoint(s) became healthy, below the required "
            f"fraction {spec.min_healthy_frac:.2f}. Servers on healthy hosts were left running; stop them with "
            f"pssh -h <hosts> -i \"pkill -f 'vllm.entrypoints.openai.api_server'\"."
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m psrl.eval.vllm_multinode",
        description="Launch a vLLM fleet on every host in a hosts file.",
    )
    parser.add_argument("--hosts-file", required=True, help="One host per line; # comments ignored.")
    parser.add_argument("--spec-json", required=True, help="Fleet spec JSON on shared storage.")
    parser.add_argument("--output-dir", required=True, help="Shared-FS directory for endpoints and logs.")
    parser.add_argument("--env-script", default="", help="Env script each remote sources before launching.")
    parser.add_argument("--ssh-user", default="", help="Optional ssh username.")
    parser.add_argument(
        "--min-healthy-frac",
        type=float,
        default=0.5,
        help="Fraction of expected endpoints required for success (default: 0.5).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the per-host command and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m psrl.eval.vllm_multinode`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_parser().parse_args(argv)

    spec = MultinodeSpec(
        hosts=read_hosts_file(args.hosts_file),
        fleet=load_spec_json(args.spec_json),
        env_script=args.env_script,
        ssh_user=args.ssh_user,
        min_healthy_frac=args.min_healthy_frac,
    )
    try:
        result = launch_multinode(spec, output_dir=args.output_dir, dry_run=args.dry_run)
    except RuntimeError as e:
        psrl_logger.error(str(e))
        return 1
    if not args.dry_run:
        psrl_logger.info(f"Endpoints written to {result.endpoints_file}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
