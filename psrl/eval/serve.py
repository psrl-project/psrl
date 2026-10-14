"""CLI for serving a checkpoint for offline evaluation.

This is the only Hydra-aware module in `psrl.eval`. It composes config, converts
it into the plain dataclasses the serving modules take, and hands off. Keeping
the boundary here is what lets `vllm_server` and `vllm_fleet` be tested without
composing a config or holding a GPU.

Three topologies, selected with `topology=`. Note that overrides address the
*group name*, not the file name: a value in `topology/fleet.yaml` is overridden as
`topology.replicas=...`, never `fleet.replicas=...`.

- `single`: one server process.
- `fleet`: N replicas on this host, each with its own port and GPU block.
- `multinode`: one fleet per host across a hosts file.

Usage::

    # 4 replicas x TP=2 on this host
    python -m psrl.eval.serve server=qwen35_9b topology=fleet \\
        topology.replicas=4 topology.tp=2 output_dir=outputs/serve/qwen35

    # every host in a hosts file
    python -m psrl.eval.serve server=qwen35_9b topology=multinode \\
        topology.hosts_file=${PSRL_WORKSPACE}/hosts/32GPUs output_dir=/shared/serve

    # check the plan without launching
    python -m psrl.eval.serve server=qwen35_9b dry_run=true output_dir=/tmp/plan

Writes `<output_dir>/endpoints.json`, which downstream eval harnesses read to
discover healthy endpoints instead of being told the topology.
"""

from __future__ import annotations

import logging
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from psrl.eval.vllm_fleet import FleetSpec, launch_fleet
from psrl.eval.vllm_multinode import MultinodeSpec, launch_multinode, read_hosts_file

psrl_logger = logging.getLogger(__file__)


def build_fleet_spec(cfg: DictConfig) -> FleetSpec:
    """Convert composed config into a `FleetSpec`.

    This is the Hydra boundary: everything downstream sees plain dataclasses.
    `single` is expressed as a one-replica fleet rather than a separate code path,
    so there is only one launch implementation to reason about.

    Args:
        cfg: Composed config with `server` and `topology` groups.

    Returns:
        The per-host fleet to launch.
    """
    server, topology = cfg.server, cfg.topology
    replicas = 1 if topology.kind == "single" else int(topology.replicas)
    base_port = int(topology.get("port", 0) or topology.get("base_port", 8000))
    return FleetSpec(
        checkpoint=str(server.checkpoint),
        served_model_name=str(server.served_model_name),
        replicas=replicas,
        tp=int(topology.tp),
        pp=int(topology.pp),
        dp=int(topology.get("dp", 1)),
        base_port=base_port,
        gpu_ids=tuple(topology.gpu_ids or ()),
        host=str(topology.host),
        max_model_len=server.max_model_len,
        gpu_memory_utilization=float(server.gpu_memory_utilization),
        tool_call_parser=str(server.tool_call_parser or ""),
        chat_template=str(server.chat_template or ""),
        extra=tuple(server.extra or ()),
        wait_ready_sec=float(topology.wait_ready_sec),
        min_healthy_frac=float(topology.get("min_healthy_frac", 1.0)),
    )


def _serve_local(cfg: DictConfig) -> int:
    """Launch a single-host topology and report where the endpoints landed."""
    fleet = build_fleet_spec(cfg)
    result = launch_fleet(
        fleet,
        env_script=str(cfg.env_script or ""),
        output_dir=str(cfg.output_dir),
        dry_run=bool(cfg.dry_run),
    )
    if cfg.dry_run:
        return 0

    psrl_logger.info(f"Endpoints written to {result.endpoints_file}.")
    psrl_logger.info(f"api_base: {result.api_base}")
    if cfg.wait_forever:
        _block_until_stopped(result.terminate)
    return 0


def _serve_multinode(cfg: DictConfig) -> int:
    """Launch one fleet per host across a hosts file."""
    topology = cfg.topology
    fleet = build_fleet_spec(cfg)
    # Remotes must bind all interfaces, otherwise the coordinator cannot reach them.
    if fleet.host in ("127.0.0.1", "localhost"):
        raise ValueError(f"Multinode requires a routable bind address, got host={fleet.host!r}. Use 0.0.0.0.")
    spec = MultinodeSpec(
        hosts=read_hosts_file(topology.hosts_file),
        fleet=fleet,
        env_script=str(cfg.env_script or ""),
        ssh_user=str(topology.get("ssh_user", "") or ""),
        min_healthy_frac=float(topology.get("min_healthy_frac", 0.5)),
    )
    result = launch_multinode(spec, output_dir=str(cfg.output_dir), dry_run=bool(cfg.dry_run))
    if cfg.dry_run:
        return 0

    psrl_logger.info(f"Endpoints written to {result.endpoints_file}.")
    psrl_logger.info(f"api_base: {result.api_base}")
    if cfg.wait_forever:
        # Remote processes outlive this one, so there is nothing local to reap.
        _block_until_stopped(lambda: None)
    return 0


def _block_until_stopped(on_stop) -> None:
    """Sleep until interrupted, then run `on_stop`.

    Only used with `wait_forever=true`, for running the launcher in the
    foreground. An eval wrapper wants the default: return immediately and leave
    the servers up.
    """
    psrl_logger.info("Serving. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        psrl_logger.info("Interrupted, stopping servers...")
        on_stop()


@hydra.main(config_path="config", config_name="serve", version_base=None)
def main(cfg: DictConfig) -> int:
    """Entry point for `python -m psrl.eval.serve`."""
    psrl_logger.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")
    kind = cfg.topology.kind
    try:
        if kind == "multinode":
            return _serve_multinode(cfg)
        if kind in ("single", "fleet"):
            return _serve_local(cfg)
        raise ValueError(f"Unknown topology kind {kind!r}, expected single, fleet, or multinode.")
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        psrl_logger.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
