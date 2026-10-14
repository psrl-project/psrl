import socket

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

_PORT_SCANNER_PREFIX = "psrl_port_scanner_"
_PORT_SCANNER_NAMESPACE = "psrl"


def _actor_name_for_ip(host_ip: str) -> str:
    """Convert an IP to a Ray actor name (dots → underscores)."""
    return _PORT_SCANNER_PREFIX + host_ip.replace(".", "_")


@ray.remote
class PortScanner:
    """Per-node port allocator pinned to a specific node.

    Serial execution guarantees no two callers on the same node ever
    receive the same port. The actor always checks localhost (127.0.0.1)
    because it is scheduled on the target node itself.
    """

    def __init__(self):
        self.min_port = 20000
        self.max_port = 60000

    def find_free_port(self):
        for port in range(self.min_port, self.max_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:  # port is available
                    self.min_port = port + 1
                    return port
        raise RuntimeError("No free ports available")


def create_port_scanners(ip_to_node_id: dict[str, str]) -> dict[str, ray.actor.ActorHandle]:
    """Create one PortScanner named actor per unique node IP, pinned to that node.

    Must be called exactly once from the driver process (ray_trainer.__init__).
    Uses lifetime="detached" + explicit namespace so that actors are visible
    across all Ray jobs in the cluster (needed because vLLM's EngineCore subprocess
    creates a new Ray job via ray.init). The driver explicitly kills them at exit.

    Args:
        ip_to_node_id: Mapping from node IP to Ray node ID.

    Returns:
        Dict mapping IP to actor handle (for explicit ray.kill at normal exit).
    """
    handles = {}
    for ip, node_id in ip_to_node_id.items():
        name = _actor_name_for_ip(ip)
        # Kill any leftover scanner from a previous crashed run.
        try:
            old = ray.get_actor(name, namespace=_PORT_SCANNER_NAMESPACE)
            ray.kill(old)
        except ValueError:
            pass
        handles[ip] = PortScanner.options(
            name=name,
            namespace=_PORT_SCANNER_NAMESPACE,
            lifetime="detached",
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote()
    return handles


def get_port_scanner(host_ip: str):
    """Get the PortScanner actor for the given node IP.

    Raises ValueError if create_port_scanners() has not been called for this IP.
    """
    name = _actor_name_for_ip(host_ip)
    return ray.get_actor(name, namespace=_PORT_SCANNER_NAMESPACE)
