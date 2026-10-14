import os
import socket
from dataclasses import dataclass
from enum import Enum

import torch


def get_local_ip() -> str:
    """
    Get the local IP address of the node.
    """
    if os.environ.get("LOCAL_NODE_IP") is not None:
        return os.environ.get("LOCAL_NODE_IP")
    else:
        return socket.gethostbyname(socket.gethostname())


def get_local_gpu_id() -> int:
    """
    Get the current GPU ID of the node.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None:
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
        if gpu_ids[0] == "":
            return -1
        local_gpu_id = torch.cuda.current_device()
        assert len(gpu_ids) > local_gpu_id, (
            f"GPU ID is out of range, current local GPU ID is {local_gpu_id}, but only {gpu_ids} GPUs are available"
        )
        return int(gpu_ids[local_gpu_id])
    else:
        return -1


class LinkType(Enum):
    """Network connection types"""

    ETH = "ethernet"  # Ethernet connection
    IB_PCIE = "ib_pcie"  # Cross-node connection with mixed IB and PCIe
    PCIE = "pcie"  # CPU-GPU connection within same node
    IB = "infiniband"  # Cross-node connection with only IB
    NVLINK = "nvlink"  # GPU-GPU connection within same node
    LOCAL = "local"  # Local memory copy


@dataclass
class NetworkLink:
    """Network connection information"""

    link_type: LinkType
    bandwidth_gbps: float  # Bandwidth in Gbps
    latency_us: float  # Latency in microseconds


class NetworkTopology:
    """Network topology manager"""

    def __init__(self):
        self._links: dict[tuple[str, str], NetworkLink] = {}
        self._client_positions: dict[str, tuple[str, int]] = {}  # client_name -> (node_ip, node_gpu_id)
        self._init_default_topology()

    def _init_default_topology(self):
        """Initialize default topology based on hostname to determine if same node"""
        # Default bandwidth configuration for 8*H20 machines
        # H20 specs: NVLink 4.0 (900 GB/s bidirectional), PCIe 5.0 x16 (128 GB/s), IB HDR200 (200 Gbps)
        self._default_links = {
            LinkType.ETH: NetworkLink(LinkType.ETH, 25.0, 20.0),  # 25 Gbps, 20us
            LinkType.IB_PCIE: NetworkLink(LinkType.IB_PCIE, 128.0, 5.0),  # 128 GB/s (PCIe 5.0 x16), 5us
            LinkType.PCIE: NetworkLink(LinkType.PCIE, 128.0, 5.0),  # 128 GB/s (PCIe 5.0 x16), 5us
            LinkType.IB: NetworkLink(LinkType.IB, 200.0, 2.0),  # 200 Gbps (HDR200), 2us
            LinkType.NVLINK: NetworkLink(LinkType.NVLINK, 900.0, 0.5),  # 900 GB/s (NVLink 4.0), 0.5us
            LinkType.LOCAL: NetworkLink(LinkType.LOCAL, 1000.0, 0.1),  # Local memory copy
        }

    def register_client(self, client_name: str, node_ip: str, node_gpu_id: int):
        """Register client and its node location"""
        self._client_positions[client_name] = (node_ip, node_gpu_id)

    def get_link(self, client1: str, client2: str) -> NetworkLink:
        """Get network connection information between two clients"""
        if client1 == client2:
            return self._default_links[LinkType.LOCAL]

        # Check if already cached
        key = tuple(sorted([client1, client2]))
        if key in self._links:
            return self._links[key]

        # Get node information
        node1, gpu_id1 = self._client_positions.get(client1, ("unknown", -1))
        node2, gpu_id2 = self._client_positions.get(client2, ("unknown", -1))

        # Determine connection type
        if node1 == node2:
            # Different devices on one node use NVLink or PCIe.
            if gpu_id1 != gpu_id2:
                if gpu_id1 == -1 or gpu_id2 == -1:
                    link_type = LinkType.PCIE
                else:
                    link_type = LinkType.NVLINK
            else:
                link_type = LinkType.LOCAL
        else:
            # Cross-node, assume InfiniBand
            if gpu_id1 != -1 and gpu_id2 != -1:
                link_type = LinkType.IB
            else:
                link_type = LinkType.IB_PCIE

        link = self._default_links[link_type]
        self._links[key] = link
        return link

    def get_bandwidth_gbps(self, client1: str, client2: str) -> float:
        """Get bandwidth between two clients (Gbps)"""
        return self.get_link(client1, client2).bandwidth_gbps

    def get_latency_us(self, client1: str, client2: str) -> float:
        """Get latency between two clients (microseconds)"""
        return self.get_link(client1, client2).latency_us

    def get_link_type(self, client1: str, client2: str) -> LinkType:
        """Get link type between two clients for sorting purposes"""
        return self.get_link(client1, client2).link_type

    def get_link_priority(self, client1: str, client2: str) -> int:
        """Get link priority for sorting (higher value = better connection)"""
        link_type = self.get_link_type(client1, client2)
        # Define priority mapping: LOCAL > NVLINK > IB > PCIE > IB_PCIE > ETH
        priority_map = {
            LinkType.LOCAL: 5,
            LinkType.NVLINK: 4,
            LinkType.IB: 3,
            LinkType.PCIE: 2,
            LinkType.IB_PCIE: 1,
            LinkType.ETH: 0,
        }
        return priority_map[link_type]

    def set_custom_link(self, client1: str, client2: str, link: NetworkLink):
        """Set custom connection information"""
        key = tuple(sorted([client1, client2]))
        self._links[key] = link
