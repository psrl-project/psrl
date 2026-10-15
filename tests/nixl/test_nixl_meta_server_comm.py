"""
Exercise native NIXL metadata exchange and CPU tensor transfers.
"""

import os
import random
import socket
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration


def _available_port():
    for port in random.sample(range(20000, 30000), 10000):
        with socket.socket() as listener:
            try:
                listener.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No NIXL metadata port is available outside the ephemeral port range.")


def _run_transfer_scenario():
    import torch
    from nixl._api import nixl_agent, nixl_agent_config
    from omegaconf import OmegaConf
    from psrl.utils.nixl.client import NIXLStorageClient
    from psrl.utils.nixl.nixl_spec import NIXLClientType, NIXLSharding
    from psrl.utils.nixl.server import NIXLMetaServer

    config = OmegaConf.create(
        {
            "server_ip": "127.0.0.1",
            "server_port": _available_port(),
            "max_pinned_temp_memory_slots": None,
            "enable_tms_for_temp_buffers": False,
            "metadata_server_info_timeout_s": 15,
        }
    )
    prefix = f"nixl-test-{uuid4().hex}"
    agents = []
    with ExitStack() as cleanup:
        server = NIXLMetaServer(f"{prefix}-server", config)
        cleanup.callback(server.shutdown)
        clients = []
        tensors = []
        for index, client_type in enumerate((NIXLClientType.PUSH_SIDE, NIXLClientType.PS_FOR_PUSH)):
            name = f"{prefix}-client-{index}"
            agent = nixl_agent(name, nixl_agent_config(True, True, _available_port()))
            agents.append(agent)
            client = NIXLStorageClient(
                name,
                server.server_name,
                use_gpu=False,
                client_type=client_type,
                nixl_config=config,
                binded_agent=agent,
            )
            cleanup.callback(client.shutdown)
            state = {"a": torch.arange(6, dtype=torch.float32), "b": torch.arange(8, dtype=torch.float32)}
            for tensor in state.values():
                tensor.add_(index * 10)
            client.register_local_tensors(state, {key: NIXLSharding.default() for key in state})
            client.connect_to_server(timeout=15)
            client.send_local_info()
            clients.append(client)
            tensors.append(state)

        server.wait_for_client_infos(expected_agents=2, timeout=15)
        server.make_comm_plan()
        server.notify_all_client_infos_and_comm_plan()
        for client in clients:
            client.wait_for_server_info(timeout=15)

        source, target = clients
        for key in tensors[0]:
            expected = tensors[1][key].clone()
            shards = source.client_read(target.client_name, target.client_name, key, "read")
            assert shards == [(0,)], "The read plan must transfer the full tensor shard."
            source.wait(key, "read", "READ", target_client=target.client_name, timeout=15)
            torch.testing.assert_close(tensors[0][key], expected)

            tensors[0][key].add_(100)
            shards = source.client_write(target.client_name, target.client_name, key, "write")
            assert shards == [(0,)], "The write plan must transfer the full tensor shard."
            source.wait(key, "write", "WRITE", target_client=target.client_name, timeout=15)
            torch.testing.assert_close(tensors[1][key], tensors[0][key])


def test_nixl_meta_comm():
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", UCX_TLS="tcp,sm,self", UCX_NET_DEVICES="all")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, f"Native NIXL transfer failed. {result.stdout} {result.stderr}"


if __name__ == "__main__":
    _run_transfer_scenario()
