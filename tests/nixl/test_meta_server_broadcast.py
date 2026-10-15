import pickle
import threading
import time

import pytest
from psrl.utils.nixl.comm_plan import NIXLCommPlan  # noqa: E402
from psrl.utils.nixl.nixl_spec import NIXLClientType  # noqa: E402
from psrl.utils.nixl.server import NIXLMetaServer  # noqa: E402

pytestmark = pytest.mark.cpu_test


class _ClientInfo:
    def __init__(self, name: str, client_type: NIXLClientType = NIXLClientType.PS_FOR_PUSH):
        self.name = name
        self.type = client_type

    def serialize(self) -> bytes:
        return self.name.encode()


class _ConcurrentAgent:
    def __init__(self, expected_concurrency: int):
        self.expected_concurrency = expected_concurrency
        self.lock = threading.Lock()
        self.all_workers_started = threading.Event()
        self.active = 0
        self.max_active = 0
        self.payloads: dict[str, bytes] = {}

    def send_notif(self, agent_name: str, payload: bytes):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active == self.expected_concurrency:
                self.all_workers_started.set()

        assert self.all_workers_started.wait(timeout=2.0)
        time.sleep(0.01)

        with self.lock:
            self.payloads[agent_name] = payload
            self.active -= 1


class _FailingAgent:
    def send_notif(self, agent_name: str, payload: bytes):
        if agent_name == "agent_2":
            raise RuntimeError("send failed")


def _make_server(agent, *, agent_count: int = 8, max_workers: int = 4) -> NIXLMetaServer:
    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server._broadcast_init_enabled = False
    server.metadata_broadcast_max_workers = max_workers
    server.connected_clients = {f"agent_{i}": [f"client_{i}"] for i in range(agent_count)}
    server.client_infos = {f"client_{i}": _ClientInfo(f"client_{i}") for i in range(agent_count)}
    server.client_info_bytes = {name: info.serialize() for name, info in server.client_infos.items()}
    server._client_temp_mappings = {name: {("weight", (0,)): name.encode()} for name in server.client_infos}
    server.comm_plan = NIXLCommPlan({}, {}, {})
    server.agent = agent
    server._is_all_client_infos_recved = True
    server._is_all_temp_mappings_recved = True
    return server


def test_client_info_broadcast_uses_bounded_parallelism():
    agent = _ConcurrentAgent(expected_concurrency=4)
    server = _make_server(agent, max_workers=4)

    server.notify_all_client_infos_and_comm_plan()

    assert agent.max_active == 4
    assert set(agent.payloads) == set(server.connected_clients)
    for agent_name, payload in agent.payloads.items():
        decoded = pickle.loads(payload)
        client_name = server.connected_clients[agent_name][0]
        assert decoded == {
            "client_infos": {client_name: client_name.encode()},
            "comm_plan": NIXLCommPlan({}, {}, {}).serialize(),
        }


def test_client_info_broadcast_propagates_send_failure():
    server = _make_server(_FailingAgent(), agent_count=4, max_workers=2)

    with pytest.raises(RuntimeError, match="send failed"):
        server.notify_all_client_infos_and_comm_plan()


def test_client_info_broadcast_includes_counterpart_types_and_per_agent_plan():
    agent = _ConcurrentAgent(expected_concurrency=3)
    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server._broadcast_init_enabled = False
    server.metadata_broadcast_max_workers = 3
    server.connected_clients = {
        "push_agent": ["push"],
        "ps_agent_0": ["ps_0"],
        "ps_agent_1": ["ps_1"],
    }
    server.client_infos = {
        "push": _ClientInfo("push", NIXLClientType.PUSH_SIDE),
        "ps_0": _ClientInfo("ps_0"),
        "ps_1": _ClientInfo("ps_1"),
        "unused_ps": _ClientInfo("unused_ps"),
    }
    server.client_info_bytes = {name: info.serialize() for name, info in server.client_infos.items()}
    server._client_temp_mappings = {name: {("weight", (0,)): name.encode()} for name in server.client_infos}
    server.comm_plan = NIXLCommPlan(
        push_to_ps_plan={"push": {"weight": {"ps_0": [(0,)]}}},
        rollout_pull_from_ps_plan={},
        train_pull_from_ps_plan={"push": {"weight": {"ps_1": [(1,)]}}},
    )
    server.agent = agent
    server._is_all_client_infos_recved = True
    server._is_all_temp_mappings_recved = True

    server.notify_all_client_infos_and_comm_plan()

    push_payload = pickle.loads(agent.payloads["push_agent"])
    assert set(push_payload["client_infos"]) == {"push", "ps_0", "ps_1", "unused_ps"}
    push_plan = NIXLCommPlan.deserialize(push_payload["comm_plan"])
    assert push_plan == server.comm_plan.for_clients(["push"])

    ps_payload = pickle.loads(agent.payloads["ps_agent_0"])
    assert set(ps_payload["client_infos"]) == {"ps_0"}
    assert NIXLCommPlan.deserialize(ps_payload["comm_plan"]) == NIXLCommPlan({}, {}, {})

    agent.payloads.clear()
    server.notify_all_client_temp_mappings()
    for agent_name, payload in agent.payloads.items():
        assert set(pickle.loads(payload)) == set(server.connected_clients[agent_name])
