import pytest
from psrl.workers.gen.vllm_async_server import GenInterface

pytestmark = pytest.mark.cpu_test


# Tests


def test_gen_interface_with_ps_manager():
    """GenInterface still works when ps_manager_handle is provided."""
    iface = GenInterface(
        role="rollout",
        rollout_replica_idx=0,
        ps_manager_handle=object(),  # any truthy object
        status_endpoint="tcp://127.0.0.1:28000",
    )
    assert iface.ps_manager_handle is not None
    assert iface.role == "rollout"


def test_gen_interface_without_ps_manager():
    """GenInterface can be constructed with ps_manager_handle=None (reward model path)."""
    iface = GenInterface(
        role="reward_model_MyRM",
        rollout_replica_idx=2,
        ps_manager_handle=None,
        status_endpoint="tcp://127.0.0.1:28001",
    )
    assert iface.ps_manager_handle is None
    assert iface.rollout_replica_idx == 2
