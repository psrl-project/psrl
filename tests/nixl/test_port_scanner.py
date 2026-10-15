"""
Verify port scanner startup waits for scheduling and cleans up failed initialization.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from psrl.utils.nixl import port_scanner

pytestmark = pytest.mark.cpu_test


def _patch_scanners(monkeypatch):
    handles = [SimpleNamespace(ready=SimpleNamespace(remote=Mock(return_value=object()))) for _ in range(2)]
    actor_class = Mock()
    actor_class.options.return_value = actor_class
    actor_class.remote.side_effect = handles
    monkeypatch.setattr(port_scanner, "PortScanner", actor_class)
    monkeypatch.setattr(port_scanner.ray, "get_actor", Mock(side_effect=ValueError("Scanner is not registered.")))
    return actor_class, handles


def test_scanner_creation_waits_for_all_node_pinned_actors(monkeypatch):
    actor_class, handles = _patch_scanners(monkeypatch)
    wait = Mock()
    monkeypatch.setattr(port_scanner.ray, "get", wait)

    scanners = port_scanner.create_port_scanners({"192.0.2.1": "node-a", "192.0.2.2": "node-b"})

    assert list(scanners.values()) == handles, "Scanners must retain their per-node handles."
    options = [call.kwargs for call in actor_class.options.call_args_list]
    assert [option["scheduling_strategy"].node_id for option in options] == ["node-a", "node-b"], (
        "Each scanner must be pinned to its target node."
    )
    assert all(option["scheduling_strategy"].soft is False for option in options), (
        "Port availability must be checked on the requested node."
    )
    assert all(option["num_cpus"] == 0 for option in options), "Scanners must not wait for a model's CPU allocation."
    wait.assert_called_once_with([handle.ready.remote.return_value for handle in handles], timeout=60)


@pytest.mark.parametrize("creation_failure", [True, False])
def test_failed_scanner_startup_releases_created_actors(monkeypatch, creation_failure):
    actor_class, handles = _patch_scanners(monkeypatch)
    error = RuntimeError("Target node is unavailable.")
    if creation_failure:
        actor_class.remote.side_effect = [handles[0], error]
    monkeypatch.setattr(port_scanner.ray, "get", Mock(side_effect=error))
    kill = Mock()
    monkeypatch.setattr(port_scanner.ray, "kill", kill)

    with pytest.raises(RuntimeError, match="Target node is unavailable"):
        port_scanner.create_port_scanners({"192.0.2.1": "node-a", "192.0.2.2": "node-b"})

    expected = handles[:1] if creation_failure else handles
    assert [call.args[0] for call in kill.call_args_list] == expected, (
        "Startup failure must release every detached scanner created by the call."
    )


def test_scanner_cleanup_failure_preserves_original_startup_error(monkeypatch):
    _, handles = _patch_scanners(monkeypatch)
    monkeypatch.setattr(port_scanner.ray, "get", Mock(side_effect=RuntimeError("Target node is unavailable.")))
    kill = Mock(side_effect=ValueError("Scanner already died."))
    monkeypatch.setattr(port_scanner.ray, "kill", kill)

    with pytest.raises(RuntimeError, match="Target node is unavailable"):
        port_scanner.create_port_scanners({"192.0.2.1": "node-a", "192.0.2.2": "node-b"})

    assert kill.call_count == len(handles), "A cleanup failure must not skip the other scanner actors."
