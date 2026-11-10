import pickle
from unittest.mock import Mock

import pytest
from psrl.utils.common.nixl_metadata import merge_client_updates
from psrl.utils.nixl.client import NIXLMultiStorageClients, NIXLStorageClient
from psrl.utils.nixl.nixl_spec import NIXLClientInfo, NIXLClientType

pytestmark = pytest.mark.cpu_test


def _info(name, *, registered=True, metadata=b"metadata"):
    return NIXLClientInfo(name, "127.0.0.1", -1, NIXLClientType.PUSH_SIDE, {}, metadata, is_registered=registered)


def _client(name):
    client = object.__new__(NIXLStorageClient)
    client.client_name = name
    client._is_connected = True
    client.local_client_info = _info(name)
    client._temp_desc_bytes_mapping = {("weight", (0,)): b"local-descriptor"}
    client._all_client_infos = {}
    client._all_temp_mappings = {}
    client._target_client_connected = {}
    client.agent = Mock()
    return client


def test_export_snapshots_current_registration_without_sending_notifications():
    client = _client("trainer")
    updates = client.get_local_client_info()
    client._temp_desc_bytes_mapping.clear()

    assert NIXLClientInfo.deserialize(updates["trainer"]["info"]) == client.local_client_info, (
        "Metadata must describe the current registration."
    )
    assert updates["trainer"]["temp_mapping"] == {("weight", (0,)): b"local-descriptor"}, (
        "The exported mapping must survive local registration changes."
    )
    client.agent.send_notif.assert_not_called()


@pytest.mark.parametrize("local_info", [None, _info("trainer", registered=False)])
def test_export_rejects_sleeping_clients(local_info):
    client = _client("trainer")
    client.local_client_info = local_info
    with pytest.raises(AssertionError, match="must register its tensors"):
        client.get_local_client_info()


def test_install_replaces_descriptors_and_preserves_other_clients():
    source = _client("trainer")
    target = _client("ps")
    untouched = _info("other")
    target._all_client_infos["other"] = untouched
    for cycle in range(32):
        descriptor = bytes([cycle])
        source.local_client_info = _info("trainer", metadata=descriptor)
        source._temp_desc_bytes_mapping = {("weight", (0,)): descriptor}
        updates = pickle.loads(pickle.dumps(source.get_local_client_info()))
        target.update_client_infos(updates)
        assert target._all_client_infos["trainer"].meta == descriptor, "Every cycle must install fresh metadata."
        assert target._all_temp_mappings["trainer"] == {("weight", (0,)): descriptor}, (
            "Temporary descriptors must advance with the registration."
        )
        assert target._all_client_infos["other"] is untouched, "Unchanged peers must retain their descriptors."
    target.agent.get_new_notifs.assert_not_called()


def test_update_reloads_remote_memory_registration_before_next_transfer():
    target = _client("ps")
    target._target_client_connected = {"trainer": True, "other": True}
    source = _client("trainer")
    source.local_client_info = _info("trainer", metadata=b"new-registration")

    target.update_client_infos(source.get_local_client_info())
    target._ensure_client_info_fetched("trainer")

    target.agent.add_remote_agent.assert_called_once_with(b"new-registration")
    assert target._target_client_connected["other"], "Unchanged remote registrations must remain connected."


def test_invalid_update_fails_without_partially_installing_descriptors():
    client = _client("ps")
    existing = _info("trainer")
    client._all_client_infos["trainer"] = existing
    updates = {
        "trainer": {"info": _info("trainer", metadata=b"new").serialize(), "temp_mapping": {}},
        "broken": {"info": b"invalid metadata", "temp_mapping": {}},
    }
    with pytest.raises(pickle.UnpicklingError):
        client.update_client_infos(updates)
    assert client._all_client_infos == {"trainer": existing}, "A failed update must preserve installed metadata."
    assert client._all_temp_mappings == {}, "A failed update must preserve installed temporary descriptors."


def test_update_reaches_all_clients_sharing_one_ps_agent():
    clients = [_client("push"), _client("pull")]
    agent = object.__new__(NIXLMultiStorageClients)
    agent._is_connected = True
    agent.multi_clients = clients
    updates = _client("trainer").get_local_client_info()

    agent.update_client_infos(updates)

    for client in clients:
        assert client._all_client_infos["trainer"].name == "trainer", "Both PS clients must receive the update."
        assert client._all_temp_mappings["trainer"] == updates["trainer"]["temp_mapping"], (
            "Both PS clients must receive temporary descriptors."
        )
    assert clients[0]._all_client_infos is clients[1]._all_client_infos, "PS clients must share updated metadata."


def test_merge_requires_exactly_one_update_per_expected_client():
    groups = [_client("rank0").get_local_client_info(), _client("rank1").get_local_client_info()]
    assert set(merge_client_updates(groups, ["rank0", "rank1"])) == {"rank0", "rank1"}, (
        "All participating ranks must be present."
    )
    with pytest.raises(ValueError, match="Duplicate.*rank0"):
        merge_client_updates([groups[0], groups[0]], ["rank0", "rank1"])
    with pytest.raises(ValueError, match="missing=\\['rank1'\\]"):
        merge_client_updates(groups[:1], ["rank0", "rank1"])
    with pytest.raises(ValueError, match="unexpected=\\['rank1'\\]"):
        merge_client_updates(groups, ["rank0"])
