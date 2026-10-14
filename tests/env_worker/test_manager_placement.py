import pytest
from omegaconf import OmegaConf
from psrl.workers.env_worker.manager import resolve_placement_node_ips


@pytest.mark.cpu_test
def test_colocated_placement_uses_every_alive_node():
    config = OmegaConf.create({"placement": "colocated", "dedicated_node_ips": []})
    alive = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

    assert resolve_placement_node_ips(config, alive) == alive


@pytest.mark.cpu_test
def test_dedicated_placement_uses_only_listed_nodes():
    config = OmegaConf.create({"placement": "dedicated", "dedicated_node_ips": ["10.0.0.2"]})
    alive = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

    assert resolve_placement_node_ips(config, alive) == ["10.0.0.2"]


@pytest.mark.cpu_test
def test_dedicated_placement_requires_a_node_list():
    config = OmegaConf.create({"placement": "dedicated", "dedicated_node_ips": []})

    with pytest.raises(ValueError, match="dedicated_node_ips"):
        resolve_placement_node_ips(config, ["10.0.0.1"])


@pytest.mark.cpu_test
def test_dedicated_placement_rejects_nodes_absent_from_the_cluster():
    config = OmegaConf.create({"placement": "dedicated", "dedicated_node_ips": ["10.9.9.9"]})

    with pytest.raises(ValueError, match="not alive"):
        resolve_placement_node_ips(config, ["10.0.0.1"])


@pytest.mark.cpu_test
def test_unknown_placement_policy_raises():
    config = OmegaConf.create({"placement": "sideways", "dedicated_node_ips": []})

    with pytest.raises(ValueError, match="Unknown placement"):
        resolve_placement_node_ips(config, ["10.0.0.1"])
