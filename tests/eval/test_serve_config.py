"""Tests for Hydra config composition in `psrl.eval.serve`.

These guard the boundary itself: composed config in, plain `FleetSpec` out. A
break here would only surface at launch time on a GPU host, which is the
expensive place to find it.
"""

import pytest
from hydra import compose, initialize_config_module
from psrl.eval.serve import build_fleet_spec

pytestmark = pytest.mark.cpu_test


def _compose(*overrides):
    with initialize_config_module(config_module="psrl.eval.config", version_base=None):
        return compose(config_name="serve", overrides=["output_dir=/tmp/test", *overrides])


class TestDefaults:
    def test_default_is_a_four_replica_fleet(self):
        """The 8-GPU host default: 4 x TP=2."""
        cfg = _compose()
        assert cfg.topology.kind == "fleet"
        spec = build_fleet_spec(cfg)
        assert (spec.replicas, spec.tp) == (4, 2)

    def test_default_server_is_qwen35_9b(self):
        """32768 is measured, not chosen: one TP=2 replica holds ~140k KV tokens
        (num_gpu_blocks=8796 x 16), so a larger window would serialize the fleet."""
        spec = build_fleet_spec(_compose())
        assert spec.served_model_name == "qwen35-9b"
        assert spec.max_model_len == 32768

    def test_hydra_run_dir_is_pinned_to_output_dir(self):
        """Otherwise Hydra scatters outputs/<date>/<time>/ in the launch directory.

        `compose()` strips the hydra node, so this asserts on the YAML source.
        """
        from pathlib import Path

        import psrl.eval.config

        text = (Path(psrl.eval.config.__file__).parent / "serve.yaml").read_text()
        assert "dir: ${output_dir}" in text
        assert "chdir: false" in text


class TestServerGroup:
    def test_switching_checkpoint_is_one_override(self):
        """The point of the server group: checkpoint, name, and window move together."""
        spec = build_fleet_spec(_compose("server=qwen3_8b"))
        assert spec.served_model_name == "qwen3-8b"
        assert spec.max_model_len == 40960

    @pytest.mark.parametrize("preset", ["qwen3_8b", "qwen35_9b"])
    def test_no_preset_enables_tool_call_extraction(self, preset):
        """Tool-call extraction would strip the agents' text protocol."""
        assert build_fleet_spec(_compose(f"server={preset}")).tool_call_parser == ""


class TestTopologyGroup:
    def test_single_is_one_replica(self):
        spec = build_fleet_spec(_compose("topology=single"))
        assert spec.replicas == 1

    def test_single_uses_port_not_base_port(self):
        """The spec normalizes `single.yaml`'s `port` key to `base_port`."""
        spec = build_fleet_spec(_compose("topology=single", "topology.port=8005"))
        assert spec.base_port == 8005

    def test_fleet_replicas_and_tp_are_overridable(self):
        """Overrides address the group name `topology`, never the file name `fleet`."""
        spec = build_fleet_spec(_compose("topology=fleet", "topology.replicas=2", "topology.tp=4"))
        assert (spec.replicas, spec.tp) == (2, 4)

    def test_file_name_is_not_a_valid_override_key(self):
        """Guards the docstring: `fleet.replicas=` is a mistake Hydra rejects."""
        with pytest.raises(Exception, match="fleet"):
            _compose("topology=fleet", "fleet.replicas=2")

    def test_fleet_requires_every_replica(self):
        """On one host a missing replica means misconfiguration, not flakiness."""
        assert build_fleet_spec(_compose("topology=fleet")).min_healthy_frac == 1.0

    def test_multinode_tolerates_node_loss(self):
        """Multinode serving degrades rather than aborts when a node is unresponsive."""
        assert build_fleet_spec(_compose("topology=multinode")).min_healthy_frac == 0.5

    def test_multinode_binds_all_interfaces(self):
        """A loopback bind would be unreachable from the coordinator."""
        assert build_fleet_spec(_compose("topology=multinode")).host == "0.0.0.0"

    def test_multinode_defaults_to_a_repo_hosts_file(self):
        cfg = _compose("topology=multinode")
        assert "hosts/" in cfg.topology.hosts_file

    def test_dp_reachable_for_single_url_consumers(self):
        """mini-swe's eval forwards one OPENAI_API_BASE per host, so parallelism
        there has to go behind a single port via dp, not across replicas."""
        spec = build_fleet_spec(_compose("topology=multinode", "topology.replicas=1", "topology.dp=4"))
        assert (spec.replicas, spec.dp) == (1, 4)
        assert spec.gpus_per_replica == 8  # tp=2 * dp=4

    def test_dp_defaults_to_one_everywhere(self):
        """DP remains disabled because the patched vLLM implementation cannot support it."""
        for topology in ("single", "fleet", "multinode"):
            assert build_fleet_spec(_compose(f"topology={topology}")).dp == 1
