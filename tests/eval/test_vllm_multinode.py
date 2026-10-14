"""Tests for `psrl.eval.vllm_multinode` that need no remote host.

The ssh path itself cannot be covered here. What is covered is everything that
decides *what* gets sent: host parsing, the remote command, and the URL rewrite
that makes a remote's loopback endpoint addressable from the coordinator.
"""

import pytest
from psrl.eval.vllm_fleet import FleetSpec
from psrl.eval.vllm_multinode import (
    MultinodeSpec,
    _rehost,
    build_remote_command,
    launch_multinode,
    read_hosts_file,
)

REPO_HOSTS_FILE = "/apdcephfs_zwfy10/share_303541817/lhy/hosts/32GPUs"


@pytest.fixture
def fleet():
    return FleetSpec(
        checkpoint="/models/Qwen3.5-9B",
        served_model_name="qwen35-9b",
        replicas=4,
        tp=2,
        max_model_len=131072,
    )


class TestReadHostsFile:
    def test_strips_comments_and_blanks(self, tmp_path):
        path = tmp_path / "hosts"
        path.write_text("# header\n29.162.247.148\n\n  28.49.195.154  \n# trailing\n")
        assert read_hosts_file(path) == ["29.162.247.148", "28.49.195.154"]

    def test_reads_the_repo_convention(self):
        """Multinode deployment consumes the existing `hosts/<N>GPUs` convention unchanged."""
        hosts = read_hosts_file(REPO_HOSTS_FILE)
        assert "29.162.247.148" in hosts
        assert "28.49.195.154" in hosts

    def test_rejects_a_file_with_no_hosts(self, tmp_path):
        path = tmp_path / "empty"
        path.write_text("# only comments\n\n")
        with pytest.raises(ValueError, match="No hosts"):
            read_hosts_file(path)

    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_hosts_file(tmp_path / "nope")


class TestBuildRemoteCommand:
    def test_invokes_the_fleet_module(self):
        cmd = build_remote_command("/shared/fleet.json", "/shared/hosts/n1")
        assert cmd.startswith("python -m psrl.eval.vllm_fleet")
        assert "--spec-json /shared/fleet.json" in cmd
        assert "--output-dir /shared/hosts/n1" in cmd

    def test_sources_env_script_first(self):
        cmd = build_remote_command("/f.json", "/o", "/env/psrl.sh")
        assert cmd.startswith("source /env/psrl.sh && python -m psrl.eval.vllm_fleet")

    def test_only_paths_are_interpolated(self):
        """Passing a spec file instead of ~17 flags is what removes the second
        layer of shell quoting the bash implementation needed."""
        cmd = build_remote_command("/has space/fleet.json", "/out dir")
        assert "'/has space/fleet.json'" in cmd
        assert "'/out dir'" in cmd


class TestRehost:
    def test_rewrites_loopback_to_the_owning_host(self):
        """A remote binds 0.0.0.0 and reports 127.0.0.1, which is useless to the
        coordinator on another machine."""
        assert _rehost("http://127.0.0.1:8001/v1", "28.49.195.154") == "http://28.49.195.154:8001/v1"

    def test_preserves_port_and_path(self):
        assert _rehost("http://127.0.0.1:8003/v1", "h") == "http://h:8003/v1"


class TestCapacityAccounting:
    def test_endpoints_are_hosts_times_replicas(self, fleet):
        assert MultinodeSpec(hosts=["h1", "h2"], fleet=fleet).n_expected == 8

    def test_scales_to_sixteen_nodes(self, fleet):
        """128 GPUs / 16 nodes is the documented scale target."""
        assert MultinodeSpec(hosts=[f"h{i}" for i in range(16)], fleet=fleet).n_expected == 64

    def test_default_quorum_tolerates_node_loss(self, fleet):
        """A long multinode evaluation must tolerate an unresponsive node."""
        assert MultinodeSpec(hosts=["h1"], fleet=fleet).min_healthy_frac == 0.5


class TestDryRun:
    def test_does_not_connect_and_yields_no_endpoints(self, tmp_path, fleet):
        spec = MultinodeSpec(hosts=["10.0.0.1", "10.0.0.2"], fleet=fleet)
        result = launch_multinode(spec, output_dir=tmp_path, dry_run=True)
        assert result.endpoints == []
        assert result.failed_hosts == []

    def test_still_writes_the_shared_spec(self, tmp_path, fleet):
        launch_multinode(MultinodeSpec(hosts=["10.0.0.1"], fleet=fleet), output_dir=tmp_path, dry_run=True)
        assert (tmp_path / "fleet.json").is_file()
