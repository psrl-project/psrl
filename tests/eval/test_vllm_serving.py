"""Tests for `psrl.eval.vllm_server` and `psrl.eval.vllm_fleet`.

Everything here runs without a GPU and without launching a process, which is the
payoff for keeping config composition out of these modules.
"""

import signal

import pytest
from psrl.eval.vllm_fleet import (
    FleetSpec,
    build_specs,
    load_spec_json,
    partition_gpus,
    read_endpoints,
    write_endpoints,
    write_spec_json,
)
from psrl.eval.vllm_server import Endpoint, ServerSpec, build_command, build_shell_command


class TestPartitionGpus:
    def test_four_replicas_of_two(self):
        assert partition_gpus([0, 1, 2, 3, 4, 5, 6, 7], 4, 2) == [(0, 1), (2, 3), (4, 5), (6, 7)]

    def test_two_replicas_of_four(self):
        assert partition_gpus([0, 1, 2, 3, 4, 5, 6, 7], 2, 4) == [(0, 1, 2, 3), (4, 5, 6, 7)]

    def test_blocks_are_contiguous(self):
        """Contiguous blocks keep each TP group inside one NVLink domain."""
        for block in partition_gpus(list(range(16)), 4, 4):
            assert list(block) == list(range(block[0], block[0] + 4))

    def test_respects_offset_gpu_ids(self):
        assert partition_gpus([4, 5, 6, 7], 2, 2) == [(4, 5), (6, 7)]

    @pytest.mark.parametrize(
        "gpu_ids, replicas, gpus_each",
        [([0, 1, 2], 2, 2), ([0, 1], 0, 2), ([0, 1], 2, 0), ([], 1, 1)],
    )
    def test_rejects_inconsistent_counts(self, gpu_ids, replicas, gpus_each):
        with pytest.raises(ValueError):
            partition_gpus(gpu_ids, replicas, gpus_each)


class TestBuildCommand:
    @pytest.fixture
    def spec(self):
        return ServerSpec(
            checkpoint="/models/Qwen3.5-9B",
            served_model_name="qwen35-9b",
            port=8001,
            tp=2,
            gpu_ids=(2, 3),
            max_model_len=131072,
        )

    def test_is_pure(self, spec):
        """Same input, same output: this is what makes --dry-run trustworthy."""
        assert build_command(spec) == build_command(spec)

    def test_forwards_parallelism_and_window(self, spec):
        cmd = build_command(spec)
        assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
        assert cmd[cmd.index("--max-model-len") + 1] == "131072"

    def test_gpu_pinning_is_env_not_argv(self, spec):
        """GPUs are pinned via CUDA_VISIBLE_DEVICES, never as a vLLM flag."""
        assert "2,3" not in build_command(spec)

    def test_tool_parser_disabled_by_default(self, spec):
        """Agents that parse their own text protocol must not get tool-call extraction."""
        assert "--tool-call-parser" not in build_command(spec)

    def test_tool_parser_opt_in(self):
        cmd = build_command(ServerSpec(checkpoint="/m", served_model_name="n", tool_call_parser="hermes"))
        assert "--enable-auto-tool-choice" in cmd
        assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"

    def test_omits_max_model_len_when_none(self):
        assert "--max-model-len" not in build_command(ServerSpec(checkpoint="/m", served_model_name="n"))

    def test_extra_args_forwarded_verbatim(self):
        cmd = build_command(ServerSpec(checkpoint="/m", served_model_name="n", extra=("--trust-remote-code",)))
        assert cmd[-1] == "--trust-remote-code"

    def test_dp_omitted_when_one(self):
        """A fleet of independent servers is preferred, so dp=1 is the norm."""
        assert "--data-parallel-size" not in build_command(ServerSpec(checkpoint="/m", served_model_name="n"))

    def test_dp_emitted_when_above_one(self):
        """Needed when a consumer can only be handed a single URL."""
        cmd = build_command(ServerSpec(checkpoint="/m", served_model_name="n", dp=4))
        assert cmd[cmd.index("--data-parallel-size") + 1] == "4"

    def test_async_scheduling_disabled_only_for_moe_under_dp(self):
        """vLLM v1 switches DP sync from NCCL to fragile gloo TCP for MoE models."""
        moe_dp = build_command(ServerSpec(checkpoint="/m", served_model_name="n", dp=2), True)
        assert moe_dp[moe_dp.index("--async-scheduling") + 1] == "false"
        # Not applied without DP, even if the caller asks.
        assert "--async-scheduling" not in build_command(ServerSpec(checkpoint="/m", served_model_name="n"), True)


class TestBuildShellCommand:
    @pytest.fixture
    def spec(self):
        return ServerSpec(checkpoint="/models/m", served_model_name="n", port=8000)

    def test_wraps_in_login_shell(self, spec):
        assert build_shell_command(spec, "/env/psrl.sh")[:2] == ["bash", "-lc"]

    def test_sources_env_script(self, spec):
        assert "source /env/psrl.sh" in build_shell_command(spec, "/env/psrl.sh")[2]

    def test_execs_so_pid_is_vllms_own(self, spec):
        """Without exec the handle would track the wrapping shell, so SIGTERM
        would kill the shell and orphan the server."""
        assert "&& exec python -m vllm" in build_shell_command(spec, "/env/psrl.sh")[2]

    def test_no_env_script_still_execs(self, spec):
        inner = build_shell_command(spec)[2]
        assert inner.startswith("exec python -m vllm")
        assert "source" not in inner

    def test_quotes_paths_with_spaces(self, spec):
        assert "'/has space/env.sh'" in build_shell_command(spec, "/has space/env.sh")[2]


class TestServerSpec:
    def test_url_avoids_bind_all_address(self):
        """0.0.0.0 binds every interface but is not curlable on some kernels."""
        spec = ServerSpec(checkpoint="/m", served_model_name="n", host="0.0.0.0", port=8001)
        assert spec.url == "http://127.0.0.1:8001/v1"

    def test_url_keeps_explicit_host(self):
        spec = ServerSpec(checkpoint="/m", served_model_name="n", host="28.49.195.154", port=8000)
        assert spec.url == "http://28.49.195.154:8000/v1"

    def test_n_gpus_is_tp_times_pp(self):
        assert ServerSpec(checkpoint="/m", served_model_name="n", tp=2, pp=2).n_gpus == 4

    def test_n_gpus_includes_dp(self):
        """DP replicas each need their own GPUs, so they count toward the footprint."""
        assert ServerSpec(checkpoint="/m", served_model_name="n", tp=2, dp=4).n_gpus == 8

    def test_rejects_gpu_count_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="Expected 2 GPU"):
            ServerSpec(checkpoint=str(tmp_path), served_model_name="n", tp=2, gpu_ids=(0, 1, 2)).validate()

    def test_rejects_bad_memory_utilization(self, tmp_path):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            ServerSpec(checkpoint=str(tmp_path), served_model_name="n", gpu_memory_utilization=1.5).validate()

    def test_rejects_missing_checkpoint(self):
        with pytest.raises(FileNotFoundError):
            ServerSpec(checkpoint="/nonexistent-checkpoint-xyz", served_model_name="n").validate()


class TestServerHandleTeardown:
    """Teardown must signal the process group, not just the launched pid.

    `launch` uses start_new_session=True, so each server leads its own process group
    and vLLM's VLLM::Worker_TPn children join it. Signalling the leader alone leaves
    workers alive holding ~90 GiB per GPU -- observed in practice.
    """

    def _handle(self, pid, alive):
        from pathlib import Path

        from psrl.eval.vllm_server import ServerHandle

        class FakeProcess:
            returncode = None

            def __init__(self):
                self.waited = False

            def poll(self):
                return None if alive else 0

            def wait(self, timeout=None):
                self.waited = True
                return 0

        spec = ServerSpec(checkpoint="/m", served_model_name="n", port=8000)
        handle = ServerHandle(spec=spec, process=FakeProcess(), log_file=Path("/tmp/x.log"))
        object.__setattr__(handle.process, "pid", pid)
        return handle

    def test_signals_the_group_when_alive(self, monkeypatch):
        sent = []
        monkeypatch.setattr("psrl.eval.vllm_server.os.getpgid", lambda pid: 4242)
        monkeypatch.setattr("psrl.eval.vllm_server.os.killpg", lambda gid, sig: sent.append((gid, sig)))
        self._handle(1234, alive=True).terminate()
        assert sent == [(4242, signal.SIGTERM)]

    def test_sweeps_the_group_even_when_leader_is_gone(self, monkeypatch):
        """The exact failure seen: leader reaped, workers still pinning GPUs."""
        sent = []
        monkeypatch.setattr("psrl.eval.vllm_server.os.getpgid", lambda pid: 99)
        monkeypatch.setattr("psrl.eval.vllm_server.os.killpg", lambda gid, sig: sent.append((gid, sig)))
        self._handle(1234, alive=False).terminate()
        assert sent == [(99, signal.SIGKILL)]

    def test_tolerates_an_already_reaped_group(self, monkeypatch):
        def boom(pid):
            raise ProcessLookupError

        monkeypatch.setattr("psrl.eval.vllm_server.os.getpgid", boom)
        self._handle(1234, alive=False).terminate()  # must not raise


class TestBuildSpecs:
    @pytest.fixture
    def fleet(self):
        return FleetSpec(
            checkpoint="/models/Qwen3.5-9B",
            served_model_name="qwen35-9b",
            replicas=4,
            tp=2,
            base_port=8000,
            gpu_ids=(0, 1, 2, 3, 4, 5, 6, 7),
            max_model_len=131072,
        )

    def test_ports_increment_from_base(self, fleet):
        assert [s.port for s in build_specs(fleet)] == [8000, 8001, 8002, 8003]

    def test_each_replica_gets_its_own_gpus(self, fleet):
        assert [list(s.gpu_ids) for s in build_specs(fleet)] == [[0, 1], [2, 3], [4, 5], [6, 7]]

    def test_replicas_share_one_served_name(self, fleet):
        """Clients must not need to know which replica answered."""
        assert {s.served_model_name for s in build_specs(fleet)} == {"qwen35-9b"}

    def test_window_propagates_to_every_replica(self, fleet):
        assert all(s.max_model_len == 131072 for s in build_specs(fleet))

    def test_explicit_gpu_ids_must_match_exactly(self):
        """Explicit GPU lists must be consumed completely to expose configuration mistakes."""
        fleet = FleetSpec(checkpoint="/m", served_model_name="n", replicas=1, tp=1, gpu_ids=(0, 1, 2, 3))
        with pytest.raises(ValueError, match="Expected 1 GPU"):
            build_specs(fleet)

    def test_discovered_gpus_are_an_upper_bound(self, monkeypatch):
        """A 1-replica probe on an 8-GPU host should take what it needs."""
        monkeypatch.setattr("psrl.eval.vllm_fleet.discover_local_gpus", lambda: [0, 1, 2, 3, 4, 5, 6, 7])
        specs = build_specs(FleetSpec(checkpoint="/m", served_model_name="n", replicas=1, tp=2))
        assert [list(s.gpu_ids) for s in specs] == [[0, 1]]

    def test_rejects_too_few_discovered_gpus(self, monkeypatch):
        monkeypatch.setattr("psrl.eval.vllm_fleet.discover_local_gpus", lambda: [0, 1])
        with pytest.raises(ValueError, match="needs 8 GPU"):
            build_specs(FleetSpec(checkpoint="/m", served_model_name="n", replicas=4, tp=2))

    def test_rejects_when_no_gpus_visible(self, monkeypatch):
        monkeypatch.setattr("psrl.eval.vllm_fleet.discover_local_gpus", lambda: [])
        with pytest.raises(ValueError, match="No GPUs available"):
            build_specs(FleetSpec(checkpoint="/m", served_model_name="n"))


class TestEndpointsFile:
    @pytest.fixture
    def endpoints(self):
        return [
            Endpoint(url="http://h:8000/v1", host="h", gpu_ids=(0, 1), pid=1, healthy=True),
            Endpoint(url="http://h:8001/v1", host="h", gpu_ids=(2, 3), pid=2, healthy=False),
        ]

    def test_unhealthy_excluded_by_default(self, tmp_path, endpoints):
        """An eval must never dispatch work to a replica that never came up."""
        name, urls = read_endpoints(write_endpoints(tmp_path / "e.json", "qwen35-9b", endpoints))
        assert name == "qwen35-9b"
        assert urls == ["http://h:8000/v1"]

    def test_healthy_only_false_keeps_all(self, tmp_path, endpoints):
        _, urls = read_endpoints(write_endpoints(tmp_path / "a.json", "n", endpoints, healthy_only=False))
        assert len(urls) == 2

    def test_gpu_ids_survive_as_json_list(self, tmp_path, endpoints):
        import json

        payload = json.loads(write_endpoints(tmp_path / "e.json", "n", endpoints).read_text())
        assert payload["endpoints"][0]["gpu_ids"] == [0, 1]


class TestFleetSpecSerialization:
    def test_round_trip_preserves_everything(self, tmp_path):
        """The remote must reconstruct byte-identical serving config."""
        fleet = FleetSpec(
            checkpoint="/models/Qwen3.5-9B",
            served_model_name="qwen35-9b",
            replicas=4,
            tp=2,
            max_model_len=131072,
            extra=("--trust-remote-code",),
            gpu_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        )
        assert load_spec_json(write_spec_json(tmp_path / "f.json", fleet)) == fleet

    def test_sequence_fields_restored_as_tuples(self, tmp_path):
        fleet = FleetSpec(checkpoint="/m", served_model_name="n", gpu_ids=(0,), extra=("--x",))
        back = load_spec_json(write_spec_json(tmp_path / "f.json", fleet))
        assert isinstance(back.gpu_ids, tuple)
        assert isinstance(back.extra, tuple)
