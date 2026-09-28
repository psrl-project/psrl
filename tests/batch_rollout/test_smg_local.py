"""Tests for the local SMG serving backend.

This backend runs the RL rollout stack without a parameter server. Two properties
make that safe, and both are easy to break silently:

- SMG must accept an empty `ps_manager_addr`, because every PS call site in its
  worker selector is guarded and version filtering falls back to the worker label.
- A PS-free engine must load real checkpoint weights. RL forces
  `load_format=dummy` and pulls weights over NIXL, so an engine that inherits that
  default without a PS serves noise and every reward is meaningless.
"""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from psrl.batch_rollout.serving.base import build_serving_backend
from psrl.workers.gen.smg_adapter import build_rollout_router_args


def _config(**serving):
    return OmegaConf.create(
        {
            "batch_rollout": {
                "output_dir": "/tmp/batch_rollout_smg_test",
                "serving": {
                    "name": "smg_local",
                    "n_instances": 1,
                    "ngpus_per_node_per_instance": 2,
                    "nnodes_per_instance": 1,
                    "model_name": "",
                    **serving,
                },
            },
            "gen_actor_rollout_ref": {
                "model": {"path": "/models/qwen"},
                "rollout": {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "prompt_length": 2048,
                    "response_length": 30720,
                    "max_model_len": 32768,
                },
            },
            "psrl": {"logging_path": "/tmp/batch_rollout_smg_test/logs"},
        }
    )


@pytest.mark.cpu_test
def test_backend_is_selectable_by_name():
    """`serving=smg_local` must resolve, or the group is dead config."""
    from psrl.batch_rollout.serving.smg_local import SMGLocalServingBackend

    backend = build_serving_backend(_config())

    assert isinstance(backend, SMGLocalServingBackend)


@pytest.mark.cpu_test
def test_smg_accepts_an_empty_parameter_server_address():
    """The whole PS-free design rests on SMG tolerating this.

    `psrl` selection stays on, so version filtering, group-sticky routing, and the
    admission gate still apply. They fall back to each worker's `weight_version`
    label instead of asking a PS that is not there.
    """
    cfg = OmegaConf.create(
        {
            "psrl": {
                "rollout_coordination": {"routing_strategy": {"method": "request_num_balance"}},
                "rollout_gateway": {"trajectory_id_strategy": "manual"},
            },
            "data": {"max_prompt_length": 2048},
        }
    )

    args = build_rollout_router_args(cfg, "127.0.0.1", 8100, "")

    assert args.psrl_ps_manager_addr == ""
    assert args.worker_selection_strategy == "psrl"
    # TITO is what makes a token-level dump possible, and is independent of the PS.
    assert args.enable_tito is True
    # The psrl selector requires the routing loop.
    assert args.enable_routing_loop is True


@pytest.mark.cpu_test
def test_replica_refuses_a_parameter_server_handle():
    """A PS handle would make the engine wait for a weight push that never comes."""
    from psrl.batch_rollout.serving.smg_local import BatchRolloutReplica

    with pytest.raises(ValueError, match="ps_manager_handle=None"):
        BatchRolloutReplica(
            replica_rank=0,
            local_replica_rank=0,
            psrl_config=SimpleNamespace(),
            config=SimpleNamespace(),
            model_config=SimpleNamespace(),
            gen_interface=SimpleNamespace(ps_manager_handle=object()),
        )


@pytest.mark.cpu_test
def test_world_size_mismatch_is_rejected_before_claiming_gpus():
    """tp * pp * dp must match the pool shape, or the group hangs on placement."""
    backend = build_serving_backend(_config(ngpus_per_node_per_instance=4))

    with pytest.raises(ValueError, match="world size"):
        backend.start()


@pytest.mark.cpu_test
def test_a_ps_free_engine_loads_real_weights():
    """RL forces `dummy` and pulls over NIXL. Without a PS that serves noise.

    Verified on the real `PSRL_vLLMHttpServer.__init__` rather than a copy of its
    logic, because the whole point is that this specific branch is what stops a
    collection run from grading incoherent output.
    """
    import inspect

    from psrl.workers.gen import vllm_async_server

    source = inspect.getsource(vllm_async_server.PSRL_vLLMHttpServer.__init__)

    assert "gen_interface.ps_manager_handle is not None" in source, (
        "load_format must be chosen from whether a PS is present."
    )
    dummy_line = source.index('self.config.load_format = "dummy"')
    auto_line = source.index('self.config.load_format = "auto"')
    # `dummy` under a PS, `auto` without one.
    assert dummy_line < auto_line


@pytest.mark.cpu_test
def test_token_capture_is_advertised_only_by_this_backend():
    """`dump_tokens` is meaningful only where TITO records the real stream."""
    from psrl.batch_rollout.serving.openai_api import OpenAIServingBackend
    from psrl.batch_rollout.serving.smg_local import SMGLocalServingBackend

    # Read the declared value without starting either stack.
    assert "supports_token_capture=True" in __import__("inspect").getsource(SMGLocalServingBackend.start)
    assert "supports_token_capture=False" in __import__("inspect").getsource(OpenAIServingBackend.start)


@pytest.mark.cpu_test
def test_rl_only_config_groups_stay_absent():
    """A rollout-only consumer must not have to declare RL subsystems it never uses.

    TMS, NIXL, LMCache, and the parameter server each own a config group. The
    engine reads them through optional accessors, so their absence is the point:
    if this list shrinks, someone made a group mandatory again and every future
    non-RL consumer inherits it.
    """
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "psrl/batch_rollout/config")
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        cfg = compose(config_name="batch_rollout", overrides=["serving=smg_local"])

    for group in ("tms", "nixl", "lmcache", "ps_mode", "ps_manager_ip"):
        assert group not in cfg.psrl, (
            f"psrl.{group} is declared again. It belongs to an RL subsystem collection "
            "does not run, so the engine should read it optionally instead."
        )

    # Resolution is still the real check for whatever remains.
    assert OmegaConf.to_container(cfg.psrl, resolve=True)


@pytest.mark.cpu_test
def test_absent_status_endpoint_disables_reporting_rather_than_crashing():
    """`GenInterface` documents None/"" as no reporting, so the code must honor it.

    It previously did `ZMQPushQueue(endpoint or "")`, and zmq rejects an empty
    address with `ZMQError: Invalid argument`. Any consumer without a coordinator
    to receive stats hits that, which is every consumer outside RL.
    """
    import inspect

    from psrl.workers.gen import vllm_async_server

    source = inspect.getsource(vllm_async_server.PSRL_vLLMHttpServer.run_server)

    assert "self.gen_interface.status_endpoint" in source
    assert 'status_endpoint or ""' not in source, (
        "An empty endpoint must skip reporting, not be passed to ZMQPushQueue."
    )


@pytest.mark.cpu_test
def test_tms_is_read_optionally_at_replica_startup():
    """The replica reads TMS while building engine env vars, before any PS exists."""
    import inspect

    from psrl.workers.gen import vllm_async_server

    source = inspect.getsource(vllm_async_server.PSRL_vLLMReplica.launch_servers)

    assert "psrl_config.tms." not in source, "Direct attribute access forces every consumer to declare psrl.tms."


@pytest.mark.cpu_test
def test_trajectory_chaining_is_automatic():
    """`manual` silently reduces a multi-turn episode to its final turn.

    SMG chains a session's turns into one trajectory by prefix under `auto`. Under
    `manual` the agent is expected to send a trajectory header, which an external
    harness does not, so every turn misses the prefix lookup and TITO keeps only
    the last one. The episode still grades correctly, which is what makes the loss
    easy to miss: `num_turns` reads 1 and the earlier turns are simply absent.
    """
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "psrl/batch_rollout/config")
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        cfg = compose(config_name="batch_rollout", overrides=["serving=smg_local"])

    assert cfg.psrl.rollout_gateway.trajectory_id_strategy == "auto"


@pytest.mark.cpu_test
def test_optional_accessor_matches_direct_attribute_access():
    """Making a group optional must not change what RL reads when it is present.

    The RL path relies on TMS being read exactly as before. This pins both halves:
    the same value when the group exists, and the default when it does not.
    """
    from psrl.workers.gen.vllm_async_server import PSRL_vLLMHttpServer

    server = object.__new__(PSRL_vLLMHttpServer)

    server.psrl_config = OmegaConf.create({"tms": {"range": "all", "enable_cuda_graph": True}})
    assert server._psrl_opt("tms", "range") == "all"
    assert server._psrl_opt("tms", "enable_cuda_graph", False) is True

    # The RL default is a null range, which must not be confused with absence.
    server.psrl_config = OmegaConf.create({"tms": {"range": None, "enable_cuda_graph": False}})
    assert server._psrl_opt("tms", "range") is None

    server.psrl_config = OmegaConf.create({})
    assert server._psrl_opt("tms", "range") is None
    assert server._psrl_opt("tms", "enable_cuda_graph", False) is False


@pytest.mark.cpu_test
def test_rl_config_still_declares_every_subsystem():
    """The RL root must keep its groups. Only non-RL consumers may omit them."""
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "psrl/trainer/config")
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        cfg = compose(config_name="ppo_trainer")

    for group in ("tms", "nixl", "lmcache", "ps_mode", "ps_manager_ip", "staleness"):
        assert group in cfg.psrl, f"RL still needs psrl.{group}."
