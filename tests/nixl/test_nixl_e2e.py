import asyncio
import dataclasses
import math
import os
import time

# Must precede anything that reaches megatron.bridge. That chain pulls in modelopt ->
# tilelang, which loads its own `libcudart_stub.so` into the process; a later
# `flashinfer.comm` import then resolves cudaDeviceReset against the stub and dies with
# "undefined symbol". vLLM's FusedMoE -> all2all_utils -> nixl_ep path imports
# flashinfer, so the whole harness fails at import time unless the real CUDA runtime is
# bound first. Importing it here is the fix; order, not presence, is what matters.
try:
    import flashinfer.comm  # noqa: F401
except Exception:  # pragma: no cover - flashinfer is optional for CPU-only runs
    pass

import hydra
import ray
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from omegaconf import DictConfig, OmegaConf
from psrl.utils.common.worker_naming import gen_client_name, train_client_name
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.fsdp_converter import convert_fsdp_inplace
from psrl.utils.converter.megatron_converter import convert_megatron_inplace
from psrl.utils.converter.vllm_converter import convert_vllm_inplace
from psrl.utils.quantization import resolve_quantization_plan
from psrl.utils.nixl import (
    NIXL_META_SERVER_NAME,
    NIXLClientType,
    NIXLMetaServer,
    NIXLStorageClient,
)
from psrl.workers.ps import (
    PSClassWithInitArgs,
    PSResourcePool,
    PSResourceSpec,
    PSStoragePlan,
    PSStorageWorker,
    PSWorkerGroup,
)
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

FP8_PLAN = resolve_quantization_plan({"scheme": "fp8"})

try:
    from tests.nixl.coord_judge import (
        check_by_layout,
        check_fp8_exact,
        fill_by_layout,
        fill_fp8_exact,
    )
except ModuleNotFoundError:
    # Loaded by file path rather than by package name. Run as a script
    # (`python tests/nixl/test_nixl_e2e.py`), sys.path[0] is this file's own directory,
    # so `tests` resolves to whatever else is on the path -- adding the repo root does
    # not help, because the shadowing entry comes first. Loading the sibling module
    # directly sidesteps package resolution entirely and works under pytest too, since
    # the try branch wins there.
    import importlib.util
    from pathlib import Path

    _spec = importlib.util.spec_from_file_location(
        "psrl_e2e_coord_judge", Path(__file__).resolve().parent / "coord_judge.py"
    )
    _coord_judge = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_coord_judge)
    fill_by_layout = _coord_judge.fill_by_layout
    check_by_layout = _coord_judge.check_by_layout
    fill_fp8_exact = _coord_judge.fill_fp8_exact
    check_fp8_exact = _coord_judge.check_fp8_exact
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import AutoConfig, AutoModelForCausalLM
from verl.utils.fs import copy_to_local
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.engine.megatron.utils import set_random_seed

NUM_GPU_PER_NODE = 8

#: Repository root, shipped to Ray actors so they import this tree rather than the
#: checkout the cluster environment puts on the default path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fully_shard_transformer(model, mesh) -> None:
    """Apply FSDP2 ``fully_shard`` bottom-up, mirroring verl's wrap policy.

    Each transformer block is sharded as its own unit before the root, which is what
    produces per-parameter DTensors rather than one flat buffer. The order matters:
    `fully_shard` must see children already sharded when it reaches the root.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        for block in layers:
            fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype:
    if dtype_name is None:
        return torch.bfloat16
    normalized = str(dtype_name).lower()
    if normalized in ("float32", "fp32"):
        return torch.float32
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    if normalized in ("float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype {dtype_name!r}.")


def validate_parallel_config(cfg: DictConfig) -> None:
    num_train = cfg.test.num_train
    num_gen = cfg.test.num_gen
    train_engine_type = cfg.test.train_engine_type
    gen_tp = cfg.test.gen.tensor_parallel_size
    gen_pp = cfg.test.gen.pipeline_parallel_size
    gen_ep = cfg.test.gen.get("expert_parallel_size", 1)
    gen_dp = cfg.test.gen.get("data_parallel_size", None)

    assert num_gen % (gen_tp * gen_pp) == 0, (
        f"Gen TP*PP does not divide num_gen. num_gen={num_gen!r}, product={gen_tp * gen_pp!r}."
    )
    computed_dp = num_gen // (gen_tp * gen_pp)
    if gen_dp is not None and gen_dp > 1:
        assert computed_dp == gen_dp, (
            f"gen data_parallel_size={gen_dp} conflicts with computed DP="
            f"{computed_dp} (num_gen={num_gen}, TP={gen_tp}, PP={gen_pp})."
        )
    if gen_ep > 1:
        # vLLM's expert parallel group spans every DP and TP rank of the engine, as in production.
        assert gen_ep == computed_dp * gen_tp, (
            f"vLLM EP spans DP x TP. EP={gen_ep!r}, DP={computed_dp!r}, TP={gen_tp!r}."
        )

    if train_engine_type == "fsdp_hybrid":
        ddp_size = cfg.test.fsdp_hybrid.ddp_size
        fsdp_size = cfg.test.fsdp_hybrid.fsdp_size
        assert num_train % (ddp_size * fsdp_size) == 0, (
            f"HSDP DDP*FSDP does not divide num_train. num_train={num_train!r}, product={ddp_size * fsdp_size!r}."
        )
    elif train_engine_type in ("fsdp", "fsdp2"):
        # FSDP2 shards over the whole train world, so num_train needs no divisibility
        # constraint -- unlike Megatron, whose TP x PP x CP must divide it. The gen-side
        # DP check above already covers this backend, and applies it the same way it does
        # for the others: only when `data_parallel_size > 1`, since 1 means "derived".
        # Asserting unconditionally here rejected the inherited `data_parallel_size: 1`,
        # which is this file's convention for unset rather than a claim of DP=1.
        pass
    elif train_engine_type == "megatron":
        megatron = cfg.test.megatron
        train_tp = megatron.tensor_model_parallel_size
        train_pp = megatron.pipeline_model_parallel_size
        train_cp = megatron.get("context_parallel_size", 1)
        model_parallel_size = train_tp * train_pp * train_cp
        assert num_train % model_parallel_size == 0, (
            f"Megatron TP*PP*CP does not divide num_train. num_train={num_train!r}, product={model_parallel_size!r}."
        )


def make_dual_print(log_path, prefix=None):
    with open(log_path, "w") as _:
        pass

    def dual_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if prefix:
            msg = f"[{prefix}] {msg}"
        print(msg, **kwargs)
        with open(log_path, "a") as f:
            print(msg, file=f, **kwargs)

    return dual_print


@ray.remote
class GlobalStore:
    def __init__(self):
        self.train_master_ip = None
        self.gen_master_ip = None
        self.train_master_ip_event = asyncio.Event()
        self.gen_master_ip_event = asyncio.Event()

    async def set_train_master_ip(self, train_master_ip):
        self.train_master_ip = train_master_ip
        self.train_master_ip_event.set()

    async def set_gen_master_ip(self, gen_master_ip):
        self.gen_master_ip = gen_master_ip
        self.gen_master_ip_event.set()

    async def get_train_master_ip(self):
        if self.train_master_ip is None:
            await self.train_master_ip_event.wait()
        return self.train_master_ip

    async def get_gen_master_ip(self):
        if self.gen_master_ip is None:
            await self.gen_master_ip_event.wait()
        return self.gen_master_ip


@ray.remote
class MetaServerActor:
    def __init__(self, server_name, psrl_config, expected_agents, log_dir):
        self.server = NIXLMetaServer(server_name, psrl_config.nixl)
        self.expected_agents = expected_agents
        self.client_name = server_name
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.client_name}.log")
        self.print = make_dual_print(log_path, prefix=self.client_name)

    def init_finished(self):
        self.print(f"Server init finished on ip {os.environ.get('LOCAL_IP')}")
        return True

    def protocol(self):
        self.print("step1: wait_for_client_shardings")
        self.server.wait_for_client_shardings(self.expected_agents)
        # self.print(f"client_sharding_dicts: {self.server.client_sharding_dicts}")
        self.print("step2: make_unified_sharding")
        self.server.make_unified_sharding()
        # self.print(f"client_unified_sharding_dicts: {self.server.client_unified_sharding_dicts}")
        self.print("step3: notify_all_client_shardings")
        self.server.notify_all_client_shardings()
        self.print("step4: wait_for_client_infos")
        self.server.wait_for_client_infos(self.expected_agents)
        # self.print(f"client_infos: {self.server.client_infos}")
        self.print("step5: make_comm_plan")
        self.server.make_comm_plan()
        self.print("step6: notify_all_client_infos_and_comm_plan")
        self.server.notify_all_client_infos_and_comm_plan()
        self.print("step7: wait_for_client_temp_mappings")
        self.server.wait_for_client_temp_mappings(self.expected_agents)
        self.print("step8: notify_all_client_temp_mappings")
        self.server.notify_all_client_temp_mappings()
        self.print("protocol done.")
        return self.server.client_unified_sharding_dicts

    def republish(self, n_agents, dst_agent_names, client_names, timeout=None):
        """Collect re-registered infos and rebroadcast them, as a colocate switch does.

        Returns what the server saw, because the failure this can have is silent from the
        driver's side: a sender that never reported and a receiver that never heard both
        present as one timeout, and which one it was decides whether the bug is in the
        switch or in the test's own fan-out.
        """
        print(f"[SERVER] republish: waiting for {n_agents} updated client infos", flush=True)
        self.server.wait_for_update_infos(n_agents, timeout=timeout)
        reached = {
            name: self.server.client_infos[name].agent_name
            for name in client_names
            if name in self.server.client_infos
        }
        missing = [name for name in client_names if name not in reached]
        print(f"[SERVER] republish: reporting senders {sorted(reached)}, missing {missing}", flush=True)
        errors = {}
        for dst in dst_agent_names:
            try:
                self.server.broadcast_update_client_infos([dst], client_names)
            except Exception as exc:  # reported rather than masking the senders' state
                errors[dst] = repr(exc)
        print(f"[SERVER] republish: broadcast errors {errors}", flush=True)
        return {"reached": reached, "missing": missing, "errors": errors}

    def shutdown(self):
        self.server.shutdown()


@ray.remote(num_cpus=1)
class TrainClientActor:
    def __init__(
        self,
        global_store,
        engine_type,
        rank,
        world_size,
        server_name,
        psrl_config,
        backend,
        torch_port,
        log_dir,
        ps_for_push_worker_handles,
        model_path,
        model_config,
        fsdp_hybrid_config=None,
        megatron_config=None,
    ):
        # `fsdp2` is the name the config matrix and `psrl.worker` use; `fsdp` is this
        # harness's older spelling for the same thing, since the branch below calls
        # `fully_shard` either way and FSDP1 is rejected outright by config validation
        # (flat_param cannot give the in-place representation NIXL registration needs).
        # Accepting both keeps one vocabulary from having to be renamed across the matrix.
        assert engine_type in ["fsdp", "fsdp2", "fsdp_hybrid", "megatron"], (
            f"Unsupported engine type: {engine_type!r}."
        )
        if rank == 0:
            train_master_ip = ray.util.get_node_ip_address()
            ray.get(global_store.set_train_master_ip.remote(train_master_ip))
        else:
            train_master_ip = ray.get(global_store.get_train_master_ip.remote())
        os.environ["MASTER_ADDR"] = train_master_ip
        os.environ["MASTER_PORT"] = str(torch_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        self.engine_type = engine_type
        self.rank = rank
        self.world_size = world_size
        self.client_name = train_client_name(rank)
        self.model_config = model_config
        self.trust_remote_code = False
        self.train_dtype = resolve_torch_dtype(model_config.get("train_dtype", "bfloat16"))
        self.gen_dtype = resolve_torch_dtype(model_config.get("gen_dtype", "bfloat16"))
        self.fp8_block = (128, 128) if model_config.get("gen_quantization") == "fp8" else None
        self.fp8_keys: list[str] = []
        # Megatron FP8 training: TE keeps linear weights as per tensor scaled FP8 (`fp8_param`).
        self.train_fp8 = model_config.get("train_fp8", None)
        assert self.train_fp8 is None or (engine_type == "megatron" and self.fp8_block is not None), (
            "train_fp8 needs a Megatron trainer and an FP8 rollout (FSDP2 has no FP8 parameters)."
        )
        self.fsdp_hybrid_config = fsdp_hybrid_config
        self.megatron_config = megatron_config
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.client_name}.log")
        self.print = make_dual_print(log_path, prefix=self.client_name)
        self.print(f"train_master_ip: {train_master_ip}")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        self.print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

        # NOTE(lhy): Create the client before loading the model.
        if engine_type == "megatron":
            self._init_megatron_parallel(megatron_config)
        self.client = NIXLStorageClient(
            client_name=self.client_name,
            server_name=server_name,
            use_gpu=True,
            client_type=NIXLClientType.PUSH_SIDE,
            nixl_config=psrl_config.nixl,
            replica_idx=0,
            worker_index=rank,
            logging_path=log_dir,
            # Mirror production: direct supply derives remote descriptors on the reader,
            # so the source's info carries geometry rather than expanded descriptors.
            pairwise_descriptors=bool(model_config.get("pairwise_descriptors", True)),
        )

        if engine_type in ("fsdp", "fsdp2", "fsdp_hybrid"):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.train_dtype,
                trust_remote_code=self.trust_remote_code,
            )
            # Set all model parameters to a deterministic source value.
            for p in model.parameters():
                p.data.fill_(1)
            torch.manual_seed(42)
            # FSDP2 (fully_shard). FSDP1 is rejected by config validation because
            # flat_param cannot guarantee the in-place parameter representation that
            # registering with NIXL requires, so the harness must not build it either.
            if engine_type in ("fsdp", "fsdp2"):
                mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
                _fully_shard_transformer(model, mesh)
                self.model = model
            elif engine_type == "fsdp_hybrid":
                # HSDP: replicate across ddp_size, shard within fsdp_size.
                ddp_size = fsdp_hybrid_config.get("ddp_size", 2)
                fsdp_size = fsdp_hybrid_config.get("fsdp_size", 4)
                assert world_size % (ddp_size * fsdp_size) == 0, (
                    f"DDP*FSDP does not divide world_size. world_size={world_size!r}, "
                    f"product={ddp_size * fsdp_size!r}."
                )
                mesh = init_device_mesh(
                    "cuda",
                    mesh_shape=(ddp_size, fsdp_size),
                    mesh_dim_names=("replicate", "shard"),
                )
                _fully_shard_transformer(model, mesh)
                self.model = model
        elif engine_type == "megatron":
            set_random_seed(42)
            self._init_megatron_model(model_path)
        self.ps_for_push_worker_handles = ps_for_push_worker_handles
        # self._log_env_info()

    def _log_env_info(self):
        for k in sorted(os.environ):
            self.print(f"{k}={os.environ[k]}")

    @property
    def _is_fsdp_family(self) -> bool:
        """One predicate for every branch that means "FSDP, any generation".

        `fsdp` and `fsdp2` are the same thing here -- both call `fully_shard`, and FSDP1 is
        rejected by config validation -- so the two names must move together. Branching on
        the raw string at each site is how the first attempt missed one: the alias was
        added to the model-construction branch but not to `protocol`, which then fell
        through every `elif` and raised `UnboundLocalError: state_dict` on the cluster
        instead of here.
        """
        return self.engine_type in ("fsdp", "fsdp2", "fsdp_hybrid")

    def _get_replica_id(self):
        if self.engine_type in ("fsdp", "fsdp2"):
            return 0
        elif self.engine_type == "fsdp_hybrid":
            return self.rank // self.fsdp_hybrid_config.get("fsdp_size", 4)
        elif self.engine_type == "megatron":
            assert mpu.is_initialized(), "Megatron parallel is not initialized"
            return mpu.get_data_parallel_rank()

    def _init_megatron_parallel(self, megatron_config):
        """Initialize Megatron parallel state."""
        virtual_pipeline_model_parallel_size = megatron_config.get("virtual_pipeline_model_parallel_size", 1)
        if virtual_pipeline_model_parallel_size == 1:
            virtual_pipeline_model_parallel_size = None
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=megatron_config.get("tensor_model_parallel_size", 4),
            pipeline_model_parallel_size=megatron_config.get("pipeline_model_parallel_size", 2),
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=megatron_config.get("context_parallel_size", 1),
            expert_model_parallel_size=megatron_config.get("expert_model_parallel_size", 1),
            expert_tensor_parallel_size=megatron_config.get("expert_tensor_parallel_size", 1),
            nccl_communicator_config_path=None,
        )
        self.print(f"[Rank {self.rank}] Megatron parallel initialized")

    def _init_megatron_model(self, model_path):
        """Initialize a Megatron-Bridge model like the current train worker."""
        local_model_path = copy_to_local(model_path)
        hf_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=self.trust_remote_code)

        dtype = PrecisionType.to_dtype(self.train_dtype)
        self.print(f"[Rank {self.rank}] Config loaded: {hf_config.model_type}")

        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_config(hf_config)
        provider = bridge.to_megatron_provider(load_weights=False)
        virtual_pipeline_model_parallel_size = self.megatron_config.get("virtual_pipeline_model_parallel_size", 1)
        if virtual_pipeline_model_parallel_size == 1:
            virtual_pipeline_model_parallel_size = None
        provider_overrides = {
            "tensor_model_parallel_size": self.megatron_config.get("tensor_model_parallel_size", 4),
            "pipeline_model_parallel_size": self.megatron_config.get("pipeline_model_parallel_size", 2),
            "expert_model_parallel_size": self.megatron_config.get("expert_model_parallel_size", 1),
            "expert_tensor_parallel_size": self.megatron_config.get("expert_tensor_parallel_size", 1),
            "virtual_pipeline_model_parallel_size": virtual_pipeline_model_parallel_size,
            "context_parallel_size": self.megatron_config.get("context_parallel_size", 1),
            "sequence_parallel": self.megatron_config.get("sequence_parallel", False),
        }
        if self.train_fp8 is not None:
            provider_overrides.update({"fp8": "hybrid", "fp8_recipe": self.train_fp8, "fp8_param": True})
        provider.apply_overrides_and_finalize(
            dtype=dtype,
            overrides=provider_overrides,
        )

        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        # No training happens here, so skip DDP: its gradient buffer is as large as the parameters
        # and at 235B scale it is what runs the card out of memory, not the weights.
        actor_module, _ = make_megatron_module(
            wrap_config=McoreModuleWrapperConfig(wrap_with_ddp=self.megatron_config.get("wrap_with_ddp", True)),
            hf_config=hf_config,
            bridge=bridge,
            provider=provider,
        )
        self.model = actor_module

        for name, param in self.model[0].named_parameters():
            if _is_te_quantized(param):
                param.quantize_(torch.ones(param.shape, dtype=dtype, device=param.device))
            else:
                param.data.fill_(1)
        quantized = sum(1 for _, p in self.model[0].named_parameters() if _is_te_quantized(p))
        self.print(f"[Rank {self.rank}] {quantized} parameters are TE quantised (train_fp8={self.train_fp8}).")
        assert self.train_fp8 is None or quantized > 0, "train_fp8 is set but TE built no FP8 parameters."
        for name, param in self.model[0].named_buffers():
            param.data.fill_(1)

        self.print(f"[Rank {self.rank}] Model initialized: {self.model}")

    def init_finished(self):
        self.print(f"Train client init finished on ip {os.environ.get('LOCAL_IP')}")
        return True

    def protocol(self, model_path):
        self.print("step0: convert_fsdp/megatron_inplace")
        if self.engine_type in ("fsdp", "fsdp2", "fsdp_hybrid"):
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=self.trust_remote_code)
            parameter_mapping = create_parameter_mapping("FSDP", model_config)
            state_dict, sharding = convert_fsdp_inplace(parameter_mapping, self.model)
        elif self.engine_type == "megatron":
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=self.trust_remote_code)
            parameter_mapping = create_parameter_mapping("Megatron", model_config)
            conversion_result = convert_megatron_inplace(parameter_mapping, self.model)
            state_dict = conversion_result.state_dict
            sharding = conversion_result.sharding_dict
        self.state_dict = state_dict
        self.sharding = sharding
        self.state_dict_keys = list(state_dict.keys())
        self.unified_sharding = None
        self.print("step1: connect_to_server")
        self.client.connect_to_server()
        self.print("step2: send_local_sharding")
        self.client.send_local_sharding(sharding)
        self.print("step3: wait_for_server_sharding")
        self.unified_sharding = self.client.wait_for_server_sharding()
        # The scale keys the server assigned name the weights the rollout loads as block FP8.
        if self.fp8_block is not None:
            self.fp8_keys = sorted(
                key[: -len("_scale_inv")]
                for key, sharding in self.unified_sharding.items()
                if key.endswith("_scale_inv") and not sharding.is_empty and key[: -len("_scale_inv")] in state_dict
            )
        self._fill(version=0)
        self.print("step4: register_local_tensors")
        self.client.set_publish_cast(self.train_dtype, self.gen_dtype)
        if self.fp8_block is not None:
            self.client.set_publish_quantization(FP8_PLAN)
        self.client.register_local_tensors(self.state_dict, self.unified_sharding)
        self.print("step5: send_local_info")
        self.client.send_local_info()
        self.print("step6: wait_for_server_info (client infos & comm plan)")
        self.client.wait_for_server_info()
        self.print("step7: send_local_temp_mapping")
        self.client.send_local_temp_mapping()
        self.print("step8: wait_for_server_temp_mappings")
        self.client.wait_for_server_temp_mappings()
        self.print("protocol done.")

    def push_to_ps(self, ps_agent_names, ps_client_names):
        futures = []
        for key in self.state_dict_keys:
            wait_operations = []
            for ps_agent_name, ps_client_name in zip(ps_agent_names, ps_client_names):
                shards_to_transfer = self.client.client_write(
                    ps_agent_name,
                    ps_client_name,
                    key,
                    "train_push",
                )
                if len(shards_to_transfer) > 0:
                    wait_operations.append((key, ps_client_name, shards_to_transfer))
                    # self.print(f"Pushing {key} to {ps_client_name}")
            # self.print(f"Waiting for {len(wait_operations)} push operations")
            for key, ps_client_name, shards_to_transfer in wait_operations:
                try:
                    self.client.wait(key, "train_push", "WRITE", target_client=ps_client_name)
                except Exception as e:
                    self.print(f"Wait failed for key {key} to {ps_client_name}. error: {e}")
                    raise e
                futures.append(
                    self.ps_for_push_worker_handles[ps_client_name].transfer_train_to_gen.remote(
                        key, shards_to_transfer
                    )
                )
        ray.get(futures)

    def advance_version(self, base: int):
        """Refill the weights at a new coordinate base, standing in for a train step.

        A second eval round is only a test if the weights it pulls are *different* from
        the first. Re-filling at the same base reproduces identical bytes, so a rollout
        that received nothing would pass exactly as a rollout that received the new
        version -- which is the confusion the steady-state switch-back check exists to
        resolve (item D4).

        Returns the new offsets because they are what the judge compares against; the
        gen side cannot derive them, since the shift is this call's argument.
        """
        self._fill(version=base)
        torch.cuda.synchronize()
        # Publishing refreshes the contiguous copies readers take fragmented shards from.
        self.client.publish_version(base)
        self.print(
            f"[MEASURE] publish refresh {self.client.last_publish_seconds * 1e3:.1f} ms, "
            f"publish copies {self.client.publish_copy_bytes / 2**20:.0f} MiB"
        )
        return self.coord_offsets

    def _fill(self, version: int) -> None:
        """Fill block FP8 keys exactly quantisable and every other key with its layout hash."""
        if self.fp8_keys:
            target, requantize = self.state_dict, None
            if self.train_fp8 is not None:
                target, requantize = _bf16_shadows(self.model[0], self.state_dict)
            fill_fp8_exact(
                target,
                self.sharding,
                self.fp8_keys,
                lambda t: (1,) * (t.dim() - 2) + self.fp8_block,
                version=version,
                uniform=self.train_fp8 is not None,
            )
            if requantize is not None:
                requantize()
        rest = {k: v for k, v in self.state_dict.items() if k not in self.fp8_keys}
        # Only tensors held in the train dtype are cast on publish. Parameters that are
        # deliberately float32 (Qwen3.5's GDN A_log, router bias) travel as they are, so
        # drawing their hash in the gen dtype would describe a conversion that never
        # happens and the judge would reconstruct different bits.
        cast, plain = {}, {}
        for key, entry in rest.items():
            tensor = _first_tensor(entry)
            (cast if tensor.is_floating_point() and tensor.dtype == self.train_dtype else plain)[key] = entry
        self.judge_version = version
        self.coord_offsets = {
            key: version for key in fill_by_layout(cast, self.sharding, version=version, wire_dtype=self.gen_dtype)
        }
        self.coord_offsets.update(
            {key: version for key in fill_by_layout(plain, self.sharding, version=version, wire_dtype=None)}
        )
        self.coord_offsets.update({key: version for key in self.fp8_keys})

    def get_coord_offsets(self):
        """The per-key global coordinate offsets this rank filled with.

        The gen side needs exactly these to judge what it pulled: recomputing them
        there would re-derive the same numbers from the same key ordering and so would
        not catch a key-set mismatch between the two sides.
        """
        return self.coord_offsets

    def build_optimizer(self):
        """An Adam with materialised state, so a switch has optimizer bytes to carry.

        `lr=0` so the step that creates the moments leaves the weights -- and therefore
        the layout judge's expectations -- untouched. The moments themselves are
        perturbed to known nonzero values, because zeros would survive a lost offload.
        """
        params = [p for p in self._model_parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=0.0)
        for p in params:
            p.grad = torch.zeros_like(p)
        self.optimizer.step()
        for p in params:
            p.grad = None
        torch.manual_seed(1234 + self.rank)
        for state in self.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                state[key].copy_(torch.rand_like(state[key]))
        self._opt_checksum = self._optimizer_checksum()
        return sum(s["exp_avg"].numel() * 2 * s["exp_avg"].element_size() for s in self.optimizer.state.values())

    def _model_parameters(self):
        if self.engine_type == "megatron":
            return [p for chunk in self.model for p in chunk.parameters()]
        return list(self.model.parameters())

    def _optimizer_checksum(self) -> float:
        """A single number over all optimizer state, for a bit-exactness comparison.

        Unwrapped to local shards first: under FSDP2 the state tensors are DTensors, and
        arithmetic on one outside its mesh's dispatch raises rather than computing.
        """
        total = torch.zeros((), dtype=torch.float64, device="cuda")
        for state in self.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state[key]
                local = tensor._local_tensor if hasattr(tensor, "_local_tensor") else tensor
                total += local.detach().double().sum()
        return float(total)

    def stage_offload(self, preserve_weights: bool):
        """The trainer half of a colocate switch, through the code production runs."""
        from psrl.utils.nixl.stage_offload import (
            TrainerStage,
            fsdp_weight_tensors,
            megatron_weight_tensors,
            optimizer_tensors,
            stage_policy,
        )

        if not hasattr(self, "_stage"):
            self._stage = TrainerStage()
        self.client.deregister_local_tensors()
        weights = megatron_weight_tensors(self.model) if self.engine_type == "megatron" else fsdp_weight_tensors(self.model)
        policy = stage_policy(has_ps_data_plane=not preserve_weights, offload_optimizer=True, tms_pause=False)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_before = torch.cuda.mem_get_info()[0]
        kept, dropped = self._stage.offload(
            weights=weights, grads=[], optimizer=optimizer_tensors(getattr(self, "optimizer", None)), policy=policy
        )
        freed = torch.cuda.mem_get_info()[0] - free_before
        self.print(
            f"[MEASURE] stage offload: kept {kept.bytes / 2**30:.2f} GiB in {kept.seconds:.3f}s "
            f"({kept.gbps:.1f} GB/s), dropped {dropped.bytes / 2**30:.2f} GiB, device freed {freed / 2**30:.2f} GiB"
        )
        return {"kept": kept.bytes, "kept_s": kept.seconds, "dropped": dropped.bytes, "freed": freed}

    def stage_restore(self, server_name):
        """Bring the weights back, re-register at the new addresses, and announce them."""
        restored = self._stage.restore()
        self.client.register_local_tensors(self.state_dict, self.unified_sharding)
        self.client.publish_version(self.judge_version)
        self.client.send_local_info_to([server_name])
        checksum_ok = True
        if getattr(self, "optimizer", None) is not None:
            checksum_ok = self._optimizer_checksum() == self._opt_checksum
        self.print(
            f"[MEASURE] stage restore: {restored.bytes / 2**30:.2f} GiB in {restored.seconds:.3f}s "
            f"({restored.gbps:.1f} GB/s), optimizer intact={checksum_ok}"
        )
        return {"restored": restored.bytes, "restored_s": restored.seconds, "optimizer_intact": checksum_ok}

    def get_nixl_names(self):
        """This rank's (agent, client) names, for rollouts reading it directly (T2).

        A train actor is its own NIXL agent, so both names are the same string. The gen
        side's pull takes an (agent, client) pair either way, which is why direct supply
        needs no second read path -- only a different pair.
        """
        return self.client_name, self.client_name

    def shutdown(self):
        self.client.shutdown()
        dist.destroy_process_group()


def _is_te_quantized(tensor) -> bool:
    try:
        from transformer_engine.pytorch.tensor import QuantizedTensor
    except ImportError:
        return False
    return isinstance(tensor, QuantizedTensor)


def _bf16_shadows(module, state_dict: dict):
    """
    A copy of `state_dict` whose FP8 parameter entries are views of a high precision shadow of their parameter.

    The converter emits a TE FP8 parameter as views of its FP8 bytes. TE cannot write into a
    part of an FP8 parameter, so a fill writes the shadow, and the returned callback quantises
    each whole parameter from it. A byte view and its shadow view share offset and strides.
    """
    from psrl.utils.nixl.moe_grouping import GroupedShards

    # Megatron may pack several FP8 parameters into one buffer, so a view is matched by address range.
    params = []
    for _, param in module.named_parameters():
        if _is_te_quantized(param):
            params.append((param._data.data_ptr(), param._data.numel(), param, param.dequantize()))

    def shadow_of(view):
        address = view.data_ptr()
        for start, numel, param, shadow in params:
            if start <= address < start + numel:  # FP8 data is one byte per element
                offset = view.storage_offset() - param._data.storage_offset()
                return shadow.as_strided(view.shape, view.stride(), shadow.storage_offset() + offset)
        raise KeyError("An FP8 parameter view lies in no FP8 parameter of the module.")

    def is_fp8(t):
        return getattr(t, "psrl_fp8_scale", None) is not None

    shadowed = {}
    for key, entry in state_dict.items():
        if isinstance(entry, GroupedShards):
            shards = tuple(shadow_of(t) if is_fp8(t) else t for t in entry.shards)
            shadowed[key] = GroupedShards(shards, entry.shard_indices, entry.shard_mesh)
        else:
            shadowed[key] = shadow_of(entry) if is_fp8(entry) else entry

    def requantize():
        for _, _, param, shadow in params:
            param.quantize_(shadow)

    return shadowed, requantize


def _assert_pynccl_loads() -> None:
    """
    Fail before vLLM starts when its NCCL binding cannot load.

    vLLM disables pynccl for every group when the NCCL library lacks a symbol it binds
    (`ncclCommSuspend` needs NCCL >= 2.29.7). TP collectives then fall back silently,
    and MoE with data_parallel_size > 1 asserts deep inside `all_gatherv`.
    """
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
    from vllm.utils.nccl import find_nccl_library

    path = find_nccl_library()
    try:
        NCCLLibrary(path)
    except Exception as e:
        raise RuntimeError(
            f"vLLM cannot load NCCL from {path}: {e}. Point VLLM_NCCL_SO_PATH at NCCL >= 2.29.7, "
            f"which scripts/nixl_env.sh does when the staged copy exists."
        ) from e


@ray.remote(num_cpus=1)
class GenClientActor:
    def __init__(
        self,
        global_store,
        rank,
        world_size,
        server_name,
        psrl_config,
        backend,
        torch_port,
        log_dir,
        model_path,
        model_config,
        gen_config,
    ):
        from vllm import LLM

        if rank == 0:
            gen_master_ip = ray.util.get_node_ip_address()
            ray.get(global_store.set_gen_master_ip.remote(gen_master_ip))
            """
            os.environ["UCX_LOG_LEVEL"] = "debug"
            os.environ["NIXL_LOG_LEVEL"] = "debug"
            f = open(os.path.join(log_dir, "ucx.log"), "w")
            os.dup2(f.fileno(), sys.stdout.fileno())
            os.dup2(f.fileno(), sys.stderr.fileno())
            """
        else:
            gen_master_ip = ray.get(global_store.get_gen_master_ip.remote())
        os.environ["MASTER_ADDR"] = gen_master_ip
        os.environ["MASTER_PORT"] = str(torch_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        self.tp_size = gen_config.get("tensor_parallel_size", 2)
        self.pp_size = gen_config.get("pipeline_parallel_size", 1)
        self.ep_size = gen_config.get("expert_parallel_size", 1)
        self.rank = rank
        self.tp_rank = rank % self.tp_size
        self.instance_world_size = self.tp_size * self.pp_size
        self.engine = None
        assert world_size % (self.tp_size * self.pp_size) == 0, (
            f"TP*PP does not divide world_size. world_size={world_size!r}, product={self.tp_size * self.pp_size!r}."
        )
        # Use explicit data_parallel_size from config if provided, otherwise compute from world_size
        config_dp_size = gen_config.get("data_parallel_size", None)
        computed_dp_size = world_size // self.instance_world_size
        if config_dp_size is not None and config_dp_size > 1:
            assert computed_dp_size == config_dp_size, (
                f"Computed and configured data parallel sizes differ. computed={computed_dp_size!r}, "
                f"configured={config_dp_size!r}, num_gen={world_size!r}, TP={self.tp_size!r}, PP={self.pp_size!r}."
            )
        self.dp_size = computed_dp_size
        self.model_config = model_config
        # None means "use vLLM's default", which is what disaggregate wants. Colocate
        # sets it, because the card is shared with a train rank.
        self.gpu_memory_utilization = gen_config.get("gpu_memory_utilization", None)
        self.trust_remote_code = False
        self.gen_dtype = model_config.get("gen_dtype", "bfloat16")
        self.client_name = gen_client_name(0, rank)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.client_name}.log")
        self.print = make_dual_print(log_path, prefix=self.client_name)
        self.print(f"gen_master_ip: {gen_master_ip}")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.manual_seed(42)
        os.environ["LOCAL_RANK"] = str(0)
        self.print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

        # NOTE(lhy): Create the client before loading the model.
        self.client = NIXLStorageClient(
            client_name=self.client_name,
            server_name=server_name,
            use_gpu=True,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=psrl_config.nixl,
            replica_idx=0,
            worker_index=rank,
            logging_path=log_dir,
            pairwise_descriptors=bool(model_config.get("pairwise_descriptors", True)),
        )

        llm_kwargs = {
            "model": model_path,
            "dtype": self.gen_dtype,
            "tensor_parallel_size": self.tp_size,
            "pipeline_parallel_size": self.pp_size,
            "distributed_executor_backend": "external_launcher",
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
            "trust_remote_code": self.trust_remote_code,
            "enable_expert_parallel": self.ep_size > 1,
            "seed": 42,
        }
        if self.dp_size > 1:
            llm_kwargs["data_parallel_size"] = self.dp_size
        # The transfer overwrites every weight, so a scale run can start from dummy weights instead
        # of reading a checkpoint that may not exist locally (only config and tokenizer are needed).
        if gen_config.get("load_format", None):
            llm_kwargs["load_format"] = gen_config.load_format
        if model_config.get("gen_quantization") == "fp8":
            # The block FP8 that verl's rollout uses, quantised online from a bf16 checkpoint.
            llm_kwargs["quantization"] = "fp8"
            ignored_layers = [
                f"model.layers.{i}.mlp.gate" for i in range(model_config.get("num_layers", 128))
            ]
            # A config may still name projections that cannot be block-quantised at all,
            # for example one whose kernel requires a larger tile than the layer has.
            for suffix in model_config.get("fp8_ignored_layer_suffixes", []):
                ignored_layers += [
                    f"model.layers.{i}.{suffix}" for i in range(model_config.get("num_layers", 128))
                ]
            llm_kwargs["hf_overrides"] = {
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "weight_block_size": [128, 128],
                    "ignored_layers": ignored_layers,
                }
            }
        if self.gpu_memory_utilization is not None:
            # Colocate shares the card with a train rank, so vLLM's default (>0.9) is not
            # available: it reserves a fraction of *total* memory and refuses to start
            # when that much is not free. With a trainer already holding its parameters
            # the startup check fails with "Free memory on device ... is less than desired
            # GPU memory utilization", which is the design's memory account showing up as
            # an error rather than a bug.
            #
            # Disaggregate leaves this None and keeps vLLM's default, so the two
            # placements do not share a tuning constant that only one of them needs.
            llm_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
        _assert_pynccl_loads()
        llm = LLM(**llm_kwargs)
        self.model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.print(f"local rank {self.rank} model: {self.model}")
        """
        seen_module_prefixes = set()
        for module_prefix, module in self.model.named_modules():
            seen_module_prefixes.add(module_prefix)
        self.print(f"seen_module_prefixes: {seen_module_prefixes}, "
                   f"has lm_head: {'lm_head' in seen_module_prefixes}, "
                   f"has lm_head module: {hasattr(self.model, 'lm_head')}")
        """

    def _get_replica_id(self):
        return self.rank // self.instance_world_size

    def _get_ep_rank(self):
        if self.ep_size <= 1:
            return 0
        try:
            from vllm.distributed.parallel_state import get_ep_group

            return get_ep_group().rank_in_group
        except AssertionError:
            return 0

    def init_finished(self):
        self.print(f"Gen client init finished on ip {os.environ.get('LOCAL_IP')}")
        return True

    def get_state_dict(self):
        return {k: v.cpu() for k, v in self.state_dict.items()}

    def judge_layout(self, keys, version):
        """
        Judge the pulled weights in place against this client's own declared sharding.
        """
        torch.cuda.synchronize()
        fp8 = [
            k
            for k in keys
            if k in self.state_dict
            and k + "_scale_inv" in self.state_dict
            and _first_tensor(self.state_dict[k]).dtype == torch.float8_e4m3fn
        ]
        judged, failure = check_fp8_exact(
            self.state_dict,
            self.sharding,
            fp8,
            (128, 128),
            version=version,
            uniform=self.model_config.get("train_fp8", None) is not None,
        )
        if failure is not None:
            return judged, failure
        rest_judged, failure = check_by_layout(
            self.state_dict, self.sharding, [k for k in keys if k not in fp8], version=version
        )
        return judged + rest_judged, failure

    def get_nixl_names(self):
        """This client's (agent, client) names, for peers reading it (item D1).

        A gen actor is its own NIXL agent, so both names are the same string -- the same
        shape the train side reports, which is what lets the peer phase reuse the pull
        path unchanged.
        """
        return self.client_name, self.client_name

    def protocol(self, model_path):
        self.print("step0: convert_vllm_inplace")
        from transformers import AutoConfig
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm_patches.interfaces import supports_weight_layout

        vllm_model = self.model
        if isinstance(vllm_model, CUDAGraphWrapper):
            vllm_model = vllm_model.unwrap()
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=self.trust_remote_code)
        param_mapping = (
            None if supports_weight_layout(vllm_model) else create_parameter_mapping(type(vllm_model), model_config)
        )
        state_dict, sharding = convert_vllm_inplace(
            vllm_model,
            tp_rank=self.tp_rank,
            ep_rank=self._get_ep_rank(),
            parameter_mapping=param_mapping,
        )
        # self.print(f"state_dict keys: {list(state_dict.keys())}")
        self.state_dict = state_dict
        self.sharding = sharding
        self.state_dict_keys = list(state_dict.keys())
        self.unified_sharding = None
        self.print("step1: connect_to_server")
        self.client.connect_to_server()
        self.print("step2: send_local_sharding")
        self.client.send_local_sharding(sharding)
        self.print("step3: wait_for_server_sharding")
        self.unified_sharding = self.client.wait_for_server_sharding()
        # self.print(f"unified_sharding: {self.unified_sharding}")
        self.print("step4: register_local_tensors")
        self.client.register_local_tensors(self.state_dict, self.unified_sharding)
        self.print("step5: send_local_info")
        self.client.send_local_info()
        self.print("step6: wait_for_server_info (client infos & comm plan)")
        self.client.wait_for_server_info()
        # self.print(f"comm plan: {self.client._comm_plan}")
        self.print("step7: send_local_temp_mapping")
        self.client.send_local_temp_mapping()
        self.print("step8: wait_for_server_temp_mappings")
        self.client.wait_for_server_temp_mappings()
        self.print("protocol done.")

    def wait_for_update_infos(self, n_agents, timeout=None):
        """Absorb re-registered source infos; drops the pull engine built on the old ones."""
        self.client.wait_for_update_infos(n_agents, timeout=timeout)
        self.engine = None
        return True

    def pull(self, epoch, participants=None):
        """Pull through the production engine and return its report."""
        if self.engine is None:
            self.engine = self.client.pull_engine()
        report = self.engine.pull(epoch, set(participants) if participants is not None else None)
        torch.cuda.synchronize()
        return dataclasses.asdict(report)

    def shutdown(self):
        self.client.shutdown()


def create_ps_worker_group(train_engine_type, num_ps, psrl_config, model_config):
    ps_model_config = OmegaConf.create(
        {
            "path": model_config.path,
            "use_shm": False,
            "trust_remote_code": False,
        }
    )
    ray_nodes = ray.nodes()
    ray_nodes_sorted = sorted(ray_nodes, key=lambda n: n["NodeManagerAddress"])
    nodes = [ray_nodes_sorted[-i - 1] for i in range(num_ps)]
    ps_resource_pool = PSResourcePool(
        [
            PSResourceSpec(
                node_ip=node["NodeManagerAddress"],
                node_id=node["NodeID"],
                attached_gpu_id=None,
            )
            for node in nodes
        ]
    )
    storage_plan = PSStoragePlan(
        train_model_dtype=resolve_torch_dtype(model_config.get("train_dtype", "bfloat16")),
        gen_model_dtype=resolve_torch_dtype(model_config.get("gen_dtype", "bfloat16")),
    )
    ps_cls_with_init = PSClassWithInitArgs(
        ray.remote(PSStorageWorker),
        storage_plan,
        ps_model_config,
        psrl_config,
    )
    ps_wg = PSWorkerGroup(ps_resource_pool, ps_cls_with_init)
    return ps_wg


def _assert_actors_run_this_tree() -> None:
    """Fail loudly if Ray actors import a different checkout than the driver.

    The cluster environment puts another `psrl` on the path, so without the PYTHONPATH
    override the actors silently run that one while the driver runs this one -- and the
    run *passes*, proving nothing about the code under test. A one-line check here is
    worth far more than the minutes it costs, because the failure it catches is
    invisible in the result.

    Checked on every node, not just one: a partial PYTHONPATH would be the same bug with
    a smaller blast radius.
    """

    @ray.remote(num_cpus=0)
    def _where() -> str:
        import psrl.utils.nixl.client as module

        return module.__file__

    nodes = [node for node in ray.nodes() if node["Alive"]]
    seen = ray.get(
        [
            _where.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ).remote()
            for node in nodes
        ]
    )
    expected = os.path.join(_REPO_ROOT, "psrl", "utils", "nixl", "client.py")
    wrong = sorted({path for path in seen if path != expected})
    assert not wrong, (
        f"Ray actors import a different checkout than the driver.\n"
        f"  expected: {expected}\n"
        f"  actors:   {wrong}\n"
        "The run would pass while measuring code this tree does not contain."
    )
    print(f"[PASS] all {len(seen)} nodes import this tree: {expected}")


def _first_tensor(entry):
    return entry.shards[0] if hasattr(entry, "shards") else entry


def _timed_pull(gen_actors, *, epoch: int, label: str, expect_peers: bool, alone=None) -> list[dict]:
    """Run one pull on every gen actor and print what it moved."""
    start = time.time()
    if alone is None:
        reports = ray.get([g.pull.remote(epoch) for g in gen_actors])
    else:
        reports = ray.get([g.pull.remote(epoch, [name]) for g, name in zip(gen_actors, alone)])
    elapsed = time.time() - start
    upstream = sum(r["upstream_bytes"] for r in reports)
    peer = sum(r["peer_bytes"] for r in reports)
    fallback = sum(r["fallback_keys"] for r in reports)
    slowest = max(reports, key=lambda r: r["seconds"])
    rates = sorted(
        r["upstream_bytes"] / max(r["upstream_seconds"], 1e-9) / 1e9 for r in reports if r["upstream_bytes"]
    )
    if rates:
        print(
            f"[MEASURE] {label}: per client upstream GB/s min {rates[0]:.1f} median {rates[len(rates) // 2]:.1f} "
            f"max {rates[-1]:.1f}, upstream seconds max {max(r['upstream_seconds'] for r in reports):.3f}"
        )
    # Per source egress across all readers: bytes served and the span from first post to last completion.
    served: dict[str, list] = {}
    for r in reports:
        for source, (nbytes, first, last) in r.get("sources", {}).items():
            agg = served.setdefault(source, [0, first, last])
            agg[0] += nbytes
            agg[1], agg[2] = min(agg[1], first), max(agg[2], last)
    if served:
        egress = sorted((v[0] / max(v[2] - v[1], 1e-9) / 1e9, v[0] / 1e9, k) for k, v in served.items())
        print(
            f"[MEASURE] {label}: per source egress GB/s "
            f"min {egress[0][0]:.1f} ({egress[0][2]}, {egress[0][1]:.1f} GB) "
            f"median {egress[len(egress) // 2][0]:.1f} "
            f"max {egress[-1][0]:.1f} ({egress[-1][2]}, {egress[-1][1]:.1f} GB), "
            f"bytes per source min {min(e[1] for e in egress):.1f} max {max(e[1] for e in egress):.1f} GB"
        )
    print(
        f"[PASS] {label} done. time: {elapsed:.3f}s (slowest client {slowest['seconds']:.3f}s, "
        f"upstream {slowest['upstream_seconds']:.3f}s, peer {slowest['peer_seconds']:.3f}s) "
        f"trainer egress {upstream / 1e9:.2f} GB, peer {peer / 1e9:.2f} GB, "
        f"descriptors max {max(r['descriptors'] for r in reports)}, "
        f"staged {max(r['staged_bytes'] for r in reports) / 2**20:.0f} MiB, fallback keys {fallback}, "
        f"ipc {sum(r.get('ipc_bytes', 0) for r in reports) / 1e9:.2f} GB"
    )
    assert fallback == 0, f"{fallback} keys fell back to upstream reads because a designate did not answer."
    if expect_peers:
        assert peer > 0, "Diffusion was planned but no client read from a peer."
    else:
        assert peer == 0, f"{peer} bytes were read from peers with diffusion off."
    return reports


def _judge_all_gen(gen_actors, filled: dict, *, version: int, label: str) -> None:
    """Run the layout judge on every gen actor and fail on the first bad client."""
    keys = sorted(filled)
    verdicts = ray.get([g.judge_layout.remote(keys, version) for g in gen_actors])
    for gen_idx, (judged, failure) in enumerate(verdicts):
        assert failure is None, f"Gen client {gen_idx}{label}: {failure}"
        assert judged > 0, f"Gen client {gen_idx}{label} holds none of the filled keys."
        print(f"[PASS] Gen client {gen_idx}{label}: {judged}/{len(keys)} filled keys match their layout.")


@hydra.main(config_path="config", config_name="nixl_e2e", version_base=None)
def test_nixl_e2e(cfg: DictConfig):
    validate_parallel_config(cfg)
    log_dir = cfg.logging.path
    os.makedirs(log_dir, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "PSRL_LOGGING_PATH": log_dir,
                # Ray worker processes resolve `psrl` through the cluster's shared
                # environment, which points at a *different* checkout
                # (.../psrl_main/psrl). Without this the driver runs this tree while
                # every actor runs that one, so the harness silently measures code the
                # working tree does not contain -- a passing run that proves nothing
                # about the change under test. Prepending the repo root makes the
                # actors import what the driver does.
                "PYTHONPATH": f"{_REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
            }
        },
    )
    _assert_actors_run_this_tree()
    listen_ip = cfg.network.listen_ip
    print(f"meta server listen_ip: {listen_ip}")
    server_name = NIXL_META_SERVER_NAME
    backend = cfg.network.backend
    torch_port_train = cfg.network.torch_port_train
    torch_port_gen = cfg.network.torch_port_gen
    # Number of train GPUs
    num_train = cfg.test.num_train
    # Number of gen GPUs
    num_gen = cfg.test.num_gen
    # T2 / D3: with direct supply the rollout reads the train clients and no PS exists.
    # Defaulting to False keeps every existing config on the PS path, so the two can be
    # compared on the same cluster rather than one replacing the other untested.
    direct_supply = bool(cfg.test.get("direct_supply", False))
    # Number of PS Nodes (align with the number of nodes used for gen)
    num_ps = 0 if direct_supply else math.ceil(num_gen / NUM_GPU_PER_NODE)
    ray_nodes = ray.nodes()
    ray_nodes_sorted = sorted(ray_nodes, key=lambda n: n["NodeManagerAddress"])
    # Use the first num_train / NUM_GPU_PER_NODE nodes for train
    train_nnodes = math.ceil(num_train / NUM_GPU_PER_NODE)
    assert len(ray_nodes_sorted) >= train_nnodes, (
        f"Ray has too few train nodes. required={train_nnodes!r}, available={len(ray_nodes_sorted)!r}."
    )
    train_nodes = [ray_nodes_sorted[i] for i in range(train_nnodes)]
    # Use the last num_gen / NUM_GPU_PER_NODE nodes for gen
    gen_nnodes = math.ceil(num_gen / NUM_GPU_PER_NODE)
    assert len(ray_nodes_sorted) >= gen_nnodes, (
        f"Ray has too few generation nodes. required={gen_nnodes!r}, available={len(ray_nodes_sorted)!r}."
    )
    gen_nodes = [ray_nodes_sorted[-i - 1] for i in range(gen_nnodes)]

    # Colocate places train and gen on the *same* GPUs, which is the arrangement every
    # conclusion in design 3.7 is about and the one this harness could not express: with
    # `num_gpus=1` each, Ray guarantees they never share a device, so the timing-exclusion
    # path was unreachable. Taking the first nodes for both and halving the request is
    # what puts a train rank and a gen rank on one card.
    #
    # The fractional request is a *scheduling* statement, not a memory split: each actor
    # still sees the whole device. That is exactly the colocate situation -- two processes
    # on one GPU, which is why the design requires the trainer to offload before rollout
    # allocates its KV cache rather than relying on them fitting together.
    colocate = bool(cfg.test.get("colocate", False))
    colocate_bundles = None
    if colocate:
        assert num_train == num_gen, (
            f"colocate pairs each train rank with a gen rank on one GPU, so the counts "
            f"must match; got num_train={num_train} num_gen={num_gen}"
        )
        gen_nodes = train_nodes
        # A placement group with one bundle per GPU, not a bare fractional request.
        #
        # `num_gpus=0.5` alone lets Ray pack *any* two actors onto a device, and it
        # chose two gen actors: 8 actors landed on 4 devices, so vLLM's TP=2 tried to
        # init two NCCL ranks on one GPU and failed with "invalid usage". The pairing
        # has to be train-with-gen, which means the placement must be expressed rather
        # than left to the scheduler.
        #
        # PACK, not STRICT_SPREAD: the latter demands one bundle per *node*, so asking
        # for 8 bundles on a 5-node cluster never becomes ready and the run dies on a
        # placement-group timeout rather than anything to do with colocate. PACK keeps
        # the bundles together, and one `GPU: 1` bundle per pair is what reserves a whole
        # card for exactly one train rank and one gen rank.
        from ray.util.placement_group import placement_group

        colocate_bundles = placement_group(
            [{"GPU": 1, "CPU": 2} for _ in range(num_train)],
            strategy="PACK",
        )
        ray.get(colocate_bundles.ready(), timeout=300)
    gpus_per_actor = 0.5 if colocate else 1

    def _placement(index: int, node_id: str):
        """Where one actor goes: a colocate bundle, or a node."""
        if colocate_bundles is not None:
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            return PlacementGroupSchedulingStrategy(
                placement_group=colocate_bundles,
                placement_group_bundle_index=index,
            )
        return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    # print(f"train_nodes: {train_nodes}, gen_nodes: {gen_nodes}")
    train_engine_type = cfg.test.train_engine_type

    psrl_config = OmegaConf.create(
        {
            "logging_path": log_dir,
            "ps_manager_ip": listen_ip,
            # The whole `nixl` section, not a hand-picked subset. Copying named keys
            # meant a new config option was silently dropped: `peer_diffusion: False`
            # was set in the YAML, never reached the planner, and the run diffused
            # anyway -- with no error, because a missing key just takes its default.
            #
            # Resolved on the way in. The section's values interpolate against siblings
            # (`server_ip: ${network.listen_ip}`), and those siblings do not exist in this
            # smaller config, so copying the node unresolved defers the lookup to a place
            # where it fails: `InterpolationKeyError: network.listen_ip not found`, raised
            # inside the server actor rather than here.
            "nixl": OmegaConf.to_container(cfg.nixl, resolve=True),
            "ps_mode": cfg.ps.mode,
        }
    )

    global_store = GlobalStore.remote()

    start_time = time.time()
    ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}

    # Create per-node PortScanner actors (required by NIXLStorageClient).
    # The comment above predates the call: every NIXLStorageClient resolves
    # `get_port_scanner(host_ip)` during construction, so without this the first client
    # dies with "Failed to look up actor with name 'psrl_port_scanner_<ip>'". Only alive
    # nodes are passed -- a dead entry would pin a detached actor to a node that cannot
    # run it, and the lookup would hang rather than fail.
    from psrl.utils.nixl.port_scanner import create_port_scanners

    alive_ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes() if node["Alive"]}
    port_scanner_handles = create_port_scanners(alive_ip_to_node_id)
    print(f"[PASS] created {len(port_scanner_handles)} PortScanner actors")

    assert listen_ip in ip_to_node_id, f"listen_ip is absent from Ray nodes: {listen_ip!r}."
    server = MetaServerActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ip_to_node_id[listen_ip], soft=False)
    ).remote(server_name, psrl_config, num_train + num_gen + num_ps, log_dir)
    ray.get(server.init_finished.remote())
    end_time = time.time()
    print(f"[PASS] server init done. time: {end_time - start_time}s")

    start_time = time.time()
    if direct_supply:
        # T2 / D3: no PS at all. The rollout reads the train clients, so there is no
        # intermediate copy to create, register, or wait for. Keeping this as a branch
        # rather than a separate harness is what makes the two arrangements comparable:
        # everything after this point is identical except which names the pull targets.
        ps_wg = None
        ps_agent_names = []
        ps_for_push_names = []
        ps_for_pull_names = []
        ps_for_push_worker_handles = {}
        print("[PASS] direct supply: no PS workers created")
    else:
        ps_wg = create_ps_worker_group(train_engine_type, num_ps, psrl_config, cfg.model)
        ray.get(ps_wg.execute_all_async("init_model"))
        ray.get(ps_wg.execute_all_async("init_nixl_client"))
        ps_agent_names = ray.get(ps_wg.execute_all_async("get_nixl_agent_name"))
        ps_for_push_names = ray.get(ps_wg.execute_all_async("get_nixl_train_storage_client_name"))
        ps_for_pull_names = ray.get(ps_wg.execute_all_async("get_nixl_gen_storage_client_name"))
        print(f"ps_for_push_names: {ps_for_push_names}")
        print(f"ps_for_pull_names: {ps_for_pull_names}")
        ps_for_push_worker_handles = {}
        for ps_for_push_name in ps_for_push_names:
            ps_for_push_worker_handles[ps_for_push_name] = ps_wg.distinguish_worker_by_method(
                lambda worker, name=ps_for_push_name: (
                    ray.get(worker.get_nixl_train_storage_client_name.remote()) == name
                )
            )
        end_time = time.time()
        print(f"[PASS] ps init done. time: {end_time - start_time}s")

    start_time = time.time()
    # Prepare configuration for train actors
    fsdp_hybrid_config = cfg.test.fsdp_hybrid if train_engine_type == "fsdp_hybrid" else None
    megatron_config = cfg.test.megatron if train_engine_type == "megatron" else None

    train_actors = [
        TrainClientActor.options(
            num_gpus=gpus_per_actor,
            scheduling_strategy=_placement(rank, train_nodes[rank // NUM_GPU_PER_NODE]["NodeID"]),
        ).remote(
            global_store,
            train_engine_type,
            rank,
            num_train,
            server_name,
            psrl_config,
            backend,
            torch_port_train,
            log_dir,
            ps_for_push_worker_handles,
            cfg.model.path,
            cfg.model,
            fsdp_hybrid_config,
            megatron_config,
        )
        for rank in range(num_train)
    ]
    gen_actors = [
        GenClientActor.options(
            num_gpus=gpus_per_actor,
            scheduling_strategy=_placement(rank, gen_nodes[rank // NUM_GPU_PER_NODE]["NodeID"]),
        ).remote(
            global_store,
            rank,
            num_gen,
            server_name,
            psrl_config,
            backend,
            torch_port_gen,
            log_dir,
            cfg.model.path,
            cfg.model,
            cfg.test.gen,
        )
        for rank in range(num_gen)
    ]
    for i in range(num_train):
        ray.get(train_actors[i].init_finished.remote())
    for i in range(num_gen):
        ray.get(gen_actors[i].init_finished.remote())
    end_time = time.time()
    print(f"[PASS] client init done. time: {end_time - start_time}s")

    start_time = time.time()
    # Run protocol for server, clients, and PS workers
    server.protocol.remote()
    futures = []
    for t in train_actors:
        futures.append(t.protocol.remote(cfg.model.path))
    for g in gen_actors:
        futures.append(g.protocol.remote(cfg.model.path))
    if ps_wg is not None:
        futures.extend(ps_wg.execute_all_async("nixl_protocol"))
    ray.get(futures)
    end_time = time.time()
    print(f"[PASS] protocol done. time: {end_time - start_time}s")

    # Warm-up
    """
    futures = []
    for g in gen_actors:
        futures.append(g.pull_from_ps.remote(ps_agent_names, ps_for_pull_names))
    ray.get(futures)
    end_time = time.time()
    print(f"[PASS] Warm-up done. time: {end_time - start_time}s")
    """

    # Each train client pushes to each PS worker. Under direct supply there is no
    # push at all -- that the step disappears rather than becoming a no-op is the
    # point of D3: the intermediate copy is gone, not merely skipped.
    if not direct_supply:
        start_time = time.time()
        futures = []
        for t in train_actors:
            futures.append(t.push_to_ps.remote(ps_agent_names, ps_for_push_names))
        ray.get(futures)
        end_time = time.time()
        print(f"[PASS] train push to all ps done. time: {end_time - start_time}s")

    # Each gen client pulls from each PS worker
    """
    for gen_idx, g in enumerate(gen_actors):
        start_time = time.time()
        ray.get(g.pull_from_ps.remote(ps_agent_names, ps_for_pull_names))
        end_time = time.time()
        print(f"[PASS] gen {gen_idx} pull from all ps done. time: {end_time - start_time}s")
    """
    diffusion_planned = bool(cfg.nixl.get("peer_diffusion", True))
    gen_names = [c for _a, c in ray.get([g.get_nixl_names.remote() for g in gen_actors])]
    _timed_pull(gen_actors, epoch=1, label="gen pull", expect_peers=diffusion_planned)
    if diffusion_planned and os.environ.get("PSRL_E2E_BASELINE", "1") == "1":
        # Same run A/B. A client that is its only participant has no peers and reads everything upstream.
        # The first baseline pull prepares its handles, so the second is the one to compare with round 2.
        for epoch in (2, 3):
            _timed_pull(gen_actors, epoch=epoch, label="no-diffusion baseline", alone=gen_names, expect_peers=False)

    # Fetch and verify gen client state_dicts against the coordinate function.
    print("[CHECK] Verifying GenClientActor state_dicts encode their global coordinates...")
    # Merge offsets from every train rank: under TP/PP each rank holds only part of the
    # key space, so one rank's map would leave most gen keys unjudged. Ranks that do
    # share a key agree on its offset (the fill is driven by the canonical key order),
    # so a plain merge is well defined.
    per_rank_offsets = ray.get([t.get_coord_offsets.remote() for t in train_actors])
    coord_offsets: dict[str, int] = {}
    for rank_offsets in per_rank_offsets:
        coord_offsets.update(rank_offsets)
    print(f"[CHECK] train side filled {len(coord_offsets)} distinct keys")

    _judge_all_gen(gen_actors, coord_offsets, version=0, label="")
    print("[PASS] All GenClientActor parameters match the coordinate function.")

    # A colocate switch cycle between rounds (design §3.7.3). Under direct supply the
    # trainer is the only copy of the weights, so they must survive the offload; the
    # proof is a pull of the *same* version after the switch, judged against round 1's
    # expectations. Advancing the version first would refill every byte and hide a lost
    # one, which is why this runs before round 2 rather than as part of it.
    if cfg.test.get("stage_switch", False):
        assert direct_supply, "stage_switch checks weight survival, which only means something under direct supply"
        check_optimizer = bool(cfg.test.get("stage_switch_optimizer", False))
        if check_optimizer:
            opt_bytes = ray.get([t.build_optimizer.remote() for t in train_actors])
            print(f"[CHECK] optimizer state per rank: {max(opt_bytes) / 2**30:.2f} GiB")
        else:
            print("[CHECK] optimizer not built (`test.stage_switch_optimizer=true` to include it)")
        offloads = ray.get([t.stage_offload.remote(True) for t in train_actors])
        republish = server.republish.remote(
            num_train, gen_names, [train_client_name(i) for i in range(num_train)]
        )
        gen_waits = [g.wait_for_update_infos.remote(1) for g in gen_actors]
        restores = ray.get([t.stage_restore.remote(server_name) for t in train_actors])
        republished, _ = ray.get([republish, *gen_waits])
        print(f"[CHECK] republish saw {republished}")
        kept = max(o["kept"] for o in offloads)
        down = max(o["kept_s"] for o in offloads)
        up = max(r["restored_s"] for r in restores)
        freed = min(o["freed"] for o in offloads)
        print(
            f"[MEASURE] stage switch: per rank kept {kept / 2**30:.2f} GiB, slowest offload {down:.3f}s "
            f"({kept / down / 1e9:.1f} GB/s), slowest restore {up:.3f}s ({kept / up / 1e9:.1f} GB/s), "
            f"least device memory freed {freed / 2**30:.2f} GiB"
        )
        assert all(r["optimizer_intact"] for r in restores), "optimizer state changed across the switch"
        assert freed >= 0.9 * kept, f"only {freed / 2**30:.2f} of {kept / 2**30:.2f} GiB went back to the device"
        _timed_pull(gen_actors, epoch=10, label="pull after stage switch", expect_peers=diffusion_planned)
        _judge_all_gen(gen_actors, coord_offsets, version=0, label=" (after stage switch)")
        print("[PASS] Stage switch: weights and optimizer survived, readers found the new addresses.")

    # Second round: a new version must actually arrive (item D4).
    #
    # One round cannot distinguish "the transfer works" from "the transfer worked once".
    # The failure this guards is silent: a steady-state switch-back that re-reads the
    # checkpoint, or a rollout that keeps serving a cached version, both produce *valid*
    # weights -- merely the wrong ones -- so nothing raises and the only visible symptom
    # is a loss curve that stops moving.
    #
    # The train side refills at a shifted coordinate base, which is what makes round two
    # a real test: re-filling at the same base would reproduce identical bytes, and a
    # rollout that received nothing would pass exactly as one that received the update.
    if cfg.test.get("rounds", 1) > 1:
        print("[ROUND 2] Advancing the train-side version...")
        version_shift = 1_000_000
        per_rank_offsets_v2 = ray.get([t.advance_version.remote(version_shift) for t in train_actors])
        coord_offsets_v2: dict[str, int] = {}
        for rank_offsets in per_rank_offsets_v2:
            coord_offsets_v2.update(rank_offsets)

        # The shift must have changed something, or the round proves nothing.
        assert coord_offsets_v2 != coord_offsets, (
            "advancing the version left the coordinate offsets unchanged, so round two "
            "cannot tell a fresh pull from a stale one"
        )

        _timed_pull(gen_actors, epoch=4, label="round 2 pull", expect_peers=diffusion_planned)

        _judge_all_gen(gen_actors, coord_offsets_v2, version=version_shift, label=" (round 2)")
        print("[PASS] Round 2: every rollout picked up the new version, not a cached one.")

    # Shutdown all actors and Ray
    for t in train_actors:
        ray.get(t.shutdown.remote())
    for g in gen_actors:
        ray.get(g.shutdown.remote())
    if ps_wg is not None:
        ray.get(ps_wg.execute_all_async("shutdown"))
    ray.get(server.shutdown.remote())
    ray.shutdown()


if __name__ == "__main__":
    test_nixl_e2e()
