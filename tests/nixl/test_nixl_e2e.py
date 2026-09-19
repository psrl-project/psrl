import asyncio
import math
import os
import time

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
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig, AutoModelForCausalLM
from verl.models.mcore import hf_to_mcore_config
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import apply_fsdp2
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.engine.megatron.utils import set_random_seed

NUM_GPU_PER_NODE = 8


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
        assert gen_ep == gen_tp, f"vLLM EP must match TP in PSRL. EP={gen_ep!r}, TP={gen_tp!r}."

    if train_engine_type == "fsdp_hybrid":
        ddp_size = cfg.test.fsdp_hybrid.ddp_size
        fsdp_size = cfg.test.fsdp_hybrid.fsdp_size
        assert num_train % (ddp_size * fsdp_size) == 0, (
            f"HSDP DDP*FSDP does not divide num_train. num_train={num_train!r}, product={ddp_size * fsdp_size!r}."
        )
    elif train_engine_type == "megatron":
        megatron = cfg.test.megatron
        train_tp = megatron.tensor_model_parallel_size
        train_pp = megatron.pipeline_model_parallel_size
        train_cp = megatron.get("context_parallel_size", 1)
        model_parallel_size = train_tp * train_pp * train_cp
        assert num_train % model_parallel_size == 0, (
            f"Megatron TP*PP*CP does not divide num_train. num_train={num_train!r}, product={model_parallel_size!r}."
        )


def assert_state_dict_all_ones(state_dict: dict[str, torch.Tensor], actor_label: str) -> None:
    assert state_dict, f"{actor_label} state_dict is empty."
    for key, tensor in state_dict.items():
        expected = torch.ones_like(tensor)
        if not torch.allclose(tensor, expected, atol=1e-6):
            max_abs_diff = (tensor - expected).abs().max().item()
            raise AssertionError(
                f"{actor_label} parameter {key!r} is not all ones. "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, max_abs_diff={max_abs_diff}."
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
        self.server.wait_for_client_shardings(self.expected_agents, timeout=120)
        # self.print(f"client_sharding_dicts: {self.server.client_sharding_dicts}")
        self.print("step2: make_unified_sharding")
        self.server.make_unified_sharding()
        # self.print(f"client_unified_sharding_dicts: {self.server.client_unified_sharding_dicts}")
        self.print("step3: notify_all_client_shardings")
        self.server.notify_all_client_shardings()
        self.print("step4: wait_for_client_infos")
        self.server.wait_for_client_infos(self.expected_agents, timeout=120)
        # self.print(f"client_infos: {self.server.client_infos}")
        self.print("step5: make_comm_plan")
        self.server.make_comm_plan()
        self.print("step6: notify_all_client_infos_and_comm_plan")
        self.server.notify_all_client_infos_and_comm_plan()
        self.print("step7: wait_for_client_temp_mappings")
        self.server.wait_for_client_temp_mappings(self.expected_agents, timeout=120)
        self.print("step8: notify_all_client_temp_mappings")
        self.server.notify_all_client_temp_mappings()
        self.print("protocol done.")
        return self.server.client_unified_sharding_dicts

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
        assert engine_type in ["fsdp", "fsdp_hybrid", "megatron"], f"Unsupported engine type: {engine_type!r}."
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
        )

        if engine_type == "fsdp" or engine_type == "fsdp_hybrid":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.train_dtype,
                trust_remote_code=self.trust_remote_code,
            )
            # Set all model parameters to a deterministic source value.
            for p in model.parameters():
                p.data.fill_(1)
            torch.manual_seed(42)
            # FSDP2
            if engine_type == "fsdp":
                mesh = init_device_mesh("cuda", mesh_shape=(world_size,))
            else:
                # HSDP: replicate across ddp_size, shard across fsdp_size.
                ddp_size = fsdp_hybrid_config.get("ddp_size", 2)
                fsdp_size = fsdp_hybrid_config.get("fsdp_size", 4)
                assert world_size % (ddp_size * fsdp_size) == 0, (
                    f"DDP*FSDP does not divide world_size. world_size={world_size!r}, "
                    f"product={ddp_size * fsdp_size!r}."
                )
                mesh = init_device_mesh("cuda", mesh_shape=(ddp_size, fsdp_size))
            apply_fsdp2(model, {"mesh": mesh}, {})
            self.model = model
        elif engine_type == "megatron":
            set_random_seed(42)
            self._init_megatron_model(model_path)
        self.ps_for_push_worker_handles = ps_for_push_worker_handles
        # self._log_env_info()

    def _log_env_info(self):
        for k in sorted(os.environ):
            self.print(f"{k}={os.environ[k]}")

    def _get_replica_id(self):
        if self.engine_type == "fsdp":
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

        use_mbridge = True

        if use_mbridge:
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
            provider.apply_overrides_and_finalize(
                dtype=dtype,
                overrides=provider_overrides,
            )
            tf_config = None
        else:
            tf_config = hf_to_mcore_config(hf_config, dtype)

        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        wrap_config = McoreModuleWrapperConfig()
        actor_module, updated_tf_config = make_megatron_module(
            wrap_config,
            tf_config,
            hf_config,
            bridge=bridge if use_mbridge else None,
            provider=provider,
        )
        if use_mbridge:
            tf_config = updated_tf_config
        self.model = actor_module

        for name, param in self.model[0].named_parameters():
            param.data.fill_(1)
        for name, param in self.model[0].named_buffers():
            param.data.fill_(1)

        self.print(f"[Rank {self.rank}] Model initialized: {self.model}")

    def init_finished(self):
        self.print(f"Train client init finished on ip {os.environ.get('LOCAL_IP')}")
        return True

    def protocol(self, model_path):
        self.print("step0: convert_fsdp/megatron_inplace")
        if self.engine_type == "fsdp" or self.engine_type == "fsdp_hybrid":
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
        for tensor in state_dict.values():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                tensor.data.fill_(1)
        # self.print(f"state_dict keys: {state_dict.keys()}")
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
        self.print("step4: register_local_tensors")
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
                    merge_and_cache_xfer=False,
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
        # self.client.merge_and_finish_cached_xfer()

    def shutdown(self):
        self.client.shutdown()
        dist.destroy_process_group()


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

    def pull_from_ps(self, ps_agent_names, ps_client_names):
        wait_operations = []
        total_start_time = time.time()
        for key in self.state_dict_keys:
            for ps_agent_name, ps_client_name in zip(ps_agent_names, ps_client_names):
                # self.print(f"pull {key} from {ps_client_name}")
                shards_to_transfer = self.client.client_read(
                    ps_agent_name,
                    ps_client_name,
                    key,
                    "gen_pull",
                    merge_and_cache_xfer=False,
                )
                if len(shards_to_transfer) > 0:
                    wait_operations.append((key, ps_client_name, shards_to_transfer))
        for key, ps_client_name, shards_to_transfer in wait_operations:
            self.client.wait(key, "gen_pull", "READ", target_client=ps_client_name)
        self.client.merge_and_finish_cached_xfer()
        total_end_time = time.time()
        # torch.cuda.synchronize()
        self.print(f"Total pull from ps done: {total_end_time - total_start_time}s")

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


@hydra.main(config_path="config", config_name="nixl_e2e", version_base=None)
def test_nixl_e2e(cfg: DictConfig):
    validate_parallel_config(cfg)
    log_dir = cfg.logging.path
    os.makedirs(log_dir, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PSRL_LOGGING_PATH": log_dir}},
    )
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
    # Number of PS Nodes (align with the number of nodes used for gen)
    num_ps = math.ceil(num_gen / NUM_GPU_PER_NODE)
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
    # print(f"train_nodes: {train_nodes}, gen_nodes: {gen_nodes}")
    train_engine_type = cfg.test.train_engine_type

    psrl_config = OmegaConf.create(
        {
            "logging_path": log_dir,
            "ps_manager_ip": listen_ip,
            "nixl": {
                "server_ip": cfg.nixl.server_ip,
                "server_port": cfg.nixl.server_port,
                "max_pinned_temp_memory_slots": cfg.nixl.max_pinned_temp_memory_slots,
                "enable_tms_for_temp_buffers": cfg.nixl.enable_tms_for_temp_buffers,
            },
            "ps_mode": cfg.ps.mode,
        }
    )

    global_store = GlobalStore.remote()

    start_time = time.time()
    ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}

    # Create per-node PortScanner actors (required by NIXLStorageClient).
    assert listen_ip in ip_to_node_id, f"listen_ip is absent from Ray nodes: {listen_ip!r}."
    server = MetaServerActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ip_to_node_id[listen_ip], soft=False)
    ).remote(server_name, psrl_config, num_train + num_gen + num_ps, log_dir)
    ray.get(server.init_finished.remote())
    end_time = time.time()
    print(f"[PASS] server init done. time: {end_time - start_time}s")

    start_time = time.time()
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
            lambda worker, name=ps_for_push_name: ray.get(worker.get_nixl_train_storage_client_name.remote()) == name
        )
    end_time = time.time()
    print(f"[PASS] ps init done. time: {end_time - start_time}s")

    start_time = time.time()
    # Prepare configuration for train actors
    fsdp_hybrid_config = cfg.test.fsdp_hybrid if train_engine_type == "fsdp_hybrid" else None
    megatron_config = cfg.test.megatron if train_engine_type == "megatron" else None

    train_actors = [
        TrainClientActor.options(
            num_gpus=1,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=train_nodes[rank // NUM_GPU_PER_NODE]["NodeID"], soft=False
            ),
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
            num_gpus=1,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=gen_nodes[rank // NUM_GPU_PER_NODE]["NodeID"], soft=False
            ),
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

    # Each train client pushes to each PS worker
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
    start_time = time.time()
    futures = []
    for g in gen_actors:
        futures.append(g.pull_from_ps.remote(ps_agent_names, ps_for_pull_names))
    ray.get(futures)
    end_time = time.time()
    print(f"[PASS] gen pull from all ps done. time: {end_time - start_time}s")

    # Fetch and verify gen client state_dicts
    print("[CHECK] Verifying GenClientActor state_dicts are all ones...")
    gen_state_dicts = ray.get([g.get_state_dict.remote() for g in gen_actors])
    for gen_idx, state_dict in enumerate(gen_state_dicts):
        assert_state_dict_all_ones(state_dict, f"Gen client {gen_idx}")
        print(f"[PASS] Gen client {gen_idx} is all ones.")
    print("[PASS] All GenClientActor parameters are all ones.")

    # Shutdown all actors and Ray
    for t in train_actors:
        ray.get(t.shutdown.remote())
    for g in gen_actors:
        ray.get(g.shutdown.remote())
    ray.get(ps_wg.execute_all_async("shutdown"))
    ray.get(server.shutdown.remote())
    ray.shutdown()


if __name__ == "__main__":
    test_nixl_e2e()
