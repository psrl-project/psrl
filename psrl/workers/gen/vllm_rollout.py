import ray
from verl.workers.rollout.vllm_rollout import ServerAdapter


class PSRL_ServerAdapter(ServerAdapter):
    def get_node_id(self) -> str:
        """Get the node id of the vllm worker."""
        if not hasattr(self, "node_id"):
            self.node_id = None

        if self.node_id is not None:
            return self.node_id
        self.node_id = ray.get_runtime_context().get_node_id()
        return self.node_id

    def get_runtime_gpu_ids(self) -> list[int]:
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        raw_gpu_ids = accelerator_ids.get("GPU", accelerator_ids.get("NPU", []))
        return [int(gpu_id) for gpu_id in raw_gpu_ids]

    # _use_compilation_mode = False


# TODO(zyf): the following code is built in `AsyncLLM`(psrl/workers/gen/vllm_async_server.py),
#            need to check the correctness and implement the missing functions.

# import asyncio
# import inspect
# import json
# import logging
# import os
# import uuid
# from collections.abc import Sequence
# from contextlib import contextmanager
# from pprint import pprint
# from typing import Any, cast

# import numpy as np
# import torch
# import vllm
# import vllm.entrypoints.cli.serve
# from omegaconf import DictConfig, ListConfig, OmegaConf
# from ray.util.queue import Queue as RayQueue
# from tensordict import TensorDict
# from verl import DataProto
# from vllm import SamplingParams
# from vllm.config import CompilationConfig
# from vllm.engine.arg_utils import AsyncEngineArgs
# from vllm.inputs import PromptType, TokensPrompt
# from vllm.outputs import PoolingRequestOutput, RequestOutput
# from vllm.pooling_params import PoolingParams
# from vllm.sampling_params import RequestOutputKind
# from vllm.usage.usage_lib import UsageContext
# from vllm.utils.argparse_utils import FlexibleArgumentParser
# from vllm.v1.engine.async_llm import AsyncLLM
# from psrl.utils.dataset.utils import _pre_process_inputs
# from psrl.utils.logger import deprecated
# from psrl.utils.rollout.reprefill import (
#     collect_reprefill_log_probs,
#     collect_token_log_probs,
#     merge_reprefill_log_probs,
#     should_update_reprefill_log_probs,
# )
# from psrl.workers.config import HFModelConfig, RolloutConfig
# from psrl.workers.gen import StatCollector

# psrl_logger = logging.getLogger(__name__)
# psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# def _to_plain_container(value):
#     if OmegaConf.is_config(value):
#         return OmegaConf.to_container(value, resolve=True)
#     return value


# def _extending_rope_scaling(rope_scaling) -> dict | None:
#     """
#     Return a RoPE-scaling dict only when it actually extends the native window.

#     Transformers 5.x GLM configs expose a default ``rope_scaling`` /
#     ``rope_parameters`` with ``rope_type=default`` and no factor. That object is
#     truthy but must not hide launch-time ``hf_overrides``.
#     """
#     rope_scaling = _to_plain_container(rope_scaling)
#     if not isinstance(rope_scaling, dict) or not rope_scaling:
#         return None
#     factor = rope_scaling.get("factor")
#     if factor is not None and float(factor) > 1.0:
#         return rope_scaling
#     rope_type = str(rope_scaling.get("type") or rope_scaling.get("rope_type") or "default").lower()
#     if rope_type in ("", "default", "none"):
#         return None
#     return rope_scaling


# def _normalize_rope_override_dict(rope_scaling: dict) -> dict:
#     """Copy a RoPE override and make Transformers 5 / vLLM keys explicit."""
#     rope_scaling = dict(rope_scaling)
#     rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
#     if rope_type is not None:
#         rope_scaling["rope_type"] = rope_type
#     return rope_scaling


# _ROPE_LEAF_KEYS = frozenset(
#     {"rope_type", "type", "factor", "rope_theta", "original_max_position_embeddings"}
# )


# def _merge_rope_parameters(base, override: dict) -> dict:
#     """
#     Overlay a YaRN override onto the checkpoint ``rope_parameters``.

#     Keep existing fields such as ``rope_theta``. Nested-by-layer-type dicts
#     get the same overlay on every layer.
#     """
#     override = _normalize_rope_override_dict(override)
#     base = _to_plain_container(base)
#     if not isinstance(base, dict) or not base:
#         return override
#     is_nested = all(isinstance(value, dict) for value in base.values()) and not (
#         set(base) & _ROPE_LEAF_KEYS
#     )
#     if is_nested:
#         return {layer: {**layer_rp, **override} for layer, layer_rp in base.items()}
#     return {**base, **override}


# def _ensure_vllm_hf_rope_overrides(engine_kwargs: dict, model_hf_config) -> None:
#     """
#     Rewrite ``hf_overrides`` so vLLM actually applies launch-time YaRN.

#     vLLM ``ModelConfig`` applies dict-valued ``hf_overrides`` after
#     ``patch_rope_parameters``, and ``_get_and_verify_max_len`` reads
#     ``rope_parameters["rope_type"]``. GLM docs still use ``rope_scaling.type``.
#     """
#     hf_overrides = _to_plain_container(engine_kwargs.get("hf_overrides")) or {}
#     if not isinstance(hf_overrides, dict):
#         return
#     scaling = _extending_rope_scaling(hf_overrides.get("rope_scaling"))
#     if scaling is None:
#         scaling = _extending_rope_scaling(hf_overrides.get("rope_parameters"))
#     if scaling is None:
#         return
#     scaling = _normalize_rope_override_dict(scaling)
#     hf_overrides = dict(hf_overrides)
#     hf_overrides["rope_scaling"] = scaling
#     hf_overrides["rope_parameters"] = _merge_rope_parameters(
#         getattr(model_hf_config, "rope_parameters", None),
#         scaling,
#     )
#     engine_kwargs["hf_overrides"] = hf_overrides


# def _resolve_rope_scaling(model_hf_config, engine_kwargs: dict) -> dict | None:
#     """
#     Resolve RoPE scaling from vLLM ``hf_overrides`` first, then the HF config.

#     Disk ``config.json`` is the official GLM switch. Launch scripts can instead
#     pass ``engine_kwargs.vllm.hf_overrides.rope_scaling`` so a shared checkpoint
#     is not mutated.
#     """
#     hf_overrides = _to_plain_container(engine_kwargs.get("hf_overrides")) or {}
#     override_scaling = None
#     if isinstance(hf_overrides, dict):
#         override_scaling = _extending_rope_scaling(hf_overrides.get("rope_scaling"))
#         if override_scaling is None:
#             override_scaling = _extending_rope_scaling(hf_overrides.get("rope_parameters"))
#     if override_scaling is not None:
#         return override_scaling
#     return _extending_rope_scaling(
#         getattr(model_hf_config, "rope_scaling", None)
#         or getattr(model_hf_config, "rope_parameters", None)
#     )


# class PSRL_vLLMRollout:
#     def __init__(
#         self,
#         psrl_config: DictConfig,
#         config: RolloutConfig,
#         model_config: HFModelConfig,
#         is_reward_model: bool = False,
#         is_teacher_model: bool = False,
#         **kwargs,
#     ):
#         """
#         Initialize PSRL vLLM rollout with specified configuration.

#         Args:
#             model_path: Path to the model to load
#             config: Configuration object containing vLLM and PSRL settings
#             tokenizer: Tokenizer instance for the model
#             **kwargs: Additional keyword arguments (trust_remote_code, lora_kwargs, etc.)
#         """

#         super().__init__()
#         self.psrl_config = psrl_config
#         self.config = config
#         self.stat_collector = None
#         self.is_reward_model = is_reward_model
#         self.is_teacher_model = is_teacher_model
#         if self.is_reward_model:
#             self.reward_model_name = kwargs.get("reward_model_name")
#             self.rm_config = kwargs.get("rm_config")
#         else:
#             self.reward_model_name = None
#             self.rm_config = None

#         self.is_validate = kwargs.get("is_validate", False)

#         # Cached vLLM server initialization artifacts for HTTP serving.
#         self._server_args = None

#         tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
#         pipeline_parallel_size = config.get("pipeline_model_parallel_size", 1)
#         model_parallel_size = tensor_parallel_size * pipeline_parallel_size
#         expert_parallel_size = config.get("expert_parallel_size", 1)
#         enable_expert_parallel = expert_parallel_size > 1

#         enable_return_routed_experts = config.get("enable_rollout_routing_replay", False)

#         # For model parallel, we only run the inference engine on the first
#         # worker-group rank. This must align with upper-layer scheduling that
#         # uses execute_rank_zero_async()/workers[0] as the representative.
#         # Keep LOCAL_RANK as fallback for safety in non-standard env setup.
#         if model_parallel_size > 1:
#             print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
#             print(f"RANK: {os.environ.get('RANK')}")
#             print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
#             print(f"RAY_LOCAL_RANK: {os.environ.get('RAY_LOCAL_RANK')}")
#             print(f"RAY_WORLD_SIZE: {os.environ.get('RAY_WORLD_SIZE')}")
#             # owner_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
#             # if owner_rank != "0":

#             if os.environ.get("RANK") != "0":
#                 self.inference_engine = None
#                 return

#         model_path = model_config.local_path
#         tokenizer = model_config.tokenizer
#         model_hf_config = model_config.hf_config
#         trust_remote_code = model_config.trust_remote_code
#         lora_kwargs = (
#             {
#                 "enable_lora": True,
#                 "max_loras": 1,
#                 "max_lora_rank": model_config.lora_rank,
#             }
#             if model_config.lora_rank > 0
#             else {}
#         )

#         max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

#         # copy it to avoid secretly modifying the engine config
#         engine_kwargs = _to_plain_container(config.get("engine_kwargs", {}).get("vllm", {}) or {}) or {}
#         if not isinstance(engine_kwargs, dict):
#             engine_kwargs = {}
#         engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
#         _ensure_vllm_hf_rope_overrides(engine_kwargs, model_hf_config)

#         rope_scaling_config = _resolve_rope_scaling(model_hf_config, engine_kwargs)
#         if not rope_scaling_config:
#             max_position_embeddings = None
#             if hasattr(model_hf_config, "max_position_embeddings"):
#                 max_position_embeddings = model_hf_config.max_position_embeddings
#             elif hasattr(model_hf_config, "llm_config") and hasattr(
#                 model_hf_config.llm_config, "max_position_embeddings"
#             ):
#                 max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
#             elif hasattr(model_hf_config, "text_config") and hasattr(
#                 model_hf_config.text_config, "max_position_embeddings"
#             ):
#                 max_position_embeddings = model_hf_config.text_config.max_position_embeddings
#             if max_position_embeddings is None:
#                 raise ValueError("max_position_embeddings not found in model_hf_config")
#             assert max_position_embeddings >= config.prompt_length + config.response_length, (
#                 "model context length should be greater than total sequence length"
#             )
#         else:
#             # handle type where there's a length extend factor
#             # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
#             # for using yarn as an example
#             rope_scaling_factor = rope_scaling_config.get("factor", 1.0)
#             original_max_position_embeddings = rope_scaling_config.get(
#                 "original_max_position_embeddings",
#                 getattr(model_hf_config, "max_position_embeddings", None),
#             )
#             assert original_max_position_embeddings is not None, (
#                 "original_max_position_embeddings is required when rope_scaling is set."
#             )
#             assert (
#                 original_max_position_embeddings * rope_scaling_factor
#                 >= config.prompt_length + config.response_length
#             ), (
#                 "model context length should be greater than total sequence length, "
#                 + f"got rope_scaling_factor={rope_scaling_factor} and "
#                 + f"original_max_position_embeddings={original_max_position_embeddings}"
#             )

#         max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

#         if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
#             raise ValueError(
#                 "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
#                              please increase max_num_batched_tokens or disable chunked prefill"
#             )

#         # Load dummy format for meta init mode to save init time
#         load_format = (
#             "dummy"
#             if (config.load_format.startswith("dummy") or kwargs.get("init_mode", "full") == "empty")
#             else config.load_format
#         )

#         # LoRA configuration
#         lora_kwargs = (
#             {
#                 "enable_lora": True,
#                 "max_loras": 1,
#                 "max_lora_rank": model_config.lora_rank,
#             }
#             if model_config.lora_rank > 0
#             else {}
#         )
#         # For each vLLM engine parameter,
#         # - `None` means not setting it, so we pop it, and leave it to vLLM default value
#         #    (which can vary across different vLLM versions);
#         # - Otherwise it's the desired value we want to explicitly set.
#         weight_arena_config = psrl_config.nixl.get("weight_arena", {})
#         if OmegaConf.is_config(weight_arena_config):
#             weight_arena_config = OmegaConf.to_container(weight_arena_config, resolve=True)
#         else:
#             weight_arena_config = dict(weight_arena_config)
#         arena_role = "reward" if self.is_reward_model else "rollout"
#         arena_enabled_key = f"{arena_role}_enabled"
#         arena_materialization_key = f"{arena_role}_materialization"
#         if self.is_reward_model:
#             from psrl.utils.node_shared_weight_cache import validate_reward_cache_config

#             validate_reward_cache_config(weight_arena_config)
#         direct_arena = weight_arena_config.get(
#             arena_enabled_key, False
#         ) and weight_arena_config.get(arena_materialization_key, "direct") == "direct"
#         if direct_arena:
#             if load_format != "dummy":
#                 raise RuntimeError(
#                     f"Direct {arena_role} weight arena requires the dummy/empty initialization path, "
#                     f"got load_format={load_format!r}."
#                 )
#             if model_config.lora_rank > 0:
#                 raise RuntimeError(f"Direct {arena_role} weight arena does not support LoRA.")
#             if getattr(config, "quantization", None) is not None or engine_kwargs.get("quantization") is not None:
#                 raise RuntimeError(f"Direct {arena_role} weight arena currently supports unquantized models only.")
#             load_format = "psrl_arena_dummy"
#         if config.get("limit_images", None):  # support for multi-image data
#             engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

#         compilation_config = {}

#         cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
#         # enforce_eager must be False to use cudagraph
#         if not config.enforce_eager and cudagraph_capture_sizes:
#             if isinstance(cudagraph_capture_sizes, ListConfig):
#                 compilation_args = {"cudagraph_capture_sizes": cudagraph_capture_sizes}
#                 if _use_compilation_mode:
#                     compilation_args["mode"] = CompilationMode.VLLM_COMPILE
#                 else:
#                     compilation_args["level"] = CompilationLevel.PIECEWISE
#                 compilation_config["compilation_config"] = CompilationConfig(**compilation_args)
#             else:
#                 psrl_logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

#         if model_parallel_size > 1:
#             # Configure vLLM for tensor/pipeline parallelism within Ray
#             # Reset CUDA_VISIBLE_DEVICES to allow vLLM to manage GPU assignment
#             os.environ.pop("CUDA_VISIBLE_DEVICES", None)
#             distributed_executor_backend = "ray"
#         else:
#             distributed_executor_backend = None  # auto detect

#         runner = config.get("runner", "generate")
#         task = config.get("task", "generate")

#         llm_kwargs = dict(
#             enable_sleep_mode=True,
#             tensor_parallel_size=tensor_parallel_size,
#             pipeline_parallel_size=pipeline_parallel_size,
#             enable_expert_parallel=enable_expert_parallel,
#             distributed_executor_backend=distributed_executor_backend,
#             dtype=config.dtype,
#             enforce_eager=config.enforce_eager,
#             gpu_memory_utilization=config.gpu_memory_utilization,
#             disable_custom_all_reduce=True,
#             skip_tokenizer_init=False,
#             max_model_len=max_model_len,
#             # max_seq_len_to_capture=max_model_len, # deprecated
#             max_num_seqs=config.max_num_seqs,
#             load_format=load_format,
#             disable_log_stats=config.disable_log_stats,
#             max_num_batched_tokens=max_num_batched_tokens,
#             enable_chunked_prefill=config.enable_chunked_prefill,
#             enable_prefix_caching=config.enable_prefix_caching,
#             trust_remote_code=trust_remote_code,
#             logprobs_mode=config.logprobs_mode,
#             runner=runner,
#             task=task,
#             worker_extension_cls="psrl.workers.gen.vllm_extension.vLLMWorkerExtension",
#             seed=kwargs.get("seed", 0),
#             enable_return_routed_experts=enable_return_routed_experts,
#             **compilation_config,
#             **lora_kwargs,
#             **engine_kwargs,
#         )

#         # Support for pooling models (e.g., reward models)
#         self.is_pooling_model = (runner == "pooling")


#         if self.config.prometheus.enable:
#             assert self.psrl_config.server_rollout.enable, (
#                 "Prometheus monitoring requires server rollout to be enabled."
#             )

#             if self.config.prometheus.served_model_name:
#                 # Extract model name from path if it's a full path
#                 served_model_name = self.config.prometheus.served_model_name
#                 if "/" in served_model_name:
#                     # If it's a full path, extract the last part as model name
#                     served_model_name = served_model_name.split("/")[-1]
#                 llm_kwargs["served_model_name"] = served_model_name

#         if config.use_psrl_scheduler:
#             llm_kwargs["scheduler_cls"] = "psrl.workers.gen.rollout_scheduler.RolloutScheduler"
#         if self.is_reward_model and self.rm_config is not None:
#             rm_routing_strategy = self.rm_config.get("routing_strategy", {}) or {}
#             max_num_waiting_reqs_after_preemption = rm_routing_strategy.get(
#                 "max_num_waiting_reqs_after_preemption",
#                 psrl_config.routing_strategy.max_num_waiting_reqs_after_preemption,
#             )
#         else:
#             max_num_waiting_reqs_after_preemption = psrl_config.routing_strategy.max_num_waiting_reqs_after_preemption
#         llm_kwargs["additional_config"] = {
#             "max_num_waiting_reqs_after_preemption": max_num_waiting_reqs_after_preemption,
#             "max_model_len_used_in_estimation": max_model_len
#             * psrl_config.routing_strategy.max_estimated_concurrent_seqs_per_instance,
#             "psrl_nixl_weight_arena": weight_arena_config,
#             "psrl_role": "reward" if self.is_reward_model else "rollout",
#             "instance_id": kwargs.get("instance_id", 0),
#         }
#         psrl_logger.info(
#             "vLLM scheduler config: role=%s use_psrl_scheduler=%s scheduler_cls=%s "
#             "max_num_waiting_reqs_after_preemption=%s",
#             "reward" if self.is_reward_model else "rollout",
#             config.use_psrl_scheduler,
#             llm_kwargs.get("scheduler_cls"),
#             max_num_waiting_reqs_after_preemption,
#         )

#         # Initialize abort queue, events, and request ids
#         self.scheduler_abort_queue = RayQueue()
#         self.scheduler_abort_events = {}
#         self.scheduler_abort_requests = set()
#         self._scheduler_abort_processor_task = None

#         if self.psrl_config.server_rollout.enable:
#             # Build and cache OpenAI server args at engine init time.
#             # GenWorker will reuse these to start an in-process OpenAI-compatible server
#             # without re-parsing or drifting defaults.
#             server_args = self._build_server_args(model_path=model_path, args=llm_kwargs)
#             self._server_args = server_args

#             engine_args = AsyncEngineArgs.from_cli_args(server_args)
#             usage_context = UsageContext.OPENAI_API_SERVER
#             vllm_config = engine_args.create_engine_config(usage_context=usage_context)
#         else:
#             llm_kwargs["model"] = model_path
#             async_engine_init_params = inspect.signature(AsyncEngineArgs.__init__).parameters
#             supports_var_keyword = any(
#                 param.kind == inspect.Parameter.VAR_KEYWORD for param in async_engine_init_params.values()
#             )
#             if not supports_var_keyword:
#                 supported_keys = {
#                     key for key, param in async_engine_init_params.items() if key != "self"
#                 }
#                 unsupported_keys = sorted(set(llm_kwargs) - supported_keys)
#                 if unsupported_keys:
#                     psrl_logger.warning(
#                         "Drop unsupported AsyncEngineArgs kwargs for compatibility: %s.",
#                         unsupported_keys,
#                     )
#                     llm_kwargs = {key: value for key, value in llm_kwargs.items() if key in supported_keys}
#             engine_args = AsyncEngineArgs(**llm_kwargs)
#             usage_context = UsageContext.ENGINE_CONTEXT
#             vllm_config = engine_args.create_engine_config()

#             stat_loggers = None
#             if not config.disable_log_stats and psrl_config.status_collection.enable:
#                 psrl_logger.info(f"Enable status collection for rollout instance {kwargs.get('instance_id', 0)}")
#                 # Use custom stat loggers to collect engine stats
#                 status_queue = kwargs["status_queue"]
#                 self.stat_collector = StatCollector(
#                     vllm_config,
#                     psrl_config,
#                     instance_id=kwargs.get("instance_id", 0),
#                     is_reward_model=self.is_reward_model,
#                     reward_model_name=self.reward_model_name,
#                 )
#                 self.stat_collector.begin_record()
#                 self.stat_collector.init_output_queue(status_queue)
#                 self.stat_collector.init_scheduler_abort_queue(self.scheduler_abort_queue)
#                 self.stat_collector.record_model_version_update(0)
#                 stat_loggers = [self.stat_collector]
#             psrl_logger.info(f"Initialize AsyncLLM for rollout instance {kwargs.get('instance_id', 0)}")
#             self.inference_engine = AsyncLLM.from_vllm_config(
#                 vllm_config=vllm_config,
#                 usage_context=usage_context,
#                 stat_loggers=stat_loggers,
#             )

#         # NOTE(lhy): sleep mode is not supported when using NIXL
#         # Because it will cause illegal memory registration
#         """
#         # Offload vllm model to reduce peak memory usage
#         if load_format == "dummy" and config.free_cache_engine:
#             self.inference_engine.sleep(level=1)
#         """

#         # Initialize parameters based on model type
#         if self.is_pooling_model:
#             # Convert PoolingConfig to PoolingParams
#             # Handle both dict (OmegaConf DictConfig) and object types
#             pooling_config = config.pooling_config
#             # Use getattr for objects/DictConfig, dict.get() for regular dicts
#             if isinstance(pooling_config, dict) and not isinstance(pooling_config, DictConfig):
#                 normalize = pooling_config.get("normalize", False)
#                 use_activation = pooling_config.get("use_activation", False)
#                 task = pooling_config.get("task", "classify")
#             else:
#                 # DictConfig and objects support attribute access
#                 normalize = getattr(pooling_config, "normalize", False)
#                 use_activation = getattr(pooling_config, "use_activation", False)
#                 task = getattr(pooling_config, "task", "classify")
#             self.pooling_params = PoolingParams(
#                 normalize=normalize,
#                 use_activation=use_activation,
#                 task=task,
#             )
#             self.sampling_params = None
#             self._sampling_params_kwargs: dict[str, Any] = {}
#             psrl_logger.info(f"pooling_params: {self.pooling_params}")
#             print(f"pooling_params: {normalize=}, {use_activation=}, {task=}")
#             psrl_logger.info(f"Initialized PoolingParams for pooling model")
#         else:
#             # For generative models, use SamplingParams
#             kwargs = dict(
#                 n=1,
#                 logprobs=0,  # can be set to 0 and let actor to recompute
#                 max_tokens=config.response_length,
#                 repetition_penalty=config.get("repetition_penalty", 1.0),
#                 output_kind=RequestOutputKind.CUMULATIVE,
#             )

#             # we may detokenize the result all together later
#             kwargs["detokenize"] = False

#             # supporting adding any sampling params from the config file
#             for k in config.keys():
#                 if hasattr(SamplingParams(), str(k)) and k != "seed" and k != "n":
#                     kwargs[k] = config.get(k)

#             # GenRM sampling belongs to the reward-model configuration rather
#             # than its rollout config. Apply it while constructing the vLLM
#             # default too, so callers without per-request params still use it.
#             if self.is_reward_model and self.rm_config is not None:
#                 sampling_config = self.rm_config.get("sampling_config", None)
#                 if sampling_config is not None:
#                     sampling_config = OmegaConf.to_container(sampling_config, resolve=True)
#                     if isinstance(sampling_config, dict):
#                         for key, value in sampling_config.items():
#                             if (
#                                 value is not None
#                                 and key != "n"
#                                 and hasattr(SamplingParams(), str(key))
#                             ):
#                                 kwargs[key] = value
#             kwargs["n"] = 1  # already repeat in ray_trainer
#             psrl_logger.info(f"kwargs: {kwargs}")
#             self._sampling_params_kwargs = dict(kwargs)
#             self.sampling_params = SamplingParams(**kwargs)
#             self.pooling_params = None

#         self.pad_token_id = tokenizer.pad_token_id

#         # Start abort processor task
#         try:
#             loop = asyncio.get_event_loop()
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#         self._scheduler_abort_processor_task = loop.create_task(self._scheduler_abort_processor_loop())
#         self._scheduler_abort_processor_task.add_done_callback(
#             lambda f: f.result()
#         )  # To avoid silent error in async tasks

#     def _build_server_args(self, model_path: str, args: dict[str, Any]):
#         """Build a CLI-like args Namespace compatible with vLLM OpenAI server.

#         Args:
#             model_path: Path to the model to load
#             args: Dictionary of vLLM engine arguments
#         """
#         server_args = ["serve", model_path]
#         for k, v in args.items():
#             if isinstance(v, bool):
#                 if v:
#                     server_args.append(f"--{k}")
#             elif v is not None:
#                 server_args.append(f"--{k}")
#                 # Use json.dumps for dict to ensure valid JSON format
#                 server_args.append(json.dumps(v) if isinstance(v, dict) else str(v))

#         pprint(server_args)

#         parser = FlexibleArgumentParser(description="vLLM CLI")
#         subparsers = parser.add_subparsers(required=False, dest="subparser")
#         cmds = {}
#         for cmd in vllm.entrypoints.cli.serve.cmd_init():
#             cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
#             cmds[cmd.name] = cmd

#         args = parser.parse_args(args=server_args)
#         args.model = getattr(args, "model_tag", model_path)
#         if getattr(args, "subparser", None) in cmds:
#             cmds[args.subparser].validate(args)
#         return args

#     @property
#     def server_args(self):
#         """Get the cached vLLM OpenAI server args."""
#         return self._server_args

#     async def _scheduler_abort_processor_loop(self):
#         """Background loop that processes abort requests from the queue."""
#         while True:
#             # Wait for abort request from queue using async method (blocking wait)
#             request_ids = await self.scheduler_abort_queue.get_async(block=True)
#             if request_ids is None:  # Sentinel value to stop the loop
#                 break

#             # Create a new event for this abort request
#             for request_id in request_ids:
#                 self.scheduler_abort_requests.add(request_id)
#             request_ids_tuple = tuple(request_ids)
#             self.scheduler_abort_events[request_ids_tuple] = asyncio.Event()

#             # Process the abort request
#             # SchedulerStats carries vLLM's internal request IDs.
#             await self.inference_engine.abort(request_ids, internal=True)

#             # Signal that this abort request is done and clean up
#             self.scheduler_abort_events[request_ids_tuple].set()
#             del self.scheduler_abort_events[request_ids_tuple]

#     async def _wait_for_all_scheduler_abort_requests_processed(self):
#         """Wait until the abort queue is empty and all pending abort requests are processed."""
#         if self.scheduler_abort_queue is None:
#             return
#         # Wait until queue is empty
#         while not self.scheduler_abort_queue.empty():
#             await asyncio.sleep(0)

#         # Wait for all pending abort requests to complete
#         if self.scheduler_abort_events:
#             events = list(self.scheduler_abort_events.values())
#             await asyncio.gather(*[event.wait() for event in events], return_exceptions=True)

#     def pre_process_inputs(self, prompts: DataProto) -> PromptType | Sequence[PromptType]:
#         """
#         Pre-process prompts to convert them into vLLM-compatible inputs.

#         This method performs several transformations:
#         1. Remove left padding from prompt token IDs
#         2. Concatenate raw prompt IDs and response IDs for continuation
#         3. Add multi-modal data if present (images, etc.)
#         4. Configure sampling parameters based on mode (sampling/validation/greedy)

#         Args:
#             prompts: DataProto containing input prompts and metadata

#         Returns:
#             vllm_inputs ready for vLLM generation
#         """

#         batch_size = len(prompts)

#         non_tensor_batch = prompts.non_tensor_batch
#         if "raw_prompt_ids" not in non_tensor_batch:
#             idx = prompts.batch["input_ids"]  # (bs, prompt_length)
#             # Remove the left padding in the prompt token_id
#             non_tensor_batch["raw_prompt_ids"] = np.array(
#                 [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
#                 dtype=object,
#             )

#         if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
#             raise RuntimeError(
#                 f"vllm sharding manager is not work properly with "
#                 f"{batch_size=} v.s. {len(non_tensor_batch['raw_prompt_ids'])=}."
#             )

#         if isinstance(non_tensor_batch["raw_prompt_ids"], np.ndarray):
#             raw_prompt_ids = non_tensor_batch["raw_prompt_ids"].tolist()
#         else:
#             raw_prompt_ids = non_tensor_batch["raw_prompt_ids"]

#         if "raw_response_ids" in non_tensor_batch:
#             raw_response_ids = non_tensor_batch["raw_response_ids"]
#             raw_response_ids = np.fromiter(raw_response_ids.tolist(), dtype=object)
#         else:
#             raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

#         if isinstance(raw_response_ids, np.ndarray):
#             raw_response_ids = raw_response_ids.tolist()

#         if "multi_modal_data" in non_tensor_batch:
#             vllm_inputs = []
#             for raw_prompt_ids_, raw_response_ids_, multi_modal_data in zip(
#                 raw_prompt_ids, raw_response_ids, non_tensor_batch["multi_modal_data"]
#             ):
#                 vllm_inputs.append(
#                     {
#                         "prompt_token_ids": raw_prompt_ids_ + raw_response_ids_,
#                         "multi_modal_data": multi_modal_data,
#                     }
#                 )
#         else:
#             vllm_inputs = [
#                 {"prompt_token_ids": raw_prompt_ids_ + raw_response_ids_}
#                 for raw_prompt_ids_, raw_response_ids_ in zip(raw_prompt_ids, raw_response_ids)
#             ]

#         # ensure the type of `prompt_token_ids` passed to vllm is list[int]
#         # https://github.com/volcengine/verl/pull/772
#         for input_data in vllm_inputs:
#             if isinstance(input_data["prompt_token_ids"], np.ndarray):
#                 input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
#             elif not isinstance(input_data["prompt_token_ids"], list):
#                 raise TypeError(
#                     f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
#                 )

#         return vllm_inputs

#     def post_process_outputs(
#         self,
#         prompts: DataProto,
#         outputs: (RequestOutput | PoolingRequestOutput | list[RequestOutput | PoolingRequestOutput]),
#     ) -> DataProto:
#         """
#         Post-process vLLM outputs to convert them back into DataProto format.

#         This method performs several transformations:
#         1. Extract response token IDs, lengths, and interruption status
#         2. Collect log probabilities if required
#         3. Concatenate new response IDs to existing raw response IDs
#         4. Build final DataProto with updated tensors and metadata

#         Args:
#             prompts: Original input DataProto containing prompts
#             outputs: vLLM generation outputs (single output or list of outputs)

#         Returns:
#             Updated DataProto with generated responses and proper formatting
#         """

#         if not isinstance(outputs, list):
#             outputs = [outputs]
#         assert len(outputs) == len(prompts), "Mismatched batch size between prompts and VLLM outputs."

#         batch_size = len(prompts)
#         non_tensor_batch = prompts.non_tensor_batch
#         uid_list = non_tensor_batch["uid"].tolist()

#         meta_info = prompts.meta_info
#         update_reprefill_log_probs = should_update_reprefill_log_probs(
#             self.psrl_config.log_prob.get("update_reprefill_log_probs", False),
#             is_reward_model=self.is_reward_model,
#             is_teacher_model=self.is_teacher_model,
#         )

#         # Keep the response ids that were present before this continuation. They
#         # are used to verify that prompt logprobs really correspond to the
#         # prefix we are about to replace.
#         prior_raw_response_ids = non_tensor_batch.get("raw_response_ids")
#         if prior_raw_response_ids is None:
#             prior_raw_response_ids = [[] for _ in range(batch_size)]
#         elif isinstance(prior_raw_response_ids, np.ndarray):
#             prior_raw_response_ids = prior_raw_response_ids.tolist()
#         else:
#             prior_raw_response_ids = list(prior_raw_response_ids)

#         response_ids_list = []
#         response_len_list = []
#         interrupted_list = []
#         interrupted_by_scheduler_list = []
#         pooling_output_list = []
#         all_log_prob_list = []
#         replace_reprefill_log_prob_list = [False] * batch_size
#         all_teacher_id_list = []
#         routed_experts_list = []

#         metrics_list = []

#         for i, uid in enumerate(uid_list):
#             sample_idx = i
#             vllm_output = outputs[i]
#             try:
#                 metrics_list.append(vllm_output.metrics)
#             except Exception as e:
#                 psrl_logger.warning(f"Failed to get metrics for request {uid}: {e}")
#                 metrics_list.append({})
#             if self.is_pooling_model:
#                 # For pooling models, outputs is a PoolingOutput object, not a list
#                 # Extract pooling output data, PoolingOutput has a data attribute (torch.Tensor)
#                 pooling_output_value = vllm_output.outputs.data
#                 pooling_output_list.append(pooling_output_value)

#                 # Pooling models don't generate tokens, so use empty response
#                 response_ids = []
#                 response_len = 0
#                 # TODO(zyf): need to check the finish_reason of the pooling model
#                 # interrupted = vllm_output.outputs[0].finish_reason == "abort"
#                 interrupted = False
#             else:
#                 assert len(vllm_output.outputs) == 1, "RolloutRouter only supports single request generation."
#                 response_ids = vllm_output.outputs[0].token_ids
#                 response_len = len(response_ids)
#                 interrupted = vllm_output.outputs[0].finish_reason == "abort"

#             response_ids_list.append(response_ids)
#             response_len_list.append(response_len)
#             interrupted_list.append(interrupted)
#             if str(uid) in self.scheduler_abort_requests:
#                 assert interrupted, "Requests interrupted by the scheduler should also have finish_reason 'abort'."
#                 interrupted_by_scheduler_list.append(True)
#                 self.scheduler_abort_requests.remove(str(uid))
#             else:
#                 psrl_logger.debug(
#                     f"Request {uid} is not interrupted by the scheduler (not in {self.scheduler_abort_requests}). "
#                     f"It is interrupted by the synchronization (i.e., partial rollout)."
#                 )
#                 interrupted_by_scheduler_list.append(False)

#             log_prob_list = []
#             teacher_id_list = []
#             sampling_params = {}
#             if "sampling_params" in non_tensor_batch:
#                 sampling_params = non_tensor_batch["sampling_params"][i]
#             teacher_topk = int(sampling_params.get("prompt_logprobs", 0) or 0)
#             # if inference logprobs is required, we need to collect the log probabilities
#             if (
#                 not self.is_pooling_model
#                 and not self.is_teacher_model
#                 and self.psrl_config.log_prob.enable_rollout_engine_log_prob
#                 and hasattr(vllm_output.outputs[0], "logprobs")
#                 and vllm_output.outputs[0].logprobs is not None
#             ):
#                 if update_reprefill_log_probs:
#                     response_lengths = non_tensor_batch.get("response_unpadded_len", [0] * batch_size)
#                     if np.isscalar(response_lengths):
#                         curr_response_len = int(response_lengths)
#                     else:
#                         curr_response_len = int(response_lengths[sample_idx])
#                     log_prob_list, replace_reprefill_log_prob_list[sample_idx] = collect_reprefill_log_probs(
#                         prior_raw_response_ids[sample_idx],
#                         vllm_output.prompt_token_ids,
#                         vllm_output.prompt_logprobs,
#                         response_ids,
#                         vllm_output.outputs[0].logprobs,
#                         curr_response_len,
#                     )
#                 elif self.psrl_config.partial_rollout.interrupt_as_prompt:
#                     response_lengths = non_tensor_batch.get("response_unpadded_len", [0] * batch_size)
#                     if np.isscalar(response_lengths):
#                         curr_response_len = int(response_lengths)
#                     else:
#                         curr_response_len = int(response_lengths[sample_idx])
#                     # Preserve the legacy interrupt-as-prompt behavior: only a
#                     # completed continuation carries the full response logprobs.
#                     if not interrupted and curr_response_len > 0 and vllm_output.prompt_logprobs is not None:
#                         prompt_token_ids = vllm_output.prompt_token_ids
#                         prefix_token_ids = prompt_token_ids[-curr_response_len:]
#                         prefix_logprobs = vllm_output.prompt_logprobs[-curr_response_len:]
#                         for token_id, logprob in zip(prefix_token_ids, prefix_logprobs, strict=False):
#                             if logprob is None:
#                                 break
#                             try:
#                                 log_prob_list.append(logprob[token_id].logprob)
#                             except (KeyError, TypeError):
#                                 log_prob_list.append(logprob[str(token_id)].logprob)
#                         log_prob_list.extend(
#                             collect_token_log_probs(response_ids, vllm_output.outputs[0].logprobs)
#                         )
#                 else:
#                     log_prob_list.extend(collect_token_log_probs(response_ids, vllm_output.outputs[0].logprobs))
#             elif (
#                 not self.is_pooling_model
#                 and self.is_teacher_model
#                 and hasattr(vllm_output, "prompt_logprobs")
#                 and vllm_output.prompt_logprobs is not None
#             ):
#                 response_unpadded_len = int(non_tensor_batch.get("response_unpadded_len", [0])[i])
#                 prompt_token_ids = vllm_output.prompt_token_ids
#                 if response_unpadded_len > 0:
#                     prompt_logprobs = vllm_output.prompt_logprobs[-response_unpadded_len:]
#                     prompt_token_ids = prompt_token_ids[-response_unpadded_len:]
#                 else:
#                     prompt_logprobs = []
#                     prompt_token_ids = []
#                 for token_id, logprob_dict in zip(prompt_token_ids, prompt_logprobs, strict=False):
#                     if logprob_dict is None:
#                         log_prob_list.append([float("nan")] * teacher_topk if teacher_topk > 0 else float("nan"))
#                         teacher_id_list.append([])
#                         continue

#                     topk_entries = sorted(
#                         logprob_dict.items(),
#                         key=lambda item: getattr(item[1], "rank", None)
#                         if getattr(item[1], "rank", None) is not None
#                         else -float(item[1].logprob),
#                     )
#                     if teacher_topk > 0:
#                         topk_entries = topk_entries[:teacher_topk]
#                         log_prob_list.append([float(entry.logprob) for _, entry in topk_entries])
#                         teacher_id_list.append([int(token_id) for token_id, _ in topk_entries])
#                     elif token_id in logprob_dict:
#                         log_prob_list.append(float(logprob_dict[token_id].logprob))
#                     else:
#                         # Some vLLM versions may stringify token ids in prompt_logprobs.
#                         log_prob_list.append(float(logprob_dict[str(token_id)].logprob))
#             all_log_prob_list.append(log_prob_list)
#             all_teacher_id_list.append(teacher_id_list)

#             routed_experts = None
#             if self.config.enable_rollout_routing_replay:
#                 routed_experts = vllm_output.outputs[0].routed_experts
#             routed_experts_list.append(routed_experts)

#         # Consolidate batch results
#         if "raw_response_ids" in non_tensor_batch:
#             raw_response_ids = non_tensor_batch.pop("raw_response_ids")
#             raw_response_ids = np.fromiter(raw_response_ids.tolist(), dtype=object)
#         else:
#             raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

#         raw_response_ids = raw_response_ids + np.fromiter(response_ids_list, dtype=object)
#         non_tensor_batch["raw_response_ids"] = raw_response_ids

#         if "response_unpadded_len" in non_tensor_batch:
#             curr_response_unpadded_len = non_tensor_batch["response_unpadded_len"]
#         else:
#             curr_response_unpadded_len = [0] * batch_size
#         response_unpadded_len = [curr_response_unpadded_len[i] + response_len_list[i] for i in range(batch_size)]
#         non_tensor_batch["response_unpadded_len"] = np.array(response_unpadded_len, dtype=int)
#         non_tensor_batch["interrupted"] = np.array(interrupted_list, dtype=bool)
#         non_tensor_batch["interrupted_by_scheduler"] = np.array(interrupted_by_scheduler_list, dtype=bool)

#         meta_info["vllm_metrics"] = np.array(metrics_list, dtype=object)
#         if update_reprefill_log_probs:
#             non_tensor_batch["reprefill_log_prob_updated"] = np.array(
#                 replace_reprefill_log_prob_list,
#                 dtype=bool,
#             )
#         # meta_info["metrics"] = metrics_list

#         # Update rollout_log_probs
#         if not self.is_pooling_model:
#             if self.psrl_config.log_prob.enable_rollout_engine_log_prob:
#                 if "rollout_log_probs" in non_tensor_batch:
#                     curr_rollout_log_probs = non_tensor_batch.pop("rollout_log_probs")
#                     curr_rollout_log_probs = np.fromiter(curr_rollout_log_probs.tolist(), dtype=object)
#                 else:
#                     curr_rollout_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
#                 for i in range(batch_size):
#                     curr_rollout_log_probs[i] = merge_reprefill_log_probs(
#                         curr_rollout_log_probs[i],
#                         all_log_prob_list[i],
#                         replace_previous=replace_reprefill_log_prob_list[i],
#                     )
#                     if (
#                         update_reprefill_log_probs
#                         and len(curr_rollout_log_probs[i]) != response_unpadded_len[i]
#                     ):
#                         raise ValueError(
#                             "Rollout logprob length mismatch after reprefill for request "
#                             f"{uid_list[i]}: got {len(curr_rollout_log_probs[i])} logprobs for "
#                             f"{response_unpadded_len[i]} response tokens"
#                         )
#                 non_tensor_batch["rollout_log_probs"] = curr_rollout_log_probs
#             elif self.is_teacher_model:
#                 if "teacher_log_probs" in non_tensor_batch:
#                     curr_teacher_log_probs = non_tensor_batch.pop("teacher_log_probs")
#                     curr_teacher_log_probs = np.fromiter(curr_teacher_log_probs.tolist(), dtype=object)
#                 else:
#                     curr_teacher_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
#                 curr_teacher_log_probs += np.fromiter(all_log_prob_list, dtype=object)
#                 non_tensor_batch["teacher_log_probs"] = curr_teacher_log_probs
#                 if any(len(ids) > 0 for ids in all_teacher_id_list):
#                     if "teacher_ids" in non_tensor_batch:
#                         curr_teacher_ids = non_tensor_batch.pop("teacher_ids")
#                         curr_teacher_ids = np.fromiter(curr_teacher_ids.tolist(), dtype=object)
#                     else:
#                         curr_teacher_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)
#                     curr_teacher_ids += np.fromiter(all_teacher_id_list, dtype=object)
#                     non_tensor_batch["teacher_ids"] = curr_teacher_ids

#         # process routed experts
#         if self.config.enable_rollout_routing_replay:
#             non_tensor_batch["routed_experts"] = np.fromiter(routed_experts_list, dtype=object)

#         if len(pooling_output_list) > 0:
#             non_tensor_batch["pooling_output"] = np.fromiter(pooling_output_list, dtype=object)

#         batch = TensorDict(
#             {
#                 "input_ids": prompts.batch["input_ids"],  # [bs, prompt_length]
#             },
#             batch_size=batch_size,
#         )
#         return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

#     @deprecated("vllm_rollout.add_requests is not used.")
#     def add_requests(self, prompts: DataProto, sampling_params: dict[str, Any]):
#         """
#         Add generation requests to the vLLM inference engine.
#         This method converts prompts to vLLM format and queues them for generation.

#         Args:
#             prompts: DataProto containing input prompts
#             sampling_params: Sampling parameters for generation
#         """
#         vllm_inputs = self.pre_process_inputs(prompts)
#         parsed_vllm_inputs = cast(PromptType | Sequence[PromptType], vllm_inputs)
#         if isinstance(parsed_vllm_inputs, (str, dict)):
#             # Convert a single prompt to a list.
#             parsed_vllm_inputs = [parsed_vllm_inputs]

#         sampling_params = SamplingParams(**sampling_params)
#         for prompt in parsed_vllm_inputs:
#             request_id = str(self.get_next_request_id())
#             self.inference_engine.llm_engine.add_request(
#                 request_id,
#                 prompt,
#                 sampling_params,
#                 priority=0,
#             )

#     @torch.no_grad()
#     async def generate_sequences_async(self, prompts: DataProto, sampling_params: dict[str, Any]) -> DataProto:
#         """
#         Generate sequences from prompts using asynchronous vLLM generation.

#         This method enables concurrent generation of multiple sequences with:
#         1. Async task creation for each prompt
#         2. Independent sampling parameter customization per prompt
#         3. Support for partial rollout (continuation from previous responses)
#         4. Efficient concurrent processing with asyncio

#         Args:
#             prompts: DataProto containing input prompts with required 'uid' field
#             sampling_params: Sampling parameters for generation

#         Returns:
#             DataProto with concatenated results from all async generations
#         """
#         vllm_inputs = self.pre_process_inputs(prompts)
#         sample_ids = prompts.non_tensor_batch.get("uid", None)
#         curr_response_unpadded_len = prompts.non_tensor_batch.get("response_unpadded_len", [0] * len(vllm_inputs))
#         assert sample_ids is not None, "sample_ids must be provided in the prompts.non_tensor_batch"

#         # For pooling models, use encode method with PoolingParams
#         if self.is_pooling_model:
#             tasks = []
#             for prompt_idx, (vllm_input, sample_id) in enumerate(zip(vllm_inputs, sample_ids)):
#                 if isinstance(vllm_input, dict):
#                     prompt = TokensPrompt(**vllm_input)
#                 else:
#                     prompt = vllm_input
#                 tasks.append(
#                     self.encode_sequence_task(
#                         prompt_idx,
#                         prompt,
#                         pooling_params=self.pooling_params,
#                         uid=str(sample_id),
#                     )
#                 )

#             completed_rollout = []
#             for completed_task in asyncio.as_completed(tasks):
#                 prompt_idx, output = await completed_task
#                 completed_rollout.append(self.post_process_outputs(prompts[prompt_idx : prompt_idx + 1], output))

#             return DataProto.concat(completed_rollout)
#         else:
#             # users can customize different sampling_params at different run

#             tasks = []
#             for prompt_idx, (vllm_input, sample_id, curr_response_len) in enumerate(
#                 zip(vllm_inputs, sample_ids, curr_response_unpadded_len)
#             ):
#                 # Each task should own its own SamplingParams because max_tokens differs per prompt.
#                 task_sampling_params = SamplingParams(**sampling_params)
#                 max_tokens = None if self.is_teacher_model else self.config.response_length - curr_response_len
#                 tasks.append(
#                     self.generate_sequence_task(
#                         prompt_idx,
#                         vllm_input,
#                         sampling_params=task_sampling_params,
#                         uid=str(sample_id),
#                         max_tokens=max_tokens,
#                     )
#                 )

#             completed_rollout = []
#             for completed_task in asyncio.as_completed(tasks):
#                 prompt_idx, output = await completed_task
#                 completed_rollout.append(self.post_process_outputs(prompts[prompt_idx : prompt_idx + 1], output))

#             return DataProto.concat(completed_rollout)


#     @torch.no_grad()
#     async def raw_generate_sequences_async(self, prompts: DataProto, sampling_params: dict[str, Any]):
#         """Generate sequences from the prompts using vLLM asynchronously without post-processing."""
#         vllm_inputs = self.pre_process_inputs(prompts)
#         sample_ids = prompts.non_tensor_batch.get("uid", None)
#         curr_response_unpadded_len = prompts.non_tensor_batch.get("response_unpadded_len", [0] * len(vllm_inputs))
#         assert sample_ids is not None, "sample_ids must be provided in the prompts.non_tensor_batch"

#         # users can customize different sampling_params at different run
#         tasks = []
#         for prompt_idx, (vllm_input, sample_id, curr_response_len) in enumerate(
#             zip(vllm_inputs, sample_ids, curr_response_unpadded_len)
#         ):
#             task_sampling_params = SamplingParams(**sampling_params)
#             max_tokens = None if self.is_teacher_model else self.config.response_length - curr_response_len
#             tasks.append(
#                 self.generate_sequence_task(
#                     prompt_idx,
#                     vllm_input,
#                     sampling_params=task_sampling_params,
#                     uid=str(sample_id),
#                     max_tokens=max_tokens,
#                 )
#             )

#         vllm_outputs = []
#         for completed_task in asyncio.as_completed(tasks):
#             output = await completed_task
#             vllm_outputs.append(output)

#         return vllm_outputs

#     async def generate_sequence_task(
#         self,
#         idx: int,
#         prompt_tokens: dict[str, Any] | list[int],
#         sampling_params: SamplingParams,
#         uid: str | None = None,
#         max_tokens: int | None = None,
#     ) -> tuple[int, RequestOutput]:
#         """
#         Generate a single sequence asynchronously using vLLM.

#         This method creates an async generation task for a single prompt and
#         waits for completion, returning the final output.

#         Args:
#             idx: Index of the prompt in the batch
#             prompt_tokens: Either token IDs list or dict with prompt data
#             sampling_params: Sampling parameters for generation
#             uid: Unique identifier for the request (optional)

#         Returns:
#             Tuple of (prompt_idx, final_request_output)
#             prompt_idx is the index of the prompt in the batch
#         """
#         # Ensure all abort requests in the queue are processed before starting generation
#         # NOTE(lhy): currently, only the preempted requests are put into the abort queue.
#         # Other requests are aborted (e.g., partial rollout) directly by the scheduler.
#         if self.scheduler_abort_queue is not None:
#             await self._wait_for_all_scheduler_abort_requests_processed()

#         if isinstance(prompt_tokens, list):
#             prompt_tokens = {"prompt_token_ids": prompt_tokens}
#         if max_tokens is not None:
#             sampling_params.max_tokens = int(max_tokens)

#         request_id = str(uuid.uuid4()) if uid is None else uid
#         task = self.inference_engine.generate(
#             prompt=TokensPrompt(**prompt_tokens),
#             sampling_params=sampling_params,
#             request_id=request_id,
#         )
#         async for output in task:
#             last_output = output
#         return idx, last_output

#     async def encode_sequence_task(
#         self,
#         idx: int,
#         prompt: PromptType,
#         pooling_params: PoolingParams,
#         uid: str | None = None,
#     ) -> tuple[int, PoolingRequestOutput]:
#         """
#         Encode a single sequence asynchronously using vLLM for pooling models.

#         This method creates an async encoding task for a single prompt and
#         waits for completion, returning the final output.

#         Args:
#             idx: Index of the prompt in the batch
#             prompt: Prompt input (TokensPrompt or other PromptType)
#             pooling_params: Pooling parameters for encoding
#             uid: Unique identifier for the request (optional)

#         Returns:
#             Tuple of (prompt_idx, final_pooling_request_output)
#             prompt_idx is the index of the prompt in the batch
#         """
#         # Ensure all abort requests in the queue are processed before starting encoding
#         if self.scheduler_abort_queue is not None:
#             await self._wait_for_all_scheduler_abort_requests_processed()

#         request_id = str(uuid.uuid4()) if uid is None else uid
#         task = self.inference_engine.encode(
#             prompt=prompt,
#             pooling_params=pooling_params,
#             request_id=request_id,
#         )
#         async for output in task:
#             last_output = output
#         return idx, last_output

#     async def interrupt_all_requests_async(self) -> int:
#         """
#         Interrupt all requests (both running and waiting) asynchronously.

#         This method aborts all currently queued and executing requests in the
#         vLLM engine, useful for emergency stops or model updates.

#         Returns:
#             Number of requests that were interrupted
#         """
#         interrupted_request_num = await self.inference_engine.abort_all()
#         if interrupted_request_num > 0:
#             psrl_logger.debug(f"Interrupted {interrupted_request_num} requests via abort_all().")
#         else:
#             psrl_logger.debug("No requests to interrupt via abort_all().")
#         return interrupted_request_num

#     async def interrupt_requests_async(self, request_ids) -> int:
#         """
#         Interrupt specific requests by their IDs asynchronously.

#         This method aborts only the specified requests, allowing selective
#         interruption based on staleness or other criteria.

#         Args:
#             request_ids: List of request IDs to interrupt

#         Returns:
#             Number of requests removed from the vLLM output processor.
#         """
#         if len(request_ids) > 0:
#             request_ids = [str(request_id) for request_id in request_ids]
#             # GenWorker passes vLLM scheduler IDs (the internal IDs).  Using
#             # the default external-ID lookup silently misses randomized IDs.
#             return int(await self.inference_engine.abort(request_ids, internal=True) or 0)
#         return 0
