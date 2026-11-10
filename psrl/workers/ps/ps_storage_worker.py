import gc
import json
import os
from dataclasses import dataclass

import ray
import torch
from accelerate import init_empty_weights
from omegaconf import DictConfig
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from verl.utils.fs import copy_to_local

from psrl.utils.common.nixl_metadata import ClientInfoUpdates
from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import ps_agent_name, ps_client_pull_name, ps_client_push_name
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.hf_converter import convert_hf_inplace, maybe_convert_to_smaller_parts
from psrl.utils.converter.model_dtypes import fix_meta_model_dtypes
from psrl.utils.logger import get_ps_logger, get_worker_info, setup_ps_logger
from psrl.utils.nixl import (
    NIXLClientType,
    NIXLMultiStorageClients,
    fingerprint_log_record,
    resolve_weight_fingerprint_options,
)

psrl_logger = get_ps_logger()


# Fallback fp32 patterns for model classes that lack `_keep_in_fp32_modules_strict`.
FP32_PATTERNS: dict[str, list[str]] = {
    # `A_log` must remain float32 for stable Qwen3.5 and Qwen3-Next GDN recurrence.
    "Qwen3_5": ["A_log"],
}


# TODO(lhy): Support configurable PS storage redundancy.
@dataclass
class PSStoragePlan:
    train_model_dtype: torch.dtype
    gen_model_dtype: torch.dtype
    storage_style: str = "hf"  # ["hf", "hsdp"]
    hsdp_pattern: dict[str, int] | None = None  # e.g. {"replicate": 1, "fully_shard": 2}

    def __post_init__(self):
        if self.storage_style == "hsdp":
            assert self.hsdp_pattern is not None, "hsdp_pattern is required."
            assert all(isinstance(value, int) for value in self.hsdp_pattern.values()), (
                "hsdp_pattern values must be integers."
            )
            assert all(value >= 0 for value in self.hsdp_pattern.values()), "hsdp_pattern values must be non-negative."

    def train_gen_model_share(self) -> bool:
        return self.train_model_dtype == self.gen_model_dtype


class PSStorageWorker:
    """A worker that only stores the data and uses NIXL to communicate."""

    def __init__(
        self,
        storage_plan: PSStoragePlan,
        model_config: DictConfig,
        psrl_config: DictConfig,
    ) -> None:
        self.storage_plan = storage_plan
        self.model_config = model_config
        self.psrl_config = psrl_config
        self.train_meta_hf_model: torch.nn.Module | None = None
        self.gen_meta_hf_model: torch.nn.Module | None = None

        # Tied checkpoint aliases are cached while the meta model still exposes them.
        self._tied_weights_alias_map: dict[str, list[str]] = {}

        # Cache for non-persistent named buffers (e.g. inv_freq), populated lazily.
        self._cached_non_persistent_buffers: dict[str, torch.Tensor] | None = None

        self.nixl_multi_storage_clients = None

        self.rank = int(os.environ.get("RANK"))
        self.log_prefix = f"PSStorageWorker_R{self.rank}"
        setup_ps_logger(self.psrl_config.logging_path, self.log_prefix)
        psrl_logger.info(f"Initialized on {get_worker_info()}.")

    def get_replica_id(self) -> int:
        """
        Get the replica id (dp id) of the storage worker.
        """
        if self.storage_plan.storage_style == "hf":
            return self.rank
        elif self.storage_plan.storage_style == "hsdp":
            return self.rank // self.storage_plan.hsdp_pattern["fully_shard"]
        else:
            raise ValueError(f"Invalid storage style: {self.storage_plan.storage_style}")

    def init_nixl_client(self):
        """Initialize the NIXL client."""
        # NOTE(lhy): Initialize NIXL before the actor module to improve UCX 1.18.0 communication performance.
        self.use_gpu = self.psrl_config.ps_mode == "nixl_gpu"
        # TODO(lhy): Support separate PS modes for training and generation.
        self.agent_name = ps_agent_name(self.rank)
        self.client_for_push_name = ps_client_push_name(self.rank)
        self.client_for_pull_name = ps_client_pull_name(self.rank)
        self.nixl_multi_storage_clients = NIXLMultiStorageClients(
            agent_name=self.agent_name,
            multi_client_names=[
                self.client_for_push_name,
                self.client_for_pull_name,
            ],
            server_name=NIXL_META_SERVER_NAME,
            use_gpu=self.use_gpu,
            multi_client_types=[
                NIXLClientType.PS_FOR_PUSH,
                NIXLClientType.PS_FOR_PULL,
            ],
            nixl_config=self.psrl_config.nixl,
            replica_idx=self.get_replica_id(),
            worker_index=self.rank,
            logging_path=self.psrl_config.logging_path,
        )
        psrl_logger.info(
            f"NIXL multi storage clients initialized on port {self.nixl_multi_storage_clients.client_port}."
        )

    def _nixl_protocol_phase1(self):
        """Execute protocol phase 1: from step 0 to step 3 (before register_local_tensors)."""
        psrl_logger.info("nixl client protocol step 0: convert_hf_inplace")
        parameter_mapping = create_parameter_mapping("HuggingFace", self.train_meta_hf_model.config)
        unified_train_meta_state_dict, local_train_sharding_dict = convert_hf_inplace(
            parameter_mapping,
            self.train_meta_hf_model,
        )
        unified_gen_meta_state_dict, local_gen_sharding_dict = convert_hf_inplace(
            parameter_mapping,
            self.gen_meta_hf_model,
        )
        unified_multi_meta_state_dicts = {
            self.client_for_push_name: unified_train_meta_state_dict,
            self.client_for_pull_name: unified_gen_meta_state_dict,
        }
        psrl_logger.info("nixl client protocol step 1: connect_to_server")
        self.nixl_multi_storage_clients.connect_to_server()
        psrl_logger.info("nixl client protocol step 2: send_local_sharding")
        multi_local_sharding_dicts = {
            self.client_for_push_name: local_train_sharding_dict,
            self.client_for_pull_name: local_gen_sharding_dict,
        }
        self.nixl_multi_storage_clients.send_local_sharding(multi_local_sharding_dicts)
        psrl_logger.info("nixl client protocol step 3: wait_for_server_sharding")
        unified_multi_sharding_dicts = self.nixl_multi_storage_clients.wait_for_server_sharding()
        for client_name, sharding_dict in unified_multi_sharding_dicts.items():
            assert sharding_dict is not None, f"Client with missing sharding dictionary: {client_name}."
        return unified_multi_meta_state_dicts, unified_multi_sharding_dicts

    def _nixl_protocol_phase2(self):
        """Execute protocol phase 2: from step 5 to step 8 (remaining steps)."""
        psrl_logger.info("nixl client protocol step 5: send_local_info")
        self.nixl_multi_storage_clients.send_local_info()
        psrl_logger.info("nixl client protocol step 6: wait_for_server_info")
        self.nixl_multi_storage_clients.wait_for_server_info()
        psrl_logger.info("nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_multi_storage_clients.send_local_temp_mapping()
        psrl_logger.info("nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_multi_storage_clients.wait_for_server_temp_mappings()
        psrl_logger.info("nixl client protocol done.")

    def nixl_protocol(self):
        psrl_logger.info("nixl protocol start with two phases.")
        unified_multi_meta_state_dicts, unified_multi_sharding_dicts = self._nixl_protocol_phase1()
        # Torch allocation is not thread-safe, so registration stays on the main thread.
        psrl_logger.info("nixl client protocol step 4: register_local_tensors")
        client_for_push = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        client_for_pull = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        client_for_push.register_local_tensors(
            unified_multi_meta_state_dicts[self.client_for_push_name],
            unified_multi_sharding_dicts[self.client_for_push_name],
        )
        if self.storage_plan.train_gen_model_share():
            original_tensor_mapping = client_for_push.get_original_tensor_mapping()
            client_for_pull.register_local_tensors(
                unified_multi_meta_state_dicts[self.client_for_pull_name],
                unified_multi_sharding_dicts[self.client_for_pull_name],
                binded_meta_tensor_mapping=original_tensor_mapping,
            )
        else:
            client_for_pull.register_local_tensors(
                unified_multi_meta_state_dicts[self.client_for_pull_name],
                unified_multi_sharding_dicts[self.client_for_pull_name],
            )
        self._nixl_protocol_phase2()

    def nixl_update_client_infos(self, updates: ClientInfoUpdates) -> None:
        """
        Acknowledge updated descriptors for both PS storage clients.
        """
        self.nixl_multi_storage_clients.update_client_infos(updates)

    def get_nixl_agent_name(self) -> str:
        """Get the name of the NIXL agent."""
        return self.agent_name

    def get_nixl_train_storage_client_name(self) -> str:
        """Get the name of the NIXL train storage client."""
        return self.client_for_push_name

    def get_nixl_gen_storage_client_name(self) -> str:
        """Get the name of the NIXL gen storage client."""
        return self.client_for_pull_name

    def get_node_id(self) -> str:
        """Return the Ray node ID of this PS storage worker."""
        return ray.get_runtime_context().get_node_id()

    def get_non_persistent_named_buffers(self) -> dict[str, torch.Tensor]:
        """
        Return cached CPU copies of non-persistent training model buffers.

        These buffers are absent from `state_dict()` but required after TMS resume.
        `init_empty_weights()` leaves them initialized on CPU.

        Returns:
            dict[str, torch.Tensor]: Mapping of dotted buffer name to CPU tensor.
        """
        if self._cached_non_persistent_buffers is not None:
            return self._cached_non_persistent_buffers

        assert self.train_meta_hf_model is not None, "train_meta_hf_model is not initialized. Call init_model first."
        persistent_names = set(self.train_meta_hf_model.state_dict().keys())
        result: dict[str, torch.Tensor] = {}
        for name, buf in self.train_meta_hf_model.named_buffers():
            if name not in persistent_names:
                # Non-persistent buffers contain real CPU data rather than meta tensors.
                result[name] = buf.detach().clone()

        psrl_logger.info(
            f"[get_non_persistent_named_buffers] Cached buffer count: {len(result)}. "
            f"Sample names: {list(result.keys())[:5]}{'...' if len(result) > 5 else ''}."
        )
        self._cached_non_persistent_buffers = result
        return result

    def init_model(self):
        """
        Initialize empty train and generation model skeletons on the meta device.

        Tied-weight aliases are recorded while the meta model still exposes them.
        """
        local_path = copy_to_local(self.model_config.path, use_shm=self.model_config.get("use_shm", False))
        model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=self.model_config.get("trust_remote_code", False),
        )
        if type(model_config) in AutoModelForImageTextToText._model_mapping:
            model_class = AutoModelForImageTextToText
        else:
            model_class = AutoModelForCausalLM

        if self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu"):
            with init_empty_weights():
                self.train_meta_hf_model = model_class.from_config(
                    model_config,
                    torch_dtype=self.storage_plan.train_model_dtype,
                    trust_remote_code=self.model_config.get("trust_remote_code", False),
                )
                if self.storage_plan.train_gen_model_share():
                    self.gen_meta_hf_model = self.train_meta_hf_model
                else:
                    self.gen_meta_hf_model = model_class.from_config(
                        model_config,
                        torch_dtype=self.storage_plan.gen_model_dtype,
                        trust_remote_code=self.model_config.get("trust_remote_code", False),
                    )
            # Restore architecture-required fp32 parameters after the uniform dtype cast.
            self._fix_meta_model_dtypes(self.train_meta_hf_model)
            if not self.storage_plan.train_gen_model_share():
                self._fix_meta_model_dtypes(self.gen_meta_hf_model)
        else:
            raise ValueError(f"Invalid PS mode: {self.psrl_config.ps_mode}")

        # Train and generation share one architecture, so one alias map suffices.
        self._tied_weights_alias_map = self._build_tied_weights_alias_map(self.train_meta_hf_model, local_path)
        if self._tied_weights_alias_map:
            psrl_logger.info(f"init_model: detected tied-weight aliases: {self._tied_weights_alias_map}")

        self.model_info = create_parameter_mapping("HuggingFace", self.train_meta_hf_model.config).get_model_info()

        psrl_logger.info(f"init_model (meta-only) done on {get_worker_info()}.")

    @staticmethod
    def _fix_meta_model_dtypes(meta_model: torch.nn.Module) -> None:
        """Restore architecture-constrained fp32 parameters on a meta model."""
        fixed_count = fix_meta_model_dtypes(meta_model)
        if fixed_count:
            psrl_logger.info("_fix_meta_model_dtypes: corrected %s tensor(s) to float32.", fixed_count)

    @staticmethod
    def _build_tied_weights_alias_map(
        meta_model: torch.nn.Module,
        local_path: str,
    ) -> dict[str, list[str]]:
        """Build a map canonical_checkpoint_key -> [alias_keys_not_in_checkpoint].

        Currently only handles the tie_word_embeddings case:
        model.embed_tokens.weight (canonical) <- lm_head.weight (alias).
        """
        cfg = getattr(meta_model, "config", None)
        if cfg is None or not getattr(cfg, "tie_word_embeddings", False):
            return {}

        ckpt_keys = PSStorageWorker._get_checkpoint_keys(local_path)
        canonical = "model.embed_tokens.weight"
        alias = "lm_head.weight"

        if alias in ckpt_keys:
            return {}

        if canonical not in ckpt_keys:
            # NOTE(zym): Qwen3.5 conditional generation stores embeddings under the language model prefix.
            canonical = "model.language_model.embed_tokens.weight"

        assert canonical in ckpt_keys, (
            f"_build_tied_weights_alias_map: tie_word_embeddings=True but "
            f"'{canonical}' not found in checkpoint under '{local_path}'."
        )
        psrl_logger.info(f"_build_tied_weights_alias_map: '{alias}' (alias) <- '{canonical}' (canonical)")
        return {canonical: [alias]}

    @staticmethod
    def _get_checkpoint_keys(local_path: str) -> set[str]:
        """
        Return the set of parameter names actually stored in the checkpoint.

        Uses the safetensors index.json weight_map when present (O(1) scan),
        otherwise opens the single shard and lists its keys.
        """
        index_json = os.path.join(local_path, "model.safetensors.index.json")
        single_sf = os.path.join(local_path, "model.safetensors")

        if os.path.isfile(index_json):
            with open(index_json) as fh:
                index = json.load(fh)
            return set(index.get("weight_map", index).keys())

        if os.path.isfile(single_sf):
            with safe_open(single_sf, framework="pt", device="cpu") as f:
                return set(f.keys())

        raise FileNotFoundError(f"No safetensors checkpoint found under '{local_path}' for key scanning.")

    @staticmethod
    def _fix_meta_model_dtypes(meta_model: torch.nn.Module) -> None:
        """
        Restore architecture-constrained fp32 parameters and buffers on a meta model.

        `_keep_in_fp32_modules_strict` applies to fp16 and bf16 models.
        `_keep_in_fp32_modules` applies only to fp16 models.
        """
        keep_in_fp32_strict: set[str] = set()
        keep_in_fp32: set[str] = set()

        for module in meta_model.modules():
            if patterns := getattr(module, "_keep_in_fp32_modules_strict", None):
                keep_in_fp32_strict.update(patterns)
            if patterns := getattr(module, "_keep_in_fp32_modules", None):
                keep_in_fp32.update(patterns)

        # Fallbacks cover model classes that omit `_keep_in_fp32_modules_strict`.
        for module in meta_model.modules():
            cls_name = type(module).__name__
            for key, patterns in FP32_PATTERNS.items():
                if key in cls_name:
                    keep_in_fp32_strict.update(patterns)

        if not keep_in_fp32_strict and not keep_in_fp32:
            return

        sample_param = next(meta_model.parameters(), None)
        if sample_param is None:
            return
        current_dtype = sample_param.dtype

        patterns_to_fix: set[str] = set()
        if current_dtype in (torch.float16, torch.bfloat16):
            patterns_to_fix.update(keep_in_fp32_strict)
        if current_dtype == torch.float16:
            patterns_to_fix.update(keep_in_fp32)

        if not patterns_to_fix:
            return

        fixed_count = 0

        for param_name, param in meta_model.named_parameters():
            if param.dtype == torch.float32:
                continue
            if any(pattern in param_name for pattern in patterns_to_fix):
                new_data = torch.empty(param.shape, dtype=torch.float32, device=param.device)
                param.data = new_data
                fixed_count += 1

        for buf_name, buf in meta_model.named_buffers():
            if buf is None or buf.dtype == torch.float32:
                continue
            if any(pattern in buf_name for pattern in patterns_to_fix):
                # Re-register buffers on their owning modules to preserve module state.
                parts = buf_name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_path, attr_name = parts
                    parent_module = meta_model.get_submodule(parent_path)
                else:
                    attr_name = parts[0]
                    parent_module = meta_model
                new_buf = torch.empty(buf.shape, dtype=torch.float32, device=buf.device)
                parent_module.register_buffer(attr_name, new_buf)
                fixed_count += 1

        if fixed_count > 0:
            psrl_logger.info(
                f"_fix_meta_model_dtypes corrected parameters to float32. Count: {fixed_count}. "
                f"Patterns: {patterns_to_fix}."
            )

    # --- Post-protocol weight loading ---

    def preload_checkpoint_to_cpu(self) -> None:
        """
        Cache checkpoint tensors on CPU before NIXL buffer allocation.

        With broadcast initialization, only rank zero reads disk. Tied aliases are
        expanded so registered buffers can be filled by direct key lookup.
        """
        if self.psrl_config.broadcast_init.enabled and self.rank != 0:
            psrl_logger.info(
                f"[preload_checkpoint_to_cpu] Broadcast initialization is enabled. "
                f"Skipping disk read on nonroot rank: {self.rank}."
            )
            return

        assert hasattr(self, "_tied_weights_alias_map") and hasattr(self, "model_info"), (
            "preload_checkpoint_to_cpu requires _tied_weights_alias_map and model_info. Call init_model first."
        )

        local_path = copy_to_local(self.model_config.path, use_shm=self.model_config.get("use_shm", False))
        shard_files = self._discover_safetensors_shards(local_path)
        psrl_logger.info(
            f"[preload_checkpoint_to_cpu] Reading checkpoint shards. Count: {len(shard_files)}. Path: {local_path}."
        )

        cache: dict[str, torch.Tensor] = {}

        for shard_file in shard_files:
            psrl_logger.debug(f"[preload_checkpoint_to_cpu] Opening shard {shard_file}.")
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    for split_key, split_tensor in maybe_convert_to_smaller_parts(
                        self.model_info, key, f.get_tensor(key)
                    ).items():
                        cache[split_key] = split_tensor

        alias_count = 0
        for canonical, aliases in self._tied_weights_alias_map.items():
            if canonical not in cache:
                continue
            for alias_key in aliases:
                cache[alias_key] = cache[canonical].clone()
                alias_count += 1

        self._checkpoint_cpu_cache = cache
        psrl_logger.info(
            f"[preload_checkpoint_to_cpu] Cached checkpoint tensors. Key count: {len(cache)}. "
            f"Tied alias count: {alias_count}."
        )

    def write_checkpoint_to_registered_tensors(self) -> None:
        """
        Copy cached checkpoint tensors into registered NIXL buffers.

        Only rank zero writes during broadcast initialization. The cache is released after the copy.
        """
        if self.psrl_config.broadcast_init.enabled and self.rank != 0:
            psrl_logger.info(
                f"[write_checkpoint_to_registered_tensors] Broadcast initialization is enabled. "
                f"Skipping the CPU-to-buffer write on nonroot rank: {self.rank}."
            )
            return

        assert self.nixl_multi_storage_clients is not None, (
            "NIXL clients must be initialized (call init_nixl_client()) before writing weights."
        )
        assert hasattr(self, "_checkpoint_cpu_cache"), (
            "write_checkpoint_to_registered_tensors requires _checkpoint_cpu_cache. "
            "Call preload_checkpoint_to_cpu first."
        )

        push_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        pull_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        shared = self.storage_plan.train_gen_model_share()

        expected_keys: set[str] = set(push_client.local_client_info.tensor_infos.keys())
        if not shared:
            expected_keys |= set(pull_client.local_client_info.tensor_infos.keys())

        # Alias keys exist in registered buffers and the expanded cache, but not checkpoint files.
        all_alias_keys: set[str] = set()
        for aliases in self._tied_weights_alias_map.values():
            all_alias_keys.update(aliases)

        direct_expected_keys: set[str] = expected_keys - all_alias_keys

        loaded_keys: set[str] = set()

        for key, src_tensor in self._checkpoint_cpu_cache.items():
            if key in direct_expected_keys:
                push_client.load_state_dict_into_registered_tensors({key: src_tensor})
                if not shared:
                    pull_client.load_state_dict_into_registered_tensors({key: src_tensor})
                loaded_keys.add(key)
            elif key in all_alias_keys and key in expected_keys:
                psrl_logger.info(
                    f"[write_checkpoint_to_registered_tensors] Writing alias '{key}' from pre-expanded cache."
                )
                push_client.load_state_dict_into_registered_tensors({key: src_tensor})
                if not shared:
                    pull_client.load_state_dict_into_registered_tensors({key: src_tensor})
                loaded_keys.add(key)

        missing = expected_keys - loaded_keys
        if missing:
            raise RuntimeError(
                f"write_checkpoint_to_registered_tensors: {len(missing)} key(s) not found "
                f"in checkpoint (and not covered by any tied-weight alias): "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )

        alias_loaded = loaded_keys & all_alias_keys
        psrl_logger.info(
            f"[write_checkpoint_to_registered_tensors] Wrote {len(loaded_keys)}/{len(expected_keys)} key(s) "
            f"({len(alias_loaded)} via tied-weight alias)."
        )

        del self._checkpoint_cpu_cache
        gc.collect()

    @staticmethod
    def _discover_safetensors_shards(local_path: str) -> list[str]:
        """
        Return safetensors shard paths in checkpoint index order.

        Raises:
            FileNotFoundError: If no supported checkpoint exists.
            RuntimeError: If the checkpoint uses the legacy PyTorch format.
        """
        index_json = os.path.join(local_path, "model.safetensors.index.json")
        single_sf = os.path.join(local_path, "model.safetensors")

        if os.path.isfile(index_json):
            with open(index_json) as fh:
                index = json.load(fh)
            weight_map: dict[str, str] = index.get("weight_map", index)
            seen: set[str] = set()
            ordered: list[str] = []
            for rel_path in weight_map.values():
                if rel_path not in seen:
                    seen.add(rel_path)
                    ordered.append(os.path.join(local_path, rel_path))
            return ordered

        if os.path.isfile(single_sf):
            return [single_sf]

        pt_bin = os.path.join(local_path, "pytorch_model.bin")
        pt_idx = os.path.join(local_path, "pytorch_model.bin.index.json")
        if os.path.isfile(pt_bin) or os.path.isfile(pt_idx):
            raise RuntimeError(
                f"Checkpoint at '{local_path}' uses pytorch_model.bin format, which is not supported. "
                "Convert it to safetensors first:\n"
                "  python -c \"from transformers import AutoModel; m = AutoModel.from_pretrained('<path>'); "
                "m.save_pretrained('<path>', safe_serialization=True)\""
            )

        raise FileNotFoundError(
            f"No safetensors checkpoint found under '{local_path}'. "
            "Expected 'model.safetensors' or 'model.safetensors.index.json'."
        )

    def _build_transfer_key_cache(self, src_original_state_dict):
        self._transfer_key_cache = {"src_dict_id": id(src_original_state_dict)}
        for key_tuple in src_original_state_dict.keys():
            k, shard_idx = key_tuple
            if k not in self._transfer_key_cache:
                self._transfer_key_cache[k] = []
            self._transfer_key_cache[k].append((k, shard_idx))

    def transfer_train_to_gen_merged(self, key_and_shards_list: list[tuple[str, list[tuple[int, ...]]]]):
        if self.storage_plan.train_gen_model_share():
            return
        for key, shards in key_and_shards_list:
            self.transfer_train_to_gen(key, shards, sync=False)
        if self.use_gpu:
            torch.cuda.synchronize()

    def transfer_train_to_gen(self, key: str, shards: list[tuple[int, ...]] | None = None, sync: bool = True):
        if self.storage_plan.train_gen_model_share():
            return
        src_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        target_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        src_original_state_dict = src_client.get_original_tensor_mapping()
        target_original_state_dict = target_client.get_original_tensor_mapping()
        if not hasattr(self, "_transfer_key_cache") or self._transfer_key_cache.get("src_dict_id") != id(
            src_original_state_dict
        ):
            self._build_transfer_key_cache(src_original_state_dict)
        matching_keys = self._transfer_key_cache.get(key, [])
        for key_shard_idx_tuple in matching_keys:
            if shards is not None and key_shard_idx_tuple[1] not in shards:
                continue
            target_original_state_dict[key_shard_idx_tuple].copy_(src_original_state_dict[key_shard_idx_tuple])
        if sync and self.use_gpu:
            torch.cuda.synchronize()

    # --- Broadcast initialization helpers ---

    def _ps_agent_name_for_rank(self, rank: int) -> str:
        """
        Return the NIXL agent name for the PS worker at the given rank.

        Args:
            rank (int): Target PS worker rank.

        Returns:
            str: Agent name, e.g. 'NIXLPSClient_1'.
        """
        return ps_agent_name(rank)

    def _ps_train_client_name_for_rank(self, rank: int) -> str:
        """
        Return the NIXL push-side (train buffer) client name for the PS worker at the given rank.

        Args:
            rank (int): Target PS worker rank.

        Returns:
            str: Push client name, e.g. 'NIXLPSClient_1_for_push'.
        """
        return ps_client_push_name(rank)

    def broadcast_send_to_children(self, round_idx: int, plan) -> None:
        """
        Broadcast this worker's registered train buffers to its children.

        The call blocks until every NIXL write completes so the manager's round barrier is safe.

        Args:
            round_idx (int): Current broadcast round index (used only for logging).
            plan: BroadcastPlan instance providing the tree topology.
        """
        children = plan.get_children(self.rank)
        if not children:
            psrl_logger.info(
                f"[broadcast_send_to_children] No children, skipping. Rank: {self.rank}. Round: {round_idx}."
            )
            return

        train_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        # NOTE(lhy): Broadcast registered keys because tied aliases are absent from checkpoint
        # keys. Omitting aliases leaves child buffers uninitialized.
        transfer_keys = list(train_client.local_client_info.tensor_infos.keys())
        psrl_logger.info(
            f"[broadcast_send_to_children] Starting transfers. Rank: {self.rank}. "
            f"Round: {round_idx}. Key count: {len(transfer_keys)}. Children: {children}."
        )

        for child_rank in children:
            child_agent = self._ps_agent_name_for_rank(child_rank)
            child_client = self._ps_train_client_name_for_rank(child_rank)
            for key in transfer_keys:
                train_client.client_write(
                    target_agent=child_agent,
                    target_client=child_client,
                    key=key,
                    tag="ps_broadcast_init",
                    use_comm_plan=False,
                )

        # NIXL writes outlive CUDA synchronization, so wait before the Ray barrier returns.
        for child_rank in children:
            child_client = self._ps_train_client_name_for_rank(child_rank)
            for key in transfer_keys:
                train_client.wait(key, "ps_broadcast_init", "WRITE", target_client=child_client)
        train_client.clear_intermediate_cached_data()

        psrl_logger.info(
            f"[broadcast_send_to_children] All transfers complete. Rank: {self.rank}. Round: {round_idx}."
        )

    def do_transfer_train_to_gen_after_broadcast(self) -> None:
        """
        Copy the train buffer to a separate generation buffer after broadcast.
        """
        if self.storage_plan.train_gen_model_share():
            return
        # NOTE(lhy): Registered keys include tied aliases absent from checkpoint files. Using
        # checkpoint keys would leave generation aliases uninitialized on nonroot workers.
        push_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        registered_keys = list(push_client.local_client_info.tensor_infos.keys())
        psrl_logger.info(
            f"[do_transfer_train_to_gen_after_broadcast] Starting transfer. Rank: {self.rank}. "
            f"Key count: {len(registered_keys)}."
        )
        for key in registered_keys:
            self.transfer_train_to_gen(key=key, sync=False)
        if self.use_gpu:
            torch.cuda.synchronize()
        psrl_logger.info(f"[do_transfer_train_to_gen_after_broadcast] rank {self.rank}: transfer done.")

    def shutdown(self):
        self.nixl_multi_storage_clients.shutdown()

    def debug_log_info(self, label: str = ""):
        """
        Log info for both push (train) and pull (gen) clients on this PS worker.
        Called via Ray RPC from the train worker for precision debugging.
        """
        push_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_push_name)
        pull_client = self.nixl_multi_storage_clients.get_client_by_name(self.client_for_pull_name)
        push_client.log_shard_info(label=label)
        pull_client.log_shard_info(label=label)

    def log_weight_fingerprints(self, model_version: int) -> list[dict]:
        """Log PS receive-side and rollout-side fingerprints before publishing a version."""
        options = resolve_weight_fingerprint_options(
            self.psrl_config,
            flow="transfer_chain",
            model_version=model_version,
        )
        if options is None:
            return []

        records = []
        for role, stage, client_name in (
            ("ps_train", "ps_after_receive", self.client_for_push_name),
            ("ps_rollout", "ps_after_layout_copy", self.client_for_pull_name),
        ):
            client = self.nixl_multi_storage_clients.get_client_by_name(client_name)
            fingerprint = client.get_weight_fingerprint(
                mode=options["mode"],
                sample_count=options["sample_count"],
                chunk_bytes=options["chunk_bytes"],
            )
            record = fingerprint_log_record(
                fingerprint,
                flow="transfer_chain",
                stage=stage,
                role=role,
                model_version=model_version,
                rank=self.rank,
                client_name=client_name,
                include_tensor_digests=options["include_tensor_digests"],
            )
            psrl_logger.warning("[WEIGHT_FINGERPRINT] %s", json.dumps(record, sort_keys=True))
            records.append({**record, "tensor_digests": fingerprint["tensor_digests"]})
        return records
