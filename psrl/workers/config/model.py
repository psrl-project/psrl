from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING
from transformers import AutoConfig
from verl.base_config import BaseConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config

from psrl.utils.common.chat_template import resolve_chat_template_value

__all__ = ["HFModelConfig"]


@dataclass
class HFModelConfig(BaseConfig):
    _mutable_fields = {
        "hf_config_path",
        "tokenizer_path",
        "hf_config",
        "generation_config",
        "tokenizer",
        "processor",
        "local_path",
        "architectures",
        "local_hf_config_path",
        "local_tokenizer_path",
    }

    path: str = MISSING
    local_path: str | None = None
    hf_config_path: str | None = None
    local_hf_config_path: str | None = None
    tokenizer_path: str | None = None
    local_tokenizer_path: str | None = None

    # Input tokenizer path. Used when the reward model's chat template differs from the policy
    input_tokenizer: str | None = None

    # whether to load tokenizer. This is useful when we only want to load model config
    load_tokenizer: bool = True

    hf_config: Any = None
    generation_config: Any = None
    tokenizer: Any = None
    processor: Any = None

    use_shm: bool = False
    trust_remote_code: bool = False

    custom_chat_template: str | None = None

    external_lib: str | None = None

    override_config: dict = field(default_factory=dict)

    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False

    use_remove_padding: bool = False

    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str | None = "all-linear"

    exclude_modules: str | None = None
    use_liger: bool = False

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    architectures: list[str] | None = None

    def __post_init__(self):
        if self.path in (MISSING, "???"):
            return
        import_external_libs(self.external_lib)

        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path

        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
            self.processor = hf_processor(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)

        resolved_template = resolve_chat_template_value(self.custom_chat_template)
        if resolved_template is not None:
            if self.processor is not None:
                self.processor.chat_template = resolved_template
            else:
                self.tokenizer.chat_template = resolved_template

        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        self.generation_config = get_generation_config(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code
        )

        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        self.hf_config = AutoConfig.from_pretrained(
            self.local_hf_config_path,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=attn_implementation,
        )

        override_config_kwargs = {}

        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )

        # TODO(vermouth1992): Align Megatron and FSDP model override configurations.
        override_config = (
            self.override_config["model_config"] if "model_config" in self.override_config else self.override_config
        )
        override_config_kwargs.update(override_config)
        update_model_config(self.hf_config, override_config_kwargs=override_config_kwargs)

        self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        self.architectures = getattr(self.hf_config, "architectures", None)
        assert self.architectures is not None and len(self.architectures) == 1, (
            f"Expect only one architecture, got {self.architectures}"
        )

        if getattr(self.hf_config, "model_type", None) == "kimi_vl":
            self.hf_config.text_config.topk_method = "greedy"

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer
