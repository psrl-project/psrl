# vLLM/FSDP/Megatron to Unified (HuggingFace) State Dict Converter

This module provides functionality to convert vLLM/FSDP/Megatron model state dicts to unified (HuggingFace) format. The key feature is **inplace conversion**, which means no new memory is allocated, and parameter splitting is achieved through tensor views.

## Core Features

- **Inplace Conversion**: Uses tensor views instead of data copying to save memory.
- **Scalable Design**: Supports several models for vLLM, all models for FSDP, and all models for Megatron (enabled through [mbridge](https://github.com/ISEEKYAN/mbridge)).
- **Auto Configuration**: Automatically retrieves model information from HuggingFace config.

## Adding New Model Support for vLLM

To add support for a new model:

1. Create a parameter mapping class:

```python
from psrl.utils.converter import register_model, ParameterMapping, create_parameter_mapping
from psrl.utils.converter.model_mappings import MappingType

@register_model(YourModelClass)  # Replace with actual model class
class YourModelParameterMapping(ParameterMapping):
    def __init__(self, config):
        super().__init__(config)

    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]

    def get_model_info(self):
        return {
            "num_heads": self.config.num_attention_heads,
            "num_kv_heads": getattr(self.config, "num_key_value_heads", self.config.num_attention_heads),
            "head_size": self.config.hidden_size // self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size,
        }
```

2. Use the new model (pass a HuggingFace config object, not a path):

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_path)
param_mapping = create_parameter_mapping(YourModelClass, config)
```
