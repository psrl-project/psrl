"""
Check the installed vLLM ModelConfig with local configs and no model weights.
"""

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "yarn,requested,expected",
    [
        (False, None, 32768),
        (False, 32768, 32768),
        (False, 32769, None),
        (True, None, 40960),
        (True, 40960, 40960),
        (True, 40961, None),
    ],
)
def test_native_model_config_context_validation(tmp_path, yarn, requested, expected):
    from transformers import LlamaConfig
    from vllm.engine.arg_utils import AsyncEngineArgs

    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=32768,
        architectures=["LlamaForCausalLM"],
    )
    config.save_pretrained(tmp_path)
    overrides = (
        {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 1.25,
                "original_max_position_embeddings": 32768,
                "rope_theta": 10000.0,
            }
        }
        if yarn
        else {}
    )
    arguments = AsyncEngineArgs(
        model=str(tmp_path),
        max_model_len=requested,
        hf_overrides=overrides,
        dtype="float32",
        skip_tokenizer_init=True,
    )
    if expected is None:
        with pytest.raises(ValueError, match="max_model_len"):
            arguments.create_model_config()
    else:
        validated = arguments.create_model_config()
        assert validated.max_model_len == expected, "vLLM must resolve the checkpoint and RoPE context ceiling."
