from enum import Enum

import torch
from psrl.utils.rollout.vision_utils import (
    image_resize_targets,
    serialize_image_inputs,
    serialize_tensor,
)
from verl.utils.tokenizer import build_multimodal_processor_inputs


class MultimodalPreprocessingMode(str, Enum):
    RUST = "rust"
    PYTHON = "python"

    @classmethod
    def parse(cls, value: str | None) -> "MultimodalPreprocessingMode":
        try:
            return cls((value or cls.RUST.value).lower())
        except ValueError as exc:
            raise ValueError("multimodal_preprocessing must be 'rust' or 'python'") from exc


class GatewayMultimodalPayloadBuilder:
    def __init__(self, mode, processor, tokenizer, executor):
        self.mode = MultimodalPreprocessingMode.parse(mode)
        self.processor = processor
        self.tokenizer = tokenizer
        self.executor = executor

    @property
    def uses_rust_preprocessing(self) -> bool:
        return self.mode is MultimodalPreprocessingMode.RUST

    async def build(self, request_input, image_refs: list, mm_data: dict) -> dict:
        if not image_refs:
            return {}
        payload = {
            "image_data": await serialize_image_inputs(image_refs),
            "multimodal_token_mode": ("unexpanded" if self.uses_rust_preprocessing else "preexpanded"),
            "modalities": ["multi-images" if len(image_refs) > 1 else "image"],
        }
        if self.uses_rust_preprocessing:
            # SMG owns placeholder expansion in this mode. PSRL needs the exact
            # expanded IDs used by vLLM for training-data alignment.
            payload["return_prompt_token_ids"] = True
            payload["image_preprocessing"] = {
                "resize_targets": image_resize_targets(mm_data.get("images") or [], len(image_refs))
            }
        else:
            payload["preprocessed_mm_inputs"] = await self._python_inputs(request_input, mm_data)
        return payload

    async def _python_inputs(self, request_input, mm_data: dict) -> dict:
        images = mm_data.get("images") or []
        if not images or self.processor is None:
            raise ValueError("python multimodal preprocessing requires processor and decoded images")
        prompt_text = await self.executor(None, self.tokenizer.decode, request_input.input_ids, True)
        inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[prompt_text],
            images=images,
            mm_processor_kwargs=request_input.mm_processor_kwargs,
        )
        inputs.pop("input_ids", None)
        inputs.pop("attention_mask", None)
        tensors = dict(inputs.convert_to_tensors("pt"))
        pixel_values = tensors.pop("pixel_values")
        flat_keys = {}
        grid = tensors.get("image_grid_thw")
        if isinstance(grid, torch.Tensor):
            tensors["pixel_values_sizes"] = grid.prod(-1).to(torch.int64)
            flat_keys["pixel_values"] = "pixel_values_sizes"

        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token = getattr(self.processor, "image_token", None)
            image_token_id = self.tokenizer.convert_tokens_to_ids(image_token) if image_token else None
        placeholders = []
        ids = request_input.input_ids
        i = 0
        while image_token_id is not None and i < len(ids):
            if ids[i] != image_token_id:
                i += 1
                continue
            start = i
            while i < len(ids) and ids[i] == image_token_id:
                i += 1
            placeholders.append((start, i - start))

        return {
            "pixel_values": serialize_tensor(pixel_values),
            "model_specific_tensors": {
                key: serialize_tensor(value) for key, value in tensors.items() if isinstance(value, torch.Tensor)
            },
            "mm_placeholders": placeholders,
            "batched_keys": ["image_grid_thw"],
            "flat_keys": flat_keys,
            "keep_on_cpu_keys": ["image_grid_thw"],
        }
