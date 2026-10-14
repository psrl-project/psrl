from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

MEMORY_TEMPLATE = (
    "You are presented with a problem, a section of an article that may contain the answer to the problem, and a "
    "previous memory. Please read the provided section carefully and update the memory with the new information that "
    "helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any "
    "new, useful information.\n\n"
    "<problem>\n{prompt}\n</problem>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "<section>\n{chunk}\n</section>\n\n"
    "Updated memory:\n"
)


FINAL_TEMPLATE = (
    "You are presented with a problem and a previous memory. Please answer the problem based on the previous memory "
    "and put the answer in \\boxed{{}}.\n\n"
    "<problem>\n{prompt}\n</problem>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "Your answer:\n"
)


@dataclass
class MemAgentRuntimeConfig:
    """Configuration owned by the external MemAgent implementation."""

    chunk_tokens: int = 2048
    max_memory_tokens: int = 1024
    max_final_tokens: int = 256
    max_chunks: int = 64
    allow_context_truncation: bool = False
    no_memory: str = "No previous memory"
    stop_token_strings: list[str] = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])
    memory_template: str = MEMORY_TEMPLATE
    final_template: str = FINAL_TEMPLATE


def build_runtime_config(value: DictConfig | dict[str, Any] | None) -> MemAgentRuntimeConfig:
    """Merge YAML values onto the structured MemAgent schema."""
    raw = OmegaConf.create(value or {})
    merged = OmegaConf.merge(OmegaConf.structured(MemAgentRuntimeConfig), raw)
    config: MemAgentRuntimeConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    return config
