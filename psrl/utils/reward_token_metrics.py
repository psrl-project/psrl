from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


def extract_reward_model_token_counts(extra_info: Any) -> tuple[int, int]:
    """Return recursively aggregated RM input and output token counts."""
    input_tokens = 0
    output_tokens = 0
    stack = [extra_info]

    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key == "rm_input_len" and isinstance(value, Real) and not isinstance(value, bool):
                    input_tokens += int(value)
                elif key == "rm_output_len" and isinstance(value, Real) and not isinstance(value, bool):
                    output_tokens += int(value)
                elif isinstance(value, Mapping) or (
                    isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
                ):
                    stack.append(value)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            stack.extend(current)

    return input_tokens, output_tokens
