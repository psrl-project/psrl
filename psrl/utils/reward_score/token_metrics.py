from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any


@dataclass
class RewardModelTokenLengths:
    """Per-call RM input and output token lengths, grouped by model name."""

    overall_input: list[int] = field(default_factory=list)
    overall_output: list[int] = field(default_factory=list)
    model_input: dict[str, list[int]] = field(default_factory=dict)
    model_output: dict[str, list[int]] = field(default_factory=dict)


def _is_token_count(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_nested_container(value: Any) -> bool:
    if isinstance(value, Mapping):
        return True
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _model_name_from_spec_key(parent_key: str | None) -> str | None:
    if not parent_key:
        return None
    parts = parent_key.split("/", 2)
    if len(parts) == 3:
        return parts[2]
    return parent_key


def _record_length(
    lengths: list[int],
    model_lengths: dict[str, list[int]],
    model_name: str | None,
    value: Any,
) -> None:
    length = int(value)
    lengths.append(length)
    if model_name is None:
        return
    model_lengths.setdefault(model_name, []).append(length)


def collect_reward_model_token_lengths(extra_info: Any) -> RewardModelTokenLengths:
    """
    Collect each nested `rm_input_len` and `rm_output_len` as one sequence length.

    A mapping key is the reward spec that owns any token-count fields inside that
    mapping. Spec keys `{loop}/{fn}/{model_name}` contribute under `model_name`.
    """
    overall_input: list[int] = []
    overall_output: list[int] = []
    model_input: dict[str, list[int]] = {}
    model_output: dict[str, list[int]] = {}
    stack: list[tuple[Any, str | None]] = [(extra_info, None)]

    while stack:
        current, parent_key = stack.pop()
        model_name = _model_name_from_spec_key(parent_key)
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key == "rm_input_len" and _is_token_count(value):
                    _record_length(overall_input, model_input, model_name, value)
                elif key == "rm_output_len" and _is_token_count(value):
                    _record_length(overall_output, model_output, model_name, value)
                elif _is_nested_container(value):
                    next_parent = key if isinstance(value, Mapping) else parent_key
                    stack.append((value, next_parent))
        elif _is_nested_container(current):
            stack.extend((item, parent_key) for item in current)

    return RewardModelTokenLengths(
        overall_input=overall_input,
        overall_output=overall_output,
        model_input=model_input,
        model_output=model_output,
    )


def extract_reward_model_token_counts(extra_info: Any) -> tuple[int, int]:
    """Return recursively aggregated RM input and output token counts."""
    lengths = collect_reward_model_token_lengths(extra_info)
    return sum(lengths.overall_input), sum(lengths.overall_output)


def _summarize_token_lengths(lengths: Sequence[int]) -> tuple[float, float, float, float]:
    if not lengths:
        return 0.0, 0.0, 0.0, 0.0
    total = float(sum(lengths))
    return float(min(lengths)), total / len(lengths), float(max(lengths)), total


def _put_length_stats(
    metrics: dict[str, float],
    key_prefix: str,
    lengths: Sequence[int],
    include_total: bool,
) -> None:
    length_min, length_mean, length_max, length_total = _summarize_token_lengths(lengths)
    metrics[f"{key_prefix}_min"] = length_min
    metrics[f"{key_prefix}_mean"] = length_mean
    metrics[f"{key_prefix}_max"] = length_max
    if include_total:
        metrics[f"{key_prefix}_total"] = length_total


def _extend_model_lengths(dst: dict[str, list[int]], src: dict[str, list[int]]) -> None:
    for model_name, values in src.items():
        dst.setdefault(model_name, []).extend(values)


def build_rm_length_metrics(extra_infos: Sequence[Any]) -> dict[str, float]:
    """
    Build batch RM length metrics from per-sample `reward_extra_info` payloads.

    `perf/rm_input_length` and `perf/rm_output_length` are batch totals.
    `rm_seq_length/*` summarizes each nested RM call as one sequence.
    """
    merged = RewardModelTokenLengths()
    for extra_info in extra_infos:
        part = collect_reward_model_token_lengths(extra_info)
        merged.overall_input.extend(part.overall_input)
        merged.overall_output.extend(part.overall_output)
        _extend_model_lengths(merged.model_input, part.model_input)
        _extend_model_lengths(merged.model_output, part.model_output)

    metrics = {
        "perf/rm_input_length": float(sum(merged.overall_input)),
        "perf/rm_output_length": float(sum(merged.overall_output)),
    }
    _put_length_stats(metrics, "rm_seq_length/overall_input", merged.overall_input, include_total=False)
    _put_length_stats(metrics, "rm_seq_length/overall_output", merged.overall_output, include_total=False)
    for model_name in sorted(set(merged.model_input) | set(merged.model_output)):
        _put_length_stats(
            metrics,
            f"rm_seq_length/{model_name}/input",
            merged.model_input.get(model_name, []),
            include_total=True,
        )
        _put_length_stats(
            metrics,
            f"rm_seq_length/{model_name}/output",
            merged.model_output.get(model_name, []),
            include_total=True,
        )
    return metrics
