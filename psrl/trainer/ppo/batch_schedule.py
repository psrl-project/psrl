"""Training batch scheduling for different aggregation modes."""

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

TRAJECTORY_AGG_MODE = "trajectory"
REQUEST_AGG_MODE = "request"


@dataclass(frozen=True)
class BatchScheduleStep:
    """One worker dispatch in a training batch schedule.

    ``sample_indices=None`` delegates mini-batch splitting and epochs to the
    worker, preserving the original trajectory-level training path. Explicit
    indices represent one optimizer update assembled by the controller.
    """

    sample_indices: tuple[int, ...] | None
    global_batch_size: int
    mini_batch_size: int | None
    num_mini_batch: int | None
    epochs: int
    seed: int
    shuffle: bool

    def worker_extra_info(self) -> dict[str, Any]:
        """Return the worker controls associated with this dispatch."""
        extra_info = {
            "global_batch_size": self.global_batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "dataloader_kwargs": {"shuffle": self.shuffle},
        }
        if self.mini_batch_size is not None:
            extra_info["mini_batch_size"] = self.mini_batch_size
        if self.num_mini_batch is not None:
            extra_info["num_mini_batch"] = self.num_mini_batch
        return extra_info


@dataclass(frozen=True)
class BatchSchedule:
    """Ordered worker dispatches for one actor or critic training call."""

    aggregation_mode: str
    entries_per_update: int
    steps: tuple[BatchScheduleStep, ...]


@dataclass(frozen=True)
class BatchScheduleStrategy:
    """Scheduling behavior for one batch aggregation mode."""

    aggregation_mode: str
    builder: Callable[..., BatchSchedule]
    dispatch_steps_individually: bool
    use_algorithm_rollout_count: bool

    def entries_per_update(
        self,
        ppo_mini_batch_size: int,
        *,
        rollout_n: int,
        algorithm_rollout_n: int,
    ) -> int:
        """Convert PPO mini-batch size to this strategy's aggregation unit."""
        rollout_count = algorithm_rollout_n if self.use_algorithm_rollout_count else rollout_n
        return ppo_mini_batch_size * rollout_count


def _build_trajectory_schedule(
    tags: list[dict[str, Any]],
    entries_per_update: int,
    *,
    epochs: int,
    shuffle: bool,
    seed: int,
) -> BatchSchedule:
    """Preserve worker-side mini-batch splitting and epoch iteration."""
    del tags
    return BatchSchedule(
        aggregation_mode=TRAJECTORY_AGG_MODE,
        entries_per_update=entries_per_update,
        steps=(
            BatchScheduleStep(
                sample_indices=None,
                global_batch_size=entries_per_update,
                mini_batch_size=entries_per_update,
                num_mini_batch=None,
                epochs=epochs,
                seed=seed,
                shuffle=shuffle,
            ),
        ),
    )


def _build_request_schedule(
    tags: list[dict[str, Any]],
    entries_per_update: int,
    *,
    epochs: int,
    shuffle: bool,
    seed: int,
) -> BatchSchedule:
    """Create one controller dispatch per request-aligned optimizer update."""
    request_to_indices: dict[Any, list[int]] = defaultdict(list)
    request_order: list[Any] = []
    for sample_index, tag in enumerate(tags):
        if tag.get("is_padding", False):
            continue
        if "uid" not in tag:
            raise ValueError(
                f"Request-aggregated batch scheduling requires tag['uid']; missing at sample index {sample_index}."
            )
        request_id = tag["uid"]
        if request_id not in request_to_indices:
            request_order.append(request_id)
        request_to_indices[request_id].append(sample_index)

    num_requests = len(request_order)
    if num_requests == 0:
        raise ValueError("Request-aggregated batch contains no non-padding requests.")
    if num_requests % entries_per_update != 0:
        raise ValueError(
            "Request count must be divisible by entries_per_update: "
            f"num_requests={num_requests}, entries_per_update={entries_per_update}. "
            "Adjust psrl.staleness_buffer_entries or the PPO mini-batch size."
        )

    steps: list[BatchScheduleStep] = []
    for epoch in range(epochs):
        epoch_request_order = request_order.copy()
        if shuffle:
            random.Random(seed + epoch).shuffle(epoch_request_order)

        for start in range(0, num_requests, entries_per_update):
            request_ids = epoch_request_order[start : start + entries_per_update]
            sample_indices = tuple(
                sample_index for request_id in request_ids for sample_index in request_to_indices[request_id]
            )
            steps.append(
                BatchScheduleStep(
                    sample_indices=sample_indices,
                    global_batch_size=len(sample_indices),
                    mini_batch_size=None,
                    num_mini_batch=1,
                    epochs=1,
                    seed=seed + len(steps),
                    shuffle=False,
                )
            )

    return BatchSchedule(
        aggregation_mode=REQUEST_AGG_MODE,
        entries_per_update=entries_per_update,
        steps=tuple(steps),
    )


_BATCH_SCHEDULE_STRATEGIES = {
    TRAJECTORY_AGG_MODE: BatchScheduleStrategy(
        aggregation_mode=TRAJECTORY_AGG_MODE,
        builder=_build_trajectory_schedule,
        dispatch_steps_individually=False,
        use_algorithm_rollout_count=False,
    ),
    REQUEST_AGG_MODE: BatchScheduleStrategy(
        aggregation_mode=REQUEST_AGG_MODE,
        builder=_build_request_schedule,
        dispatch_steps_individually=True,
        use_algorithm_rollout_count=True,
    ),
}
SUPPORTED_BATCH_AGG_MODES = frozenset(_BATCH_SCHEDULE_STRATEGIES)


def resolve_sample_keys(keys: Sequence[str], sample_indices: Sequence[int]) -> list[str]:
    """Resolve schedule-local sample positions to concrete batch keys."""
    invalid_indices = [index for index in sample_indices if index < 0 or index >= len(keys)]
    if invalid_indices:
        raise ValueError(f"Sample indices must be in range [0, {len(keys)}), got {invalid_indices[:20]}.")
    return [keys[index] for index in sample_indices]


def get_batch_schedule_strategy(batch_agg_mode: str) -> BatchScheduleStrategy:
    """Resolve and validate a batch scheduling strategy."""
    try:
        return _BATCH_SCHEDULE_STRATEGIES[batch_agg_mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported batch_agg_mode={batch_agg_mode!r}; expected one of {sorted(SUPPORTED_BATCH_AGG_MODES)}."
        ) from exc


def build_batch_schedule(
    tags: list[dict[str, Any]],
    batch_agg_mode: str,
    entries_per_update: int,
    *,
    epochs: int,
    shuffle: bool,
    seed: int,
) -> BatchSchedule:
    """Build a schedule through the strategy registered for `batch_agg_mode`."""
    strategy = get_batch_schedule_strategy(batch_agg_mode)
    if entries_per_update <= 0:
        raise ValueError(f"entries_per_update must be positive, got {entries_per_update}.")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")

    return strategy.builder(
        tags,
        entries_per_update,
        epochs=epochs,
        shuffle=shuffle,
        seed=seed,
    )
