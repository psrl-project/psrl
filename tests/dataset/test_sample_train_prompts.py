"""Unit tests for on-demand refill sampling in `DataProcessor.sample_train_prompts`.

`sample_train_prompts` is the only source of replacement prompts when a rollout
group fails, so returning `None` costs a training buffer slot permanently. These
tests pin the distinction between an epoch boundary, which is recoverable because
`_get_train_next` rebuilds its iterators before re-raising, and a genuinely empty
dataset, which is not.

`DataProcessor` is decorated with `@ray.remote`, so the tests reach the undecorated
class through `__ray_metadata__.modified_class` and build instances with
`object.__new__` to skip the heavy dataset and tokenizer setup in `__init__`.
"""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from psrl.utils.dataset import data_processor as data_processor_module

pytestmark = pytest.mark.cpu_test


_RAW_DATA_PROCESSOR = data_processor_module.DataProcessor.__ray_metadata__.modified_class


class _EpochDataLoader:
    """Yield a fixed list of batch dicts once per iteration, like a real epoch."""

    def __init__(self, batches: list[dict]) -> None:
        self.batches = batches
        self.epochs_started = 0

    def __iter__(self):
        self.epochs_started += 1
        return iter(self.batches)


def _make_batch(n_rows: int, offset: int = 0) -> dict:
    """Build one dataloader-shaped batch dict with `n_rows` prompt rows."""
    return {
        "input_ids": torch.arange(offset, offset + n_rows * 2).reshape(n_rows, 2),
        "raw_prompt": np.array([f"prompt-{offset + i}" for i in range(n_rows)], dtype=object),
    }


def _make_processor(dataloader: _EpochDataLoader, rollout_n: int = 2) -> object:
    """Build a `DataProcessor` with only the attributes `sample_train_prompts` touches."""
    processor = object.__new__(_RAW_DATA_PROCESSOR)
    processor.rollout_n = rollout_n
    processor.global_steps = 0
    processor._train_sample_idx = 0
    processor.MAX_TRAIN_ID = 10**9
    processor.dataloader_lock = threading.Lock()
    processor.retry_buffer = None
    processor.train_dataloaders = [dataloader]
    processor.train_dataloader_iters = None
    processor.ps_manager_handle = MagicMock()
    return processor


def _sample(processor, n_prompts: int):
    """Call `sample_train_prompts` with the `ray.get` request registration stubbed out."""
    with patch.object(data_processor_module.ray, "get", return_value=None):
        return processor.sample_train_prompts(n_prompts=n_prompts)


class TestSampleTrainPromptsEpochRollover:
    """Behavioral tests for how `sample_train_prompts` reads `StopIteration`."""

    def test_refill_after_epoch_boundary_returns_a_full_batch(self):
        """A refill landing exactly on an epoch boundary must still produce prompts.

        `_get_train_next` resets `train_dataloader_iters` before re-raising
        `StopIteration`, so the exception means "this epoch ended", not "the dataset
        is finished". Treating it as terminal is what silently drops a buffer slot.
        """
        rollout_n = 2
        dataloader = _EpochDataLoader([_make_batch(n_rows=2)])
        processor = _make_processor(dataloader, rollout_n=rollout_n)

        # Drain the first epoch exactly, leaving `retry_buffer` empty.
        first = _sample(processor, n_prompts=2)
        assert first is not None
        assert len(first) == 2 * rollout_n

        # The next refill hits the boundary. It must roll into a fresh epoch.
        second = _sample(processor, n_prompts=2)
        assert second is not None, "A refill on an epoch boundary returned no prompts."
        assert len(second) == 2 * rollout_n
        assert dataloader.epochs_started == 2

    def test_refill_spanning_an_epoch_boundary_fills_every_requested_prompt(self):
        """A refill larger than the epoch remainder must span into the next epoch.

        `handle_waiting_buffer` refills `aborted_entry_num` prompts at once, which can
        exceed what is left in the current epoch.
        """
        rollout_n = 2
        dataloader = _EpochDataLoader([_make_batch(n_rows=2)])
        processor = _make_processor(dataloader, rollout_n=rollout_n)

        _sample(processor, n_prompts=1)  # Leaves 1 row buffered from epoch 1.
        batch = _sample(processor, n_prompts=3)

        assert batch is not None
        assert len(batch) == 3 * rollout_n
        assert dataloader.epochs_started == 2

    def test_empty_dataset_returns_none_without_spinning(self):
        """A dataset that yields nothing must terminate instead of retrying forever.

        This is the case the terminal-`StopIteration` reading exists to handle, and
        the rollover fix must not turn it into an unbounded loop. Bounding epoch
        restarts is what proves it.
        """
        dataloader = _EpochDataLoader([])
        processor = _make_processor(dataloader)

        assert _sample(processor, n_prompts=2) is None
        assert dataloader.epochs_started <= 3, (
            f"Empty dataset restarted the epoch {dataloader.epochs_started} times, which indicates a spin."
        )
