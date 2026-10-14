"""Tests for request-aware optimizer batch scheduling."""

import pytest
from psrl.trainer.ppo.batch_schedule import (
    REQUEST_AGG_MODE,
    TRAJECTORY_AGG_MODE,
    build_batch_schedule,
    resolve_sample_keys,
)

pytestmark = pytest.mark.cpu_test


def test_request_schedule_keeps_sibling_trajectories_in_one_dispatch():
    tags = [
        {"uid": 10},
        {"uid": 10},
        {"uid": 10},
        {"uid": 11},
        {"uid": 12},
        {"uid": 12},
        {"uid": 13},
    ]
    keys = ["10_0", "10_1", "10_2", "11", "12_0", "12_1", "13"]

    schedule = build_batch_schedule(
        tags,
        REQUEST_AGG_MODE,
        entries_per_update=2,
        epochs=1,
        shuffle=False,
        seed=0,
    )

    assert len(schedule.steps) == 2
    first_keys = resolve_sample_keys(keys, schedule.steps[0].sample_indices)
    second_keys = resolve_sample_keys(keys, schedule.steps[1].sample_indices)
    assert first_keys == ["10_0", "10_1", "10_2", "11"]
    assert second_keys == ["12_0", "12_1", "13"]


def test_request_schedule_ignores_physical_padding_when_counting_requests():
    tags = [
        {"uid": 20},
        {"uid": 20},
        {"uid": 21},
        {"uid": "padding", "is_padding": True},
    ]

    schedule = build_batch_schedule(
        tags,
        REQUEST_AGG_MODE,
        entries_per_update=2,
        epochs=1,
        shuffle=False,
        seed=0,
    )

    assert schedule.steps[0].sample_indices == (0, 1, 2)
    assert schedule.steps[0].global_batch_size == 3


def test_trajectory_schedule_delegates_epoch_and_minibatch_work_to_worker():
    schedule = build_batch_schedule(
        [],
        TRAJECTORY_AGG_MODE,
        entries_per_update=8,
        epochs=3,
        shuffle=True,
        seed=7,
    )

    assert len(schedule.steps) == 1
    step = schedule.steps[0]
    assert step.sample_indices is None
    assert step.mini_batch_size == 8
    assert step.epochs == 3
    assert step.shuffle is True


def test_request_schedule_rejects_partial_logical_update():
    tags = [{"uid": 30}, {"uid": 30}, {"uid": 31}, {"uid": 32}]

    with pytest.raises(ValueError, match="Request count must be divisible"):
        build_batch_schedule(
            tags,
            REQUEST_AGG_MODE,
            entries_per_update=2,
            epochs=1,
            shuffle=False,
            seed=0,
        )
