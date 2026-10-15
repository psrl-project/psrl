from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from psrl.utils.rollout.reprefill import (
    collect_reprefill_log_probs,
    filter_rollout_request_ids,
    merge_reprefill_log_probs,
    reprefill_prefix_matches,
    should_update_reprefill_log_probs,
)
from verl import DataProto

pytestmark = pytest.mark.cpu_test


def test_reprefill_merge_replaces_only_when_enabled_and_aligned():
    assert reprefill_prefix_matches([11, 12], [4, 5, 11, 12], 2)
    assert not reprefill_prefix_matches([11, 13], [4, 5, 11, 12], 2)
    assert merge_reprefill_log_probs([-1.0, -2.0], [-1.1, -2.1, -3.0], replace_previous=True) == [
        -1.1,
        -2.1,
        -3.0,
    ]
    assert merge_reprefill_log_probs([-1.0, -2.0], [-3.0], replace_previous=False) == [-1.0, -2.0, -3.0]


def test_router_inflight_snapshot_excludes_validation():
    assert filter_rollout_request_ids({0: [10, 11], 1: [20]}, {10: False, 11: True, 20: False}) == {
        0: [10],
        1: [20],
    }
    assert filter_rollout_request_ids({0: [11]}, {11: True}) == {}


def test_reprefill_update_marker_concatenates_per_request():
    first = DataProto(
        batch=None,
        non_tensor_batch={"reprefill_log_prob_updated": np.array([False])},
        meta_info={},
    )
    second = DataProto(
        batch=None,
        non_tensor_batch={"reprefill_log_prob_updated": np.array([True])},
        meta_info={},
    )

    merged = DataProto.concat([first, second])

    assert merged.non_tensor_batch["reprefill_log_prob_updated"].tolist() == [False, True]


def test_reprefill_logprob_update_is_disabled_for_reward_and_teacher_models():
    assert should_update_reprefill_log_probs(True, is_reward_model=False, is_teacher_model=False)
    assert not should_update_reprefill_log_probs(True, is_reward_model=True, is_teacher_model=False)
    assert not should_update_reprefill_log_probs(True, is_reward_model=False, is_teacher_model=True)
    assert not should_update_reprefill_log_probs(False, is_reward_model=False, is_teacher_model=False)


def test_reprefill_update_collects_decode_logprobs_for_initial_chunk():
    log_probs, replace_previous = collect_reprefill_log_probs(
        previous_response_ids=[],
        prompt_token_ids=[1, 2],
        prompt_log_probs=None,
        current_response_ids=[11, 12],
        current_log_probs=[
            {11: SimpleNamespace(logprob=-0.1)},
            {12: SimpleNamespace(logprob=-0.2)},
        ],
        previous_response_len=0,
    )

    assert log_probs == [-0.1, -0.2]
    assert replace_previous is False


def test_reprefill_update_replaces_matching_continuation_prefix():
    log_probs, replace_previous = collect_reprefill_log_probs(
        previous_response_ids=[11, 12],
        prompt_token_ids=[1, 2, 11, 12],
        prompt_log_probs=[
            None,
            None,
            {11: SimpleNamespace(logprob=-0.11)},
            {12: SimpleNamespace(logprob=-0.22)},
        ],
        current_response_ids=[13],
        current_log_probs=[{13: SimpleNamespace(logprob=-0.3)}],
        previous_response_len=2,
    )

    assert log_probs == [-0.11, -0.22, -0.3]
    assert replace_previous is True


def test_reprefill_update_keeps_old_prefix_when_prompt_mismatches():
    chunk_log_probs, replace_previous = collect_reprefill_log_probs(
        previous_response_ids=[11, 12],
        prompt_token_ids=[1, 2, 11, 99],
        prompt_log_probs=[
            None,
            None,
            {11: SimpleNamespace(logprob=-0.11)},
            {99: SimpleNamespace(logprob=-0.99)},
        ],
        current_response_ids=[13],
        current_log_probs=[{13: SimpleNamespace(logprob=-0.3)}],
        previous_response_len=2,
    )

    merged = merge_reprefill_log_probs([-0.1, -0.2], chunk_log_probs, replace_previous=replace_previous)

    assert merged == [-0.1, -0.2, -0.3]
    assert replace_previous is False


def test_rollout_log_probs_are_enabled_by_default():
    config_path = Path(__file__).resolve().parents[3] / "psrl/trainer/config/psrl/log_prob.yaml"
    config = OmegaConf.load(config_path)
    assert config.enable_rollout_engine_log_prob is True, "Rollout engines must return log probabilities."
