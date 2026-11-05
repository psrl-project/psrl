"""The trainer reads session fields from every rollout row."""

from psrl.workers.gen.utils import TokenOutput


def test_session_fields_default_on_a_single_row():
    output = TokenOutput(prompt_ids=[1], response_ids=[2], response_mask=[1]).as_dict()

    assert output["session_primary"] is True
    assert output["loss_weight"] == 1.0


def test_a_multi_trajectory_loop_can_override_the_session_fields():
    output = TokenOutput(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        extra_fields={"session_primary": False, "loss_weight": 0.25},
    ).as_dict()

    assert output["session_primary"] is False
    assert output["loss_weight"] == 0.25


def test_none_session_fields_are_replaced_with_defaults():
    output = TokenOutput(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        extra_fields={"session_primary": None, "loss_weight": None},
    ).as_dict()

    assert output["session_primary"] is True
    assert output["loss_weight"] == 1.0
