import base64

import numpy as np
import pytest
from psrl.utils.sampler_distribution import (
    MAX_SAMPLER_CAPTURE_WIDTH,
    SamplerCapture,
    SamplerDistribution,
    configure_capture,
)
from psrl.utils.tito.training_data import build_training_data


def wire(ids, probabilities, complete=False):
    return {
        "offsets": [0, len(ids)],
        "token_ids": base64.b64encode(np.asarray(ids, dtype="<i4").tobytes()).decode(),
        "log_probs": base64.b64encode(np.log(np.asarray(probabilities, dtype="<f4")).tobytes()).decode(),
        "complete": [complete],
    }


def test_compact_layout_has_no_prompt_or_environment_entries():
    head = SamplerDistribution.from_wire(wire([2, 8], [0.4, 0.3]), [[np.log(0.4), 2]])
    support = SamplerDistribution.from_wire(wire([3], [1], True), [[0, 3]])
    merged = SamplerDistribution.concatenate([head, SamplerDistribution.empty(2), support])
    merged.validate_capture(SamplerCapture("topk", 2), [1, 0, 0, 1])
    assert merged.offsets.tolist() == [0, 2, 2, 2, 3]
    assert merged.token_ids.tolist() == [2, 8, 3]
    assert merged.complete.tolist() == [False, False, False, True]
    assert merged.prefix(3).token_ids.tolist() == [2, 8]


def test_support_budget_never_silently_truncates():
    head = SamplerDistribution.from_wire(wire([2], [0.999999]), [[np.log(0.999999), 2]])
    with pytest.raises(ValueError, match="capture budget"):
        head.validate_capture(SamplerCapture("support", 128), [1])


def test_capture_width_matches_mode_and_budget():
    assert SamplerCapture("disabled", 128).width == 0
    assert SamplerCapture("topk", 128).width == 128
    assert SamplerCapture("support", 128).width == 129
    assert not SamplerCapture().enabled
    assert SamplerCapture("topk", 4).enabled
    assert SamplerCapture("topk", MAX_SAMPLER_CAPTURE_WIDTH).width == MAX_SAMPLER_CAPTURE_WIDTH
    with pytest.raises(ValueError, match="protocol limit"):
        _ = SamplerCapture("support", MAX_SAMPLER_CAPTURE_WIDTH).width


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_generate_rows_reject_nonfinite_values_other_than_negative_infinity(invalid):
    with pytest.raises(ValueError, match="NaN or positive infinity"):
        SamplerDistribution.from_topk_rows([[2, 3]], [[0.0, invalid]], [[0.0, 2]])


def test_generate_support_validates_mass_and_sampled_token():
    with pytest.raises(ValueError, match="mass"):
        SamplerDistribution.from_topk_rows(
            [[2, 3, 0]],
            [[np.log(0.6), np.log(0.3), -np.inf]],
            [[np.log(0.6), 2]],
        )
    with pytest.raises(ValueError, match="omits"):
        SamplerDistribution.from_topk_rows(
            [[2, 3, 0]],
            [[np.log(0.6), np.log(0.4), -np.inf]],
            [[0.0, 4]],
        )


def test_generate_rows_reject_duplicate_and_out_of_range_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        SamplerDistribution.from_topk_rows([[2, 2]], [[np.log(0.6), np.log(0.4)]], [[np.log(0.6), 2]])
    with pytest.raises(ValueError, match="invalid IDs"):
        SamplerDistribution.from_topk_rows(
            [[2, np.iinfo(np.int32).max + 1]],
            [[np.log(0.6), np.log(0.4)]],
            [[np.log(0.6), 2]],
        )


def test_generate_rows_strict_false_skips_row_level_checks():
    # The per-row checks are O(rows) Python and run only in strict/CI mode, so a duplicate id that
    # fails under strict is accepted with strict off (structural row checks still run).
    distribution = SamplerDistribution.from_topk_rows(
        [[2, 2]], [[np.log(0.6), np.log(0.4)]], [[np.log(0.6), 2]], strict=False
    )
    assert distribution.token_ids.tolist() == [2, 2]


def test_generate_rows_reject_empty_width_and_noninteger_ids():
    with pytest.raises(ValueError, match=r"\[R, k\]"):
        SamplerDistribution.from_topk_rows([[]], [[]], [[0.0, 2]])
    with pytest.raises(ValueError, match="must be integers"):
        SamplerDistribution.from_topk_rows([[2.0, 3.0]], [[np.log(0.6), np.log(0.4)]], [[np.log(0.6), 2]])


@pytest.mark.parametrize(
    "payload,sampled,match",
    [
        (wire([2, 2], [0.3, 0.4]), [[np.log(0.3), 2]], "Duplicate"),
        (wire([2], [0.3], True), [[np.log(0.3), 2]], "mass"),
        (wire([2], [1], True), [[0, 3]], "omits"),
        (wire([2], [0.3]), [[-1, 2]], "disagree"),
        (wire([2], [0.3]), [[float("nan"), 3]], "Invalid sampled"),
    ],
)
def test_invalid_wire_fails_closed(payload, sampled, match):
    with pytest.raises(ValueError, match=match):
        SamplerDistribution.from_wire(payload, sampled)


def test_tito_trim_and_multiturn_environment_alignment():
    first = wire([10, 99], [1, 1])
    first["offsets"] = [0, 1, 2]
    first["complete"] = [False, False]
    records = [
        {"prompt_token_count": 2, "output_logprobs": [[0, 10], [0, 99]], "sampler_distribution": first},
        {"prompt_token_count": 4, "output_logprobs": [[0, 20]], "sampler_distribution": wire([20], [1], True)},
    ]
    data = build_training_data([1, 2, 10, 15, 20], records, max_trim_tokens=1)
    distribution = data["sampler_distribution"]
    assert data["response_mask"] == [1, 0, 1]
    assert distribution.offsets.tolist() == [0, 1, 1, 2]
    assert distribution.token_ids.tolist() == [10, 20]


def test_missing_capture_is_not_silently_disabled():
    # One turn carries a distribution, so the trajectory is a capture. A trainable
    # turn that produced logprobs without its own distribution must fail rather
    # than train against a silently disabled capture.
    records = [
        {
            "prompt_token_count": 1,
            "output_logprobs": [[0.0, 2]],
            "sampler_distribution": wire([2], [1.0], True),
        },
        {"prompt_token_count": 2, "output_logprobs": [[-0.1, 10]]},
    ]
    with pytest.raises(ValueError, match="Missing sampler distribution"):
        build_training_data([1, 2, 10], records)


def test_capture_does_not_duplicate_head_in_agent_response():
    body = {"top_p": 0.9, "temperature": 0.7}
    configure_capture(body, SamplerCapture("support", 128), protocol="chat")
    assert body["top_logprobs"] == 0
    assert body["sampler_top_logprobs"] == 129
    with pytest.raises(ValueError, match="Truncated sampling"):
        configure_capture(body, SamplerCapture("topk", 128), protocol="chat")


def test_a_responses_request_carries_the_logprobs_include():
    capture = SamplerCapture(mode="topk", budget=32)
    body = {"input": "hi", "include": ["reasoning.encrypted_content"]}
    configure_capture(body, capture, protocol="responses")
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 0
    assert body["include"] == ["reasoning.encrypted_content", "message.output_text.logprobs"]
    assert body["sampler_top_logprobs"] == capture.width


def test_a_responses_request_that_already_asks_for_logprobs_is_not_duplicated():
    body = {"input": "hi", "include": ["message.output_text.logprobs"]}
    configure_capture(body, SamplerCapture(mode="topk", budget=32), protocol="responses")
    assert body["include"] == ["message.output_text.logprobs"]


def test_a_chat_request_does_not_gain_an_include_field():
    body = {"messages": []}
    configure_capture(body, SamplerCapture(mode="topk", budget=32), protocol="chat")
    assert body["top_logprobs"] == 0
    assert "include" not in body
