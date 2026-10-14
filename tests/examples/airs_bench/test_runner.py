import pytest
from examples.airs_bench.runner import (
    build_mlgym_model_config,
    collect_scores_from_info,
)


class FakeInfo:
    def __init__(self, score, exit_status="submitted"):
        self.score = score
        self.exit_status = exit_status


@pytest.mark.cpu_test
def test_collect_scores_separates_submit_from_validate():
    # MLGym appends one entry per submit or validate, tagged by the caller.
    info = FakeInfo(score=[{"MAE": 0.5}, {"MAE": 0.3}])

    collected = collect_scores_from_info(info, submit_count=1, validate_count=1)

    assert collected["airs_submit_count"] == 1
    assert collected["airs_validate_count"] == 1
    assert collected["airs_scores"] == [{"MAE": 0.5}, {"MAE": 0.3}]


@pytest.mark.cpu_test
def test_collect_scores_handles_no_submission():
    info = FakeInfo(score=[], exit_status="exit_context")

    collected = collect_scores_from_info(info, submit_count=0, validate_count=0)

    assert collected["airs_scores"] == []
    assert collected["airs_submit_count"] == 0


@pytest.mark.cpu_test
def test_model_config_points_litellm_at_the_session_url():
    payload = {
        "base_url": "http://10.0.0.1:9000/sessions/abc/v1",
        "model": "Qwen3-4B",
        "sampling_params": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 4096},
        "trajectory_id_strategy": "manual",
    }

    model_config = build_mlgym_model_config(payload)

    assert model_config["host_url"] == payload["base_url"], (
        "MLGym reaches the policy via host_url, which it forwards to litellm api_base."
    )
    assert model_config["temperature"] == 1.0
    assert model_config["top_p"] == 0.95
    completion_kwargs = model_config["completion_kwargs"]
    assert completion_kwargs["logprobs"] is True, "TITO needs per-token logprobs."
    assert completion_kwargs["top_logprobs"] == 1
    assert completion_kwargs["max_tokens"] == 4096
    assert completion_kwargs["extra_body"]["top_k"] == 20
    headers = completion_kwargs["extra_headers"]
    assert headers["x-smg-tito-trajectory-id"] == "0", "Manual trajectory id strategy requires the TITO header."


@pytest.mark.cpu_test
def test_model_config_omits_trajectory_header_for_auto_strategy():
    payload = {
        "base_url": "http://host/v1",
        "model": "m",
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 128},
        "trajectory_id_strategy": "auto",
    }

    model_config = build_mlgym_model_config(payload)

    headers = model_config["completion_kwargs"].get("extra_headers", {})
    assert "x-smg-tito-trajectory-id" not in headers


@pytest.mark.cpu_test
def test_model_config_drops_negative_top_k():
    payload = {
        "base_url": "http://host/v1",
        "model": "m",
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 128},
        "trajectory_id_strategy": "auto",
    }

    model_config = build_mlgym_model_config(payload)

    assert "extra_body" not in model_config["completion_kwargs"], (
        "A negative top_k means unset and must not be forwarded."
    )
