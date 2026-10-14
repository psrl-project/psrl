import pytest
from psrl.trainer.ppo.utils import compute_rm_token_metrics

pytestmark = pytest.mark.cpu_test


def test_compute_rm_token_metrics_adds_rm_tokens_to_overall():
    extra_infos = [
        {"gen/default/model_a": {"rm_input_len": 10, "rm_output_len": 20}},
        {"rule": {"score": 1.0}},
        {"ensemble": [{"gen/default/model_b": {"rm_input_len": 5, "rm_output_len": 7}}]},
    ]
    metrics = compute_rm_token_metrics(
        extra_infos,
        global_token_num=[100, 50, 25],
        timing_raw={"step": 2.0},
        n_gpus=4,
    )

    assert metrics["rm_input_length"] == 15.0
    assert metrics["rm_output_length"] == 27.0
    assert metrics["overall_token_num"] == 217.0
    assert metrics["perf/overall_throughput"] == 217.0 / (2.0 * 4)


def test_compute_rm_token_metrics_treats_missing_rm_fields_as_zero():
    metrics = compute_rm_token_metrics(
        [{}, None],
        global_token_num=[8, 2],
        timing_raw={"step": 1.0},
        n_gpus=2,
    )

    assert metrics["rm_input_length"] == 0.0
    assert metrics["rm_output_length"] == 0.0
    assert metrics["overall_token_num"] == 10.0
    assert metrics["perf/overall_throughput"] == 5.0


def test_compute_rm_token_metrics_rejects_mismatched_sample_counts():
    with pytest.raises(AssertionError, match="does not match"):
        compute_rm_token_metrics(
            [{}],
            global_token_num=[1, 2],
            timing_raw={"step": 1.0},
            n_gpus=1,
        )
