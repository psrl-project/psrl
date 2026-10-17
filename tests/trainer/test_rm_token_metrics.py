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

    assert metrics["perf/rm_input_length"] == 15.0
    assert metrics["perf/rm_output_length"] == 27.0
    assert metrics["perf/overall_token_num"] == 217.0
    assert metrics["perf/overall_throughput"] == 217.0 / (2.0 * 4)
    assert metrics["rm_seq_length/overall_input_min"] == 5.0
    assert metrics["rm_seq_length/overall_input_mean"] == 7.5
    assert metrics["rm_seq_length/overall_input_max"] == 10.0
    assert metrics["rm_seq_length/overall_output_min"] == 7.0
    assert metrics["rm_seq_length/overall_output_mean"] == 13.5
    assert metrics["rm_seq_length/overall_output_max"] == 20.0
    assert metrics["rm_seq_length/model_a/input_min"] == 10.0
    assert metrics["rm_seq_length/model_a/input_mean"] == 10.0
    assert metrics["rm_seq_length/model_a/input_max"] == 10.0
    assert metrics["rm_seq_length/model_a/input_total"] == 10.0
    assert metrics["rm_seq_length/model_a/output_min"] == 20.0
    assert metrics["rm_seq_length/model_a/output_mean"] == 20.0
    assert metrics["rm_seq_length/model_a/output_max"] == 20.0
    assert metrics["rm_seq_length/model_a/output_total"] == 20.0
    assert metrics["rm_seq_length/model_b/input_min"] == 5.0
    assert metrics["rm_seq_length/model_b/input_mean"] == 5.0
    assert metrics["rm_seq_length/model_b/input_max"] == 5.0
    assert metrics["rm_seq_length/model_b/input_total"] == 5.0
    assert metrics["rm_seq_length/model_b/output_min"] == 7.0
    assert metrics["rm_seq_length/model_b/output_mean"] == 7.0
    assert metrics["rm_seq_length/model_b/output_max"] == 7.0
    assert metrics["rm_seq_length/model_b/output_total"] == 7.0


def test_compute_rm_token_metrics_aggregates_per_model_across_samples():
    extra_infos = [
        {"gen/default/model_a": {"rm_input_len": 10, "rm_output_len": 4}},
        {"gen/default/model_a": {"rm_input_len": 30, "rm_output_len": 8}},
    ]
    metrics = compute_rm_token_metrics(
        extra_infos,
        global_token_num=[1, 2],
        timing_raw={"step": 1.0},
        n_gpus=1,
    )

    assert metrics["perf/rm_input_length"] == 40.0
    assert metrics["perf/rm_output_length"] == 12.0
    assert metrics["rm_seq_length/overall_input_min"] == 10.0
    assert metrics["rm_seq_length/overall_input_mean"] == 20.0
    assert metrics["rm_seq_length/overall_input_max"] == 30.0
    assert metrics["rm_seq_length/overall_output_min"] == 4.0
    assert metrics["rm_seq_length/overall_output_mean"] == 6.0
    assert metrics["rm_seq_length/overall_output_max"] == 8.0
    assert metrics["rm_seq_length/model_a/input_min"] == 10.0
    assert metrics["rm_seq_length/model_a/input_mean"] == 20.0
    assert metrics["rm_seq_length/model_a/input_max"] == 30.0
    assert metrics["rm_seq_length/model_a/input_total"] == 40.0
    assert metrics["rm_seq_length/model_a/output_min"] == 4.0
    assert metrics["rm_seq_length/model_a/output_mean"] == 6.0
    assert metrics["rm_seq_length/model_a/output_max"] == 8.0
    assert metrics["rm_seq_length/model_a/output_total"] == 12.0


def test_compute_rm_token_metrics_treats_missing_rm_fields_as_zero():
    metrics = compute_rm_token_metrics(
        [{}, None],
        global_token_num=[8, 2],
        timing_raw={"step": 1.0},
        n_gpus=2,
    )

    assert metrics["perf/rm_input_length"] == 0.0
    assert metrics["perf/rm_output_length"] == 0.0
    assert metrics["perf/overall_token_num"] == 10.0
    assert metrics["perf/overall_throughput"] == 5.0
    assert metrics["rm_seq_length/overall_input_min"] == 0.0
    assert metrics["rm_seq_length/overall_input_mean"] == 0.0
    assert metrics["rm_seq_length/overall_input_max"] == 0.0
    assert metrics["rm_seq_length/overall_output_min"] == 0.0
    assert metrics["rm_seq_length/overall_output_mean"] == 0.0
    assert metrics["rm_seq_length/overall_output_max"] == 0.0
    model_keys = [
        key for key in metrics if key.startswith("rm_seq_length/") and not key.startswith("rm_seq_length/overall_")
    ]
    assert model_keys == []


def test_compute_rm_token_metrics_rejects_mismatched_sample_counts():
    with pytest.raises(AssertionError, match="does not match"):
        compute_rm_token_metrics(
            [{}],
            global_token_num=[1, 2],
            timing_raw={"step": 1.0},
            n_gpus=1,
        )
