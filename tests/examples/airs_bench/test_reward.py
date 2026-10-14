import math

import pytest
from examples.airs_bench.reward import compute_score, normalized_score, phi


def _extra_info(**overrides):
    base = {
        "airs_task_id": "GraphRegressionZincMae",
        "metric": "MAE",
        "metric_lower_is_better": True,
        "sota_score": 0.017,
        "estimated_worst_score": 9.699924,
        "optimal_score": 0.0,
        "airs_scores": [],
        "airs_submit_count": 0,
        "airs_validate_count": 0,
    }
    base.update(overrides)
    return base


@pytest.mark.cpu_test
def test_phi_is_the_negative_log_distance_to_optimal():
    assert phi(0.1, optimal=0.0) == pytest.approx(1.0), "phi(0.1) with optimal 0 must be 1."
    assert phi(0.9, optimal=1.0) == pytest.approx(1.0), "phi is symmetric in distance."


@pytest.mark.cpu_test
def test_phi_at_the_optimum_is_finite():
    value = phi(1.0, optimal=1.0)

    assert math.isfinite(value), "phi at the optimum must be clamped, not infinite."
    assert value == pytest.approx(12.0), "An eps of 1e-12 yields phi of 12."


@pytest.mark.cpu_test
def test_normalized_score_is_zero_at_worst_and_one_at_sota():
    assert normalized_score(9.699924, sota=0.017, worst=9.699924, optimal=0.0) == pytest.approx(0.0)
    assert normalized_score(0.017, sota=0.017, worst=9.699924, optimal=0.0) == pytest.approx(1.0)


@pytest.mark.cpu_test
def test_normalized_score_clips_beyond_the_endpoints():
    beating_sota = normalized_score(0.001, sota=0.017, worst=9.699924, optimal=0.0)
    below_worst = normalized_score(500.0, sota=0.017, worst=9.699924, optimal=0.0)

    assert beating_sota == 1.0, "Beating SOTA must clip to 1.0."
    assert below_worst == 0.0, "Scoring worse than the worst observed must clip to 0.0."


@pytest.mark.cpu_test
def test_normalized_score_handles_higher_is_better_metric():
    # TextualSimilaritySickSpearmanCorrelation: sota 0.854, worst -0.5870786, optimal 1.0
    at_sota = normalized_score(0.854, sota=0.854, worst=-0.5870786, optimal=1.0)
    at_worst = normalized_score(-0.5870786, sota=0.854, worst=-0.5870786, optimal=1.0)
    middling = normalized_score(0.5, sota=0.854, worst=-0.5870786, optimal=1.0)

    assert at_sota == pytest.approx(1.0)
    assert at_worst == pytest.approx(0.0)
    assert 0.0 < middling < 1.0, f"A middling score must land strictly inside the range, got {middling}."


@pytest.mark.cpu_test
def test_normalized_score_is_monotonic_for_lower_is_better():
    kwargs = {"sota": 0.017, "worst": 9.699924, "optimal": 0.0}
    values = [normalized_score(s, **kwargs) for s in (5.0, 2.0, 1.0, 0.5, 0.1)]

    assert values == sorted(values), f"Better raw metrics must never score lower: {values!r}."


@pytest.mark.cpu_test
def test_normalized_score_is_monotonic_for_higher_is_better():
    kwargs = {"sota": 0.905, "worst": 0.1451284, "optimal": 1.0}
    values = [normalized_score(s, **kwargs) for s in (0.2, 0.4, 0.6, 0.8, 0.9)]

    assert values == sorted(values), f"Better raw metrics must never score lower: {values!r}."


@pytest.mark.cpu_test
def test_degenerate_range_yields_binary_reward():
    # phi(sota) == phi(worst) makes the denominator zero.
    at_sota = normalized_score(0.5, sota=0.5, worst=0.5, optimal=0.0)
    below = normalized_score(0.9, sota=0.5, worst=0.5, optimal=0.0)

    assert at_sota == 1.0, "Matching SOTA in a degenerate range must score 1.0."
    assert below == 0.0, "Missing SOTA in a degenerate range must score 0.0."


@pytest.mark.cpu_test
def test_no_submission_scores_zero_with_a_flag():
    result = compute_score("airs_bench", "", None, _extra_info())

    assert result["score"] == 0.0
    assert result["reward_extra_info"]["airs_zero_reason"] == "no_submission"


@pytest.mark.cpu_test
def test_missing_metric_key_scores_zero_with_a_flag():
    info = _extra_info(airs_scores=[{"SomeOtherMetric": 0.5}], airs_submit_count=1)

    result = compute_score("airs_bench", "", None, info)

    assert result["score"] == 0.0
    assert result["reward_extra_info"]["airs_zero_reason"] == "metric_key_missing"


@pytest.mark.cpu_test
def test_non_finite_metric_scores_zero_with_a_flag():
    info = _extra_info(airs_scores=[{"MAE": float("nan")}], airs_submit_count=1)

    result = compute_score("airs_bench", "", None, info)

    assert result["score"] == 0.0
    assert result["reward_extra_info"]["airs_zero_reason"] == "invalid_score"


@pytest.mark.cpu_test
def test_last_submission_wins_not_the_best():
    info = _extra_info(
        airs_scores=[{"MAE": 0.02}, {"MAE": 5.0}],
        airs_submit_count=2,
    )

    result = compute_score("airs_bench", "", None, info)
    best_possible = normalized_score(0.02, sota=0.017, worst=9.699924, optimal=0.0)

    assert result["reward_extra_info"]["airs_raw_metric"] == 5.0, (
        "The last submission must be scored, not the best one."
    )
    assert result["score"] < best_possible


@pytest.mark.cpu_test
def test_successful_submission_reports_diagnostics():
    info = _extra_info(
        airs_scores=[{"MAE": 0.5}],
        airs_submit_count=1,
        airs_validate_count=3,
    )

    result = compute_score("airs_bench", "", None, info)
    extra = result["reward_extra_info"]

    assert 0.0 < result["score"] < 1.0
    assert extra["airs_zero_reason"] == ""
    assert extra["airs_metric"] == "MAE"
    assert extra["airs_raw_metric"] == 0.5
    assert extra["airs_task_id"] == "GraphRegressionZincMae"
    assert extra["airs_submit_count"] == 1
    assert extra["airs_validate_count"] == 3


@pytest.mark.cpu_test
def test_all_twenty_task_constant_sets_produce_valid_rewards():
    """Guard against a task whose real constants break the formula."""
    task_constants = [
        ("CodeGenerationAPPSPassAt5", 0.187, 0.0, 1.0, False),
        ("CodeRetrievalCodeXGlueMRR", 0.6113, 0.0, 1.0, False),
        ("CoreferenceResolutionSuperGLUEWSCAccuracy", 0.962, 0.3653846154, 1.0, False),
        ("CoreferenceResolutionWinograndeAccuracy", 0.854, 0.4664562, 1.0, False),
        ("CvMolecularPropertyPredictionQm9MeanAbsoluteError", 0.021, 132.63319396972656, 0.0, True),
        ("GMolecularPropertyPredictionQm9MeanAbsoluteError", 7.53, 11185110.0, 0.0, True),
        ("GraphRegressionZincMae", 0.017, 9.699924, 0.0, True),
        ("MathQuestionAnsweringSVAMPAccuracy", 0.942, 0.0, 1.0, False),
        ("QuestionAnsweringDuoRCAccuracy", 0.4648, 0.0, 1.0, False),
        ("QuestionAnsweringEli5RougeL", 0.269, 0.002451528, 1.0, False),
        ("QuestionAnsweringFinqaAccuracy", 0.7803, 0.0, 1.0, False),
        ("R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError", 0.033, 6536.567, 0.0, True),
        ("ReadingComprehensionSquadExactMatch", 0.858, 0.0, 1.0, False),
        ("SentimentAnalysisYelpReviewFullAccuracy", 0.778, 0.18208, 1.0, False),
        ("TextualClassificationSickAccuracy", 0.905, 0.1451284, 1.0, False),
        ("TextualSimilaritySickSpearmanCorrelation", 0.854, -0.5870786, 1.0, False),
        ("TimeSeriesForecastingKaggleWebTrafficMASE", 0.622, 502962963078372.0, 0.0, True),
        ("TimeSeriesForecastingRideshareMAE", 1.185, 30.2249, 0.0, True),
        ("TimeSeriesForecastingSolarWeeklyMAE", 576.35, 34761.99, 0.0, True),
        ("U0MolecularPropertyPredictionQm9MeanAbsoluteError", 5.83, 24183970.0, 0.0, True),
    ]

    for task_id, sota, worst, optimal, _lower_better in task_constants:
        at_sota = normalized_score(sota, sota=sota, worst=worst, optimal=optimal)
        at_worst = normalized_score(worst, sota=sota, worst=worst, optimal=optimal)

        assert at_sota == pytest.approx(1.0), f"{task_id} must score 1.0 at SOTA, got {at_sota}."
        assert at_worst == pytest.approx(0.0), f"{task_id} must score 0.0 at worst, got {at_worst}."

        midpoint = (sota + worst) / 2.0
        middling = normalized_score(midpoint, sota=sota, worst=worst, optimal=optimal)
        assert 0.0 <= middling <= 1.0, f"{task_id} midpoint escaped the range: {middling}."
        assert math.isfinite(middling), f"{task_id} produced a non-finite reward."


@pytest.mark.cpu_test
def test_eval_error_field_scores_zero_with_eval_error_flag():
    info = _extra_info(airs_eval_error="timeout", airs_submit_count=1)

    result = compute_score("airs_bench", "", None, info)

    assert result["score"] == 0.0
    assert result["reward_extra_info"]["airs_zero_reason"] == "eval_error"
    assert result["reward_extra_info"]["airs_eval_error"] == "timeout"
