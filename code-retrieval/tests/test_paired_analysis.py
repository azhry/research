import numpy as np

from paired_analysis import _classify_interval, _paired_effect


def test_paired_effect_reports_mean_and_query_outcomes():
    # These small values exercise arithmetic only; they are not benchmark evidence.
    candidate = np.asarray([0.8, 0.3, 0.2, 0.6])
    reference = np.asarray([0.8, 0.2, 0.4, 0.4])
    sample_indices = np.asarray([[0, 1, 2, 3], [1, 1, 1, 1], [2, 2, 2, 2]])

    effect = _paired_effect(candidate, reference, sample_indices)

    assert np.isclose(effect["mean_delta"], 0.025)
    assert effect["query_wins"] == 2
    assert effect["query_ties"] == 1
    assert effect["query_losses"] == 1
    assert effect["ci95"][0] < effect["mean_delta"] < effect["ci95"][1]


def test_interval_reading_distinguishes_improvement_uncertainty_and_regression():
    assert _classify_interval(0.01, 0.04) == "improvement"
    assert _classify_interval(-0.04, -0.01) == "regression"
    assert _classify_interval(-0.01, 0.04) == "uncertain"
    assert _classify_interval(0.0, 0.0) == "no_observed_difference"
