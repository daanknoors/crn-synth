import pandas as pd
import pytest

from crnsynth.metrics.performance.survival import (
    MedianSurvivalScore,
    PredictedMedianSurvivalScore,
)


@pytest.fixture
def data_train():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [6, 7, 8, 9, 10],
            "duration": [5, 10, 15, 20, 25],
            "event": [1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def data_synth():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [6, 7, 8, 9, 10],
            "duration": [6, 11, 16, 21, 26],
            "event": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def data_holdout():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [6, 7, 8, 9, 10],
            "duration": [7, 12, 17, 22, 27],
            "event": [1, 0, 1, 0, 1],
        }
    )


def test_median_survival_score(data_train, data_synth, data_holdout):
    metric = MedianSurvivalScore(duration_column="duration", event_column="event")
    result = metric.compute(data_train, data_synth, data_holdout)
    assert "score" in result


def test_predicted_median_survival_score(data_train, data_synth, data_holdout):
    metric = PredictedMedianSurvivalScore(
        duration_column="duration",
        event_column="event",
        feature_columns=["feature1", "feature2"],
    )
    result = metric.compute(data_train, data_synth, data_holdout)
    assert "score" in result
