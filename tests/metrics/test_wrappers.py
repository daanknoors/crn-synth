import pandas as pd
import pytest
from synthcity.metrics.eval_performance import PerformanceEvaluatorLinear
from synthcity.metrics.eval_sanity import DataMismatchScore
from synthcity.metrics.eval_statistical import JensenShannonDistance

from crnsynth.metrics.wrappers import SynthcityMetricWrapper


@pytest.fixture
def data_train():
    return pd.DataFrame(
        {"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"], "C": [0.1, 0.2, 0.3, 0.4]}
    )


@pytest.fixture
def data_synth():
    return pd.DataFrame(
        {"A": [2, 2, 3, 4], "B": ["b", "b", "c", "d"], "C": [0.3, 0.2, 0.3, 0.4]}
    )


@pytest.fixture
def data_holdout():
    return pd.DataFrame(
        {"A": [4, 3, 2, 1], "B": ["a", "b", "c", "e"], "C": [0.1, 0.2, 0.3, 0.5]}
    )


def test_synthcity_metric_wrapper_jensenshannon(data_train, data_synth, data_holdout):
    metric = JensenShannonDistance()
    wrapper = SynthcityMetricWrapper(metric, include_holdout=False, encoder="ordinal")
    scores = wrapper.compute(data_train, data_synth, data_holdout)
    assert isinstance(scores, dict)


def test_synthcity_metric_wrapper_performanceevaluatorlinear(
    data_train, data_synth, data_holdout
):
    metric = PerformanceEvaluatorLinear()
    wrapper = SynthcityMetricWrapper(metric, include_holdout=True, encoder="ordinal")
    scores = wrapper.compute(data_train, data_synth, data_holdout)
    assert isinstance(scores, dict)


def test_synthcity_metric_wrapper_datamismatchscore(
    data_train, data_synth, data_holdout
):
    metric = DataMismatchScore()
    wrapper = SynthcityMetricWrapper(metric, include_holdout=False, encoder="ordinal")
    scores = wrapper.compute(data_train, data_synth, data_holdout)
    assert isinstance(scores, dict)
