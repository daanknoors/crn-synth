import numpy as np
import pandas as pd
import pytest
from synthcity.metrics.eval_performance import PerformanceEvaluatorLinear
from synthcity.metrics.eval_sanity import DataMismatchScore
from synthcity.metrics.eval_statistical import JensenShannonDistance

from crnsynth.integration.metrics import SynthcityMetricWrapper


@pytest.fixture
def data_train():
    return pd.DataFrame(
        {
            "A": np.random.choice(["a", "b", "c"], 100),
            "B": np.random.rand(100),
            "C": np.random.rand(100),
        }
    )


@pytest.fixture
def data_synth():
    return pd.DataFrame(
        {
            "A": np.random.choice(["a", "b", "c"], 100),
            "B": np.random.rand(100),
            "C": np.random.rand(100),
        }
    )


@pytest.fixture
def data_holdout():
    return pd.DataFrame(
        {
            "A": np.random.choice(["a", "b", "c"], 20),
            "B": np.random.rand(20),
            "C": np.random.rand(20),
        }
    )


def test_synthcity_metric_wrapper_jensenshannon(data_train, data_synth, data_holdout):
    metric = JensenShannonDistance()
    wrapper = SynthcityMetricWrapper(metric, encoder="ordinal")
    scores = wrapper.compute(data_train, data_synth, data_holdout)
    assert isinstance(scores, dict)


def test_synthcity_metric_wrapper_performanceevaluatorlinear(
    data_train, data_synth, data_holdout
):
    metric = PerformanceEvaluatorLinear()
    wrapper = SynthcityMetricWrapper(
        metric, include_holdout=True, encoder="ordinal", target_column="A"
    )
    scores = wrapper.compute(data_train, data_synth, data_holdout)
    assert isinstance(scores, dict)


def test_synthcity_metric_wrapper_datamismatchscore(
    data_train, data_synth, data_holdout
):
    metric = DataMismatchScore()
    wrapper = SynthcityMetricWrapper(metric, include_holdout=False, encoder="ordinal")
    scores = wrapper.compute(data_train, data_synth, data_holdout)
    assert isinstance(scores, dict)
