import numpy as np
import pandas as pd
import pytest

from crnsynth.metrics.privacy.catknn import CategoricalKNNScore


def test_catknn_initialization():
    """Test the initialization of the catknn metric."""
    catknn = CategoricalKNNScore()

    assert catknn.categorical_columns is None
    assert catknn.random_state is None

    # categorical_columns need to be set
    with pytest.raises(ValueError):
        catknn._check_params()


def test_catknn_compute():
    """Test the computation of the catknn metric."""

    # define datasets
    data_train = pd.DataFrame(
        {
            "a": ["a", "b", "c", "d", "e"],
            "b": ["a", "b", "c", "d", "e"],
            "c": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    data_synth = pd.DataFrame(
        {
            "a": ["a", "b", "b", "c", "e"],
            "b": ["a", "b", "a", "c", "b"],
            "c": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    data_holdout = None

    # set params
    categorical_columns = ["a", "b"]
    frac_sensitive = 0.5
    random_state = 42

    catknn = CategoricalKNNScore(
        categorical_columns=categorical_columns,
        frac_sensitive=frac_sensitive,
        random_state=random_state,
    )
    catknn_score = catknn.compute(data_train, data_synth, data_holdout)
    assert isinstance(catknn_score["score"], float)
