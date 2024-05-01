"""This metric describes how difficult it is for an attacker to correctly guess the sensitive information using an algorithm called Correct Attribution Probability (CAP)"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sdmetrics.single_table import CategoricalCAP
from sklearn.preprocessing import OrdinalEncoder

from crnsynth.metrics.base import BaseMetric
from crnsynth.processing.utils import sample_subset


class CategoricalCAPScore(BaseMetric):
    """Categorical Correct Attribution Probability (CAP) score metric. This metric describes how difficult it is for an
    attacker to correctly guess the sensitive information based on a synthetic dataset and a fraction of the variables
    in the original dataset.
    """

    def __init__(
        self,
        categorical_columns=None,
        frac_sensitive=0.5,
        encoder="ordinal",
        random_state=None,
        **kwargs: Any
    ) -> None:
        """
        Args:
            categorical_columns: List of columns to consider for the metric.
            frac_sensitive: Fraction of sensitive columns to consider.
            encoder: Encoder to use for encoding the data.
            random_state: Random seed.
        """
        super().__init__(encoder=encoder, **kwargs)
        self.categorical_columns = categorical_columns
        self.frac_sensitive = frac_sensitive
        self.random_state = random_state

    @staticmethod
    def type() -> str:
        return "privacy"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        data_train: pd.DataFrame,
        data_synth: pd.DataFrame,
        data_holdout: Union[pd.DataFrame, None] = None,
    ) -> Dict:
        self._check_params()

        # subset data to categorical columns
        data_train = data_train[self.categorical_columns].copy()
        data_synth = data_synth[self.categorical_columns].copy()

        # select sensitive and known columns
        n_sensitive = int(len(self.categorical_columns) * self.frac_sensitive)
        sensitive_columns, known_columns = sample_subset(
            self.categorical_columns,
            size=n_sensitive,
            replace=False,
            random_state=self.random_state,
            return_residual=True,
        )

        # optional: encode data using encoder
        data_train, data_synth, _ = self.encode(
            data_train, data_synth, data_holdout, return_dataframe=True
        )

        # compute CAP score
        score = CategoricalCAP.compute(
            real_data=data_train,
            synthetic_data=data_synth,
            key_fields=known_columns,
            sensitive_fields=sensitive_columns,
        )
        return {"score": score}

    def _check_params(self):
        if self.categorical_columns is None:
            raise ValueError("categorical_columns is required.")
        if not 0 < self.frac_sensitive < 1:
            raise ValueError("frac_sensitive must be between 0 and 1.")
