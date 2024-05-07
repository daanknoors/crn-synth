from typing import Any, Dict, List, Tuple

import numpy as np

from crnsynth.metrics.base import BaseMetric

from .utils import fit_cox, propensity_weights


def cox_beta_score(
    data_hybrid,
    data_real,
    feature_columns,
    target_column,
    duration_column,
    event_column=None,
    clip_value=4,
):
    """Compute hybrid data quality score based on Cox coefficients
    similarity.

    Assumes you are working with an augmented dataset, where a part of the original real dataset is replaced with
    synthetic records. For example, creating a synthetic control arm for a clinical trial.

    Args:
        data_hybrid: Pre-processed hybrid data (real + synthetic).
        data_real: Pre-processed real data (full real data).
        feature_columns: Covariate names.
        target_column: Test column name (eg treatment).
        duration_column: Time to event column for Cox model.
        event_column: Optional event indicator column for Cox model.
        clip_value: Clip value for propensity weights.
    Returns:
        Score value.
    """
    data_hybrid["weights"] = propensity_weights(
        data_hybrid[feature_columns].values,
        data_hybrid[target_column].values,
        clip_value,
    )
    data_real["weights"] = propensity_weights(
        data_real[feature_columns].values, data_real[target_column].values, clip_value
    )

    cox_cols = [target_column, duration_column, "weights"]
    if event_column is not None:
        cox_cols.append(event_column)

    cox_original = fit_cox(
        data_real,
        duration_column,
        cox_cols,
        weights_column="weights",
        event_column=event_column,
    )
    cox_hybrid = fit_cox(
        data_hybrid,
        duration_column,
        cox_cols,
        weights_column="weights",
        event_column=event_column,
    )

    mean_original = cox_original.params_[target_column]
    mean_hybrid = cox_hybrid.params_[target_column]

    return mean_original - mean_hybrid


class CoxBetaScore(BaseMetric):
    """Cox beta score evaluator class."""

    def __init__(
        self,
        clip_value,
        target_column,
        duration_column,
        event_column,
        encoder=None,
        **kwargs: Any
    ) -> None:
        """
        Args:
            clip_value: Clip value for propensity weights.
            target_column: Target column name.
            duration_column: Duration column name.
            event_column: Event column name.
            encoder: Encoder to use for encoding the data.
        """
        super().__init__(encoder=encoder, **kwargs)
        self.clip_value = clip_value
        self.target_column = target_column
        self.duration_column = duration_column
        self.event_column = event_column

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def type() -> str:
        return "performance"

    def compute(self, data_train, data_synth, data_holdout=None, **kwargs):
        feature_columns = list(
            set(data_train.columns)
            - set([self.duration_column, self.target_column, self.event_column])
        )
        score = cox_beta_score(
            data_hybrid=data_synth,
            data_real=data_train,
            feature_columns=feature_columns,
            target_column=self.target_column,
            duration_column=self.duration_column,
            clip_value=self.clip_value,
            event_column=self.event_column,
        )

        # minimize absolute difference
        return {"score": abs(score)}
