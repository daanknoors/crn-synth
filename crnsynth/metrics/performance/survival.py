from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from pydantic import validate_arguments
from scipy.integrate import trapezoid

from crnsynth.metrics.base import BaseMetric

from .utils import fit_flexible_parametric_model, fit_kaplanmeier


def median_survival_score(data_hybrid, data_real, duration_column, event_column):
    """Deviation between the median survival times in the original and
    synthetic data. Survival curves are estimated with the Kaplan-Meier method.

    Optimal score value is zero.
    """
    km_original = fit_kaplanmeier(data_real[duration_column], data_real[event_column])
    km_hybrid = fit_kaplanmeier(data_hybrid[duration_column], data_hybrid[event_column])

    S_original = km_original.median_survival_time_
    S_hybrid = km_hybrid.median_survival_time_

    # scale to unit range
    Tmax = max(km_original.timeline.max(), km_hybrid.timeline.max())
    return abs(S_hybrid - S_original) / Tmax


class MedianSurvivalScore(BaseMetric):
    """Median survival score."""

    def __init__(
        self, duration_column, event_column, encoder=None, **kwargs: Any
    ) -> None:
        super().__init__(encoder=encoder, **kwargs)
        self.duration_column = duration_column
        self.event_column = event_column

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def type() -> str:
        return "performance"

    def compute(
        self,
        data_train: pd.DataFrame,
        data_synth: pd.DataFrame,
        data_holdout: Optional[pd.DataFrame] = None,
    ) -> dict:
        data_train, data_synth, data_holdout = self.encode(
            data_train=data_train,
            data_synth=data_synth,
            data_holdout=data_holdout,
            return_dataframe=True,
        )

        score = median_survival_score(
            data_hybrid=data_synth,
            data_real=data_train,
            duration_column=self.duration_column,
            event_column=self.event_column,
        )
        return {"score": score}


def predicted_median_survival_score(
    data_train,
    data_synth,
    data_holdout,
    feature_columns,
    duration_column,
    event_column,
):
    """Predicted median survival score."""
    fit_cols = list(feature_columns) + [event_column, duration_column]

    fpm_real = fit_flexible_parametric_model(
        data_train, duration_column, fit_cols, event_column=event_column
    )
    fpm_synth = fit_flexible_parametric_model(
        data_synth, duration_column, fit_cols, event_column=event_column
    )

    Tmax = max(data_train[duration_column].max(), data_synth[duration_column].max())
    Tmin = min(data_train[duration_column].min(), data_synth[duration_column].min())
    Tmin = max(0, Tmin)

    times = np.linspace(Tmin, Tmax, 200)

    # predict median survival for each data point
    S_real = fpm_real.predict_survival_function(data_holdout[fit_cols], times=times)
    S_synth = fpm_synth.predict_survival_function(data_holdout[fit_cols], times=times)

    if np.invert(np.isfinite(S_real.values)).any():
        raise ValueError("predicted median: non-finite in S_real")
    if np.invert(np.isfinite(S_synth.values)).any():
        raise ValueError("predicted median: non-finte in S_synth")

    score = trapezoid(abs(S_synth.values - S_real.values)) / Tmax
    return np.mean(score)


class PredictedMedianSurvivalScore(BaseMetric):
    """Predicted median survival score."""

    def __init__(
        self,
        feature_columns,
        duration_column,
        event_column,
        encoder=None,
        **kwargs: Any
    ) -> None:
        super().__init__(encoder=encoder, **kwargs)
        self.feature_columns = feature_columns
        self.duration_column = duration_column
        self.event_column = event_column

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def type() -> str:
        return "performance"

    def compute(
        self,
        data_train: pd.DataFrame,
        data_synth: pd.DataFrame,
        data_holdout: Optional[pd.DataFrame] = None,
    ) -> dict:
        data_train, data_synth, data_holdout = self.encode(
            data_train=data_train,
            data_synth=data_synth,
            data_holdout=data_holdout,
            return_dataframe=True,
        )

        score = predicted_median_survival_score(
            data_train=data_train,
            data_synth=data_synth,
            data_holdout=data_holdout,
            feature_columns=self.feature_columns,
            duration_column=self.duration_column,
            event_column=self.event_column,
        )
        return {"score": score}
