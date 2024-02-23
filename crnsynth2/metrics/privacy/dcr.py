from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from crnsynth2.metrics.base import BaseMetric
from crnsynth2.metrics.privacy.utils import compute_distance_nn


def compute_closest_distances(
    data_train, data_synth, data_holdout, categorical_columns, distance_metric="gower"
):
    distances_test, distances_synth = compute_distance_nn(
        data_train=data_train,
        data_synth=data_synth,
        data_holdout=data_holdout,
        categorical_columns=categorical_columns,
        n_neighbors=1,
        normalize=True,
        distance_metric=distance_metric,
    )

    return distances_test, distances_synth


class DistanceClosestRecord(BaseMetric):
    """Measures the distance from synthetic records to the closest real record.
    The lower the distance, the more similar the synthetic data is to the real data.

    Privacy risk: DCR close to 0, where synthetic data points are close to real data points.
    Compare to holdout to determine an acceptable level. DCR of synthetic data should be equal or higher than the DCR of the
    holdout test set to the training data.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        metric: str = "gower",
        categorical_columns: Union[List[str], None] = None,
        **kwargs: Any
    ) -> None:
        """
        Args:
            quantile (float): Quantile of distances to closest real record to take.
            metric (str): Distance metric to use.
            categorical_columns (List or None): List of categorical columns.
        """
        super().__init__(**kwargs)

        self.quantile = quantile
        self.metric = metric
        self.categorical_columns = categorical_columns

    @staticmethod
    def type() -> str:
        return "privacy"

    @staticmethod
    def name() -> str:
        return "distance_closest_record"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        data_real: pd.DataFrame,
        data_synth: pd.DataFrame,
        data_holdout: pd.DataFrame,
    ) -> Dict:
        if data_holdout is None:
            raise ValueError("Holdout data is required for computing this metric.")

        # compute distances to closest real record
        distances_holdout, distances_synth = compute_closest_distances(
            data_train=data_real,
            data_holdout=data_holdout,
            data_synth=data_synth,
            categorical_columns=self.categorical_columns,
            distance_metric=self.metric,
        )

        # take the quantile of distances to closest real record
        dcr_holdout = np.quantile(distances_holdout[:, 0], self.quantile)
        dcr_synth = np.quantile(distances_synth[:, 0], self.quantile)
        return {"holdout": dcr_holdout, "synth": dcr_synth}
