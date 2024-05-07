from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from crnsynth.metrics.base import BaseMetric
from crnsynth.metrics.privacy.utils import SMOOTHING_FACTOR, compute_distance_nn


def compute_ratio_distances(
    data_train,
    data_synth,
    data_holdout,
    categorical_columns,
    n_neighbors,
    distance_metric="gower",
):
    distances_test, distances_synth = compute_distance_nn(
        data_train=data_train,
        data_synth=data_synth,
        data_holdout=data_holdout,
        categorical_columns=categorical_columns,
        n_neighbors=n_neighbors,
        normalize=False,
        distance_metric=distance_metric,
    )

    # get the ratio of closest real record by the distance to the n_neighbors (e.g. 1st to 5th) closest real record
    # note add smoothing factor:
    # - to numerator in case of 0 distance (exact match) which otherwise would result in NNDR of 0 regardless of second closest distance
    # - to denominator to avoid division by 0
    ratio_distances_holdout = np.maximum(
        distances_test[:, 0], SMOOTHING_FACTOR
    ) / np.maximum(distances_test[:, n_neighbors - 1], SMOOTHING_FACTOR)

    ratio_distances_synth = np.maximum(
        distances_synth[:, 0], SMOOTHING_FACTOR
    ) / np.maximum(distances_synth[:, n_neighbors - 1], SMOOTHING_FACTOR)
    return ratio_distances_holdout, ratio_distances_synth


class NearestNeighborDistanceRatio(BaseMetric):
    """
    Ratio of the distance between synthetic data's record to the closest and second closest real record.
    Allows comparing outliers and inliers in the population.

    Privacy risk: NNDR close to 0, where synthetic data points are close to real data points in sparse areas of the data space (outliers).
    Compare to holdout to determine an acceptable level. NNDR of synthetic data should be equal or higher than the NNDR of the
    holdout test set to the training data.
    """

    def __init__(
        self,
        encoder="ordinal",
        quantile: float = 0.5,
        distance_metric: str = "gower",
        n_neighbors: int = 2,
        categorical_columns: Union[List[str], None] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            quantile (float): Quantile to take distances to closest real record to take.
            distance_metric (str): Distance metric to use for computing nearest neighbors.
            n_neighbors (int): Number of nearest neighbors to use for computing NNDR.
            categorical_columns (List or None): List of categorical columns.
        """
        super().__init__(encoder=encoder, **kwargs)
        self.quantile = quantile
        self.metric = distance_metric
        self.n_neighbors = n_neighbors
        self.categorical_columns = categorical_columns

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
        data_holdout: Optional[pd.DataFrame] = None,
        **kwargs: Optional[Dict],
    ) -> Dict:
        if data_holdout is None:
            raise ValueError("Holdout data is required for computing this metric.")

        # encode data using encoder
        data_train, data_synth, data_holdout = self.encode(
            data_train, data_synth, data_holdout, return_dataframe=True
        )

        # compute distances to closest real record
        distances_holdout, distances_synth = compute_ratio_distances(
            data_train=data_train,
            data_holdout=data_holdout,
            data_synth=data_synth,
            categorical_columns=self.categorical_columns,
            n_neighbors=self.n_neighbors,
            distance_metric=self.metric,
        )

        # take the quantile of distances to the closest real record
        nndr_holdout = np.quantile(distances_holdout, self.quantile)
        nndr_synth = np.quantile(distances_synth, self.quantile)
        return {"holdout": nndr_holdout, "synth": nndr_synth}
