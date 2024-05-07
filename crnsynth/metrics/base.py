"""Base metric class for CRN synthesis metrics."""

from typing import Optional, Union

import pandas as pd

from crnsynth.processing.encoding import encode_data


class BaseMetric:
    """Base metric class for CRN synthesis metrics."""

    def __init__(self, encoder=None, **kwargs):
        """Initialize the metric."""
        self.encoder = encoder

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def type() -> str:
        """Return the type of the metric."""
        raise NotImplementedError("type() method not implemented")

    @staticmethod
    def direction() -> str:
        """Return the direction of the metric."""
        raise NotImplementedError("direction() method not implemented")

    def compute(
        self,
        data_train: pd.DataFrame,
        data_synth: pd.DataFrame,
        data_holdout: Optional[pd.DataFrame] = None,
        **kwargs: Optional[dict],
    ) -> dict:
        """Compute the metric."""
        raise NotImplementedError("compute() method not implemented")

    def encode(self, data_train, data_synth, data_holdout=None, return_dataframe=True):
        """Encode datasets. Fit encoder on train. Transform train, synth and holdout (optional)."""
        if self.encoder is not None:
            # fit and transform training data
            data_train, self.encoder = encode_data(
                data_train,
                encoder=self.encoder,
                refit=True,
                return_dataframe=return_dataframe,
            )

            # transform synthetic data using the encoder fitted on the training data
            data_synth, _ = encode_data(
                data_synth,
                encoder=self.encoder,
                refit=False,
                return_dataframe=return_dataframe,
            )

            # optional: transform holdout data using the encoder fitted on the training data
            if data_holdout is not None:
                data_holdout, _ = encode_data(
                    data_holdout,
                    encoder=self.encoder,
                    refit=False,
                    return_dataframe=return_dataframe,
                )
        return data_train, data_synth, data_holdout

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
