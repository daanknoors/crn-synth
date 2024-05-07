"""Wrapper class for metrics from other libraries, so they can be used in the CRN-Synth framework

Ensures that metrics have a functioning compute() method.
"""
import pandas as pd

from crnsynth.integration.dataloader import (
    GenericDataLoaderSplit,
    SurvivalAnalysisDataLoader,
)
from crnsynth.metrics.base import BaseMetric


class SynthcityMetricWrapper(BaseMetric):
    """Wrapper class for SynthCity metrics to be used in the benchmarking framework.

    This class wraps a SynthCity metric object and provides .compute() method that can be used to evaluate the metric
    on real and synthetic data. The class also handles the encoding of the data prior to computing the metric.

    It converts the data to the appropriate data loader type based on the data_loader_type parameter, which are
    required by the SynthCity metrics. The data loader types are 'generic' and 'survival', and the appropriate data
    loader is selected based on the data_loader_type parameter.

    The Split data loader allows you to provide your own desired train and test data sets, instead of randomly
    sampling splits from the data in order to be consistent with the train/holdout split used during training of
    the synthetic data generator.
    """

    def __init__(self, metric, data_loader_type="generic", encoder=None, **kwargs):
        """Initialize the metric wrapper.

        Args:
            metric: SynthCity metric object
            data_loader_type: str, type of data loader to use, either 'generic' or 'survival'
            encoder: str or encoder object, used for encoding the data prior to computing the metric
        """
        super().__init__(encoder=encoder)
        self.data_loader_type = data_loader_type
        self.metric = metric
        self.scores_ = {}

    @property
    def name(self) -> str:
        return self.metric.__class__.__name__

    @staticmethod
    def type(self) -> str:
        return self.metric.type()

    @staticmethod
    def direction(self) -> str:
        return self.metric.direction()

    def _init_data_loader(self, data_train, data_holdout, data_loader_kwargs):
        """Initialize the data loader based on the data_loader_type parameter.

        Use custom data loader for already split data to ensure consistency with the train/holdout split used during
        training of the synthetic data generator.
        """
        if self.data_loader_type == "generic":
            data_loader = GenericDataLoaderSplit(
                data=data_train, data_test=data_holdout, **data_loader_kwargs
            )
        elif self.data_loader_type == "survival":
            data_loader = SurvivalAnalysisDataLoader(
                data=data_train, data_test=data_holdout, **data_loader_kwargs
            )
        else:
            raise ValueError(
                f'Invalid data loader type: {self.data_loader_type}, must be "generic" or "survival"'
            )
        return data_loader

    def compute(self, data_train, data_synth, data_holdout=None, **kwargs):
        """Compute the synthcity metric."""
        # encode the data
        data_train, data_synth, data_holdout = self.encode(
            data_train, data_synth, data_holdout
        )

        # initialize the data loaders
        data_loader_real = self._init_data_loader(data_train, data_holdout, kwargs)
        data_loader_synth = self._init_data_loader(data_synth, data_holdout, kwargs)

        # compute the metric
        self.scores_ = self.metric.evaluate(data_loader_real, data_loader_synth)
        return self.scores_
