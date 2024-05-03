"""Wrapper class for metrics from other libraries, so they can be used in the CRN-Synth framework

Ensures that metrics have a functioning compute() method.
"""
import pandas as pd
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
)

from crnsynth.metrics.base import BaseMetric


class SynthcityMetricWrapper(BaseMetric):
    """Wrapper class for SynthCity metrics to be used in the benchmarking framework.

    This class wraps a SynthCity metric object and provides a compute() method that can be used to evaluate the metric
    on real and synthetic data. The class also handles the encoding of the data prior to computing the metric.

    It converts the data to the appropriate data loader type based on the data_loader_type parameter, which are
    required by the SynthCity metrics. The data loader types are 'generic' and 'survival', and the appropriate data
    loader is selected based on the data_loader_type parameter.

    Note: This class is not intended to be used directly, but rather as a base class for specific metric wrappers.
    """

    def __init__(
        self, metric, data_loader_type="generic", include_holdout=False, encoder=None
    ):
        """Initialize the metric wrapper.

        Args:
            metric: SynthCity metric object
            data_loader_type: str, type of data loader to use, either 'generic' or 'survival'
            include_holdout: bool, whether to include holdout data in the evaluation, only supported by some metrics
            encoder: str or encoder object, used for encoding the data prior to computing the metric
        """
        super().__init__(encoder=encoder)
        self.data_loader_type = data_loader_type
        self.metric = metric
        self.include_holdout = include_holdout
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

    def _init_data_loader(self, data):
        if self.data_loader_type == "generic":
            data_loader = GenericDataLoader(data)
        elif self.data_loader_type == "survival":
            data_loader = SurvivalAnalysisDataLoader(data)
        else:
            raise ValueError(
                f'Invalid data loader type: {self.data_loader_type}, must be "generic" or "survival"'
            )
        return data_loader

    def compute(self, data_real, data_synth, data_holdout=None):
        data_real, data_synth, data_holdout = self.encode(
            data_real, data_synth, data_holdout
        )

        # compute metric with holdout data - note: only supported for some metrics
        if self.include_holdout:
            data_loader_real = self._init_data_loader(data_real)
            data_loader_synth = self._init_data_loader(data_synth)
            data_loader_holdout = self._init_data_loader(data_holdout)

            self.scores_ = self.metric.evaluate(
                data_loader_real, data_loader_synth, data_loader_holdout
            )
        # compute metric without holdout data
        else:
            data_loader_real = self._init_data_loader(data_real)
            data_loader_synth = self._init_data_loader(data_synth)

            self.scores_ = self.metric.evaluate(data_loader_real, data_loader_synth)
        return self.scores_
