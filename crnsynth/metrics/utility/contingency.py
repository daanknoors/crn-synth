"""Contingency similarity metric"""
from sdmetrics.column_pairs import ContingencySimilarity

from crnsynth.metrics.base import BaseMetric


class ContingencySimilarityScore(BaseMetric):
    """Contingency Similarity by SDMetrics (SDV)

    This metric computes the similarity of a pair of categorical columns between the real and synthetic
    datasets -- aka it compares 2D distributions.

    Computes the difference between contingency tables using total variation distance and
    converts it to a similarity score."""

    def __init__(self, encoder=None, categorical_columns=None, **kwargs):
        """Initialize the metric.

        Args:
            encoder: Encoder to use for encoding the data.
            categorical_columns: List of columns to consider for the metric.
        """
        super().__init__(encoder, **kwargs)
        self.categorical_columns = categorical_columns

    @staticmethod
    def type() -> str:
        return "utility"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(self, data_train, data_synth, data_holdout=None, **kwargs) -> dict:
        """Compute the metric."""
        assert (
            self.categorical_columns is not None
        ), "Categorical columns must be provided"

        # compute the score
        score = ContingencySimilarity.compute(
            real_data=data_train[self.categorical_columns],
            synthetic_data=data_synth[self.categorical_columns],
        )
        return {"score": abs(score)}
