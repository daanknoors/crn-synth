from sdmetrics.column_pairs import CorrelationSimilarity

from crnsynth.metrics.base import BaseMetric


def mean_features_correlation(data_real, data_synth):
    """Mean pair-wise feature correlations."""
    return data_real.corrwith(data_synth, axis=1, method="pearson").mean()


class FeatureCorrelation(BaseMetric):
    """Compute mean pair-wise feature correlations between real and synthetic data."""

    def __init__(self, encoder=None, numerical_columns=None, **kwargs):
        """
        Initialize the metric.

        Args:
            encoder: Encoder to use for encoding the data.
            numerical_columns: List of numerical columns to consider for the metric.
        """
        super().__init__(encoder=encoder, **kwargs)
        self.numerical_columns = numerical_columns

    @staticmethod
    def type() -> str:
        return "utility"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        data_train,
        data_synth,
        data_holdout=None,
    ) -> dict:
        """Compute the metric."""
        assert self.numerical_columns is not None, "Numerical columns must be provided"

        # compute the score
        score = mean_features_correlation(data_real=data_train, data_synth=data_synth)
        return {"score": score}


class CorrelationSimilarityScore(BaseMetric):
    """Correlation Similarity by SDMetrics (SDV)

    This metric measures the correlation between a pair of numerical columns and computes the similarity between the
    real and synthetic data -- aka it compares the trends of 2D distributions.
    """

    def __init__(self, encoder=None, numerical_columns=None, **kwargs):
        super().__init__(encoder, **kwargs)
        self.numerical_columns = numerical_columns

    @staticmethod
    def type() -> str:
        return "utility"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        data_train,
        data_synth,
        data_holdout=None,
    ) -> dict:
        """Compute the metric."""
        assert self.numerical_columns is not None, "Numerical columns must be provided"

        # compute the score
        score = CorrelationSimilarity.compute(
            real_data=data_train[self.numerical_columns],
            synthetic_data=data_synth[self.numerical_columns],
        )
        return {"score": abs(score)}
