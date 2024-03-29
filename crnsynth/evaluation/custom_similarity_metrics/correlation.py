"""This metric describes how difficult it is for an attacker to correctly guess the sensitive information using an algorithm called Correct Attribution Probability (CAP)"""
from typing import Any, Dict, List

from pydantic import validate_arguments
from sdmetrics.column_pairs import CorrelationSimilarity
from synthcity.metrics.eval_statistical import StatisticalEvaluator
from synthcity.plugins.core.dataloader import DataLoader


def mean_features_correlation(real_data, synthetic_data):
    """Mean pair-wise feature correlations."""
    return real_data.corrwith(synthetic_data, axis=1, method="pearson").mean()


class FeatureCorrelation(StatisticalEvaluator):
    NUMERICAL_COLS = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "feature_corr"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        score = mean_features_correlation(
            real_data=X_gt.data, synthetic_data=X_syn.data
        )
        # maximize average feature correlation
        return {"score": abs(score)}


class CorrelationSimilarityScore(StatisticalEvaluator):
    """TODO"""

    NUMERICAL_COLS = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "correlation_similarity_score"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @classmethod
    def update_cls_params(cls, params):
        """Update the clip value class method without
        instantiating the class."""
        for name, value in params.items():
            setattr(cls, name, value)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        numerical_cols = [col for col in X_gt.data.columns if "num" in col]
        numerical_cols.append("os_42")
        score = CorrelationSimilarity.compute(
            real_data=X_gt.data[numerical_cols],  # [self.NUMERICAL_COLS],
            synthetic_data=X_syn.data[numerical_cols],  # [self.NUMERICAL_COLS],
            coefficient="Pearson",
        )
        # maximize feature correlation score
        return {"score": abs(score)}
