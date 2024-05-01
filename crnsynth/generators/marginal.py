from typing import Any, List

import pandas as pd
from synthesis.synthesizers.marginal import MarginalSynthesizer

from crnsynth.generators.base import BaseGenerator


class MarginalGenerator(BaseGenerator):
    """Marginal Generator based on the MarginalSynthesizer from synthesis

    Generate records based on marginal distribution of columns independently.
    """

    def __init__(self, epsilon: float, verbose=1, **kwargs: Any) -> None:
        """Initialize the generator.

        Args:
            epsilon: privacy budget for differential privacy.
            verbose: Verbosity level.
        """

        super().__init__()
        self.model = MarginalSynthesizer(epsilon=epsilon, verbose=verbose)
        self.epsilon = epsilon

    def fit(self, data_real: pd.DataFrame) -> None:
        """Fit the model to the real data."""
        self.model.fit(data_real)

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate records based on the marginal distribution."""
        return self.model.sample(n_records)
