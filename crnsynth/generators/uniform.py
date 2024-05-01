import numpy as np
import pandas as pd
from synthesis.synthesizers.marginal import UniformSynthesizer

from crnsynth.generators.base import BaseGenerator


class UniformGenerator(BaseGenerator):
    """Generate synthetic data by sampling columns using the uniform distribution."""

    def __init__(self, epsilon: float, **kwargs) -> None:
        """Initialize the generator.

        Args:
            epsilon (float): privacy budget for differential privacy.
        """
        super().__init__()
        self.epsilon = epsilon
        self.model = UniformSynthesizer(epsilon=epsilon)

    def fit(self, data_real) -> None:
        """Fit the model to the real data."""
        self.model.fit(data_real)

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate records based on the uniform distribution."""
        return self.model.sample(n_records)
