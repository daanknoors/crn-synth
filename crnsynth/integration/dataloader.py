"""Data loader integration for data that was already split in train and test sets"""

from typing import Any

from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
)


class GenericDataLoaderSplit(GenericDataLoader):
    """Generic data loader that has already been split in train and test sets"""

    def __init__(self, data, data_test, **kwargs):
        """Initialize the data loader

        Args:
            data: training data
            data_test: holdout data
        """
        super().__init__(data=data, **kwargs)
        self.data_test = data_test

    def train(self) -> "DataLoader":
        return self.decorate(self.data)

    def test(self) -> "DataLoader":
        return self.decorate(self.data_test)


class SurvivalDataLoaderSplit(SurvivalAnalysisDataLoader):
    """Survival analysis data loader that has already been split in train and test sets"""

    def __init__(self, data, data_test, **kwargs):
        """Initialize the data loader

        Args:
           data: training data
           data_test: holdout data
        """
        super().__init__(data=data, **kwargs)
        self.data_test = data_test

    def train(self) -> "DataLoader":
        return self.decorate(self.data)

    def test(self) -> "DataLoader":
        return self.decorate(self.data_test)
