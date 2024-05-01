from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from crnsynth.metrics.base import BaseMetric


class ClassifierComparison(BaseMetric):
    """Train a classifier on real and synthetic data and compare performance on holdout data."""

    def __init__(
        self,
        target_column,
        model=None,
        score_fn=None,
        encoder=None,
        random_state=42,
        verbose=1,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            target_column: Target column name.
            model: Model to use for training.
            score_fn: Score function to use for evaluation.
            encoder: Encoder to use for encoding the data.
            random_state: Random seed.
            verbose: Verbosity level.
        """
        super().__init__(encoder, **kwargs)
        self.target_column = target_column
        self.model = model
        self.score_fn = score_fn
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def type() -> str:
        return "performance"

    def compute(self, data_train, data_synth, data_holdout=None) -> Dict:
        self._check_args(data_train, data_synth, data_holdout)

        # encode data using encoder
        data_train, data_synth, data_holdout = self.encode(
            data_train, data_synth, data_holdout, return_dataframe=True
        )

        # split datasets
        X_train, y_train = self._split_data(data_train)
        X_synth, y_synth = self._split_data(data_synth)
        X_holdout, y_holdout = self._split_data(data_holdout)

        # train model real training, predict on real holdout
        self.model.fit(X_train, y_train)
        y_real_pred = self.model.predict(X_holdout)

        # train model synth, predict on real holdout
        self.model.fit(X_synth, y_synth)
        y_synth_pred = self.model.predict(X_holdout)

        # compute score
        return {
            "real": self.score_fn(y_holdout, y_real_pred),
            "synth": self.score_fn(y_holdout, y_synth_pred),
        }

    def _split_data(self, data):
        """Split data into features and target."""
        y = data[self.target_column]
        X = data.drop(self.target_column, axis=1)
        return X, y

    def _check_args(self, data_train, data_synth, data_holdout):
        """Check arguments."""
        if data_holdout is None:
            raise ValueError("Holdout data is required for computing this metric.")

        if self.model is None:
            if self.verbose:
                print("No model provided. Using default model: RandomForestClassifier")
            self.model = RandomForestClassifier(
                max_depth=3,
                n_estimators=50,
                random_state=self.random_state,
                warm_start=False,
            )

        if self.score_fn is None:
            if self.verbose:
                print(
                    "No score function provided. Using default score function: accuracy_score"
                )
            self.score_fn = accuracy_score

        if self.target_column not in data_train.columns:
            raise ValueError(
                f"Target column {self.target_column} not found in training data."
            )
        if self.target_column not in data_synth.columns:
            raise ValueError(
                f"Target column {self.target_column} not found in synthetic data."
            )
        if self.target_column not in data_holdout.columns:
            raise ValueError(
                f"Target column {self.target_column} not found in holdout data."
            )
