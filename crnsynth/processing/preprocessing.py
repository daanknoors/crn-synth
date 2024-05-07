"""Pre-processing functions for real data before synthesis"""

from datetime import datetime
from typing import List, Optional, Union

import pandas as pd
from matplotlib.dates import date2num
from sklearn.model_selection import train_test_split

from crnsynth.processing.generalization import BaseGeneralizationMech


def split_train_holdout(
    data_real: pd.DataFrame,
    target_column=None,
    holdout_size: float = 0.2,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and holdout sets. Optionally stratify based on target column."""
    if holdout_size > 0:
        # stratify based on target column if specified
        stratify = None if target_column is None else data_real[target_column]

        data_train, data_holdout = train_test_split(
            data_real,
            test_size=holdout_size,
            stratify=stratify,
            random_state=random_state,
        )
    else:
        data_train = data_real
        data_holdout = pd.DataFrame()
    return data_train, data_holdout


def generalize_data(
    data_real: pd.DataFrame, generalizers: List[BaseGeneralizationMech]
) -> pd.DataFrame:
    """Generalize data using a list of generalization mechanisms."""
    data_gen = data_real.copy()
    for gen_mech in generalizers:
        data_gen = gen_mech.fit_transform(data_gen)
    return data_gen, generalizers


def date_to_numeric(dates, date_format="%Y-%m-%d"):
    """Convert date string to numeric date"""
    return date2num(
        dates.apply(lambda row: datetime.strptime(row, date_format))
    ).astype(int)
