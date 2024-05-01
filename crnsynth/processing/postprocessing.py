"""Post-processing functions for synthetic data after synthesis"""

from typing import List

import numpy as np
import pandas as pd
from matplotlib.dates import num2date

from crnsynth.processing.generalization import BaseGeneralizationMech


def reverse_generalization(
    data_synth: pd.DataFrame, generalizers: List[BaseGeneralizationMech]
) -> pd.DataFrame:
    """Reverse generalization of synthetic data using a list of generalization mechanisms."""
    data_synth_rev = data_synth.copy()
    for gen_mech in generalizers:
        data_synth_rev = gen_mech.inverse_transform(data_synth_rev)
    return data_synth_rev, generalizers


def numeric_to_date(num, date_format="%Y-%m-%d"):
    """Convert numeric date to date string"""
    return num2date(num).strftime(date_format)


def ensure_column_bound(
    df, column_name, column_bound_name, missing_value=None, random_state=None
):
    """Ensures that values in column_name never exceeds column_bound_name."""
    np.random.seed(random_state)
    df = df.copy()

    # replace column_name with 0 when column_bound_name== 0
    mask_zero_ond = (df[column_name] > df[column_bound_name]) & (
        df[column_bound_name] == 0
    )
    df.loc[mask_zero_ond, column_name] = 0

    if missing_value:
        # replace column_name with missing_value when column_bound_name = missing_value
        mask_missing = (df[column_bound_name] == missing_value) | (
            df[column_bound_name].isna()
        )
        df.loc[mask_missing, column_name] = missing_value

    # replace column_name > column_bound_name with sample(0, column_bound_name)
    mask_invalid_values = df[column_name] > df[column_bound_name]
    df.loc[mask_invalid_values, column_name] = np.random.randint(
        0, df[mask_invalid_values][column_bound_name] + 1
    )
    return df


def skew_lowerbound_to_upperbound(
    df, column_lowerbound, column_upperbound, min_ratio=0.8, random_state=None
):
    """Skews numeric values in the lowerbound are close to the upperbound to become equal."""
    np.random.seed(random_state)
    df = df.copy()

    # get ratio of lowerbound to upperbound
    ratio = df[column_lowerbound] / df[column_upperbound]

    # get indices of rows where ratio exceeds min_ratio
    idx_skew = ratio > min_ratio

    # convert lowerbound to upperbound
    df.loc[idx_skew, column_lowerbound] = df.loc[idx_skew, column_upperbound]
    return df
