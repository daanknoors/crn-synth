"""Encoding data for synthesis or metrics"""

from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class DefaultEncoder(ColumnTransformer):
    """Default encoder for preparing data for synthesis or metrics"""

    def __init__(
        self,
        categorical_columns: Union[Iterable, None] = None,
        numerical_columns: Union[Iterable, None] = None,
    ):
        """Initialize the encoder"""
        super().__init__(transformers=[], remainder="passthrough")

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.encoder = None

    def fit(self, data: pd.DataFrame, **kwargs):
        # derive column types from data if not specified
        if self.categorical_columns is None:
            self.categorical_columns = data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        if self.numerical_columns is None:
            self.numerical_columns = data.select_dtypes(
                include=["float64", "int64"]
            ).columns.tolist()

        # assign transformers to ColumnTransformer
        self.transformers = [
            ("cat", OneHotEncoder(drop="if_binary"), self.categorical_columns),
            ("num", StandardScaler(), self.numerical_columns),
        ]
        super().fit(data)
        return self

    def transform(self, data: pd.DataFrame):
        return super().transform(data)


def get_encoder(encoder: str):
    """Get an encoder by name"""
    encoders = {
        "onehot": OneHotEncoder(drop="if_binary"),
        "ordinal": OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        ),
        "default": DefaultEncoder(),
    }

    if encoder not in encoders:
        raise ValueError(f"Encoder {encoder} not found")
    return encoders[encoder]


def convert_encoded_data_to_dataframe(data_enc, encoder, column_names=None):
    """Convert encoded data back to a dataframe"""
    # get the feature names from the encoder
    new_column_names = encoder.get_feature_names_out(input_features=column_names)

    # remove 'remainder__' prefix (in case you used a ColumnTransformer with passthrough)
    new_column_names = [name.replace("remainder__", "") for name in new_column_names]

    # check if the data is sparse and transform it to a dense array, so it can be converted to a DataFrame
    if issparse(data_enc):
        # convert the sparse data to a dense array
        data_enc = data_enc.toarray()

    # convert the encoded data to a DataFrame
    df_enc = pd.DataFrame(
        data_enc,
        columns=new_column_names,
    )
    return df_enc


def encode_data(data, encoder, refit: bool = False, return_dataframe: bool = True):
    """Encode data using a ColumnTransformer"""
    assert (
        encoder is not None
    ), "Encoder is required, specify encoder class or name (str)"
    assert isinstance(data, pd.DataFrame), "Data must be a DataFrame"

    # get encoder if it is a string
    if isinstance(encoder, str):
        encoder = get_encoder(encoder)

    # make a copy of the data to prevent modifying the original
    data = data.copy()

    # ensure data columns are of type string to avoid type errors
    data.columns = data.columns.astype(str)

    # fit the encoder to the data if refit is True or if it is not fitted yet
    if refit:
        encoder.fit(data)
    else:
        try:
            check_is_fitted(encoder)
        except NotFittedError:
            encoder.fit(data)

    # transform the data
    data_enc = encoder.transform(data)

    # convert the transformed data back to a DataFrame
    if return_dataframe:
        data_enc = convert_encoded_data_to_dataframe(
            data_enc, encoder, column_names=data.columns
        )
    return data_enc, encoder
