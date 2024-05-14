from typing import Optional, Union

import numpy as np
import pandas as pd

from crnsynth.generators.base import BaseGenerator


class SchematicGenerator(BaseGenerator):
    """Schematic synthetic data generator. Generates synthetic data based on a schema of the data. Schema
    can be learned from real data but also defined by the user. Removes the need for access to real data
    and thus minimize privacy risks. However, statistical properties of the synthetic data are likely not
    preserved"""

    def __init__(
        self,
        schema: Optional[dict] = None,
        min_support_categorical: Union[int, float] = 0,
        date_format: Optional[str] = "%Y-%m-%d %H:%M:%S",
        verbose: int = 1,
    ):
        """
        Args:
            schema (dict, optional): Schema of the data. If not provided, schema will be learned from input data
            min_support_categorical (int, float): Minimum support for categorical values. If int, it is the minimum number of
                occurrences of a category to be included in the schema. If float, it is the minimum fraction of occurrences
                of a category to be included in the schema. Default is 0, meaning all categories are included.
            date_format (str, optional): Format of datetime columns.
            verbose (int, optional): Verbosity level.
        """
        self.schema = schema
        self.min_support_categorical = min_support_categorical
        self.date_format = date_format
        self.verbose = verbose

    def fit(self, data_real: pd.DataFrame) -> None:
        """Learn schema from input data"""
        if (self.verbose) and (self.schema is not None):
            print("Schema already defined. Overwriting schema based on input data.")

        self.schema = {}
        for col in data_real.columns:
            # check if column is categorical or string
            if data_real[col].dtype in ["object", "category"]:
                counts = data_real[col].value_counts()

                # get categories but remove those with less than min_support_categorical
                if self.min_support_categorical > 1:
                    categories = counts[
                        counts >= self.min_support_categorical
                    ].index.tolist()
                elif 0 < self.min_support_categorical < 1:
                    categories = counts[
                        counts >= self.min_support_categorical * len(data_real)
                    ].index.tolist()
                else:
                    categories = counts.index.tolist()

                self.schema[col] = {"dtype": "object", "categories": categories}
            else:
                self.schema[col] = {
                    "dtype": data_real[col].dtype,
                    "min": data_real[col].min(),
                    "max": data_real[col].max(),
                }

        if self.verbose:
            print("Schema learned from input data:")
            for col, info in self.schema.items():
                print(f"{col}: {info}")

    def set_schema(self, schema: dict) -> None:
        """Set schema manually"""
        self.check_schema(schema)
        self.schema = schema

    @staticmethod
    def check_schema(schema) -> None:
        """Check if schema is valid"""
        pass

    def generate(self, n_records: int) -> pd.DataFrame:
        """Generate synthetic data based on schema"""
        if self.schema is None:
            raise ValueError(
                "Schema not defined. Call fit() method first on input data or "
                "set schema manually using set_schema()."
            )

        self.check_schema(self.schema)

        data = {}
        for col, info in self.schema.items():
            if info["dtype"] == "object":
                data[col] = np.random.choice(info["categories"], size=n_records)
            elif info["dtype"] == "int":
                data[col] = np.random.randint(info["min"], info["max"], size=n_records)
            elif info["dtype"] == "float":
                data[col] = np.random.uniform(info["min"], info["max"], size=n_records)
            elif info["dtype"] == "datetime64[ns]":
                data[col] = np.random.choice(
                    pd.date_range(info["min"], info["max"], periods=1000).strftime(
                        self.date_format
                    ),
                    size=n_records,
                )
            else:
                raise ValueError(f"Column {col} has unsupported dtype {info['dtype']}")
        # convert to DataFrame
        dtypes = {col: info["dtype"] for col, info in self.schema.items()}
        df_synth = pd.DataFrame(data).astype(dtypes)
        return df_synth
