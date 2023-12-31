"""Utility for custom privacy metrics."""
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


def compute_distance_nn(df_train, df_test, df_synth, categorical_cols=None):
    """Compute distance to closest real record for each synthetic record.
    Normalize using holdout test data."""
    # convert data loader to dataframe
    if not isinstance(df_train, pd.DataFrame):
        df_train = df_train.dataframe()
        df_test = df_test.dataframe()
        df_synth = df_synth.dataframe()

    # define column types
    if categorical_cols is None:
        categorical_cols = df_train.select_dtypes(exclude=np.number).columns
        numeric_cols = df_train.select_dtypes(include=np.number).columns
    else:
        numeric_cols = df_train.columns.difference(categorical_cols)

    # process data before model training
    transformer = make_column_transformer(
        (SimpleImputer(missing_values=np.nan, strategy="mean"), numeric_cols),
        (OneHotEncoder(), categorical_cols),
        remainder="passthrough",
    )

    # run on all data to ensure all categories are encoded
    transformer.fit(pd.concat([df_train, df_test, df_synth], axis=0))
    df_train_hot = transformer.transform(df_train)
    df_test_hot = transformer.transform(df_test)
    df_synth_hot = transformer.transform(df_synth)

    # train nearest neighbors on training data
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="l2", n_jobs=-1)
    nn.fit(df_train_hot)

    # nearest-neighbor search for test and synthetic data
    dist_test, _ = nn.kneighbors(df_test_hot)
    dist_synth, _ = nn.kneighbors(df_synth_hot)

    # normalize DCR using 0.95 quantile of test data
    # use smoothing factor to avoid division by zero
    dist_test = np.square(dist_test)
    dist_synth = np.square(dist_synth)
    bound = np.maximum(np.quantile(dist_test[~np.isnan(dist_test)], 0.95), 1e-8)
    norm_dist_test = np.where(dist_test <= bound, dist_test / bound, 1)
    norm_dist_synth = np.where(dist_synth <= bound, dist_synth / bound, 1)

    return norm_dist_test, norm_dist_synth
