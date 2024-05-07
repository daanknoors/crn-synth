from typing import Dict, List, Optional, Union

import pandas as pd

from crnsynth.checks.params import check_param_consistency
from crnsynth.generators.base import BaseGenerator
from crnsynth.processing import postprocessing, preprocessing, utils
from crnsynth.processing.generalization import BaseGeneralizationMech
from crnsynth.synthesization.synthesization import generate_synth_data


def run_synth_pipeline(
    data_real: pd.DataFrame,
    generator: BaseGenerator,
    preprocess_func: Optional[callable] = None,
    postprocess_func: Optional[callable] = None,
    generalizers: Optional[List[BaseGeneralizationMech]] = None,
    holdout_size=0.2,
    target_column: Optional[str] = None,
    n_records: Optional[int] = None,
    output_keys: List[str] = ["train", "holdout", "synth", "generator"],
    random_state: Optional[int] = None,
    verbose: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Generic synthesis pipeline from raw to synth data.

    Flexibility to specify custom preprocessing, generalization, and postprocessing functions.
    Can be used as an example for how to run a synthesis pipeline. Customize to own needs.
    """
    # check if random state is consistent for all classes
    check_param_consistency(
        classes=[generator, *generalizers],
        parameter_name="random_state",
        parameter_value=random_state,
    )

    # split into training and testing data to allow for evaluation with unseen data
    data_train, data_holdout = preprocessing.split_train_holdout(
        data_real,
        holdout_size=holdout_size,
        target_column=target_column,
        random_state=random_state,
    )

    # processing real data
    if preprocess_func is not None:
        data_train = preprocess_func(data_train)

    # generalize data by binning numeric columns or grouping nominal columns
    # note: assign to new dataframe data_train_input to be able to output training data prior to generalization later
    if generalizers is not None:
        data_train_input, generalizers = preprocessing.generalize_data(
            data_train, generalizers
        )
    else:
        data_train_input = data_train.copy()

    # use number of records of real data prior to splitting train/holdout
    n_records = n_records if n_records is not None else data_real.shape[0]

    # generate synthetic data
    data_synth, generator = generate_synth_data(
        data_train_input, generator, n_records=n_records, verbose=verbose
    )

    # reverse generalization of synthetic data
    if generalizers is not None:
        data_synth, generalizers = postprocessing.reverse_generalization(
            data_synth, generalizers
        )

    # postprocess synthetic data
    if postprocess_func is not None:
        data_synth = postprocess_func(data_synth)

    # return output in a dictionary
    output = {
        "train": data_train,
        "input": data_train_input,
        "holdout": data_holdout,
        "synth": data_synth,
        "generator": generator,
        "generalizers": generalizers,
    }
    # reduce output dictionary to only include specified keys
    utils.reduce_dict(output, output_keys)
    return output
