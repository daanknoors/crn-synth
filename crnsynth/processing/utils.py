"""Utility functions for processing data."""
import numpy as np


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reduce_dict(dictionary, keys_to_keep):
    """Reduce dictionary to only include specified keys."""
    return {k: v for k, v in dictionary.items() if k in keys_to_keep}


def sample_subset(items, size, replace=False, random_state=None, return_residual=True):
    """Sample a subset from a collection of items."""

    # sample a subset of items
    if random_state is not None:
        rnd = np.random.RandomState(random_state)
        sample = rnd.choice(items, size=size, replace=replace)

    else:
        sample = np.random.choice(items, size=size, replace=replace)

    # convert to list
    sample = list(sample)

    # optional: return items not in the sample
    if return_residual:
        return sample, list(set(items) - set(sample))

    return sample
