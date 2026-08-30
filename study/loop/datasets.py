"""
Dataset registry (thesis 5.4).

Every task uses the identical split protocol: one stratified 70/30 partition
fixed with the same seed across all runs, so the split is a constant of the
study rather than a source of variation.

Only loaders for datasets named in study_config.DATASETS are used. Adding a
dataset here does not put it in the study; the suite is whatever the config says.
"""
import sys
from pathlib import Path
from functools import lru_cache

from sklearn import datasets as skds
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from study_config import TEST_SIZE, SPLIT_SEED  # noqa: E402

# name -> zero-argument loader returning (X, y)
LOADERS = {
    "breast_cancer": lambda: (lambda d: (d.data, d.target))(skds.load_breast_cancer()),
    "wine":          lambda: (lambda d: (d.data, d.target))(skds.load_wine()),
    "digits":        lambda: (lambda d: (d.data, d.target))(skds.load_digits()),
    "iris":          lambda: (lambda d: (d.data, d.target))(skds.load_iris()),
}


@lru_cache(maxsize=None)
def load(name):
    """Return the fixed split for one dataset as a 4-tuple."""
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")
    X, y = LOADERS[name]()
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )


def describe(name):
    X_tr, X_va, y_tr, y_va = load(name)
    import numpy as np
    return {
        "dataset": name,
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_va.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "n_classes": int(len(np.unique(y_tr))),
    }
