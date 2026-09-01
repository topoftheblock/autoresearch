"""
prepare.py -- the data and the score. Nobody edits this file.

The first of the three files of the autoresearch contract (thesis, Introduction).
It fixes what every experiment is measured on and how it is scored, so that a
score is comparable across configurations, across replicates and across the whole
study. Neither the agent nor the instruction file can reach it: the agent never
sees this module, and no level of any component in program.md changes what it
returns. That is what makes a number in a transcript a measurement rather than a
claim.

Two responsibilities, and no others:

  load(dataset)              the fixed train/validation partition for one task
  score(estimator, dataset)  the record every experiment is scored by

Every task uses the identical split protocol: one stratified 70/30 partition
fixed with the same seed across all runs, so the split is a constant of the study
rather than a source of variation (thesis, Experimental setup). The score is the
same record whatever a configuration's evaluation paragraph emphasises:
cross-validated accuracy on the training split, its across-fold standard
deviation, and accuracy on the held-out validation split.

Adding a dataset here does not put it in the study; the suite is whatever
study_config.DATASETS says.
"""
import sys
from functools import lru_cache
from pathlib import Path

from sklearn import datasets as skds
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent / "config"))
from study_config import CV_FOLDS, SCORING, SPLIT_SEED, TEST_SIZE  # noqa: E402

# name -> zero-argument loader returning (X, y)
LOADERS = {
    "breast_cancer": lambda: (lambda d: (d.data, d.target))(skds.load_breast_cancer()),
    "wine":          lambda: (lambda d: (d.data, d.target))(skds.load_wine()),
    "digits":        lambda: (lambda d: (d.data, d.target))(skds.load_digits()),
    "iris":          lambda: (lambda d: (d.data, d.target))(skds.load_iris()),
}


@lru_cache(maxsize=None)
def load(name):
    """Return the fixed split for one dataset as (X_train, X_val, y_train, y_val)."""
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")
    X, y = LOADERS[name]()
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )


def score(estimator, dataset):
    """The one scoring procedure of the study. Returns the record, or raises."""
    X_tr, X_va, y_tr, y_va = load(dataset)
    cv = cross_val_score(estimator, X_tr, y_tr, cv=CV_FOLDS, scoring=SCORING)
    estimator.fit(X_tr, y_tr)
    val = estimator.score(X_va, y_va)
    return {
        "cv_accuracy_mean": round(float(cv.mean()), 6),
        "cv_accuracy_std": round(float(cv.std()), 6),
        "val_accuracy": round(float(val), 6),
    }


def describe(name):
    import numpy as np
    X_tr, X_va, y_tr, y_va = load(name)
    return {
        "dataset": name,
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_va.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "n_classes": int(len(np.unique(y_tr))),
    }
