"""
Experiment executor, dataset-parameterised (thesis, Experimental setup).

This is the one part of the loop that is NOT a language-model call. The model
chooses what to run; this module runs it and returns real measurements. The same
record is returned regardless of what a configuration's evaluation paragraph
emphasises: cross-validated accuracy on the training split, its across-fold
standard deviation, and accuracy on the held-out validation split.
"""
import sys
import time
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from datasets import load  # noqa: E402
import study_config  # noqa: E402  (read live, so WORKERS can set EXECUTOR_N_JOBS)
from study_config import ALLOWED_PARAMS, CV_FOLDS, SCORING  # noqa: E402


def build(model, params, seed):
    if model == "random_forest":
        # n_jobs is set to 1 when the outer loop is parallelised, so that the
        # forest's internal threads do not oversubscribe the cores.
        return RandomForestClassifier(
            random_state=seed, n_jobs=study_config.EXECUTOR_N_JOBS, **params)
    if model == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"unknown model {model!r}")


def run_experiment(model, params, seed, dataset):
    """Fit one estimator on one dataset. Returns a record or an {'error': ...}."""
    if model not in ALLOWED_PARAMS:
        return {"error": f"unknown model '{model}', must be one of {sorted(ALLOWED_PARAMS)}"}
    bad = set(params) - ALLOWED_PARAMS[model]
    if bad:
        return {"error": f"unsupported params for {model}: {sorted(bad)}"}

    X_tr, X_va, y_tr, y_va = load(dataset)
    t0 = time.time()
    try:
        clf = build(model, params, seed)
        cv = cross_val_score(clf, X_tr, y_tr, cv=CV_FOLDS, scoring=SCORING)
        clf.fit(X_tr, y_tr)
        val = clf.score(X_va, y_va)
    except Exception as exc:                      # noqa: BLE001 - reported to the model
        return {"error": f"experiment failed: {exc}"}

    return {
        "model": model,
        "params": params,
        "dataset": dataset,
        "cv_accuracy_mean": round(float(cv.mean()), 6),
        "cv_accuracy_std": round(float(cv.std()), 6),
        "val_accuracy": round(float(val), 6),
        "wall_time_s": round(time.time() - t0, 3),
    }
