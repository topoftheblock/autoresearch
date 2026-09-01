"""
train.py -- the model and the training loop.

The second file of the autoresearch contract (thesis, Introduction). In
Karpathy's loop this is the file the agent edits, and it is the only one it may
edit. The loop studied here narrows the search space from arbitrary code to a
hyperparameter space, so that the instruction file stays the only free variable
(thesis, The loop under study). The narrowing changes how the agent reaches this
file, not which file it acts on: instead of rewriting the source, the agent names
a model family and hyperparameter values in a JSON proposal, and this module
builds and trains exactly that estimator. Everything the agent can vary about the
model lives here; nothing it can vary about the measurement does, that being
prepare.py.

The whitelist is the boundary of the search space. A proposal naming an unknown
family, or a hyperparameter outside the whitelist for its family, is rejected
here and never reaches the scoring procedure, which is what keeps a malformed
proposal from costing an experiment.

    run_experiment(model, params, seed, dataset) -> record | {"error": ...}
"""
import sys
import time
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / "config"))
import prepare  # noqa: E402
import study_config  # noqa: E402  (read live, so WORKERS can set EXECUTOR_N_JOBS)
from study_config import ALLOWED_PARAMS  # noqa: E402


def build(model, params, seed):
    """The estimator named by a proposal. Raises on an unknown family."""
    if model == "random_forest":
        # n_jobs is set to 1 when the outer loop is parallelised, so that the
        # forest's internal threads do not oversubscribe the cores.
        return RandomForestClassifier(
            random_state=seed, n_jobs=study_config.EXECUTOR_N_JOBS, **params)
    if model == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"unknown model {model!r}")


def validate(model, params):
    """Whitelist check. Returns an error string, or None if the proposal is legal."""
    if model not in ALLOWED_PARAMS:
        return f"unknown model '{model}', must be one of {sorted(ALLOWED_PARAMS)}"
    bad = set(params) - ALLOWED_PARAMS[model]
    if bad:
        return f"unsupported params for {model}: {sorted(bad)}"
    return None


def run_experiment(model, params, seed, dataset):
    """Train one estimator on one dataset and score it. Record, or {'error': ...}."""
    err = validate(model, params)
    if err:
        return {"error": err}

    t0 = time.time()
    try:
        record = prepare.score(build(model, params, seed), dataset)
    except Exception as exc:                      # noqa: BLE001 - reported to the model
        return {"error": f"experiment failed: {exc}"}

    return {"model": model, "params": params, "dataset": dataset, **record,
            "wall_time_s": round(time.time() - t0, 3)}
