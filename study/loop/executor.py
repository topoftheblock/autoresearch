"""
Executes real experiments (Random Forest / Gradient Boosting fits on the fixed
breast-cancer split). This is the one part of the loop that is NOT an LLM call -
the harness, not the model, actually runs sklearn. The model only chooses what
to run and reads back real numbers.
"""
import time

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

from dataset import X_train, X_val, y_train, y_val

ALLOWED_PARAMS = {
    "random_forest": {"n_estimators", "max_depth", "min_samples_leaf", "max_features"},
    "gradient_boosting": {"n_estimators", "max_depth", "learning_rate", "subsample"},
}


def run_experiment(model: str, params: dict, seed: int) -> dict:
    if model not in ALLOWED_PARAMS:
        return {"error": f"unknown model '{model}', must be random_forest or gradient_boosting"}
    bad = set(params) - ALLOWED_PARAMS[model]
    if bad:
        return {"error": f"unsupported params for {model}: {sorted(bad)}"}

    t0 = time.time()
    try:
        if model == "random_forest":
            clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
        else:
            clf = GradientBoostingClassifier(random_state=seed, **params)

        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
        clf.fit(X_train, y_train)
        val_accuracy = clf.score(X_val, y_val)
    except Exception as e:
        return {"error": f"experiment failed: {e}"}

    return {
        "model": model,
        "params": params,
        "cv_accuracy_mean": round(float(cv_scores.mean()), 5),
        "cv_accuracy_std": round(float(cv_scores.std()), 5),
        "val_accuracy": round(float(val_accuracy), 5),
        "wall_time_s": round(time.time() - t0, 3),
    }
