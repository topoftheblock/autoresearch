"""
Step 6: Causal validation.

Held-out configuration (M=0,B=0,S=0,O=0,E=1 -> "narrow, unspecified metric,
fixed 3-experiment budget, terse, exploit-first") was NOT among the 8
Plackett-Burman configurations used to fit the surrogate in Step 5. We use the
fitted linear surrogate to predict its best_cv_accuracy and n_experiments
BEFORE running it, then compare against the actually observed values from two
fresh runs (seeds 2 and 3, not reused from the ablation study).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from surrogate import AXES, load_config_level_table, fit_linear_main_effects

HELD_OUT = {"M": 0, "B": 0, "S": 0, "O": 0, "E": 1}


def predict_held_out(y_col: str) -> float:
    df = load_config_level_table()
    model = fit_linear_main_effects(df, y_col)
    x_row = {axis: (1.0 if HELD_OUT[axis] == 1 else -1.0) for axis in AXES}
    x_row["const"] = 1.0
    import pandas as pd
    X = pd.DataFrame([x_row])[["const"] + AXES]
    pred = model.predict(X).iloc[0]
    return float(pred)


if __name__ == "__main__":
    for y_col in ["best_cv_accuracy", "n_experiments"]:
        pred = predict_held_out(y_col)
        print(f"predicted {y_col} for held-out config {HELD_OUT}: {pred:.5f}")
