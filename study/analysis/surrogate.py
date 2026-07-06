"""
Step 5: Surrogate model (the white box).

Fits, for each behavioral feature y, a linear model y ~ M + B + S + O + E using
+/-1 coded axes (so the design is orthogonal and coefficients are directly
interpretable as main effects), plus a shallow decision tree as a
non-parametric cross-check. Because the ablation design is resolution III
(Plackett-Burman, 8 runs for 5 factors), only main effects are identifiable;
two-way interactions are confounded with each other and are NOT estimated
here. That is stated explicitly in every report this script produces.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor, export_text

AXES = ["M", "B", "S", "O", "E"]
TARGETS = [
    "best_cv_accuracy",
    "n_experiments",
    "n_distinct_models",
    "breadth_spread",
    "mean_proposal_chars",
]

TABLE_PATH = Path(__file__).parent.parent / "encoding" / "behavioral_table.json"


def load_config_level_table() -> pd.DataFrame:
    rows = json.loads(TABLE_PATH.read_text())
    df = pd.DataFrame(rows)
    # average over seeds/repeats per configuration, per Step 3's design
    agg = df.groupby(["config_id"] + AXES)[TARGETS].mean().reset_index()
    return agg


def fit_linear_main_effects(df: pd.DataFrame, y_col: str):
    X = df[AXES].replace({0: -1, 1: 1}).astype(float)
    X = sm.add_constant(X)
    y = df[y_col].astype(float)
    model = sm.OLS(y, X).fit()
    return model


def fit_tree(df: pd.DataFrame, y_col: str, max_depth=3):
    X = df[AXES].values
    y = df[y_col].values
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    tree.fit(X, y)
    return tree


def variance_shares(model) -> dict:
    """Approximate variance share per axis for an orthogonal +/-1 design:
    since regressors are mutually orthogonal, each main-effect coefficient's
    contribution to R^2 is beta_i^2 / sum(beta_j^2 for j != const)."""
    params = model.params.drop("const")
    sq = params ** 2
    total = sq.sum()
    if total == 0:
        return {k: 0.0 for k in params.index}
    return (sq / total).to_dict()


def run():
    df = load_config_level_table()
    report_lines = ["# Surrogate model report (Step 5)\n"]
    report_lines.append(
        "Design: Plackett-Burman resolution III, 5 axes, 8 configurations, "
        "2 repeats each, averaged over repeats before fitting.\n"
    )
    report_lines.append("Configuration-level table:\n")
    report_lines.append(df.to_string(index=False))
    report_lines.append("\n")

    results = {}
    for y_col in TARGETS:
        model = fit_linear_main_effects(df, y_col)
        shares = variance_shares(model)
        tree = fit_tree(df, y_col)

        report_lines.append(f"\n## Target: {y_col}\n")
        report_lines.append(f"R^2 (main-effects linear model) = {model.rsquared:.3f}\n")
        report_lines.append("Main effect coefficients (change in y from level 0 -> level 1, "
                             "i.e. 2x the OLS coefficient on the +/-1 coded axis):\n")
        for axis in AXES:
            coef = model.params[axis]
            pval = model.pvalues[axis]
            share = shares.get(axis, 0.0)
            report_lines.append(
                f"  {axis}: effect={2*coef:+.5f}  (p={pval:.3f}, variance share={share:.1%})"
            )
        report_lines.append("\nShallow decision tree (depth<=3):\n")
        report_lines.append(export_text(tree, feature_names=AXES))

        results[y_col] = {
            "r_squared": model.rsquared,
            "effects": {axis: float(2 * model.params[axis]) for axis in AXES},
            "pvalues": {axis: float(model.pvalues[axis]) for axis in AXES},
            "variance_share": {axis: float(shares.get(axis, 0.0)) for axis in AXES},
        }

    report_path = Path(__file__).parent / "surrogate_report.txt"
    report_path.write_text("\n".join(report_lines))

    results_path = Path(__file__).parent / "surrogate_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"wrote {report_path}")
    print(f"wrote {results_path}")
    return results


if __name__ == "__main__":
    run()
