"""
Estimation and inference (thesis, Regression model and inference).

Fits, on the individual runs rather than the cell means, the two models the
thesis specifies. With X_i the +/-1 coding of eq. (1) and D_j an indicator for
dataset j (the first entry of study_config.DATASETS is the omitted reference
level), the main-effects model of eq. (2) is

    Y = b0 + sum_i b_i X_i + sum_j g_j D_j + eps

and the saturated two-way model of eq. (3) adds every X_i X_j product.

The g_j are dataset fixed effects. Running the factorial across a task suite
without them leaves the baseline difficulty of each task -- its headroom above
a0(d), its response to the whitelist -- inside eps, where it inflates every
standard error and, in any design that is not perfectly balanced across tasks,
biases the b_i as well. Because the factorial is crossed with the suite, each
X_i is orthogonal to every D_j by construction, so the b_i estimated here are
within-task effects of the components.

Inference follows the thesis:
  * heteroskedasticity-consistent (HC3) standard errors throughout, because the
    stopping rule pins the experiment count at one level and frees it at the
    other, so within-cell variance differs systematically between the halves of
    the design;
  * the wasted trial ratio and the improvement rate are fitted on their natural
    scale, both being proportions;
  * Benjamini-Hochberg adjustment for multiplicity, the family being the
    treatment coefficients of one model on one metric. The dataset fixed effects
    are controls, not hypotheses, and are excluded from the family.

    python3 analysis/regression.py                     results_full/
    python3 analysis/regression.py --table PATH.csv    any regression table
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "config"))
import study_config as C  # noqa: E402

AXES = C.AXIS_ORDER
XCOLS = [f"X_{a}" for a in AXES]
METRICS = ["gain_over_default", "wasted_trial_ratio", "improvement_rate", "cost_to_best"]
WINDOWS = {"window": "", "full": "_full"}      # thesis, Performance metrics


# ------------------------------------------------------------- design matrix --
def dataset_dummies(df):
    """D_j for j = 2..D, with DATASETS[0] as the omitted reference level.

    Returns (DataFrame, reference_level). Empty frame when only one task is
    present in the data, in which case the fixed effect is not identified and
    there is nothing to control for.
    """
    levels = [d for d in C.DATASETS if d in set(df["dataset_id"])]
    levels += sorted(set(df["dataset_id"]) - set(levels))
    if len(levels) < 2:
        return pd.DataFrame(index=df.index), (levels[0] if levels else None)
    ref, rest = levels[0], levels[1:]
    out = pd.DataFrame({f"D_{d}": (df["dataset_id"] == d).astype(float) for d in rest},
                       index=df.index)
    return out, ref


def design(df, interactions):
    """[const | X_i | X_i X_j (optional) | D_j] and the treatment column names."""
    X = df[XCOLS].astype(float)
    treat = list(XCOLS)
    if interactions:
        pairs = {}
        for a, b in itertools.combinations(AXES, 2):
            name = f"X_{a}:X_{b}"
            pairs[name] = df[f"X_{a}"].astype(float) * df[f"X_{b}"].astype(float)
            treat.append(name)
        X = pd.concat([X, pd.DataFrame(pairs, index=df.index)], axis=1)
    dummies, ref = dataset_dummies(df)
    X = pd.concat([X, dummies], axis=1)
    return sm.add_constant(X, has_constant="add"), treat, list(dummies.columns), ref


# ------------------------------------------------------------------ fitting --
def fit(df, ycol, interactions=False):
    """One OLS fit with HC3 errors and BH adjustment over the treatment family."""
    d = df.dropna(subset=[ycol]).copy()
    if d.empty:
        return None
    X, treat, controls, ref = design(d, interactions)
    # A column that never varies (a single task, or a metric constant in this
    # subset) is not identified; drop it rather than returning a singular fit.
    keep = [c for c in X.columns if c == "const" or X[c].nunique() > 1]
    X, treat = X[keep], [c for c in treat if c in keep]
    controls = [c for c in controls if c in keep]

    y = d[ycol].astype(float)
    if y.nunique() < 2:
        return None                           # constant outcome: nothing to explain
    m = sm.OLS(y, X).fit(cov_type="HC3")
    padj = multipletests(m.pvalues[treat].to_numpy(), method="fdr_bh")[1] if treat else []

    rows = []
    for i, name in enumerate(treat + controls):
        is_treat = i < len(treat)
        rows.append({
            "term": name,
            "kind": "treatment" if is_treat else "dataset_control",
            # 2*beta is the average change in Y from switching the component
            # (eq. 1 codes the two levels as -1 and +1); a dataset dummy is 0/1,
            # so its coefficient is already the level difference.
            "estimate": float(2 * m.params[name]) if is_treat else float(m.params[name]),
            "std_err": float(2 * m.bse[name]) if is_treat else float(m.bse[name]),
            "t": float(m.tvalues[name]),
            "p": float(m.pvalues[name]),
            "p_adj": float(padj[i]) if is_treat else None,
        })
    return {"model": m, "coefs": rows, "n": int(m.nobs), "r2": float(m.rsquared),
            "reference_dataset": ref, "controls": controls}


def fit_all(df):
    """Every metric x window x {main effects, two-way interactions}."""
    main, inter = [], []
    targets = [(m, w, m + suf) for m in METRICS for w, suf in WINDOWS.items()]
    # the robustness scale of thesis, Performance metrics, defined only where a0 < 1
    rob = f"gain_over_default_{C.GAIN_SCALE_ROBUSTNESS}"
    targets += [(rob, w, rob + suf) for w, suf in WINDOWS.items()]

    for metric, window, col in targets:
        if col not in df.columns:
            continue
        for interactions, sink in ((False, main), (True, inter)):
            res = fit(df, col, interactions=interactions)
            if res is None:
                continue
            for r in res["coefs"]:
                if interactions and r["kind"] == "treatment" and ":" not in r["term"]:
                    continue          # main effects already reported in the other table
                sink.append({"metric": metric, "window": window, "n": res["n"],
                             "r2": res["r2"], "reference_dataset": res["reference_dataset"],
                             **r})
    return pd.DataFrame(main), pd.DataFrame(inter)


# --------------------------------------------------------------- compliance --
def compliance(df):
    """thesis, Results: share of runs whose behaviour matched the instruction.

    B, S and E constrain an observable action and get a rate. O constrains
    wording, so it gets the prose-length manipulation check instead. M leaves no
    behavioural trace, because the executor computes the score regardless.
    """
    rows = []
    for a in ("B", "S", "E"):
        col = f"complied_{a}"
        if col not in df.columns:
            continue
        for level in (0, 1):
            sub = df[(df[a] == level)][col].dropna()
            rows.append({"component": a, "level": level, "check": "compliance_rate",
                         "value": float(sub.mean()) if len(sub) else None, "n": int(len(sub))})
    if "prose_chars" in df.columns:
        for level in (0, 1):
            sub = df[df["O"] == level]["prose_chars"].dropna()
            rows.append({"component": "O", "level": level, "check": "mean_prose_chars",
                         "value": float(sub.mean()) if len(sub) else None, "n": int(len(sub))})
    rows.append({"component": "M", "level": None, "check": "not_observable",
                 "value": None, "n": 0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report ---
def _fmt(df, cols):
    return df[cols].to_string(index=False, float_format=lambda v: f"{v: .5f}")


def report(main, inter, comp, table_path, n_rows):
    L = [f"Regression report -- {table_path}",
         f"{n_rows} runs; datasets {C.DATASETS}; reference level "
         f"{main['reference_dataset'].iloc[0] if len(main) else 'n/a'}",
         "",
         "Model (thesis eq. 2):  Y = b0 + sum_i b_i X_i + sum_j g_j D_j + eps",
         "HC3 standard errors; Benjamini-Hochberg over the treatment coefficients",
         "of one model on one metric. Treatment estimates are 2*beta, the average",
         "change in Y from switching a component; dataset estimates are the level",
         "difference relative to the reference task.", ""]
    cols = ["metric", "window", "term", "kind", "estimate", "std_err", "p", "p_adj", "r2", "n"]
    L += ["=" * 78, "MAIN EFFECTS AND DATASET FIXED EFFECTS", "=" * 78, _fmt(main, cols), ""]
    L += ["=" * 78, "TWO-WAY INTERACTIONS (thesis eq. 3)", "=" * 78, _fmt(inter, cols), ""]
    L += ["=" * 78, "COMPLIANCE AND MANIPULATION CHECKS", "=" * 78,
          comp.to_string(index=False), ""]
    return "\n".join(L)


def run(results_dir=None, table=None):
    table_path = Path(table) if table else \
        STUDY / (results_dir or C.RESULTS_DIRNAME) / f"{C.TABLE_BASENAME}.csv"
    if not table_path.exists():
        raise SystemExit(f"no regression table at {table_path}; run run_experiment.py first")
    df = pd.read_csv(table_path)
    if "dataset_id" not in df.columns:
        raise SystemExit(
            f"{table_path} has no 'dataset_id' column, so the dataset fixed effect of "
            "eq. (2) cannot be fitted. Re-encode the transcripts with the current "
            "encoding/metrics.py.")

    main, inter = fit_all(df)
    comp = compliance(df)
    out = table_path.parent
    main.to_csv(out / "regression_main.csv", index=False)
    inter.to_csv(out / "regression_interactions.csv", index=False)
    comp.to_csv(out / "regression_compliance.csv", index=False)
    text = report(main, inter, comp, table_path, len(df))
    (out / "regression_report.txt").write_text(text)
    print(text)
    print(f"\nwrote regression_main.csv, regression_interactions.csv, "
          f"regression_compliance.csv and regression_report.txt to {out}")
    return main, inter, comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--table", default=None)
    a = ap.parse_args()
    run(results_dir=a.results_dir, table=a.table)
