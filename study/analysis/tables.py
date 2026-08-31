"""
Render the results tables of thesis_new.tex from the fitted regression output.

Formatting only: every number here comes from regression.py, so the tables in
the paper cannot drift from the estimates. Writes a LaTeX fragment per table
plus the summary quantities the prose placeholders need.

    python3 analysis/tables.py [--results-dir results_full]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "config"))
import study_config as C  # noqa: E402

LABEL = {"gain_over_default": "Gain over default",
         "wasted_trial_ratio": "Wasted trial ratio",
         "improvement_rate": "Improvement rate",
         "cost_to_best": "Cost to best"}
ORDER = list(LABEL)
AXES = C.AXIS_ORDER


def stars(p):
    if pd.isna(p):
        return ""
    return "$^{***}$" if p < .001 else "$^{**}$" if p < .01 else "$^{*}$" if p < .05 else ""


def cell(est, se, p, dp=3):
    return f"{est:+.{dp}f}{stars(p)}"


def main_table(main):
    """Body rows of tab:main: 2*beta per component, the dataset effect, and R^2."""
    d = main[main.window == "window"]
    out = []
    for m in ORDER:
        sub = d[d.metric == m].set_index("term")
        if sub.empty:
            continue
        dp = 3 if m != "cost_to_best" else 2
        cells = [cell(sub.loc[f"X_{a}", "estimate"], sub.loc[f"X_{a}", "std_err"],
                      sub.loc[f"X_{a}", "p_adj"], dp) for a in AXES]
        gcol = [c for c in sub.index if c.startswith("D_")]
        g = cell(sub.loc[gcol[0], "estimate"], sub.loc[gcol[0], "std_err"],
                 sub.loc[gcol[0], "p"], dp) if gcol else "--"
        r2 = sub["r2"].iloc[0]
        out.append(f"{LABEL[m]:<19s} & " + " & ".join(cells) + f" & {g} & {r2:.3f} \\\\")
    return "\n".join(out)


def interaction_table(inter, comp, k=4):
    """Body rows of tab:interactions: the k largest |2*beta_ij|, and compliance."""
    d = inter[(inter.window == "window") & (inter.kind == "treatment")
              & (inter.metric.isin(ORDER))].copy()
    d["abs"] = d.estimate.abs()
    top = d.sort_values("abs", ascending=False).head(k)

    rows_i = []
    for _, r in top.iterrows():
        a, b = r.term.replace("X_", "").split(":")
        dp = 3 if r.metric != "cost_to_best" else 2
        rows_i.append((f"${a} \\times {b}$ ({LABEL[r.metric].lower()})",
                       f"{r.estimate:+.{dp}f}", f"{r.p_adj:.3f}"))

    rows_c = []
    for a in ("B", "S", "E"):
        s = comp[(comp.component == a) & (comp.check == "compliance_rate")]
        if len(s) == 2:
            v0 = s[s.level == 0].value.iloc[0]
            v1 = s[s.level == 1].value.iloc[0]
            rows_c.append((f"${a}$", f"{v0:.2f} / {v1:.2f}"))
    o = comp[comp.check == "mean_prose_chars"]
    if len(o) == 2:
        rows_c.append(("$O$", f"{o[o.level==0].value.iloc[0]:.0f} / "
                              f"{o[o.level==1].value.iloc[0]:.0f} chars"))
    rows_c.append(("$M$", "not observable"))

    out = []
    for i in range(max(len(rows_i), len(rows_c))):
        li = rows_i[i] if i < len(rows_i) else ("", "", "")
        lc = rows_c[i] if i < len(rows_c) else ("", "")
        out.append(f"{li[0]} & {li[1]} & {li[2]} & {lc[0]} & {lc[1]} \\\\")
    return "\n".join(out)


def coefficient_plot(main, w=5.3, h=3.0, pad=0.55):
    """Figure fig:coefficients as native TikZ: 2*beta with 95% CIs, one panel per metric.

    Drawn rather than plotted so the paper gains no new Python dependency; every
    coordinate comes from regression_main.csv. The dataset fixed effect is in the
    model, so these are within-task effects.
    """
    d = main[(main.window == "window") & (main.kind == "treatment")]
    L = [r"\begin{tikzpicture}[font=\scriptsize]"]
    for k, m in enumerate(ORDER):
        sub = d[d.metric == m].set_index("term")
        if sub.empty:
            continue
        ox, oy = (k % 2) * (w + 1.5), -(k // 2) * (h + 1.3)
        est = [sub.loc[f"X_{a}", "estimate"] for a in AXES]
        se = [sub.loc[f"X_{a}", "std_err"] for a in AXES]
        lo = [e - 1.96 * s_ for e, s_ in zip(est, se)]
        hi = [e + 1.96 * s_ for e, s_ in zip(est, se)]
        xmin, xmax = min(lo + [0]), max(hi + [0])
        span = (xmax - xmin) or 1.0
        xmin, xmax = xmin - .08 * span, xmax + .08 * span
        span = xmax - xmin
        X = lambda v: ox + (v - xmin) / span * w
        L.append(f"  \\node[anchor=west] at ({ox:.2f},{oy + h/2 + .45:.2f}) "
                 f"{{\\textbf{{{LABEL[m]}}}}};")
        z = X(0)
        L.append(f"  \\draw[gray!55,dashed] ({z:.2f},{oy - h/2:.2f}) -- ({z:.2f},{oy + h/2:.2f});")
        for i, a in enumerate(AXES):
            y = oy + h/2 - (i + .5) * h / len(AXES)
            L.append(f"  \\node[anchor=east] at ({ox - .12:.2f},{y:.2f}) {{${a}$}};")
            L.append(f"  \\draw[thick] ({X(lo[i]):.2f},{y:.2f}) -- ({X(hi[i]):.2f},{y:.2f});")
            sig = "" if pd.isna(sub.loc[f"X_{a}", "p_adj"]) or sub.loc[f"X_{a}", "p_adj"] >= .05 else ",fill=black"
            L.append(f"  \\node[circle,draw,inner sep=1.3pt{sig}] at ({X(est[i]):.2f},{y:.2f}) {{}};")
        L.append(f"  \\draw[->] ({ox:.2f},{oy - h/2 - .1:.2f}) -- ({ox + w:.2f},{oy - h/2 - .1:.2f});")
        for v in (xmin + .12 * span, 0.0, xmax - .12 * span):
            L.append(f"  \\node[anchor=north] at ({X(v):.2f},{oy - h/2 - .15:.2f}) "
                     f"{{{v:+.3f}}};" if m != "cost_to_best" else
                     f"  \\node[anchor=north] at ({X(v):.2f},{oy - h/2 - .15:.2f}) {{{v:+.2f}}};")
    L.append(r"\end{tikzpicture}")
    return "\n".join(L)


def summary(main, inter, comp, df):
    """The quantities the prose placeholders refer to."""
    d = main[main.window == "window"]
    L = ["Per metric, the largest component effect (window, BH-adjusted):"]
    for m in ORDER:
        sub = d[(d.metric == m) & (d.kind == "treatment")].copy()
        if sub.empty:
            continue
        sub["abs"] = sub.estimate.abs()
        b = sub.sort_values("abs", ascending=False).iloc[0]
        sig = sub[sub.p_adj < .05].term.str.replace("X_", "").tolist()
        g = d[(d.metric == m) & (d.kind == "dataset_control")]
        L.append(f"  {m:20s} {b.term.replace('X_','')}={b.estimate:+.4f} "
                 f"(adj p={b.p_adj:.4f}); significant: {sig or 'none'}; "
                 f"dataset effect={g.estimate.iloc[0]:+.4f} (p={g.p.iloc[0]:.4f}); "
                 f"R2={b.r2:.3f}")
    ii = inter[(inter.window == "window") & (inter.kind == "treatment")
               & (inter.metric.isin(ORDER))]
    sig_i = ii[ii.p_adj < .05]
    L.append(f"\nTwo-way interactions significant after BH: {len(sig_i)} of {len(ii)}")
    for _, r in sig_i.sort_values("p_adj").head(8).iterrows():
        L.append(f"  {r.metric:20s} {r.term:12s} {r.estimate:+.4f} (adj p={r.p_adj:.4f})")

    # robustness: does M agree on both accuracy scales?
    rob = main[(main.metric.str.startswith("gain_over_default_")) & (main.window == "window")]
    if len(rob):
        r = rob[rob.term == "X_M"]
        if len(r):
            L.append(f"\nM on the validation scale (breast_cancer only): "
                     f"{r.estimate.iloc[0]:+.4f} (adj p={r.p_adj.iloc[0]:.4f})")
    L.append(f"\nRuns: {len(df)}; cells: {df.groupby(['config_id','dataset_id']).ngroups}; "
             f"runs per cell: {df.groupby(['config_id','dataset_id']).size().unique().tolist()}")
    L.append(f"terminated_by: {df.terminated_by.value_counts().to_dict()}")
    return "\n".join(L)


def run(results_dir=None):
    out = STUDY / (results_dir or C.RESULTS_DIRNAME)
    main = pd.read_csv(out / "regression_main.csv")
    inter = pd.read_csv(out / "regression_interactions.csv")
    comp = pd.read_csv(out / "regression_compliance.csv")
    df = pd.read_csv(out / f"{C.TABLE_BASENAME}.csv")

    t1, t2 = main_table(main), interaction_table(inter, comp)
    fig = coefficient_plot(main)
    (STUDY / (results_dir or C.RESULTS_DIRNAME) / "figure_coefficients.tex").write_text(fig + "\n")
    s = summary(main, inter, comp, df)
    (out / "table_main.tex").write_text(t1 + "\n")
    (out / "table_interactions.tex").write_text(t2 + "\n")
    (out / "summary.txt").write_text(s + "\n")
    print("=== tab:main body ===\n" + t1)
    print("\n=== tab:interactions body ===\n" + t2)
    print("\n=== summary ===\n" + s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    run(**vars(ap.parse_args()))
