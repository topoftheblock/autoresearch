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
    # Rank by adjusted significance, not raw magnitude: the four metrics live on
    # different scales, so |2*beta| is not comparable across them and would just
    # surface whichever metric has the largest units.
    d["abs"] = d.estimate.abs()
    top = d.sort_values(["p_adj", "abs"], ascending=[True, False]).head(k)

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


def _nice(x):
    """Largest round tick value (1, 2, 2.5 or 5 x 10^k) not exceeding x.

    Rounds down, not up: an outer tick placed beyond the data extent lands flush
    against the axis edge and pgfplots silently drops the label.
    """
    import math
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x))
    f = x / 10 ** e
    for c in (5, 2.5, 2, 1):
        if f >= c:
            return c * 10 ** e
    return 10 ** (e - 1) * 5


def coefficient_plot(main, ci=1.96):
    """Figure fig:coefficients as a pgfplots forest plot: 2*beta with 95% CIs.

    One panel per metric, five components per panel, a zero reference line, and
    a filled marker where the effect survives Benjamini-Hochberg adjustment.
    The panels carry independent horizontal scales because the four metrics are
    in different units; the caption says so. Every coordinate comes from
    regression_main.csv, so the figure cannot drift from the estimates.
    """
    d = main[(main.window == "window") & (main.kind == "treatment")]
    metrics = [m for m in ORDER if not d[d.metric == m].empty]

    L = [r"\begin{tikzpicture}[font=\scriptsize]",
         r"\begin{groupplot}[",
         r"  group style={group size=2 by 2, horizontal sep=1.85cm, vertical sep=1.5cm},",
         r"  width=6.0cm, height=3.9cm,",
         r"  y dir=reverse, ytick={1,2,3,4,5},",
         r"  yticklabels={$M$,$B$,$S$,$O$,$E$},",
         r"  ymin=0.45, ymax=5.55,",
         r"  ymajorgrids, major grid style={gray!22, dotted},",
         r"  axis line style={gray!55}, tick align=outside, tick pos=left,",
         r"  every tick/.style={gray!55},",
         r"  title style={font=\scriptsize\bfseries, yshift=-1pt},",
         r"  scaled x ticks=false,",
         r"]"]

    for m in metrics:
        sub = d[d.metric == m].set_index("term")
        est = [sub.loc[f"X_{a}", "estimate"] for a in AXES]
        err = [ci * sub.loc[f"X_{a}", "std_err"] for a in AXES]
        # bool(): pandas yields numpy.bool_, which fails an "is True" identity test
        sig = [bool((not pd.isna(sub.loc[f"X_{a}", "p_adj"]))
                    and sub.loc[f"X_{a}", "p_adj"] < .05) for a in AXES]
        # Symmetric about zero with rounded ticks: the convention for a
        # coefficient plot, and it keeps the zero line central so the sign of
        # each effect reads off immediately.
        ext = max(abs(e) + s_ for e, s_ in zip(est, err)) or .01
        lo, hi = -1.12 * ext, 1.12 * ext
        t = _nice(.75 * ext)
        dp = 2 if m == "cost_to_best" else 3
        ticks = ",".join(f"{v:.{dp}f}" for v in (-t, 0.0, t))

        xlab = r", xlabel={$2\hat\beta$}" if metrics.index(m) >= len(metrics) - 2 else ""
        L += [f"\\nextgroupplot[title={{{LABEL[m]}}}{xlab}, xmin={lo:.5f}, xmax={hi:.5f},",
              f"  xtick={{{ticks}}}, xticklabel style={{/pgf/number format/fixed,",
              f"    /pgf/number format/precision={dp}, /pgf/number format/fixed zerofill}},]",
              r"\addplot[gray!70, dashed, forget plot] coordinates {(0,0.45) (0,5.55)};"]
        for want, mark in ((True, "*"), (False, "o")):
            pts = [f"({est[i]:.6f},{i+1}) +- ({err[i]:.6f},0)"
                   for i in range(len(AXES)) if sig[i] is want]
            if pts:
                L += [f"\\addplot[only marks, mark={mark}, mark size=1.6pt, black, thick,",
                      r"  error bars/.cd, x dir=both, x explicit, error bar style={black}]",
                      "  coordinates {" + " ".join(pts) + "};"]
    L += [r"\end{groupplot}", r"\end{tikzpicture}"]
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
