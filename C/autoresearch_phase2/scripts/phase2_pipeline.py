"""
phase2_pipeline.py
==================
Phase 2: Outcome simulation and statistical analysis.

Since real GPU runs are not available, this module generates
*principled synthetic trajectories* from each program.md's
φ-vector, runs k=3 stochastic repeats per variant, then
executes the full statistical analysis that would be run on
real results.tsv data — so the framework is plug-and-replace
ready when real runs are available.

Generative model
----------------
Each experiment step i in a trajectory is drawn as follows:

1. p_improve  — probability of improvement per step — is a
   sigmoidal function of φ features with known-direction
   coefficients (strategy_has_hint ↑, constraint_prohibitive ↓
   past a threshold, eval_multi_objective ↑, token_count ∪,
   has_budget_counter → tighter steps early on).

2. Δloss | improve ~ Exp(rate) clipped to (0, 0.05].
   rate is faster when scope is narrow (more reliable steps)
   and slower when scope is broad (higher variance).

3. Δloss | not improve is 0 (agent reverts).

4. Hypothesis-type distribution h_t is drawn from a Dirichlet
   parameterised by the φ vector (e.g. strategy_has_hint →
   concentrates on single-HP moves; broad scope → spreads over
   architecture/preprocessing/HP categories).

Budget: 40 experiments per run (≈ one overnight session).

Outputs
-------
- trajectories.json   — raw simulated results per file × run
- outcomes.tsv        — aggregate outcome metrics
- anova_results.txt   — main-effects ANOVA on key outcomes
- phi_regression.txt  — OLS regression φ features → improvement_rate
- strategy_entropy.tsv— hypothesis-type diversity per run
- epsilon_sweep.tsv   — how many ~φ classes remain at each ε level
- phase2_report.txt   — human-readable narrative summary
"""

import json
import math
import random
import itertools
import statistics
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR         = Path(__file__).resolve().parent.parent
RECORDS_PATH     = ROOT_DIR / "phase1_output" / "records.json"
OUTPUT_DIR       = ROOT_DIR / "phase2_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS        = 3        # repeated runs per variant
N_EXPERIMENTS = 40       # experiments per run
BASELINE_LOSS = 0.310    # starting val_loss for all runs
RANDOM_SEED   = 42

HYPOTHESIS_TYPES = [
    "single_hp_increment",   # change one HP by a small amount
    "single_hp_reset",       # revert a HP to default
    "feature_engineering",   # add/modify a feature transform
    "preprocessing",         # change data preprocessing
    "ensemble_composition",  # change n_estimators / depth jointly
    "regularisation",        # add/change regularisation term
]

# ── Generative model helpers ──────────────────────────────────────────────────

def p_improve(phi: dict, rng: np.random.Generator) -> float:
    """
    Sigmoid probability of improvement per experiment.
    Coefficients reflect prior beliefs about direction of each feature.
    """
    # Linear predictor (log-odds scale)
    eta = 0.0

    # Strategy hint → more disciplined, higher hit rate
    eta += 0.60 * phi["strategy_has_hint"]

    # Multi-objective → agent spends effort on two signals; slight cost
    eta -= 0.15 * phi["eval_multi_objective"]

    # Composite score → clearer keep/discard signal, small benefit
    eta += 0.10 * phi["eval_has_composite"]

    # Prohibitive constraints → fewer dead-ends
    # Optimal ~4-5; too many hurts exploration
    prohib = phi["constraint_prohibitive"]
    eta += 0.15 * prohib - 0.02 * prohib ** 2

    # Permissive statements → broader exploration, higher variance
    eta -= 0.08 * phi["constraint_permissive"]

    # Budget counter → agent paces itself, slightly better late efficiency
    eta += 0.20 * phi["loop_has_budget_counter"]

    # Token count → richer context up to a point; then diminishing returns
    tok = phi["global_token_count"]
    eta += 0.0015 * tok - 0.000003 * tok ** 2

    # Numeric ranges → well-bounded search space → better hit rate
    eta += 0.08 * phi["scope_numeric_ranges"]

    # Ordering keywords → sequential discipline
    eta += 0.10 * min(phi["strategy_ordering_kw"], 3)

    # Baseline
    eta -= 0.50

    # Add noise (agent is stochastic)
    eta += rng.normal(0, 0.10)

    return 1.0 / (1.0 + math.exp(-eta))

def improvement_magnitude(phi: dict, rng: np.random.Generator) -> float:
    """
    Exp(rate) improvement in val_loss when an experiment succeeds.
    Narrow scope → more reliable, smaller but more certain steps.
    Broad scope  → rarer but occasionally larger jumps.
    """
    if phi["strategy_has_hint"] and phi["scope_numeric_ranges"] >= 3:
        rate = 35.0  # narrow: small, reliable steps
    elif phi["constraint_permissive"] >= 3:
        rate = 18.0  # broad: occasional big jumps
    else:
        rate = 25.0  # mid

    delta = rng.exponential(1.0 / rate)
    return float(np.clip(delta, 1e-4, 0.05))

def hypothesis_dirichlet(phi: dict) -> np.ndarray:
    """
    Dirichlet concentration vector over HYPOTHESIS_TYPES.
    Reflects how the instruction document shapes agent choices.
    """
    # Base: uniform
    alpha = np.ones(len(HYPOTHESIS_TYPES))

    if phi["strategy_has_hint"]:
        # Sequential hint → concentrate on single-HP moves
        alpha[0] += 5.0   # single_hp_increment
        alpha[1] += 2.0   # single_hp_reset
    else:
        # No hint → spread across types
        alpha[2] += 2.0   # feature_engineering
        alpha[3] += 1.5   # preprocessing
        alpha[4] += 1.5   # ensemble_composition

    if phi["eval_multi_objective"]:
        alpha[5] += 2.0   # regularisation more attractive with multi-metric

    if phi["constraint_permissive"] >= 3:
        # Permissive → anything goes
        alpha[2] += 2.0
        alpha[3] += 2.0
        alpha[4] += 2.0

    return alpha

# ── Trajectory simulation ─────────────────────────────────────────────────────

def simulate_run(phi: dict, run_id: int, rng: np.random.Generator) -> dict:
    """Simulate one autoresearch run of N_EXPERIMENTS steps."""
    current_loss = BASELINE_LOSS
    best_loss    = BASELINE_LOSS
    improvements = 0
    first_improvement_step = None
    hp_alpha = hypothesis_dirichlet(phi)
    hp_probs = rng.dirichlet(hp_alpha)  # sample once per run

    experiments = []
    for step in range(1, N_EXPERIMENTS + 1):
        # Draw hypothesis type
        hyp_type = rng.choice(HYPOTHESIS_TYPES, p=hp_probs)

        # Draw outcome
        p_imp = p_improve(phi, rng)
        improved = rng.random() < p_imp

        if improved:
            delta = improvement_magnitude(phi, rng)
            new_loss = current_loss - delta
            current_loss = new_loss
            best_loss = min(best_loss, current_loss)
            improvements += 1
            if first_improvement_step is None:
                first_improvement_step = step
            status = "keep"
        else:
            delta = 0.0
            status = "discard"

        experiments.append({
            "step":      step,
            "hyp_type":  hyp_type,
            "delta":     round(delta, 6),
            "val_loss":  round(current_loss, 6),
            "best_loss": round(best_loss, 6),
            "status":    status,
        })

    hyp_counts = Counter(e["hyp_type"] for e in experiments)
    hyp_freq   = {k: hyp_counts.get(k, 0) / N_EXPERIMENTS
                  for k in HYPOTHESIS_TYPES}
    hyp_entropy = _entropy(list(hyp_freq.values()))

    return {
        "run_id":                  run_id,
        "experiments":             experiments,
        "n_improvements":          improvements,
        "improvement_rate":        round(improvements / N_EXPERIMENTS, 4),
        "best_val_loss":           round(best_loss, 6),
        "loss_reduction":          round(BASELINE_LOSS - best_loss, 6),
        "steps_to_first_impr":     first_improvement_step,
        "hypothesis_freq":         hyp_freq,
        "hypothesis_entropy":      round(hyp_entropy, 4),
    }

def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p + 1e-12) for p in probs)

# ── Aggregation helpers ───────────────────────────────────────────────────────

def aggregate_runs(runs: list[dict]) -> dict:
    def mean(key):
        return statistics.mean(r[key] for r in runs)
    def stdev(key):
        vals = [r[key] for r in runs]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0
    def mean_opt(key):
        vals = [r[key] for r in runs if r[key] is not None]
        return statistics.mean(vals) if vals else None

    return {
        "mean_improvement_rate":    round(mean("improvement_rate"), 4),
        "std_improvement_rate":     round(stdev("improvement_rate"), 4),
        "mean_best_val_loss":       round(mean("best_val_loss"), 6),
        "std_best_val_loss":        round(stdev("best_val_loss"), 6),
        "mean_loss_reduction":      round(mean("loss_reduction"), 6),
        "mean_steps_to_first":      round(mean_opt("steps_to_first_impr"), 2)
                                    if mean_opt("steps_to_first_impr") else None,
        "mean_hyp_entropy":         round(mean("hypothesis_entropy"), 4),
        "std_hyp_entropy":          round(stdev("hypothesis_entropy"), 4),
    }

# ── ε-sweep: how many ~φ classes remain? ─────────────────────────────────────

def epsilon_sweep(records: list[dict]) -> list[dict]:
    """
    For a range of ε values, count distinct ~φ equivalence classes.
    Two programs merge when ALL features differ by < ε.
    """
    sweep_results = []
    epsilons = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    feature_keys = sorted(records[0]["phi"].keys())
    feature_vecs = {r["file"]: np.array([r["phi"][k] for k in feature_keys])
                    for r in records}
    files = [r["file"] for r in records]

    for eps in epsilons:
        # Union-Find to merge equivalent programs
        parent = {f: f for f in files}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for a, b in itertools.combinations(files, 2):
            diff = np.abs(feature_vecs[a] - feature_vecs[b])
            if np.all(diff <= eps):
                union(a, b)

        classes = defaultdict(list)
        for f in files:
            classes[find(f)].append(f)

        sweep_results.append({
            "epsilon": eps,
            "n_classes": len(classes),
            "classes": [sorted(v) for v in classes.values()],
        })

    return sweep_results

# ── Statistical analysis ──────────────────────────────────────────────────────

def run_anova(outcomes: list[dict], sigma_classes: dict[str, list[str]]):
    """
    One-way ANOVA: does σ-class explain variance in improvement_rate?
    Also run pairwise Welch t-tests between σ-classes.
    """
    lines = ["=" * 60,
             "ANOVA: σ-class × improvement_rate",
             "=" * 60]

    # Group by σ-class
    groups = defaultdict(list)
    for rec in outcomes:
        groups[rec["sigma_class"]].append(rec["mean_improvement_rate"])

    group_names = sorted(groups.keys())
    group_vals  = [groups[g] for g in group_names]

    lines.append(f"\nGroups (σ-class → mean_improvement_rate values):")
    for name, vals in zip(group_names, group_vals):
        lines.append(f"  {name}: {[round(v,4) for v in vals]}  "
                     f"mean={round(statistics.mean(vals),4)}")

    if all(len(v) >= 2 for v in group_vals) and len(group_vals) >= 2:
        try:
            f_stat, p_val = stats.f_oneway(*group_vals)
            lines.append(f"\nOne-way ANOVA:")
            lines.append(f"  F = {f_stat:.4f},  p = {p_val:.4f}")
            if p_val < 0.05:
                lines.append("  → Significant difference across σ-classes (p < 0.05)")
            else:
                lines.append("  → No significant difference (p ≥ 0.05)")
                lines.append("    Interpretation: σ-class alone may not explain")
                lines.append("    outcome variance; φ-level features needed.")
        except Exception as e:
            lines.append(f"  ANOVA failed: {e}")
    else:
        lines.append("\nInsufficient replication for formal ANOVA.")
        lines.append("(Need ≥ 2 members per σ-class with ≥ 2 runs each.)")
        lines.append("Reporting descriptive statistics only.")

    lines.append("")
    return "\n".join(lines)

def run_phi_regression(records: list[dict], outcomes: list[dict]):
    """
    OLS: φ features → mean_improvement_rate.
    Reports coefficients and R².
    """
    lines = ["=" * 60,
             "OLS Regression: φ features → mean_improvement_rate",
             "=" * 60]

    feature_keys = [
        "strategy_has_hint", "eval_multi_objective", "eval_has_composite",
        "constraint_prohibitive", "constraint_permissive",
        "loop_has_budget_counter", "scope_numeric_ranges",
        "strategy_ordering_kw", "global_token_count",
    ]

    file_to_phi = {r["file"]: r["phi"] for r in records}
    file_to_out = {o["file"]: o["mean_improvement_rate"] for o in outcomes}

    files = [r["file"] for r in records]
    X = np.array([[file_to_phi[f][k] for k in feature_keys] for f in files])
    y = np.array([file_to_out[f] for f in files])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model  = LinearRegression().fit(X_sc, y)
    y_pred = model.predict(X_sc)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    lines.append(f"\nR² = {r2:.4f}  (n={len(files)} files, "
                 f"{len(feature_keys)} features)")
    lines.append(f"\nStandardised coefficients "
                 f"(+ = higher feature → higher improvement rate):")
    coef_pairs = sorted(zip(feature_keys, model.coef_), key=lambda x: -abs(x[1]))
    for feat, coef in coef_pairs:
        bar = "█" * int(abs(coef) * 20)
        sign = "+" if coef >= 0 else "-"
        lines.append(f"  {feat:<32s} {sign}{abs(coef):.4f}  {bar}")

    lines.append(f"\nIntercept: {model.intercept_:.4f}")
    lines.append("")
    return "\n".join(lines)

def factorisation_test(records: list[dict], outcomes: list[dict]):
    """
    Check: does σ alone predict outcomes as well as the full φ?
    Compare R² of σ-only OHE regression vs φ regression.
    """
    lines = ["=" * 60,
             "FACTORISATION TEST: σ-only vs full φ",
             "=" * 60]

    file_to_sigma = {r["file"]: frozenset(r["sigma"]) for r in records}
    file_to_out   = {o["file"]: o["mean_improvement_rate"] for o in outcomes}
    files         = [r["file"] for r in records]

    # Unique σ values
    sigmas   = sorted(set(file_to_sigma[f] for f in files), key=str)
    sig_idx  = {s: i for i, s in enumerate(sigmas)}

    X_sigma = np.array([[1 if file_to_sigma[f] == s else 0
                         for s in sigmas] for f in files])
    y       = np.array([file_to_out[f] for f in files])

    if X_sigma.shape[1] < len(files):
        m_sigma = LinearRegression().fit(X_sigma, y)
        yp = m_sigma.predict(X_sigma)
        ss_res = np.sum((y - yp) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_sigma = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        r2_sigma = 1.0   # perfect fit (one-hot fully spans)

    feature_keys = [
        "strategy_has_hint", "eval_multi_objective", "eval_has_composite",
        "constraint_prohibitive", "constraint_permissive",
        "loop_has_budget_counter", "scope_numeric_ranges",
    ]
    file_to_phi = {r["file"]: r["phi"] for r in records}
    X_phi = np.array([[file_to_phi[f][k] for k in feature_keys] for f in files])
    m_phi = LinearRegression().fit(X_phi, y)
    yp2   = m_phi.predict(X_phi)
    ss_r2 = np.sum((y - yp2) ** 2)
    ss_t2 = np.sum((y - y.mean()) ** 2)
    r2_phi = 1 - ss_r2 / ss_t2 if ss_t2 > 0 else float("nan")

    lines.append(f"\n  R²(σ-only OHE)  = {r2_sigma:.4f}")
    lines.append(f"  R²(full φ)      = {r2_phi:.4f}")
    delta = r2_phi - r2_sigma
    lines.append(f"  ΔR²             = {delta:+.4f}")
    lines.append("")
    if delta > 0.05:
        lines.append("  Interpretation: φ explains substantially more variance")
        lines.append("  than σ alone. The factorisation 𝒫/~ ≅ ∏ τᵢ/~τᵢ does")
        lines.append("  NOT hold — feature magnitudes matter beyond section presence.")
    elif delta < -0.05:
        lines.append("  Interpretation: σ explains outcomes well; φ adds noise.")
        lines.append("  The coarse section-type partition may be sufficient.")
    else:
        lines.append("  Interpretation: φ adds modest explanatory power over σ.")
        lines.append("  Both levels capture complementary information.")
    lines.append("")
    return "\n".join(lines)

def anytime_curves(all_trajectories: dict) -> str:
    """Compute anytime best-loss curve statistics per file."""
    lines = ["=" * 60,
             "ANYTIME PERFORMANCE CURVES",
             "=" * 60,
             f"{'File':<20} {'Step':>5}  " +
             "  ".join(f"s{s:02d}" for s in [1,5,10,20,40]),
             "─" * 70]

    for fname, runs in all_trajectories.items():
        row = f"{fname:<20} {'':>5}  "
        checkpoints = [1, 5, 10, 20, 40]
        vals = []
        for cp in checkpoints:
            losses_at_cp = []
            for run in runs:
                exps = run["experiments"]
                best = BASELINE_LOSS
                for e in exps[:cp]:
                    if e["status"] == "keep":
                        best = e["best_loss"]
                losses_at_cp.append(best)
            vals.append(statistics.mean(losses_at_cp))
        row += "  ".join(f"{v:.4f}" for v in vals)
        lines.append(row)
    lines.append("")
    return "\n".join(lines)

def hypothesis_diversity(all_trajectories: dict) -> str:
    """Entropy of hypothesis-type distribution per file."""
    lines = ["=" * 60,
             "HYPOTHESIS-TYPE DIVERSITY (Shannon entropy)",
             "=" * 60,
             f"{'File':<20} {'Mean H':>8}  {'Std H':>7}  Dominant hypothesis type",
             "─" * 70]
    for fname, runs in all_trajectories.items():
        entropies = [r["hypothesis_entropy"] for r in runs]
        mean_h = statistics.mean(entropies)
        std_h  = statistics.stdev(entropies) if len(entropies) > 1 else 0.0
        # Pool hypothesis frequencies across runs
        pooled = defaultdict(float)
        for r in runs:
            for k, v in r["hypothesis_freq"].items():
                pooled[k] += v
        dominant = max(pooled, key=pooled.get)
        lines.append(f"{fname:<20} {mean_h:>8.4f}  {std_h:>7.4f}  {dominant}")
    lines.append("")
    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    records = json.loads(RECORDS_PATH.read_text())

    # Assign σ-class labels (recompute from sigma sets)
    sigma_groups = defaultdict(list)
    for r in records:
        sigma_groups[frozenset(r["sigma"])].append(r["file"])
    sigma_label = {}
    for i, (sig, members) in enumerate(
        sorted(sigma_groups.items(), key=lambda x: -len(x[1])), 1
    ):
        for m in members:
            sigma_label[m] = f"σ-{i:02d}"

    print("=" * 60)
    print("  Phase 2 — Simulation & Analysis")
    print("=" * 60)
    print(f"  Variants: {len(records)}  |  Runs/variant: {N_RUNS}"
          f"  |  Experiments/run: {N_EXPERIMENTS}\n")

    # ── Simulate trajectories ─────────────────────────────────────────────────
    all_trajectories = {}
    outcome_rows     = []

    for record in records:
        fname = record["file"]
        phi   = record["phi"]
        runs  = []
        for run_id in range(1, N_RUNS + 1):
            run = simulate_run(phi, run_id, rng)
            runs.append(run)
        all_trajectories[fname] = runs

        agg = aggregate_runs(runs)
        outcome_rows.append({
            "file":        fname,
            "sigma_class": sigma_label[fname],
            **agg,
        })
        print(f"  {fname:<20}  σ={sigma_label[fname]}  "
              f"impr_rate={agg['mean_improvement_rate']:.3f}±{agg['std_improvement_rate']:.3f}  "
              f"best_loss={agg['mean_best_val_loss']:.4f}±{agg['std_best_val_loss']:.4f}  "
              f"H={agg['mean_hyp_entropy']:.3f}")

    print()

    # ── ε-sweep ───────────────────────────────────────────────────────────────
    sweep = epsilon_sweep(records)
    print("  ε-sweep (how many ~φ classes remain):")
    for s in sweep:
        merged = [c for c in s["classes"] if len(c) > 1]
        print(f"    ε={s['epsilon']:>3d}  →  {s['n_classes']} classes", end="")
        if merged:
            print(f"  merged: {merged}", end="")
        print()
    print()

    # ── Statistical analyses ──────────────────────────────────────────────────
    anova_text   = run_anova(outcome_rows, sigma_groups)
    reg_text     = run_phi_regression(records, outcome_rows)
    factor_text  = factorisation_test(records, outcome_rows)
    curve_text   = anytime_curves(all_trajectories)
    div_text     = hypothesis_diversity(all_trajectories)

    # ── Narrative report ──────────────────────────────────────────────────────
    best_file    = max(outcome_rows, key=lambda r: r["mean_improvement_rate"])
    worst_file   = min(outcome_rows, key=lambda r: r["mean_improvement_rate"])
    highest_div  = max(outcome_rows, key=lambda r: r["mean_hyp_entropy"])
    lowest_div   = min(outcome_rows, key=lambda r: r["mean_hyp_entropy"])

    report_lines = [
        "=" * 60,
        "PHASE 2 NARRATIVE REPORT",
        "=" * 60,
        "",
        "1. Overview",
        "─" * 40,
        f"   {len(records)} program.md variants were simulated with {N_RUNS} runs each",
        f"   of {N_EXPERIMENTS} experiments.  Baseline val_loss = {BASELINE_LOSS}.",
        "",
        "2. Best and worst performers",
        "─" * 40,
        f"   Highest improvement rate: {best_file['file']}",
        f"     mean_impr_rate = {best_file['mean_improvement_rate']:.4f}",
        f"     mean_best_loss = {best_file['mean_best_val_loss']:.6f}",
        f"   Lowest improvement rate:  {worst_file['file']}",
        f"     mean_impr_rate = {worst_file['mean_improvement_rate']:.4f}",
        f"     mean_best_loss = {worst_file['mean_best_val_loss']:.6f}",
        f"   Δ improvement_rate (best − worst) = "
        f"{best_file['mean_improvement_rate'] - worst_file['mean_improvement_rate']:.4f}",
        "",
        "3. Hypothesis diversity",
        "─" * 40,
        f"   Most diverse agent behaviour: {highest_div['file']}",
        f"     H = {highest_div['mean_hyp_entropy']:.4f}",
        f"   Least diverse (most disciplined): {lowest_div['file']}",
        f"     H = {lowest_div['mean_hyp_entropy']:.4f}",
        "   Interpretation: lower entropy signals the program.md",
        "   successfully constrains the agent to a narrow strategy.",
        "",
        "4. Factorisation finding",
        "─" * 40,
        "   See factorisation_test block below for R²(σ) vs R²(φ).",
        "   If ΔR² > 0.05, φ-level features matter beyond section presence.",
        "   This is the primary answer to your research question.",
        "",
        "5. ε-sweep interpretation",
        "─" * 40,
        f"   At ε=0:  {sweep[0]['n_classes']} classes (all singletons).",
        f"   At ε=5:  {next(s['n_classes'] for s in sweep if s['epsilon']==5)} classes.",
        f"   At ε=21: {next(s['n_classes'] for s in sweep if s['epsilon']==21)} classes.",
        "   The ε value at which two programs that produced DIFFERENT",
        "   outcomes first merge is the critical threshold — that is",
        "   where the ~φ partition becomes too coarse to predict F([p]).",
        "",
        "6. Plug-and-replace note",
        "─" * 40,
        "   Replace simulate_run() with a real results.tsv reader.",
        "   All downstream analysis (ANOVA, regression, curves, entropy)",
        "   operates on the same outcome_rows dict and will run unchanged.",
        "",
    ]

    full_report = (
        "\n".join(report_lines) + "\n\n" +
        anova_text + "\n\n" +
        reg_text   + "\n\n" +
        factor_text + "\n\n" +
        curve_text  + "\n\n" +
        div_text
    )
    print(full_report)

    # ── Write outputs ─────────────────────────────────────────────────────────
    (OUTPUT_DIR / "phase2_report.txt").write_text(full_report)

    # trajectories.json
    serialisable = {}
    for fname, runs in all_trajectories.items():
        serialisable[fname] = [
            {k: v for k, v in r.items() if k != "experiments"} | 
            {"experiments": r["experiments"]}
            for r in runs
        ]
    (OUTPUT_DIR / "trajectories.json").write_text(
        json.dumps(serialisable, indent=2))

    # outcomes.tsv
    tsv_header = (
        "file\tsigma_class\tmean_improvement_rate\tstd_improvement_rate\t"
        "mean_best_val_loss\tstd_best_val_loss\tmean_loss_reduction\t"
        "mean_steps_to_first\tmean_hyp_entropy\tstd_hyp_entropy\n"
    )
    tsv_rows = "".join(
        f"{r['file']}\t{r['sigma_class']}\t"
        f"{r['mean_improvement_rate']}\t{r['std_improvement_rate']}\t"
        f"{r['mean_best_val_loss']}\t{r['std_best_val_loss']}\t"
        f"{r['mean_loss_reduction']}\t{r.get('mean_steps_to_first','')}\t"
        f"{r['mean_hyp_entropy']}\t{r['std_hyp_entropy']}\n"
        for r in outcome_rows
    )
    (OUTPUT_DIR / "outcomes.tsv").write_text(tsv_header + tsv_rows)

    # epsilon_sweep.tsv
    eps_lines = "epsilon\tn_classes\tmerged_pairs\n"
    for s in sweep:
        merged = [c for c in s["classes"] if len(c) > 1]
        eps_lines += f"{s['epsilon']}\t{s['n_classes']}\t{merged}\n"
    (OUTPUT_DIR / "epsilon_sweep.tsv").write_text(eps_lines)

    print(f"\n  Outputs written to {OUTPUT_DIR}/")
    print(f"    phase2_report.txt  — full narrative + statistics")
    print(f"    outcomes.tsv       — aggregate outcome metrics per variant")
    print(f"    trajectories.json  — full simulated experiment logs")
    print(f"    epsilon_sweep.tsv  — ~φ class count at each ε")
    print("=" * 60)


if __name__ == "__main__":
    main()
