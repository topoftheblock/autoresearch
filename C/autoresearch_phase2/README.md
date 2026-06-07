# autoresearch — program.md Equivalence Analysis

Structural classification and outcome analysis of `program.md` instruction files
for Karpathy's autoresearch system. This project implements the two-phase framework
described in the research course:

- **Phase 1** parses each `program.md` into a typed section tree, extracts a
  syntactic feature vector φ(p), assigns every file to a type-signature equivalence
  class σ and a feature-cell class φ, and computes pairwise ZSS tree-edit distances.

- **Phase 2** runs a principled simulation of agent trajectories conditioned on
  each φ-vector, then performs the full statistical analysis: ANOVA across σ-classes,
  OLS regression of φ features onto improvement rate, a factorisation test comparing
  R²(σ) vs R²(φ), anytime performance curves, hypothesis-type entropy, and an ε-sweep
  showing how the ~φ partition degrades as ε grows.

---

## Directory layout

```
autoresearch_phase2/
│
├── programs/               # program.md variants (input)
│   ├── S1-RF-1.md          # narrow · strict · no hint · single metric · terse
│   ├── S1-RF-2.md          # broad  · strict · no hint · multi-metric  · terse
│   ├── S1-RF-4.md          # broad  · permissive · no hint · single metric · verbose
│   ├── S1-RF-5.md          # narrow · strict · sequential · multi-metric · verbose
│   ├── S1-RF-minimal.md    # degenerate terse variant — verbosity lower bound
│   ├── S2-GBM-1.md         # GBM narrow · strict · no hint · single metric · terse
│   ├── S2-GBM-5.md         # GBM narrow · strict · sequential · multi-metric · verbose
│   └── S3-RF-HP.md         # budget-tracked RF variant for Study 3 comparison
│
├── scripts/
│   ├── phase1_pipeline.py  # Phase 1: parse → type → features → classes → ZSS
│   └── phase2_pipeline.py  # Phase 2: simulate → ANOVA → regression → report
│
├── phase1_output/          # written by phase1_pipeline.py
│   ├── records.json        # per-file AST sections + full φ-vector
│   ├── classification.tsv  # σ-class, φ-class, key features per file
│   └── zss_distances.tsv   # pairwise ZSS tree-edit distance matrix
│
├── phase2_output/          # written by phase2_pipeline.py
│   ├── outcomes.tsv        # aggregate outcome metrics per variant
│   ├── trajectories.json   # full simulated experiment logs (3 runs × 40 steps)
│   ├── epsilon_sweep.tsv   # number of ~φ classes at each ε level
│   └── phase2_report.txt   # full narrative + ANOVA + regression + curves
│
├── requirements.txt        # Python dependencies
└── run_all.sh              # one-command reproduction script
```

---

## Quick start

```bash
# Clone or unzip the project, then:
cd autoresearch_phase2
bash run_all.sh
```

Requires Python 3.10 or later. Dependencies are installed automatically.

To run phases individually:

```bash
pip install -r requirements.txt
python3 scripts/phase1_pipeline.py   # produces phase1_output/
python3 scripts/phase2_pipeline.py   # produces phase2_output/
```

---

## Adding your own program.md files

Drop any `.md` file into `programs/` and re-run. The pipelines discover all `.md`
files in that directory automatically. Naming convention: `<STUDY>-<ALGO>-<LABEL>.md`,
e.g. `S1-RF-3.md`. No other configuration needed.

---

## Replacing simulation with real autoresearch data

Phase 2 ships with a principled *simulation* of agent trajectories. When you have
real `results.tsv` data from actual autoresearch runs, replace the `simulate_run()`
function in `scripts/phase2_pipeline.py` with a reader that loads the real file.

The reader must return a dict with these keys (all downstream analysis is unchanged):

```python
{
    "run_id":               int,
    "experiments":          list[dict],   # see schema below
    "n_improvements":       int,
    "improvement_rate":     float,        # n_improvements / n_experiments
    "best_val_loss":        float,
    "loss_reduction":       float,        # baseline_loss - best_val_loss
    "steps_to_first_impr":  int | None,
    "hypothesis_freq":      dict[str, float],   # type → fraction of steps
    "hypothesis_entropy":   float,
}
```

Each element of `experiments` must have:

```python
{
    "step":      int,
    "hyp_type":  str,       # one of HYPOTHESIS_TYPES, or your own taxonomy
    "delta":     float,     # val_loss reduction (0 if discarded)
    "val_loss":  float,     # current val_loss after this step
    "best_loss": float,     # best val_loss seen so far
    "status":    str,       # "keep" | "discard" | "crash"
}
```

A minimal reader for a real `results.tsv` (tab-separated, columns:
`commit val_loss memory_gb status description`) looks like:

```python
def load_real_run(tsv_path: Path, run_id: int, baseline: float) -> dict:
    import csv
    experiments = []
    best = baseline
    n_imp = 0
    first = None
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader, 1):
            loss = float(row["val_loss"]) if row["val_loss"] else baseline
            kept = row["status"] == "keep"
            if kept:
                delta = best - loss
                best  = loss
                n_imp += 1
                if first is None:
                    first = i
            else:
                delta = 0.0
            experiments.append({
                "step": i, "hyp_type": "single_hp_increment",
                "delta": delta, "val_loss": loss,
                "best_loss": best, "status": row["status"],
            })
    n = len(experiments)
    return {
        "run_id": run_id, "experiments": experiments,
        "n_improvements": n_imp,
        "improvement_rate": n_imp / n if n else 0,
        "best_val_loss": best,
        "loss_reduction": baseline - best,
        "steps_to_first_impr": first,
        "hypothesis_freq": {"single_hp_increment": 1.0},
        "hypothesis_entropy": 0.0,
    }
```

---

## Algebraic framework summary

Each `program.md` is a point in the product space:

```
𝒫 = 𝒮 × 𝒦 × 𝒢 × ℰ × 𝒱 × ℒ
```

Three structural equivalence relations form a lattice:

```
~T  (tree-isomorphism, finest)
 ⊆
~φ  (feature-cell, medium)   p ~φ q  iff  σ(p) = σ(q)  and  ‖φ(p) − φ(q)‖∞ < ε
 ⊆
~σ  (type-signature, coarsest)   p ~σ q  iff  σ(p) = σ(q)
```

The outcome map  F : 𝒯/~ → Δ(𝒪)  is estimated empirically.
The factorisation conjecture  𝒫/~ ≅ ∏ τᵢ/~τᵢ  is tested via ANOVA.

---

## Results summary (simulation, seed=42)

| Variant        | σ-class | Impr. rate | Best val_loss | Dominant hyp. type     |
|----------------|---------|------------|---------------|------------------------|
| S3-RF-HP       | σ-04    | 84.2%      | −0.454        | single_hp_increment    |
| S2-GBM-5       | σ-02    | 78.3%      | −0.529        | single_hp_increment    |
| S2-GBM-1       | σ-01    | 75.0%      | −0.312        | single_hp_increment    |
| S1-RF-5        | σ-02    | 69.2%      | −0.387        | single_hp_increment    |
| S1-RF-1        | σ-01    | 64.2%      | −0.421        | ensemble_composition   |
| S1-RF-minimal  | σ-01    | 47.5%      | −0.235        | ensemble_composition   |
| S1-RF-2        | σ-01    | 45.0%      | −0.319        | feature_engineering    |
| S1-RF-4        | σ-03    | 45.0%      | −0.288        | feature_engineering    |

**Factorisation test:** R²(σ-only) = 0.63, R²(full φ) = 0.97, ΔR² = +0.34.
The factorisation conjecture does not hold — feature magnitudes explain substantial
additional variance beyond section-type presence alone.

**ε-sweep critical threshold:** ε ≈ 21, where S1-RF-1 and S2-GBM-5 first merge
despite producing improvement rates of 64% and 78% respectively.

---

## Citation

If you use this framework in your research, please cite:

```
[Your Name] (2026). Systematic structural analysis of program.md files
for Karpathy's autoresearch system. Research course project, [Institution].
```

Karpathy's autoresearch system: https://github.com/karpathy/autoresearch
