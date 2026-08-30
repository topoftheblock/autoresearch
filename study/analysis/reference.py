"""
Per-dataset reference constants for the normalised regret (thesis 5.5).

    reg(tau, d) = (a*(d) - a_best(tau)) / (a*(d) - a0(d))

a0(d) is a default-hyperparameter fit. a*(d) is an exhaustive grid search over
the same whitelist the loop may use. Both are computed once per dataset, outside
the loop, by procedures that involve no language model.

Both are computed on cross-validated accuracy AND on held-out validation
accuracy, because thesis 5.5 requires regret to be reported on both scales: the
evaluation-criterion component at its precise level names cross-validated
accuracy, so an effect measured only on that scale is partly an alignment
between treatment and instrument.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "loop"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from experiment import run_experiment  # noqa: E402
import study_config as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "reference_constants.json"


def _grid_points(family):
    grid = C.REFERENCE_GRID[family]
    keys = sorted(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def compute_for(dataset, seed=C.SPLIT_SEED, verbose=True):
    """Returns a0 and a* on both scales for one dataset."""
    C.validate(need_api=False)
    base, best = {}, {"cv": None, "val": None}

    for family in sorted(C.ALLOWED_PARAMS):
        rec = run_experiment(family, {}, seed, dataset)     # sklearn defaults
        if "error" in rec:
            raise RuntimeError(f"default fit failed for {family} on {dataset}: {rec['error']}")
        base[family] = rec
    a0_cv = max(r["cv_accuracy_mean"] for r in base.values())
    a0_val = max(r["val_accuracy"] for r in base.values())

    n = 0
    for family in sorted(C.REFERENCE_GRID):
        for params in _grid_points(family):
            rec = run_experiment(family, params, seed, dataset)
            if "error" in rec:
                continue
            n += 1
            for scale, key in (("cv", "cv_accuracy_mean"), ("val", "val_accuracy")):
                if best[scale] is None or rec[key] > best[scale]:
                    best[scale] = rec[key]
            if verbose and n % 25 == 0:
                print(f"    {dataset}: {n} grid points, best cv={best['cv']:.5f}")

    return {
        "dataset": dataset,
        "a0_cv": a0_cv, "a0_val": a0_val,
        "a_star_cv": best["cv"], "a_star_val": best["val"],
        "grid_points_evaluated": n,
    }


def load_or_compute(datasets, force=False, verbose=True):
    cache = json.loads(OUT.read_text()) if (OUT.exists() and not force) else {}
    for d in datasets:
        if d in cache:
            continue
        if verbose:
            print(f"  computing reference constants for {d} ...")
        cache[d] = compute_for(d, verbose=verbose)
        OUT.write_text(json.dumps(cache, indent=2))
    return cache


if __name__ == "__main__":
    C.validate(need_api=False)
    print(json.dumps(load_or_compute(C.DATASETS), indent=2))
