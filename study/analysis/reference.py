"""
Per-dataset baseline for the gain over default (thesis sec:setup).

    gain(tau, d) = score(final configuration of tau) - a0(d)

a0(d) is the accuracy of a default-hyperparameter fit, taken as the better of the
two model families. It is computed once per dataset, outside the loop, by a
procedure that involves no language model.

The exhaustive grid search that earlier versions of this study used is gone. The
current metric set needs no ranking of the whitelist, so the reference costs two
fits per dataset instead of 136 and puts no estimated quantity into a divisor.

Both scales are recorded. GAIN_SCALE selects the one the metrics use, and
GAIN_SCALES_EXCLUDED marks a dataset and scale where a0 is already at the
ceiling, so a gain cannot be measured there.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "loop"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from experiment import run_experiment  # noqa: E402
import study_config as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "reference_constants.json"


def compute_for(dataset, seed=C.SPLIT_SEED):
    """a0(d) on both scales, from a default fit of each model family."""
    fits = {}
    for family in sorted(C.ALLOWED_PARAMS):
        rec = run_experiment(family, {}, seed, dataset)
        if "error" in rec:
            raise RuntimeError(f"default fit failed for {family} on {dataset}: {rec['error']}")
        fits[family] = rec
    return {
        "dataset": dataset,
        "a0_cv": max(r["cv_accuracy_mean"] for r in fits.values()),
        "a0_val": max(r["val_accuracy"] for r in fits.values()),
        "a0_by_family": {k: {"cv": v["cv_accuracy_mean"], "val": v["val_accuracy"]}
                         for k, v in fits.items()},
    }


def load_or_compute(datasets, force=False, verbose=True):
    cache = json.loads(OUT.read_text()) if (OUT.exists() and not force) else {}
    # entries written by an earlier version carried grid fields; recompute those.
    cache = {k: v for k, v in cache.items() if "a0_by_family" in v}
    for d in datasets:
        if d in cache:
            continue
        if verbose:
            print(f"  computing a0 for {d} ...")
        cache[d] = compute_for(d)
        OUT.write_text(json.dumps(cache, indent=2))
    return cache


if __name__ == "__main__":
    C.validate(need_api=False)
    print(json.dumps(load_or_compute(C.DATASETS), indent=2))
