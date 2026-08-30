"""
Dependent variables (thesis 5.5).

Computed mechanically from the structured transcript; no natural-language
processing and no judge model is involved, because every step of the loop is
emitted as JSON with a declared action type.

Measurement window (thesis 5.5, "Measurement window"): the stopping-rule
component sets how many experiments a run performs, and several metrics are
aggregates over those experiments, so each metric is computed over the first
WINDOW experiments -- the number the fixed-budget level performs and therefore
the number available in every run. The complete-run value is computed alongside
it and carries the suffix _full.

Note on n_experiments: over a fixed prefix this is degenerate by construction
(it equals WINDOW whenever the run got that far). Its informative form is the
_full column, which is the budget the run actually spent and is the designated
behavioural counterpart of the stopping-rule component.
"""
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
import study_config as C  # noqa: E402


def _steps(tr, action):
    return [s for s in tr["steps"] if s["action"] == action]


def _same_value(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= C.REFINEMENT_FLOAT_TOL
        except (TypeError, ValueError):
            return a == b
    return a == b


def _is_refinement(cur, prev):
    """Same model family and at most one hyperparameter moved (thesis 5.5)."""
    if cur["model"] != prev["model"]:
        return False
    keys = set(cur["params"]) | set(prev["params"])
    moved = 0
    for k in keys:
        if k not in cur["params"] or k not in prev["params"]:
            moved += 1
        elif not _same_value(cur["params"][k], prev["params"][k]):
            moved += 1
    return moved <= 1


def _window(tr, w):
    """The first w executed experiments and the proposals that produced them."""
    execs, props, seen = [], [], 0
    for s in tr["steps"]:
        if s["action"] == "propose":
            pending = s
        elif s["action"] == "execute":
            if seen >= w:
                break
            execs.append(s)
            props.append(pending)
            seen += 1
    return props, execs


def _block(tr, props, execs, ref, suffix):
    n = len(execs)
    out = {}
    out[f"n_experiments{suffix}"] = n
    out[f"n_model_families{suffix}"] = len({p["model"] for p in props}) if props else 0

    ne = [p["params"]["n_estimators"] for p in props if "n_estimators" in p["params"]]
    lo, hi = C.N_ESTIMATORS_RANGE
    out[f"search_width{suffix}"] = round((max(ne) - min(ne)) / (hi - lo), 6) if len(ne) >= 2 else 0.0

    if len(props) >= 2:
        hits = sum(_is_refinement(props[t], props[t - 1]) for t in range(1, len(props)))
        out[f"refinement_rate{suffix}"] = round(hits / (len(props) - 1), 6)
    else:
        out[f"refinement_rate{suffix}"] = None      # undefined; excluded for this metric only

    out[f"mean_hypothesis_chars{suffix}"] = round(
        mean(len(p.get("content") or "") for p in props), 2) if props else None

    excluded = C.REGRET_SCALES_EXCLUDED.get(ref["dataset"], [])
    for scale, key in (("cv", "cv_accuracy_mean"), ("val", "val_accuracy")):
        best = max((e["content"][key] for e in execs), default=None)
        out[f"best_{scale}{suffix}"] = best
        a_star, a0 = ref[f"a_star_{scale}"], ref[f"a0_{scale}"]
        denom = a_star - a0
        if scale in excluded or best is None or denom <= 0:
            # thesis 5.5: a dataset with a*(d) == a0(d) offers no headroom, and
            # is excluded rather than assigned an arbitrary value. The exclusion
            # is declared per dataset and scale in study_config, so that a blank
            # column is a recorded decision rather than a silent failure.
            out[f"regret_{scale}{suffix}"] = None
        else:
            out[f"regret_{scale}{suffix}"] = round((a_star - best) / denom, 6)

    reg = out[f"regret_cv{suffix}"]
    out[f"regret_per_call{suffix}"] = round(reg / n, 6) if (reg is not None and n) else None
    return out


def encode_run(tr, ref):
    """One transcript + that dataset's reference constants -> one regression row."""
    row = {
        "run_id": tr["run_id"],
        "config_id": tr["config_id"],
        "dataset": tr["dataset"],
        "seed": tr["seed"],
    }
    row.update({a: tr["config"][a] for a in C.AXIS_ORDER})
    row.update({f"X_{a}": (1 if tr["config"][a] == 1 else -1) for a in C.AXIS_ORDER})

    props_w, execs_w = _window(tr, C.WINDOW)
    row.update(_block(tr, props_w, execs_w, ref, ""))

    props_f, execs_f = _window(tr, 10 ** 9)
    row.update(_block(tr, props_f, execs_f, ref, "_full"))

    # thesis 5.11: compliance, so that "no effect" is distinguishable from
    # "not followed"; and how the run ended, since the cap is a property of the
    # harness and not of the treatment.
    n_full = row["n_experiments_full"]
    row["complied_S"] = bool(n_full == 3) if tr["config"]["S"] == 0 else bool(n_full <= C.HARD_CAP)
    row["complied_B"] = bool(row["n_model_families_full"] == 1) if tr["config"]["B"] == 0 else True
    row["terminated_by"] = tr.get("terminated_by")
    row["n_bad_replies"] = tr.get("n_bad", 0)
    return row


FIELDS_ORDER = None  # set by the writer from the first row


def write_table(rows, out_dir):
    """Write the regression table as CSV and JSON (thesis 4.3 / 5.8)."""
    import csv
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{C.TABLE_BASENAME}.json").write_text(json.dumps(rows, indent=2))
    if rows:
        cols = list(rows[0].keys())
        with (out_dir / f"{C.TABLE_BASENAME}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return out_dir / f"{C.TABLE_BASENAME}.csv"
