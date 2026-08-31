"""
Dependent variables (thesis sec:metrics).

Four metrics, computed mechanically from the structured transcript. No evaluation
framework, scoring library or judge model is involved: each is a difference in
accuracy, a share of the proposals, or a count of executor calls.

  gain_over_default   score of the finally recommended configuration minus a0(d)
  wasted_trial_ratio  (rejected or malformed + exact duplicates) / attempts
  improvement_rate    share of executed trials beating the best score so far
  cost_to_best        executor calls up to and including the run's best score

Measurement window (thesis sec:metrics): the stopping rule sets how many
experiments a run performs and every metric aggregates over them, so each is
computed over the first WINDOW experiments, the number the fixed-budget level
performs and hence the number every run has. The complete-run value carries the
suffix _full.

Two further things this module writes into the row, both demanded by the thesis:

  * dataset_id, the task the run was made on. It is the categorical control D_j
    in eq. (2); without it on every single row the dataset fixed effect cannot be
    fitted and the baseline difficulty of a task leaks into the beta_i.
  * compliance, so that "this component had no effect" is distinguishable from
    "the model ignored this component" (thesis sec:results). Only B, S and E
    constrain an observable action; O constrains wording, so it gets the prose
    length as a manipulation check; M leaves no behavioural trace, because the
    executor computes the score whatever the instruction says.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
import study_config as C  # noqa: E402


# --------------------------------------------------------------- transcript --
def _executed(tr):
    """Executed experiments in order, each as (step, record)."""
    return [(s["step"], s["content"]) for s in tr["steps"] if s["action"] == "execute"]


def _final_choice(tr):
    """The structured configuration the run recommended, if it gave one."""
    for s in reversed(tr["steps"]):
        if s["action"] == "decide" and s.get("decision") == "stop":
            if s.get("final_model") and s.get("final_params") is not None:
                return s["final_model"], s["final_params"]
            return None
    return None


def _key(model, params):
    return (model, tuple(sorted((str(k), str(v)) for k, v in params.items())))


def _scale_key(scale):
    return "cv_accuracy_mean" if scale == "cv" else "val_accuracy"


# ------------------------------------------------------------------ metrics --
def _gain(tr, ref, ex, scale):
    """Gain over default on one scale, or None where that scale is excluded."""
    if scale in C.GAIN_SCALES_EXCLUDED.get(tr["dataset"], []):
        return None, None
    key, a0 = _scale_key(scale), ref[f"a0_{scale}"]
    scores = [r[key] for _, r in ex]
    if not scores:
        return None, None
    # The paper defines this on the configuration the run finally recommends.
    # If the run named one and it was actually executed, that score is used.
    # Otherwise the best executed score stands in, and the basis records which,
    # so a run that recommended something it never ran is never silently scored.
    chosen, basis = max(scores), "best_executed"
    fc = _final_choice(tr)
    if fc is not None:
        want = _key(*fc)
        for _, r in ex:
            if _key(r["model"], r["params"]) == want:
                chosen, basis = r[key], "final_config"
                break
    return round(chosen - a0, 6), basis


def _block(tr, ref, window, suffix):
    ex = _executed(tr)[:window]
    key = _scale_key(C.GAIN_SCALE)
    a0 = ref[f"a0_{C.GAIN_SCALE}"]
    scores = [r[key] for _, r in ex]
    best = max(scores) if scores else None
    out = {}

    # ---- gain over default -------------------------------------------------
    gain, basis = _gain(tr, ref, ex, C.GAIN_SCALE)
    out[f"gain_over_default{suffix}"] = gain
    out[f"gain_basis{suffix}"] = basis
    out[f"best_score{suffix}"] = best

    # Robustness scale (thesis sec:metrics): the same quantity on the other
    # accuracy scale, so an effect of M can be required to agree on both.
    rob, _ = _gain(tr, ref, ex, C.GAIN_SCALE_ROBUSTNESS)
    out[f"gain_over_default_{C.GAIN_SCALE_ROBUSTNESS}{suffix}"] = rob

    # ---- improvement rate --------------------------------------------------
    # "Strictly higher than the previous best-known score in that session."
    # At the first trial there is no previous score, so the run of comparisons
    # starts at a0(d): program.md hardcodes the default as the baseline to beat,
    # which makes the first trial meaningful instead of undefined.
    if ex:
        prev, hits = a0, 0
        for _, r in ex:
            if r[key] > prev:
                hits += 1
                prev = r[key]
        out[f"improvement_rate{suffix}"] = round(hits / len(ex), 6)
    else:
        out[f"improvement_rate{suffix}"] = None

    # ---- cost to best ------------------------------------------------------
    # 1-based index of the executor call that produced the run's highest score,
    # earliest index on ties. Exploration after that point does not count.
    out[f"cost_to_best{suffix}"] = (
        min(i for i, (_, r) in enumerate(ex, 1) if r[key] == best) if ex else None)

    out[f"n_executed{suffix}"] = len(ex)
    return out


def _wasted(tr, window):
    """Wasted trial ratio over the attempts that fall inside the window.

    An attempt is wasted when it never reached the executor (malformed reply or
    a shape or whitelist rejection) or when it exactly duplicates an earlier
    attempt in the same run. The denominator is every attempt made.
    """
    atts = tr.get("attempts")
    if not atts:
        return {"wasted_trial_ratio": None, "n_attempts": None,
                "n_rejected": None, "n_duplicate": None}
    # window: keep attempts up to and including the WINDOW-th executed one
    seen_exec, cut = 0, len(atts)
    for i, a in enumerate(atts):
        if a["outcome"] == "executed":
            seen_exec += 1
            if seen_exec == window:
                cut = i + 1
                break
    sel = atts[:cut] if window < 10 ** 9 else atts
    rejected = sum(1 for a in sel if a["outcome"] != "executed")
    dup = sum(1 for a in sel if a["outcome"] == "executed" and a.get("duplicate"))
    return {"wasted_trial_ratio": round((rejected + dup) / len(sel), 6) if sel else None,
            "n_attempts": len(sel), "n_rejected": rejected, "n_duplicate": dup}


# --------------------------------------------------------------- compliance --
# thesis sec:results. One mechanical check per component whose instruction
# constrains an action; the other two are handled as documented in the module
# docstring. Every check reads the complete run, not the measurement window,
# because the instruction governs the whole session.
STOP_TOLERANCE = 0.002          # the adaptive level's threshold, from axes.py S/1


def _complied_B(tr, ex):
    """B/0 says stay in one family for the session; B/1 says compare both."""
    fams = {r["model"] for _, r in ex}
    if not fams:
        return None
    return len(fams) == (2 if tr["config"]["B"] == 1 else 1)


def _complied_S(tr, ref, ex):
    """S/0 is exactly three experiments; S/1 stops on the >0.002 rule or at the cap."""
    n = len(ex)
    if not n:
        return None
    if tr["config"]["S"] == 0:
        return n == 3
    key, prev = _scale_key(C.GAIN_SCALE), ref[f"a0_{C.GAIN_SCALE}"]
    for i, (_, r) in enumerate(ex, 1):
        if r[key] - prev <= STOP_TOLERANCE:
            return n == i                     # the rule fired here; the run should end here
        prev = max(prev, r[key])
    return n == C.HARD_CAP                    # never fired, so it runs to the cap


def _refines(a, b):
    """b is a nearby variation of a: same family, at least one value carried over."""
    if a["model"] != b["model"]:
        return False
    return bool(set(a["params"].items()) & set(b["params"].items()))


def _complied_E(tr, ex):
    """E/1 refines the first promising configuration at once; E/0 does not."""
    if len(ex) < 2:
        return None
    refined = _refines(ex[0][1], ex[1][1])
    return refined if tr["config"]["E"] == 1 else not refined


def _prose_chars(tr):
    """Manipulation check for O: mean characters of model-written text per step."""
    txt = [len(s.get("content", "") or "") for s in tr["steps"]
           if s["action"] in ("propose", "interpret")]
    return round(sum(txt) / len(txt), 1) if txt else None


# ---------------------------------------------------------------- encoding ---
def encode_run(tr, ref):
    """One transcript plus that dataset's baseline -> one regression row."""
    row = {"run_id": tr["run_id"], "config_id": tr["config_id"],
           "dataset_id": tr["dataset"], "seed": tr["seed"]}
    row.update({a: tr["config"][a] for a in C.AXIS_ORDER})
    row.update({f"X_{a}": (1 if tr["config"][a] == 1 else -1) for a in C.AXIS_ORDER})

    row.update(_block(tr, ref, C.WINDOW, ""))
    row.update(_wasted(tr, C.WINDOW))
    row.update(_block(tr, ref, 10 ** 9, "_full"))
    full = _wasted(tr, 10 ** 9)
    row.update({f"{k}_full": v for k, v in full.items()})

    ex_full = _executed(tr)
    row["complied_B"] = _complied_B(tr, ex_full)
    row["complied_S"] = _complied_S(tr, ref, ex_full)
    row["complied_E"] = _complied_E(tr, ex_full)
    row["prose_chars"] = _prose_chars(tr)

    # how the run ended, since the cap belongs to the harness, not the treatment
    row["terminated_by"] = tr.get("terminated_by")
    row["n_bad_replies"] = tr.get("n_bad", 0)
    return row


def write_table(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{C.TABLE_BASENAME}.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (out_dir / f"{C.TABLE_BASENAME}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return out_dir / f"{C.TABLE_BASENAME}.csv"
