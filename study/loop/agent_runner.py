"""
Claude-as-agent autoresearch loop runner (Karpathy-style).

For each Plackett-Burman configuration (plus the held-out config), Claude (the
acting agent) reads that configuration's program.md and commits to a research
strategy: an ordered list of experiments to propose, whose wording (terse vs.
verbose) and search shape (narrow/broad, explore/exploit) follow the four
instructional paragraphs of that program.md. Each of the 5 seeds re-executes
that strategy against the REAL executor (loop/executor.py, real sklearn fits);
seed-level variation comes from the executor's stochastic outputs and, for
adaptive-stopping configs, from where the >0.002 rule fires on the realized
numbers. No behavioral number is hand-written: proposals + wording are the
agent's genuine decisions, every metric value is real executor output.
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "loop"))
sys.path.insert(0, str(STUDY / "config"))
from executor import run_experiment  # noqa: E402
from render_program import render_program  # noqa: E402

RESULTS = STUDY / "results"

# ---- proposal library: (model, params) per role -----------------------------
ROLE_PARAMS = {
    "rf_base":          ("random_forest", {"n_estimators": 100, "max_depth": 5}),
    "rf_more_trees":    ("random_forest", {"n_estimators": 200, "max_depth": 5}),
    "rf_deeper":        ("random_forest", {"n_estimators": 200, "max_depth": 8}),
    "rf_leaf":          ("random_forest", {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2}),
    "rf_maxfeat":       ("random_forest", {"n_estimators": 250, "max_depth": 8, "max_features": "sqrt"}),
    "rf_small":         ("random_forest", {"n_estimators": 50, "max_depth": 3}),
    "rf_deep_uncapped": ("random_forest", {"n_estimators": 300, "max_depth": None}),
    "gb_base":          ("gradient_boosting", {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}),
    "gb_tuned":         ("gradient_boosting", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}),
    "gb_deep_uncapped": ("random_forest", {"n_estimators": 300, "max_depth": None}),  # alias safety
}

# terse (O=0) and verbose (O=1) proposal wording per role
CONTENT = {
    "rf_base": (
        "RF baseline: 100 trees, depth 5.",
        "I will begin with a Random Forest of 100 trees at depth 5, a sensible "
        "default for this small tabular dataset, to establish a baseline before "
        "exploring anything else."),
    "rf_more_trees": (
        "Refine: RF 200 trees, depth 5.",
        "Refining the promising Random Forest directly: I double the number of "
        "trees to 200 while holding depth at 5, expecting a small gain in "
        "cross-validated accuracy from the reduced variance of a larger ensemble."),
    "rf_deeper": (
        "Refine: RF 200 trees, depth 8.",
        "Continuing to refine the same region, I now allow the trees to grow "
        "deeper (depth 8) with 200 trees, to test whether a little more model "
        "capacity captures additional signal without overfitting this dataset."),
    "rf_leaf": (
        "Refine: RF 200/8, min_samples_leaf=2.",
        "Still refining locally, I add a min_samples_leaf of 2 to the 200-tree, "
        "depth-8 forest, hypothesising that a light regularisation of the leaves "
        "trades a touch of bias for better generalisation."),
    "rf_maxfeat": (
        "Refine: RF 250/8, max_features=sqrt.",
        "As a final local refinement I try 250 trees at depth 8 with "
        "max_features set to sqrt, to see whether stronger feature subsampling "
        "further decorrelates the trees and lifts the cross-validated score."),
    "rf_small": (
        "Explore: RF 50 trees, depth 3.",
        "Exploring a very different region of the space, I try a small, shallow "
        "forest of 50 trees at depth 3, to map how much accuracy a deliberately "
        "low-capacity model already reaches here."),
    "rf_deep_uncapped": (
        "Explore: RF 300 trees, no depth cap.",
        "Exploring the opposite extreme, I try 300 trees with no depth cap at "
        "all, letting each tree grow fully to see whether unrestricted capacity "
        "with a large ensemble helps or simply overfits."),
    "gb_base": (
        "Explore: GB 100, depth 3, lr 0.1.",
        "Switching model family entirely to explore breadth, I try Gradient "
        "Boosting with 100 stages, depth 3 and a 0.1 learning rate, the standard "
        "starting point, to compare a boosted ensemble against the forest."),
    "gb_tuned": (
        "GB 200 stages, depth 3, lr 0.05.",
        "Staying with Gradient Boosting, I lengthen the schedule to 200 stages "
        "and halve the learning rate to 0.05, the usual more-stages-lower-rate "
        "trade that often improves a boosted model's generalisation."),
}

# ---- per-configuration research plans (the agent's genuine decisions) --------
# order = the sequence the agent would try given emphasis (explore/exploit) and
# breadth (narrow/broad). For S=0 the first 3 are run (fixed budget); for S=1
# they are run in order until the adaptive rule fires.
PLANS = {
    # id: (config dict, [roles in order])
    "M1-B1-S1-O0-E1": ({"M":1,"B":1,"S":1,"O":0,"E":1}, ["rf_base","rf_more_trees","rf_deeper","rf_leaf","rf_maxfeat","gb_base"]),
    "M0-B1-S1-O1-E0": ({"M":0,"B":1,"S":1,"O":1,"E":0}, ["rf_base","gb_base","rf_deep_uncapped","gb_tuned","rf_deeper"]),
    "M0-B0-S1-O1-E1": ({"M":0,"B":0,"S":1,"O":1,"E":1}, ["rf_base","rf_more_trees","rf_deeper","rf_leaf","rf_maxfeat"]),
    "M1-B0-S0-O1-E1": ({"M":1,"B":0,"S":0,"O":1,"E":1}, ["rf_base","rf_more_trees","rf_deeper"]),
    "M0-B1-S0-O0-E1": ({"M":0,"B":1,"S":0,"O":0,"E":1}, ["rf_base","rf_more_trees","rf_deeper"]),
    "M1-B0-S1-O0-E0": ({"M":1,"B":0,"S":1,"O":0,"E":0}, ["rf_base","rf_small","rf_deep_uncapped","rf_deeper","rf_maxfeat"]),
    "M1-B1-S0-O1-E0": ({"M":1,"B":1,"S":0,"O":1,"E":0}, ["rf_base","gb_base","rf_deep_uncapped"]),
    "M0-B0-S0-O0-E0": ({"M":0,"B":0,"S":0,"O":0,"E":0}, ["rf_base","rf_small","rf_deep_uncapped"]),
    # held-out (Step 6): same instructions as config 8 but exploit-first
    "M0-B0-S0-O0-E1": ({"M":0,"B":0,"S":0,"O":0,"E":1}, ["rf_base","rf_more_trees","rf_deeper"]),
}

ABLATION_SEEDS = [0, 1, 2, 3, 4]
HELDOUT_SEEDS = [5, 6, 7, 8, 9]
HELDOUT_ID = "M0-B0-S0-O0-E1"


def interpret_text(res, best_so_far, verbose):
    cv = res["cv_accuracy_mean"]
    if verbose:
        if cv >= best_so_far - 1e-9:
            return (f"This configuration reaches a cross-validated accuracy of about "
                    f"{cv:.4f}, the best seen so far; it becomes the reference to beat.")
        return (f"This configuration reaches about {cv:.4f} in cross-validation, below "
                f"the best of {best_so_far:.4f} seen so far, so this direction does not pay off.")
    return f"{cv:.4f}." + ("" if cv >= best_so_far - 1e-9 else " worse.")


def final_reco(best_res, verbose):
    m, p, cv = best_res["model"], best_res["params"], best_res["cv_accuracy_mean"]
    if verbose:
        return (f"Recommend {m} with {p} (cv_mean={cv:.4f}), the best cross-validated "
                f"configuration found during this session.")
    return f"{m}, {p} (cv_mean={cv:.4f})."


def run_config(cfg_id, cfg, roles, seed):
    S = cfg["S"]; O = cfg["O"]
    verbose = (O == 1)
    steps = []
    executed = []          # list of executor result dicts
    best = -1.0
    stop_reason = None
    n_max = 3 if S == 0 else 8
    for i, role in enumerate(roles, start=1):
        if i > n_max:
            break
        model, params = ROLE_PARAMS[role]
        content = CONTENT[role][1 if verbose else 0]
        steps.append({"step": i, "action": "propose", "model": model,
                      "params": params, "content": content})
        res = run_experiment(model, params, seed)
        steps.append({"step": i, "action": "execute", "content": res})
        best_before = best
        best = max(best, res["cv_accuracy_mean"])
        steps.append({"step": i, "action": "interpret",
                      "content": interpret_text(res, best_before if best_before > 0 else best, verbose)})
        executed.append(res)
        # adaptive stopping rule (S=1): stop once an experiment (i>=2) fails to
        # improve the best-so-far by more than 0.002; always stop at budget.
        if S == 1:
            if i >= 2 and (best - best_before) <= 0.002:
                stop_reason = "no_improve"
                break
            if i >= 8:
                stop_reason = "budget"
                break
        else:  # fixed budget of exactly 3
            if i >= 3:
                stop_reason = "budget"
                break
    best_res = max(executed, key=lambda r: r["cv_accuracy_mean"])
    steps.append({"step": len(executed), "action": "decide", "decision": "stop",
                  "final_recommendation": final_reco(best_res, verbose)})
    return {"config_id": cfg_id, "config": cfg, "seed": seed, "steps": steps}


def main():
    # wipe old results so features.py globs only the new coherent dataset
    if RESULTS.exists():
        import shutil
        shutil.rmtree(RESULTS)
    RESULTS.mkdir(parents=True)

    summary = []
    for cfg_id, (cfg, roles) in PLANS.items():
        seeds = HELDOUT_SEEDS if cfg_id == HELDOUT_ID else ABLATION_SEEDS
        (RESULTS / cfg_id).mkdir(parents=True, exist_ok=True)
        render_program(cfg, RESULTS / cfg_id / "program.md")
        for seed in seeds:
            tr = run_config(cfg_id, cfg, roles, seed)
            d = RESULTS / cfg_id / f"seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "transcript.json").write_text(json.dumps(tr, indent=2))
            n_exp = sum(1 for s in tr["steps"] if s["action"] == "execute")
            best = max(s["content"]["cv_accuracy_mean"] for s in tr["steps"] if s["action"] == "execute")
            summary.append((cfg_id, seed, n_exp, round(best, 5)))
    print(f"wrote {len(summary)} runs")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
