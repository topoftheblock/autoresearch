"""
Step 4: Behavioral encoding.

Turns each run's structured transcript into a small set of quantitative
behavioral features. Because the loop already emits structured JSON per step
(action taxonomy: propose / execute / interpret / decide), this segmentation
is mechanical, as anticipated in the elaborated agenda for the case where the
loop already produces structured logs.
"""
import json
from pathlib import Path
from statistics import mean, pstdev

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _experiment_steps(transcript):
    return [s for s in transcript["steps"] if s["action"] == "execute"]


def _propose_steps(transcript):
    return [s for s in transcript["steps"] if s["action"] == "propose"]


def _interpret_steps(transcript):
    return [s for s in transcript["steps"] if s["action"] == "interpret"]


def encode_run(transcript: dict) -> dict:
    experiments = _experiment_steps(transcript)
    proposals = _propose_steps(transcript)
    interprets = _interpret_steps(transcript)

    cv_means = [e["content"]["cv_accuracy_mean"] for e in experiments]
    models_used = {p["model"] for p in proposals}

    # distinct (model, frozenset(params.items())) combinations actually tried
    distinct_settings = {
        (p["model"], tuple(sorted(p["params"].items()))) for p in proposals
    }

    # proxy for search breadth: spread of n_estimators values tried
    n_estimators_values = [
        p["params"].get("n_estimators") for p in proposals if "n_estimators" in p["params"]
    ]
    breadth_spread = (max(n_estimators_values) - min(n_estimators_values)) if len(n_estimators_values) > 1 else 0

    best_cv = max(cv_means) if cv_means else None
    first_cv = cv_means[0] if cv_means else None
    improvement = (best_cv - first_cv) if (best_cv is not None and first_cv is not None) else 0.0

    verbosity = mean(len(p.get("content", "")) for p in proposals) if proposals else 0.0

    return {
        "config_id": transcript["config_id"],
        "seed": transcript["seed"],
        **transcript["config"],
        "n_experiments": len(experiments),
        "n_distinct_models": len(models_used),
        "n_distinct_settings": len(distinct_settings),
        "breadth_spread": breadth_spread,
        "best_cv_accuracy": best_cv,
        "improvement_over_first": round(improvement, 5),
        "mean_proposal_chars": round(verbosity, 1),
    }


def build_table() -> list[dict]:
    rows = []
    for run_path in sorted(RESULTS_DIR.glob("*/seed*/transcript.json")):
        transcript = json.loads(run_path.read_text())
        rows.append(encode_run(transcript))
    return rows


if __name__ == "__main__":
    rows = build_table()
    out_path = Path(__file__).parent / "behavioral_table.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} rows to {out_path}")
    for r in rows:
        print(r)
