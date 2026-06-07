# autoresearch — Random Forest Hyperparameter Tuning (Study 3)

## Overview

This is a budget-controlled hyperparameter tuning session for a Random Forest classifier. The experiment count is tracked explicitly because this run will be compared against classical hyperparameter optimisation methods under the same budget. Every decision should maximise the best val_loss achieved within the fewest possible experiments — sample efficiency matters here as much as final quality.

## Setup

1. Agree on a run tag (e.g. `rf-hp`). Create branch `autoresearch/<tag>` from master.
2. Read `README.md`, `prepare.py` (do not modify), and `train.py` in full.
3. Verify data in `~/.cache/autoresearch/`. If absent, ask user to run `uv run prepare.py`.
4. Create `results.tsv` with extended header including `exp_count`.
5. Confirm readiness and await go-ahead.

## Scope

Change exactly one per experiment:
- `n_estimators`: integer in [50, 500]
- `max_depth`: integer in [3, 20] or None
- `min_samples_leaf`: integer in [1, 20]
- `min_samples_split`: integer in [2, 20]
- `max_features`: one of {"sqrt", "log2", 0.3, 0.5, 0.7}
- `bootstrap`: True or False
- `class_weight`: None or "balanced"

Do NOT change the model class, dataset, train/val split, `prepare.py`, or the evaluation function. No new packages. The sole objective is to minimise val_loss.

## Budget tracking

Total budget: 60 experiments. After every 10 experiments, record a checkpoint summary in `results.tsv`:
"--- checkpoint: best val_loss so far = X.XXXX at exp N ---"

Write a final summary at experiment 60, then continue if not interrupted.

## Output format

```
val_loss:         0.2314
training_seconds: 300.1
peak_vram_mb:     1820.4
```

Extract: `grep "^val_loss:" run.log`

## Logging

Extended `results.tsv` schema (tab-separated):

```
exp_count  commit  val_loss  memory_gb  status  description
```

Do not commit `results.tsv`.

## Experiment loop

LOOP (target: 60 experiments):

1. Increment exp_count. Review `train.py` and history.
2. Choose the single change most likely to improve val_loss. Prefer testing unexplored regions of the HP space over fine-tuning already-tested values.
3. Change exactly one hyperparameter.
4. `git commit -m "[exp N] <HP: old->new, hypothesis>"`
5. `uv run train.py > run.log 2>&1`
6. `grep "^val_loss:" run.log`
7. On crash: log crash (exp_count increments), fix once if trivial, else move on.
8. Record in extended `results.tsv`.
9. Keep if improved; `git reset --hard HEAD~1` otherwise.

After experiment 60, write final summary. Then continue if not interrupted. NEVER STOP prematurely.
