# autoresearch — Random Forest Optimisation

## Overview

This is a structured, sequential hyperparameter optimisation session. You will improve a Random Forest classifier by changing exactly one hyperparameter per experiment. This discipline is intentional: it produces a clean causal record of which changes are responsible for which improvements, and it prevents confounded results that arise from changing multiple things at once.

## Setup

1. Agree on a run tag (e.g. `rf5`). Create branch `autoresearch/<tag>` from master.
2. Read `README.md`, `prepare.py` (do not modify), and `train.py` in full.
3. Verify data in `~/.cache/autoresearch/`. If absent, ask the user to run `uv run prepare.py`.
4. Create `results.tsv` with the header row only.
5. Confirm setup and await go-ahead.

## Scope

Change exactly one of the following hyperparameters per experiment:
- `n_estimators`: integer in [50, 500]
- `max_depth`: integer in [3, 20] or None
- `min_samples_leaf`: integer in [1, 20]
- `min_samples_split`: integer in [2, 20]
- `max_features`: one of {"sqrt", "log2", 0.3, 0.5, 0.7}
- `bootstrap`: True or False

Each experiment must change exactly one hyperparameter from the current best configuration. Do not bundle changes. If you want to test an interaction between two hyperparameters, test each one individually first.

## Constraints

Do not modify `prepare.py`, install packages, or change the evaluation function. Do not change the model class, the dataset, or the train/validation split.

## Evaluation

Optimise two objectives jointly:
1. Minimise val_loss (primary).
2. Minimise model complexity, defined as `n_estimators * max_depth` where max_depth defaults to 20 if set to None (secondary).

Composite score: `score = val_loss + 0.0001 * (n_estimators * max_depth)`

Keep the change if the composite score improves. This weighting means a val_loss reduction of 0.001 is worth approximately a 10x reduction in the complexity proxy.

## Output format

```
val_loss:         0.2314
training_seconds: 300.1
peak_vram_mb:     1820.4
num_params_M:     0.4
```

Extract: `grep "^val_loss:\|^num_params_M:" run.log`

## Logging

Log to `results.tsv` (tab-separated):

```
commit  val_loss  memory_gb  status  description
```

In the description field, always state: (a) which hyperparameter was changed, (b) from what value to what value, and (c) the composite score. Do not commit `results.tsv`.

## Experiment loop

LOOP FOREVER:

1. Review `train.py` and recent history. Identify which hyperparameter to probe next and in which direction, based on the pattern of past results.
2. Change exactly one hyperparameter.
3. `git commit -m "<HP changed: old->new, hypothesis>"`
4. `uv run train.py > run.log 2>&1`
5. `grep "^val_loss:\|^num_params_M:" run.log`
6. Compute the composite score. On crash: read log, fix if trivial, else log crash.
7. Record in `results.tsv` with the full description format above.
8. Keep if composite score improved; `git reset --hard HEAD~1` otherwise.

NEVER STOP. Run until manually interrupted.
