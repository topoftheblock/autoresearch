# autoresearch — Gradient Boosting Optimisation

## Overview

This is a structured sequential hyperparameter optimisation session for a Gradient Boosting classifier. You will change exactly one hyperparameter per experiment. Pay particular attention to the learning_rate / n_estimators trade-off: these two parameters are closely coupled and should be tested individually before drawing conclusions about their joint effect.

## Setup

1. Agree on a run tag (e.g. `gbm5`). Create branch `autoresearch/<tag>` from master.
2. Read `README.md`, `prepare.py` (do not modify), and `train.py` in full.
3. Verify data in `~/.cache/autoresearch/`. If absent, ask user to run `uv run prepare.py`.
4. Create `results.tsv` with header only.
5. Confirm readiness and await go-ahead.

## Scope

Change exactly one of the following per experiment:
- `n_estimators`: integer in [50, 500]
- `max_depth`: integer in [2, 10]
- `learning_rate`: float in [0.005, 0.3]
- `subsample`: float in [0.5, 1.0]
- `colsample_bytree`: float in [0.5, 1.0]
- `reg_alpha`: float in [0.0, 1.0]
- `reg_lambda`: float in [0.5, 5.0]

## Constraints

Do NOT change more than one hyperparameter per experiment. Do NOT change the model class, dataset, train/val split, `prepare.py`, or evaluation function. No new packages.

## Evaluation

Optimise two objectives:
1. Minimise val_loss (primary).
2. Minimise complexity proxy: `n_estimators * max_depth` (secondary).

Composite score: `score = val_loss + 0.0001 * (n_estimators * max_depth)`

Keep if composite score improves.

## Output format

```
val_loss:         0.2201
training_seconds: 300.1
peak_vram_mb:     920.3
num_params_M:     0.2
```

Extract: `grep "^val_loss:\|^num_params_M:" run.log`

## Logging

Log to `results.tsv` (tab-separated):

```
commit  val_loss  memory_gb  status  description
```

Description format: "learning_rate 0.1->0.05, val_loss 0.2201->0.2183, score 0.2201->0.2184 keep". Do not commit `results.tsv`.

## Experiment loop

LOOP FOREVER:

1. Review `train.py` and recent history. Decide which single hyperparameter to probe next.
2. Change exactly one hyperparameter.
3. `git commit -m "<HP: old->new, hypothesis>"`
4. `uv run train.py > run.log 2>&1`
5. `grep "^val_loss:\|^num_params_M:" run.log`
6. Compute composite score. On crash: read log tail, fix if trivial, else log crash.
7. Record in `results.tsv` with full description format.
8. Keep if composite score improved; `git reset --hard HEAD~1` otherwise.

NEVER STOP. Run until manually interrupted.
