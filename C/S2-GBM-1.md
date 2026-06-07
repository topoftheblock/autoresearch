# autoresearch — Gradient Boosting Optimisation

## Setup

1. Agree on a run tag (e.g. `gbm1`). Create branch `autoresearch/<tag>` from master.
2. Read `README.md`, `prepare.py` (do not modify), and `train.py`.
3. Verify data in `~/.cache/autoresearch/`. If absent, ask user to run `uv run prepare.py`.
4. Create `results.tsv` with header only.
5. Confirm and begin.

## Experimentation

Each run: `uv run train.py`. Fixed 5-minute wall-clock budget. Edit only `train.py`.

You may only modify these hyperparameters of the XGBClassifier:
- `n_estimators`: integer in [50, 500]
- `max_depth`: integer in [2, 10]
- `learning_rate`: float in [0.01, 0.3]
- `subsample`: float in [0.5, 1.0]
- `colsample_bytree`: float in [0.5, 1.0]

Do NOT change the model class, the dataset, the train/val split, `prepare.py`, or the evaluation function. Do NOT install new packages. One change per experiment.

Goal: minimise val_loss (lower is better).

## Output format

```
val_loss:         0.2201
training_seconds: 300.1
peak_vram_mb:     920.3
```

Extract: `grep "^val_loss:" run.log`

## Logging

Log to `results.tsv` (tab-separated):

```
commit  val_loss  memory_gb  status  description
```

Status: `keep`, `discard`, or `crash`. Crash: 0.000000 / 0.0. Do not commit file.

## Experiment loop

LOOP FOREVER:

1. Inspect `train.py` and recent results.
2. Change exactly one hyperparameter.
3. `git commit -m "<HP: old->new>"`
4. `uv run train.py > run.log 2>&1`
5. `grep "^val_loss:" run.log`
6. On crash: `tail -n 50 run.log`, fix once if trivial, else log crash.
7. Record in `results.tsv`.
8. Keep if improved; `git reset --hard HEAD~1` otherwise.

NEVER STOP. Run until manually interrupted.
