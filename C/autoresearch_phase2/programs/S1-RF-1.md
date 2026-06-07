# autoresearch — Random Forest Optimisation

## Setup

1. Agree on a run tag (e.g. `rf1`). Create branch `autoresearch/<tag>` from master.
2. Read `README.md`, `prepare.py` (do not modify), and `train.py`.
3. Verify data exists in `~/.cache/autoresearch/`. If not, ask the user to run `uv run prepare.py`.
4. Create `results.tsv` with the header row only.
5. Confirm setup and begin.

## Experimentation

Each run uses `uv run train.py`. The time budget is 5 minutes wall-clock training time. The only file you may edit is `train.py`.

You may only modify these hyperparameters of the RandomForestClassifier:
- `n_estimators`: integer in [50, 500]
- `max_depth`: integer in [3, 20] or None
- `min_samples_leaf`: integer in [1, 20]
- `max_features`: one of {"sqrt", "log2", 0.3, 0.5, 0.7}

Do NOT change the model class, the dataset, the train/val split, or any code in `prepare.py`. Do NOT install new packages. Do NOT modify the evaluation function.

The goal is to minimise val_loss (lower is better). Run one change at a time.

## Output format

After each run the script prints:

```
val_loss:         0.2314
training_seconds: 300.1
peak_vram_mb:     1820.4
```

Extract with: `grep "^val_loss:" run.log`

## Logging

Log every experiment to `results.tsv` (tab-separated). Schema:

```
commit  val_loss  memory_gb  status  description
```

Status is `keep`, `discard`, or `crash`. Use 0.000000 / 0.0 for crashes. Do not commit `results.tsv`.

## Experiment loop

LOOP FOREVER:

1. Inspect current branch and `train.py`.
2. Edit one hyperparameter in `train.py`.
3. `git commit -m "<short description>"`
4. `uv run train.py > run.log 2>&1`
5. `grep "^val_loss:" run.log`
6. If grep is empty, run `tail -n 50 run.log`, attempt a fix, re-run once. If still broken, log crash and move on.
7. Record in `results.tsv`.
8. If val_loss improved (lower): keep the commit and advance. If not: `git reset --hard HEAD~1` and discard.

NEVER STOP to ask the user for permission to continue. Run until manually interrupted.
