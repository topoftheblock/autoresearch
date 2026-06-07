# autoresearch — Random Forest Optimisation

## Setup

1. Agree on a run tag. Create branch `autoresearch/<tag>` from master.
2. Read `README.md`, `prepare.py` (do not modify), and `train.py`.
3. Verify data in `~/.cache/autoresearch/`. If absent, ask user to run `uv run prepare.py`.
4. Create `results.tsv` with header only.
5. Confirm and begin.

## Experimentation

Each run uses `uv run train.py`. Fixed 5-minute wall-clock training budget. The only file you may edit is `train.py`.

You may modify anything in `train.py` that could improve model quality: all hyperparameters, preprocessing steps, feature selection, ensemble composition, class weighting, or calibration. There is no predefined list of allowed changes.

Do NOT modify `prepare.py`, install new packages, or change the evaluation function.

The primary goal is to minimise val_loss. A secondary goal is to keep the model simple: prefer solutions with fewer trees or shallower depth when val_loss is equal. A simplification that preserves val_loss is always worth keeping.

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

Do not commit `results.tsv`.

## Experiment loop

LOOP FOREVER:

1. Inspect current `train.py`.
2. Formulate a hypothesis and edit `train.py`.
3. `git commit -m "<hypothesis and expected effect>"`
4. `uv run train.py > run.log 2>&1`
5. Extract val_loss and model size. Evaluate both objectives.
6. On crash: read log tail, fix if trivial, else log crash and move on.
7. Record in `results.tsv`.
8. Keep if val_loss improved or simplicity improved without loss degradation. Otherwise `git reset --hard HEAD~1`.

NEVER STOP. Run until manually interrupted.
