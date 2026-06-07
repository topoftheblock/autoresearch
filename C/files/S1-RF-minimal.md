# autoresearch — Random Forest

## Setup

1. Agree tag. Branch from master.
2. Read `prepare.py` and `train.py`.
3. Create `results.tsv` header.

## Experimentation

`uv run train.py`. 5-minute budget. Edit only `train.py`.
Modify n_estimators, max_depth, min_samples_leaf, max_features only.
Do NOT touch `prepare.py`. Minimise val_loss.

## Output format

```
val_loss: 0.2314
```

Extract: `grep "^val_loss:" run.log`

## Logging

`results.tsv`: `commit  val_loss  memory_gb  status  description`

## Experiment loop

LOOP: inspect -> change -> commit -> run -> log -> keep/reset. NEVER STOP.
