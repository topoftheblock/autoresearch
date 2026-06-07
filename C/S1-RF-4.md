# autoresearch — Random Forest Optimisation

## Overview

This is an open-ended autonomous research session. You are a creative ML researcher with full latitude to improve a Random Forest classifier by any means you judge promising. There is no predefined list of allowed changes. Your only hard constraint is the evaluation function defined in `prepare.py`, which you must not touch.

Approach this as you would a competitive ML benchmark: explore widely, follow promising leads, and do not limit yourself to textbook hyperparameter tuning. Feature engineering, preprocessing, ensemble composition, calibration, and any other technique that can be expressed as a change to `train.py` is fair game.

## Setup

1. Agree on a run tag. Create branch `autoresearch/<tag>` from master.
2. Read `README.md`, `prepare.py`, and `train.py` in full before proposing any changes.
3. Verify `~/.cache/autoresearch/` contains required data. If not, ask the user to run `uv run prepare.py`.
4. Create `results.tsv` with the header row only.
5. Confirm readiness and begin as soon as the user gives the go-ahead.

## Experimentation

Each run uses `uv run train.py`. Fixed 5-minute wall-clock training budget. You may edit `train.py` freely. You may not modify `prepare.py`, install new packages, or change the evaluation function.

The sole objective is to minimise val_loss. Do not apply any secondary criteria unless they help val_loss directly. When you feel you have exhausted incremental improvements, try qualitatively different approaches: change the feature representation, the ensemble strategy, or the training objective.

## Output format

```
val_loss:         0.2314
training_seconds: 300.1
peak_vram_mb:     1820.4
```

Extract: `grep "^val_loss:" run.log`

## Logging

Log to `results.tsv` (tab-separated):

```
commit  val_loss  memory_gb  status  description
```

Status: `keep`, `discard`, or `crash`. Crash entries use 0.000000 / 0.0. Do not commit `results.tsv`.

## Experiment loop

LOOP FOREVER:

1. Review `train.py` and recent `results.tsv` history.
2. Propose and implement the change you judge most likely to improve val_loss. Write a clear commit message explaining the idea and the expected effect.
3. `git commit -m "<idea and rationale>"`
4. `uv run train.py > run.log 2>&1`
5. `grep "^val_loss:" run.log`
6. On crash: `tail -n 50 run.log`, fix if trivial, else log crash and continue.
7. Record in `results.tsv`.
8. Keep if val_loss improved; `git reset --hard HEAD~1` otherwise.

NEVER STOP. If you run out of ideas, think harder: revisit near-misses, try combinations, attempt radical changes. Run until manually interrupted.
