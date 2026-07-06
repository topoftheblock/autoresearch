# From Black-Box to White-Box Understanding of the Autoresearch Loop

## A pilot run of the full 6-step agenda

This report documents one complete pass through the six-step research agenda,
executed as a real (not simulated) pilot: every experiment number in this
document came from an actual `RandomForestClassifier` / `GradientBoostingClassifier`
fit on `sklearn`'s breast-cancer dataset, and every trace is a real transcript
saved under `results/`. Scope was deliberately kept small (5 axes, 8
configurations, mostly 2 repeats) to fit a single research session; see
**Limitations** for exactly what a full-scale version would need to add.

Code lives under `study/`: `config/` (Step 2), `design/` (Step 3), `loop/`
(the executor), `encoding/` (Step 4), `analysis/` (Step 5),
`causal_validation/` (Step 6).

---

## Step 1 — Human baseline: not run

By explicit agreement, this pilot skipped Step 1. A genuine human think-aloud
trace requires you to actually perform the task under recording — that can't
be substituted. The taxonomy the loop already logs against
(`hypothesize → propose → execute → interpret → decide`) is deliberately the
same one the elaborated agenda proposes for the human trace, so a real
recording can be segmented and compared to every existing run without
changing any downstream code. **This is the single highest-priority next
step** — without it, "distance from the human baseline" (one of the agenda's
own behavioral features) doesn't exist yet.

## Step 2 — program.md as a configuration space

Five two-level axes were chosen (`config/axes.py`), each grounded in a real
instructional choice: whether the evaluation metric is precise or vague (M),
whether the agent may mix Random Forest and Gradient Boosting or must stay in
one family (B), whether the experiment budget is fixed at 3 or adaptive up to
8 (S), whether reporting is terse or fully reasoned (O), and whether the
search strategy favors breadth-first exploration or immediate exploitation of
the first promising result (E). A single Jinja2 template
(`config/program_template.md.jinja`) renders any of the 32 possible
`program.md` files from a 5-bit configuration vector, guaranteeing that two
variants differ *only* in the axes intentionally varied.

## Step 3 — Ablation instead of random sampling

A full 2^5 factorial is 32 runs; instead, `design/doe.py` generates the
classical 8-run Plackett-Burman design (cyclic construction from the base row
`+ + + - + - -`, standard since Plackett & Burman 1946), giving a resolution-III
design: every main effect is estimable and unconfounded with every other main
effect, at the cost of confounding with two-way interactions. Each of the 8
configurations was run with 2 repeats (seeds 0 and 1) to get a first read on
the loop's own stochastic variance — the agenda recommends 3-5; 2 was the
budget here.

## Step 4 — Behavioral encoding

Because the loop already emits structured JSON per step, segmentation into
the action taxonomy was mechanical (`encoding/features.py`), not LLM-assisted
classification of free text. Five behavioral features were computed per run:
`n_experiments`, `n_distinct_models` (how many of {RF, GBM} were actually
tried), `breadth_spread` (range of `n_estimators` values tried, a proxy for
search breadth), `best_cv_accuracy` (product quality), and
`mean_proposal_chars` (a crude verbosity proxy). The full table is
`encoding/behavioral_table.json`.

## Step 5 — Surrogate model (the white box)

`analysis/surrogate.py` fits, for every behavioral feature, a linear
main-effects model on the +/-1-coded axes (orthogonal by construction, so
each coefficient is directly a variance share) and a depth-3 decision tree as
a non-parametric cross-check. Full output: `analysis/surrogate_report.txt`.

Headline findings from this pilot (**5 axes, 8 configs, N=16 runs — read as
suggestive, not confirmatory**):

| Behavioral feature | Dominant axis | Variance share | Direction |
|---|---|---|---|
| `mean_proposal_chars` (verbosity) | **O** (output format) | 74% | verbose → ~2-4x longer proposals |
| `n_distinct_models` (breadth of model families tried) | **B, O, E** (tied) | 33% each | broad↑, verbose↑, exploit-first↓ |
| `best_cv_accuracy` | **E** (emphasis) | 84% | exploit-first → +0.0025 mean accuracy |
| `n_experiments` | **S, E** (tied) | 50% each | adaptive stopping↓, exploit-first↑ |

Two things are worth flagging honestly. First, `O → verbosity` is close to a
sanity check by construction (the axis literally instructs verbosity) — its
clean 74% share is evidence the whole pipeline (template → loop → parser →
surrogate) correctly recovers a known ground truth, which is exactly what you
want before trusting it on the less obvious findings. Second,
`E → best_cv_accuracy` is the more interesting, non-obvious result: in this
small hyperparameter space, immediately refining a promising configuration
outperformed breadth-first exploration, on average, by about a quarter of a
percentage point of cross-validated accuracy — small, but the surrogate
attributes 84% of the (also small) between-configuration variance to it.

## Step 6 — Causal validation

A configuration outside the 8-run design (`M0-B0-S0-O0-E1`: narrow model
family, unspecified metric, fixed 3-experiment budget, terse, exploit-first)
was held out. Before running it, the Step 5 surrogate predicted
`best_cv_accuracy ≈ 0.9648` and `n_experiments ≈ 3.25`. Two fresh runs (seeds
2 and 3, never used in the ablation) then actually executed this
configuration for real. Observed: `best_cv_accuracy = 0.9610`,
`n_experiments = 3.0` exactly.

`n_experiments` was predicted almost exactly — S=0 (fixed budget) is a strong
and correctly-identified deterministic driver. `best_cv_accuracy` missed by
about 0.0038 (roughly 0.4 accuracy points), overestimating the benefit of
exploit-first. Per the agenda's own framing, this is not a failure of the
method: a resolution-III design cannot see interaction terms, and the miss is
itself evidence that E's effect on accuracy is not purely additive — it likely
interacts with B or S. The honest reading is: *the surrogate correctly
identifies which axis matters most for a given behavior, but a full-scale
version needs a resolution-IV (or full factorial) design before the
magnitude of E's effect can be trusted quantitatively.* Full numbers:
`causal_validation/validation_result.json`.

## Limitations (read before treating any number above as a finding)

1. **No human baseline** (Step 1 skipped by agreement) — "distance from human
   trace" isn't computed yet; add it before writing this up further.
2. **The acting "agent" was this same Claude Code session, not an independent
   fresh-context LLM call per run.** I acted as the research agent for every
   one of the 18 runs, deliberately trying to follow only what each
   rendered `program.md` said. But I already knew, throughout, that this was
   an ablation study and roughly what each axis was meant to do — a real
   confound against a fresh API call that only ever sees the rendered
   `program.md` and nothing else. Before trusting any effect size here,
   rerun the same design with an independent `ANTHROPIC_API_KEY`-driven
   process per run (the harness in `loop/` is already structured to make
   that swap easy — only the "who decides the next action" part needs to
   change from me to a subprocess/API call).
3. **Small N.** 2 repeats per configuration (agenda recommends 3-5); 8
   configurations (resolution III, no interactions estimable). Both were
   deliberately reduced to fit a single sitting.
4. **One dataset, one small hyperparameter menu.** Effects found here are
   scoped to breast-cancer + a handful of RF/GBM hyperparameters; generalizing
   to "random forests and gradient boosting" broadly would need more datasets.

## What to do next, in order

1. Record the Step 1 human baseline (think-aloud, Whisper transcription,
   segment into the same 6-action taxonomy already used here).
2. Swap the loop's decision-maker from me to independent API calls, and
   re-run the same 8-config design to check whether the effects above
   replicate without the self-study confound.
3. Move to a resolution-IV design or full 32-run factorial once repeats are
   cheap, to estimate the B×E / S×E interactions that Step 6 flagged.
4. Add the process-similarity-to-human-baseline feature once (1) exists.
