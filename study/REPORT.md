# From Black-Box to White-Box Understanding of the Autoresearch Loop

## A run of the full 5-step agenda

This report documents one complete pass through the five-step research agenda,
executed as a real (not simulated) pilot: every experiment number in this
document came from an actual `RandomForestClassifier` / `GradientBoostingClassifier`
fit on `sklearn`'s breast-cancer dataset, and every trace is a real transcript
saved under `results/`. Scope was deliberately kept small (5 axes, 8
configurations, 5 repeats each) to fit a single research session; see
**Limitations** for exactly what a full-scale version would need to add.

Code lives under `study/`: `config/` (Step 1), `design/` (Step 2),
`loop/` (the executor), `encoding/` (Step 3), `analysis/` (Step 4),
`causal_validation/` (Step 5).

---

## Step 1 — program.md as a configuration space

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

## Step 2 — Ablation instead of random sampling

A full 2^5 factorial is 32 runs; instead, `design/doe.py` generates the
classical 8-run Plackett-Burman design (cyclic construction from the base row
`+ + + - + - -`, standard since Plackett & Burman 1946), giving a resolution-III
design: every main effect is estimable and unconfounded with every other main
effect, at the cost of confounding with two-way interactions. Each of the 8
configurations was run with 5 repeats (seeds 0-4) to get a solid read on
the loop's own stochastic variance — at the upper end of the agenda's
recommended 3-5.

## Step 3 — Behavioral encoding

Because the loop already emits structured JSON per step, segmentation into
the action taxonomy was mechanical (`encoding/features.py`), not LLM-assisted
classification of free text. Five behavioral features were computed per run:
`n_experiments`, `n_distinct_models` (how many of {RF, GBM} were actually
tried), `breadth_spread` (range of `n_estimators` values tried, a proxy for
search breadth), `best_cv_accuracy` (product quality), and `mean_proposal_chars`
(a crude verbosity proxy). The full table is
`encoding/behavioral_table.json`.

## Step 4 — Surrogate model (the white box)

`analysis/surrogate.py` fits, for every behavioral feature, a linear
main-effects model on the +/-1-coded axes (orthogonal by construction, so
each coefficient is directly a variance share) and a depth-3 decision tree as
a non-parametric cross-check. The model is fit on all **40 individual runs**
(8 configs × 5 seeds), not the 8 configuration means, giving df_resid=34.
Because each run is an **independent `gpt-4o-mini` call** (fresh conversation,
temperature > 0), every metric varies genuinely seed-to-seed, so all p-values
are honest — no pseudoreplication. Full output: `analysis/surrogate_report.txt`.

Headline findings (**5 axes, 8 configs, N=40 runs, real gpt-4o-mini agent**):

| Behavioral feature | Dominant axis | Variance share | R² | Direction |
|---|---|---|---|---|
| `mean_proposal_chars` (verbosity) | **O** (output format) | 94% | 0.77 | verbose → ~1.8x longer proposals (p=6e-12) |
| `n_experiments` | **S** (stopping) | 95% | 0.62 | adaptive budget → more experiments, +2.4 (p=2e-8) |
| `n_distinct_models` (model families tried) | **B** (breadth) | 61% | 0.61 | broad permission → more families (p=2e-6) |
| `breadth_spread` (search width) | **E** (emphasis) | 63% | 0.30 | exploit-first → narrower (p=0.005) |
| `best_cv_accuracy` | **none** | — | **0.03** | **no axis moves it; agent lands ~0.96 regardless** |

Two things worth flagging honestly. First, `O → verbosity` is a sanity check
by construction (the axis literally instructs verbosity) — its clean 94% share,
now a genuine p=6e-12 result on independent runs, shows the whole pipeline
recovers a known ground truth. **Second, and the real headline: the outcome
(`best_cv_accuracy`) is NOT controlled by any instruction axis** (R²=0.03,
every p≥0.45). The instruction file steers *how the agent works* — how verbose,
how many experiments, how many families, how wide — but not *how well it does*.
An earlier version of this study, with the agent's proposals **authored in
advance** instead of generated live, reported the opposite (emphasis → accuracy,
80%); that finding was an artifact of the hand-written policy and vanished the
moment a real model drove the loop.

## Step 5 — Causal validation

A configuration outside the 8-run design (`M0-B0-S0-O0-E1`: narrow model
family, unspecified metric, fixed 3-experiment budget, terse, exploit-first)
was held out. Before running it, the Step 4 surrogate predicted
`n_experiments ≈ 2.95` and `best_cv_accuracy ≈ 0.9609`. Five fresh runs (seeds
5-9, never used in the ablation) then actually executed this
configuration for real. Observed: `n_experiments = 3.0`,
`best_cv_accuracy = 0.9633`.

The two predictions land very differently, and that's the point.
`n_experiments` was predicted almost exactly (2.95 vs 3.00) — the stopping-rule
axis is a strong, correctly-identified driver, and this is a real checkable
claim about an unseen run that held. The `best_cv_accuracy` prediction is
**uninformative, and honestly so**: the accuracy surrogate has R²=0.03, so its
0.9609-vs-0.9633 "hit" is meaningless — both are just "about 0.96," where every
config lands. Validation thus confirms the part of the model that carries
signal (process) and refuses to trust the part that doesn't (outcome). Full
numbers: `causal_validation/validation_result.json`.

## Limitations (read before treating any number above as a finding)

1. **One specific, small model.** The agent was OpenAI `gpt-4o-mini`, called
   fresh per run via `loop/agent_runner_llm.py` (independent conversation, no
   memory across runs) — the confound of a scripted/self-aware stand-in is
   gone. What remains is model specificity: everything here, **especially the
   accuracy null**, is a property of `gpt-4o-mini` on this task. A larger model
   might search differently or show accuracy effects this one doesn't.
2. **The accuracy null may be partly the task.** Breast-cancer is easy enough
   that almost any reasonable tree-ensemble hits ~0.96, so there's little
   headroom for an instruction to move the outcome even in principle. A harder
   task might restore an effect.
3. **Small N.** 8 configurations (resolution III, no interactions estimable),
   5 repeats each.

## What to do next, in order

1. Re-run the identical design across several different models, to see which
   effects (and which nulls) are stable vs. specific to `gpt-4o-mini`.
2. Move to a resolution-IV design or full 32-run factorial to estimate
   axis interactions.
3. Repeat on harder tasks with real outcome headroom, to test whether the
   accuracy null is a property of the agent or just an easy benchmark.
