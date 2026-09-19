# Autoresearch Loops: A Causal Study of the Instruction File

Research project with Prof. Ramesh — Christian Block

An autoresearch loop is a system in which a language model proposes, executes and
interprets experiments without a human in the loop. In the form popularized by
Karpathy's `autoresearch`, the whole experimental procedure is written in English
in a single instruction file, `program.md`, which therefore acts as the controller
of the system. This study treats that file as a configuration vector: five
components are each written at two levels, all `2^5 = 32` combinations are
generated and executed against a real executor on two tabular classification tasks
with twenty replicates each, giving 1,280 runs, and four metrics read from the
structured transcripts are regressed on an orthogonal ±1 coding of the components
with a dataset fixed effect.

Manuscript: `thesis_new_compiled.tex`, built with `pdflatex` (run twice) to
`thesis_new_compiled.pdf`. Defence deck: `presentation/presentation.tex`.

## Repository layout

```
study/
  run_experiment.py        master script; every stage is a flag on this file
  prepare.py               the data and the score. Nobody edits this.
  train.py                 the estimator and the whitelist. The agent reaches
                           this through a JSON proposal, never by editing it.
  config/
    study_config.py        every constant the thesis fixes, in one place
    axes.py                the ten treatment paragraphs (see "Treatment set")
    program_template.md.jinja   the fixed part of program.md, with five slots
    render_program.py      configuration vector + dataset -> one program.md
  design/full_factorial.py the 32 configurations, in a fixed order
  loop/runner.py           one run: propose / execute / interpret / decide
  encoding/metrics.py      transcripts -> one regression row per run
  analysis/
    reference.py           the baseline a0(d), two fits per dataset
    regression.py          eq. (2) and eq. (3), HC3 errors, BH adjustment
    tables.py              the LaTeX tables and figure used in the thesis
  results_full/            1,280 transcripts + the analysis outputs
```

Each of the 1,280 result directories holds the `transcript.json` of one run and
the exact `program.md` that produced it, so any single run can be inspected
without re-deriving anything.

## Requirements

Python 3.12.7 and the five pinned libraries in `study/requirements.txt`.

```bash
cd study
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Re-executing the loop calls the OpenAI chat completions endpoint and needs a key:

```bash
export OPENAI_API_KEY=sk-...
```

Nothing below the "Re-run the study" heading needs a key. The transcripts are in
the repository, so every number in the thesis can be regenerated offline.

## Reproduce the thesis from the committed transcripts

No API key and no network access required.

```bash
cd study
.venv/bin/python run_experiment.py --check            # validate config, print the plan
.venv/bin/python run_experiment.py --encode           # transcripts -> regression_table.csv
.venv/bin/python run_experiment.py --analyze          # fit eq. (2) and eq. (3)
.venv/bin/python analysis/tables.py                   # LaTeX tables + coefficient figure
```

`--encode` rewrites `results_full/regression_table.{csv,json}` from the 1,280
transcripts. `--analyze` writes `regression_main.csv`,
`regression_interactions.csv`, `regression_compliance.csv` and
`regression_report.txt` beside it. `tables.py` writes `table_main.tex`,
`table_interactions.tex` and `figure_coefficients.tex`, which are pasted into the
manuscript unchanged — they should come back byte-identical to the committed
copies, and Tables 3 and 4 and Figure 2 of the thesis are those files.

The baseline is cached in `analysis/reference_constants.json` and recomputes in
seconds:

```bash
.venv/bin/python analysis/reference.py
```

## Re-run the study

Needs `OPENAI_API_KEY`. The full study is roughly 9,900 model calls.

```bash
cd study
.venv/bin/python run_experiment.py --dry-run          # one cell, no API calls
.venv/bin/python run_experiment.py --reference        # compute a0(d) only
.venv/bin/python run_experiment.py                    # the full 1,280 runs
```

A cell that already holds a transcript is skipped, so the run resumes after an
interruption. Executor time is minor — 4,768 estimator fits took 74 minutes in
total — and wall-clock is dominated by the model interface.

`study_config.py` refuses to start when a constant the thesis leaves unspecified
has not been set: such constants are marked `REQUIRED` and rejected by
`validate()`, so the study cannot silently run on an invented value.

## What is fixed, and what is not

Held fixed across every run: the agent model, pinned to the dated snapshot
`gpt-4o-mini-2024-07-18`; the decoding temperature at 0.7; the 70/30 stratified
split and its seed; the executor and its whitelist; the fixed part of the
template; the JSON response format; and the two harness caps of eight executed
experiments and four rejected replies. Runs are separate conversations.

The temperature is deliberately not zero. Run-to-run variance is the error term
against which every coefficient is tested, and greedy decoding would drive it to
zero along with every standard error. Each run passes an explicit sampling seed,
so any individual run can be regenerated exactly while replicates still differ.

Nine runs required a replacement seed after producing no valid experiment; they
are identifiable by `seed != planned_seed` in the transcript. All nine fall in
two cells, `M0-B0-S1-O0-E1` and `M0-B0-S1-O1-E0` on Breast Cancer. Every cell
holds exactly twenty completed runs.

## Treatment set

`config/axes.py` holds the ten paragraphs. They were written by a language model
distinct from the agent, from meta-prompts that were **not preserved**: the texts
can be inspected and reused, but not regenerated. Two defects are documented in
that file's docstring and in the thesis — the levels of a slot are not
length-matched, and the level-1 stopping paragraph refers to the evaluation slot,
which leaves it underspecified in the sixteen configurations with `M=0`. Neither
orders the results; both are reported rather than silently corrected, since
editing the strings now would invalidate the 1,280 runs executed against them.
