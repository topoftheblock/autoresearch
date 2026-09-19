"""
The configuration space for program.md, per Step 2 of the research agenda.

Five two-level axes were chosen from reading through prior program.md variants
(see git history commit 286e998, C/autoresearch_phase2/programs/) and the
elaborated agenda. Each axis is a plausible lever on research *behavior*:

  M  metric        - is the evaluation criterion vague or precise/stability-aware?
  B  breadth       - is the agent told to stay within one model family, or free to
                     compare Random Forest and Gradient Boosting at will?
  S  stopping      - is the budget fixed (3 experiments) or adaptive (stop on
                     diminishing returns, budget 8)?
  O  output_format - terse numeric answers, or fully reasoned prose?
  E  emphasis      - explore breadth first, or exploit/refine the first promising
                     result immediately?

Each axis has exactly two levels, coded 0/1, matching the +/-1 coding of eq. (1)
used over the complete 2^5 factorial in design/full_factorial.py.

Provenance and two known defects of this treatment set (thesis, The Instruction
File as a Configuration). The ten paragraphs were written by a language model
distinct from the agent, from meta-prompts that were NOT preserved: the texts
below can be inspected and reused, but not regenerated. Two properties such a
set would ideally have do not hold, and both are reported in the thesis rather
than corrected here, because correcting them now would invalidate the 1,280
runs already executed against these exact strings:

  1. The two levels of a slot are not length-matched. Token counts, level 0 /
     level 1: M 10/56, B 32/26, S 32/59, O 20/34, E 24/32. Paragraph length
     does not order the results -- total treatment length correlates with the
     wasted trial ratio at r = -0.05 across the 112..213 token range spanned by
     the 32 files -- but the levels are not matched.
  2. Slot S level 1 refers to slot M: "the evaluation metric (defined above)".
     Under M=0, which leaves the criterion to the loop, that phrase names
     nothing definite, so S=1 is underspecified in 16 of the 32 configurations.
     M and S remain orthogonal by construction and their interaction is null
     (-0.004 on the wasted trial ratio, adj. p = 0.54), so this did not distort
     the estimates; its visible consequence is the low S=1 compliance rate.

Anyone regenerating this set should sample several wordings per level, which is
the future work the thesis names, and should control length and cross-slot
reference across that family.
"""

AXES = {
    "M": {
        "name": "metric",
        "levels": {
            0: "unspecified",
            1: "precise",
        },
        "text": {
            0: "Evaluate model quality in whatever way you judge appropriate.",
            1: (
                "Evaluate every model using 5-fold cross-validated accuracy on the "
                "training split, AND report the standard deviation across folds as "
                "a measure of stability. A configuration is only better than another "
                "if its mean accuracy is higher; use the standard deviation only to "
                "break ties or to flag instability."
            ),
        },
    },
    "B": {
        "name": "breadth",
        "levels": {
            0: "narrow",
            1: "broad",
        },
        "text": {
            0: (
                "Stay within a single model family per experiment. Pick either "
                "RandomForestClassifier or GradientBoostingClassifier for the whole "
                "session unless you have exhausted reasonable hyperparameters for it "
                "and explicitly decide to switch."
            ),
            1: (
                "You are free to compare RandomForestClassifier and "
                "GradientBoostingClassifier side by side within the same session, "
                "and to vary multiple hyperparameters of each at once."
            ),
        },
    },
    "S": {
        "name": "stopping",
        "levels": {
            0: "fixed_budget",
            1: "adaptive",
        },
        "text": {
            0: (
                "You have a fixed budget of exactly 3 experiments in total. Run "
                "exactly 3, then stop and report your final recommendation, "
                "regardless of the outcome of experiment 3."
            ),
            1: (
                "You have a budget of up to 8 experiments. After each experiment, "
                "decide whether to continue: stop as soon as an experiment fails to "
                "improve the evaluation metric (defined above) by more than 0.002 "
                "over the best result so far, or when you reach 8 experiments, "
                "whichever comes first."
            ),
        },
    },
    "O": {
        "name": "output_format",
        "levels": {
            0: "terse",
            1: "verbose",
        },
        "text": {
            0: (
                "Keep every response short: state the action you are taking and the "
                "numbers involved, with minimal prose."
            ),
            1: (
                "Explain your reasoning at every step in full sentences: what you "
                "hypothesize, why, what you expect to observe, and how you interpret "
                "the actual result relative to that expectation."
            ),
        },
    },
    "E": {
        "name": "emphasis",
        "levels": {
            0: "explore_first",
            1: "exploit_first",
        },
        "text": {
            0: (
                "Prioritize breadth of search: try several meaningfully different "
                "configurations before you spend more than one experiment refining "
                "any single one of them."
            ),
            1: (
                "Prioritize depth: as soon as one configuration looks promising, "
                "immediately refine it with nearby variations (e.g. slightly "
                "different hyperparameter values) before trying anything unrelated."
            ),
        },
    },
}

AXIS_ORDER = ["M", "B", "S", "O", "E"]
