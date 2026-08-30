"""
Single source of truth for every constant the thesis fixes, and an explicit
register of the ones it does not.

Every value below is traceable to a section of thesis_new.tex. Values the thesis
leaves unspecified are set to REQUIRED and are rejected by validate(); they must
be filled in before a full run, and the reason each one matters is given inline.
Nothing here is invented: if the thesis does not state it, it is REQUIRED.
"""

class _Required:
    def __repr__(self):
        return "<REQUIRED: not specified in thesis_new.tex>"
    def __bool__(self):
        return False

REQUIRED = _Required()


# ---------------------------------------------------------------- treatment --
# thesis 5.2: five binary components, complete 2^5 factorial, all 32 executed.
AXIS_ORDER = ["M", "B", "S", "O", "E"]
N_CONFIGS = 2 ** len(AXIS_ORDER)


# ----------------------------------------------------------------- controls --
# thesis 5.11: agent model pinned to a dated snapshot, not a moving alias.
AGENT_MODEL = "gpt-4o-mini-2024-07-18"   # pinned dated snapshot (thesis 5.11)

# thesis 5.11: temperature held at ONE value in every run, strictly above zero.
# It is NOT varied with the replicate index, and it is NOT zero: the run-to-run
# variance is the error term epsilon against which every coefficient in eq. (2)
# is tested, and greedy decoding would drive it to zero. See thesis 5.11 para 3.
TEMPERATURE = 0.7

# thesis 5.11: an explicit sampling seed is passed to the model interface so any
# run can be regenerated exactly, and the same seed list is used in every cell.
SEND_API_SEED = True

API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
API_KEY_ENV = "OPENAI_API_KEY"

# thesis 5.7: R independent seeds per configuration per dataset.
REPLICATES_R = 20               # MDE = 0.22 sigma per dataset (thesis 5.8)
SEED_LIST = None                # derived from REPLICATES_R by seeds(); same list in every cell

# thesis 5.7 / harness: safety bounds, properties of the harness not the treatment.
HARD_CAP = 8                    # max executor calls per run
MAX_BAD = 4                     # max malformed/rejected replies per run
API_RETRIES = 4

# thesis 5.4: identical split protocol for every dataset and every run.
TEST_SIZE = 0.30
SPLIT_SEED = 0
CV_FOLDS = 5
SCORING = "accuracy"


# ------------------------------------------------------------------- tasks ---
# thesis 5.4: "several canonical tabular classification datasets", chosen to vary
# sample size, dimensionality, number of classes and headroom. The thesis does
# NOT name them, so the suite is REQUIRED. "breast_cancer" is listed because it
# is the task the existing pilot ran, not because the thesis prescribes it.
DATASETS = ["breast_cancer", "wine"]


# ------------------------------------------------------- executor whitelist --
# thesis 5.4: same whitelisted hyperparameters on every task.
ALLOWED_PARAMS = {
    "random_forest": {"n_estimators", "max_depth", "min_samples_leaf", "max_features"},
    "gradient_boosting": {"n_estimators", "max_depth", "learning_rate", "subsample"},
}

# thesis 5.5: search width is "normalised per dataset by the admissible range" of
# the one hyperparameter shared by both families. The range is REQUIRED.
N_ESTIMATORS_RANGE = (10, 500)


# --------------------------------------------------------- reference points --
# thesis 5.5: regret is measured against a*(d), an exhaustive grid search over the
# whitelist, and a0(d), a default-hyperparameter fit. a0 needs no choice (sklearn
# defaults). The GRID defining a* is a modelling choice the thesis does not make,
# so it is REQUIRED.
REFERENCE_GRID = {
    # Spans the whitelist of ALLOWED_PARAMS; n_estimators spans N_ESTIMATORS_RANGE.
    # PROPOSED, not taken from the thesis: a* (and therefore every regret value)
    # depends on this grid, so it needs explicit sign-off before the full run.
    "random_forest": {
        "n_estimators": [10, 100, 250, 500],
        "max_depth": [None, 4, 8, 16],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", 0.5],
    },                                              # 64 points
    "gradient_boosting": {
        "n_estimators": [10, 100, 250, 500],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.03, 0.1, 0.3],
        "subsample": [0.7, 1.0],
    },                                              # 72 points
}


# ------------------------------------------------------- measurement window --
# thesis 5.5 "Measurement window": every metric is computed over the first three
# experiments, three being what the fixed-budget level performs.
WINDOW = 3

# thesis 5.5 / appendix note: two proposals count as a refinement step if they
# share a model family and differ in at most one hyperparameter. Float comparison
# needs a tolerance; the thesis does not fix one.
REFINEMENT_FLOAT_TOL = 1e-9      # exact comparison up to float representation noise


# --------------------------------------------------------------- execution --
# Runs are mutually independent (thesis 5.11), so the outer loop parallelises.
# With more than one worker the estimators are given n_jobs=1: RandomForest with
# n_jobs=-1 already saturates the cores, and nesting the two oversubscribes.
WORKERS = 1
EXECUTOR_N_JOBS = -1

# Scales on which regret is undefined for a given dataset, because a*(d) = a0(d)
# leaves no headroom. thesis 5.5 excludes such a case rather than assigning a
# value. Measured, not assumed: see analysis/reference_constants.json.
#   wine: the 54-row validation split is perfectly classified by defaults, so
#   a0_val = a*_val = 1.0000 and only the cross-validated scale is usable.
REGRET_SCALES_EXCLUDED = {"wine": ["val"]}


# ------------------------------------------------------------------ output ---
RESULTS_DIRNAME = "results_full"
TABLE_BASENAME = "regression_table"


def seeds():
    """The seed list, identical in every cell (thesis 5.11)."""
    if REPLICATES_R is REQUIRED:
        raise ValueError("REPLICATES_R is not set")
    return list(range(REPLICATES_R))


def missing():
    """Names of constants the thesis leaves unspecified."""
    g = globals()
    return [k for k in (
        "AGENT_MODEL", "REPLICATES_R", "DATASETS",
        "N_ESTIMATORS_RANGE", "REFERENCE_GRID", "REFINEMENT_FLOAT_TOL",
    ) if g[k] is REQUIRED]


def validate(need_api=True):
    """Raise unless every thesis-unspecified constant has been supplied."""
    miss = missing()
    if not need_api and "AGENT_MODEL" in miss:
        miss.remove("AGENT_MODEL")
    if miss:
        raise ValueError(
            "The following constants are not specified in thesis_new.tex and must "
            "be set in config/study_config.py before running:\n  - " + "\n  - ".join(miss)
        )
