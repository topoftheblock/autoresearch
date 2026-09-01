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
# thesis, The instruction file as a configuration: five binary components, complete 2^5 factorial, all 32 executed.
AXIS_ORDER = ["M", "B", "S", "O", "E"]
N_CONFIGS = 2 ** len(AXIS_ORDER)


# ----------------------------------------------------------------- controls --
# thesis, Experimental setup: agent model pinned to a dated snapshot, not a moving alias.
AGENT_MODEL = "gpt-4o-mini-2024-07-18"   # pinned dated snapshot (thesis, Experimental setup)

# thesis, Experimental setup: temperature held at ONE value in every run, strictly above zero.
# It is NOT varied with the replicate index, and it is NOT zero: the run-to-run
# variance is the error term epsilon against which every coefficient in eq. (2)
# is tested, and greedy decoding would drive it to zero. See thesis, Experimental setup para 3.
TEMPERATURE = 0.7

# thesis, Experimental setup: an explicit sampling seed is passed to the model interface so any
# run can be regenerated exactly, and the same seed list is used in every cell.
SEND_API_SEED = True

API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
API_KEY_ENV = "OPENAI_API_KEY"

# thesis, Experimental setup: R independent seeds per configuration per dataset.
REPLICATES_R = 20               # MDE = 0.22 sigma per dataset (thesis, Experimental setup)
SEED_LIST = None                # derived from REPLICATES_R by seeds(); same list in every cell

# thesis, Experimental setup / harness: safety bounds, properties of the harness not the treatment.
HARD_CAP = 8                    # max executor calls per run
MAX_BAD = 4                     # max malformed/rejected replies per run
API_RETRIES = 4                 # transient server faults (500/502/503, timeouts)

# A 429 under a tokens-per-minute cap is self-clearing: the bucket refills on a
# fixed window. Blind exponential backoff (1+2+4+8 = 15s) gives up before a 60s
# window resets, so rate limits get a separate, patient budget and the retry
# delay is taken from the server's own Retry-After / x-ratelimit-reset headers.
RATE_LIMIT_RETRIES = 12
MAX_BACKOFF_S = 90

# thesis, Experimental setup: a run that fails to produce a valid experiment is repeated
# with a fresh seed until the cell holds exactly R completed runs. This bounds
# how many replacements one cell may draw before the study stops rather than
# quietly running short and breaking balance.
CELL_RETRIES = 3

# thesis, Experimental setup: identical split protocol for every dataset and every run.
TEST_SIZE = 0.30
SPLIT_SEED = 0
CV_FOLDS = 5
SCORING = "accuracy"


# ------------------------------------------------------------------- tasks ---
# thesis, Experimental setup names the suite exactly: Breast Cancer Wisconsin (569 x 30,
# two classes) and Wine (178 x 13, three classes), picked to differ in size,
# dimensionality and class count. D = len(DATASETS) enters N = 2^5 * D * R, and
# the dataset is a fixed effect in eq. (2), so this list is also the reference
# level ordering: the first entry is the omitted category.
DATASETS = ["breast_cancer", "wine"]


# ------------------------------------------------------- executor whitelist --
# thesis, Experimental setup: same whitelisted hyperparameters on every task.
ALLOWED_PARAMS = {
    "random_forest": {"n_estimators", "max_depth", "min_samples_leaf", "max_features"},
    "gradient_boosting": {"n_estimators", "max_depth", "learning_rate", "subsample"},
}

# ------------------------------------------------------------- baseline ------
# thesis, Experimental setup: one reference quantity per dataset, a0(d), the accuracy of a
# default-hyperparameter fit taken as the better of the two model families.
# No exhaustive grid search is needed by the current metric set.

# Scale on which the gain over default is computed.
GAIN_SCALE = "cv"               # "cv" or "val"

# Scales on which the gain is undefined for a dataset, because a0(d) is already
# at the ceiling and no run can improve on it (thesis, Experimental setup). Measured, not assumed:
#   wine: the 54-instance validation split is classified perfectly by defaults,
#   so a0_val = 1.0 and only the cross-validated scale is usable there.
GAIN_SCALES_EXCLUDED = {"wine": ["val"]}

# Second scale, reported alongside the first as the robustness check of thesis
# Sec. "Performance metrics": the evaluation-criterion component at level 1 names
# cross-validated accuracy, so an effect of M on the cv-scale gain is partly the
# treatment agreeing with the instrument. An effect is claimed for M only where
# both scales agree. Undefined wherever GAIN_SCALES_EXCLUDED excludes it.
GAIN_SCALE_ROBUSTNESS = "val"


# ------------------------------------------------------- measurement window --
# thesis, Performance metrics "Measurement window": every metric is computed over the first three
# experiments, three being what the fixed-budget level performs.
WINDOW = 3

# --------------------------------------------------------------- execution --
# Runs are mutually independent (thesis, Experimental setup), so the outer loop parallelises.
# With more than one worker the estimators are given n_jobs=1: RandomForest with
# n_jobs=-1 already saturates the cores, and nesting the two oversubscribes.
WORKERS = 1
EXECUTOR_N_JOBS = -1


# ------------------------------------------------------------------ output ---
RESULTS_DIRNAME = "results_full"
TABLE_BASENAME = "regression_table"


def seeds():
    """The seed list, identical in every cell (thesis, Experimental setup)."""
    if REPLICATES_R is REQUIRED:
        raise ValueError("REPLICATES_R is not set")
    return list(range(REPLICATES_R))


def missing():
    """Names of constants the thesis leaves unspecified."""
    g = globals()
    return [k for k in (
        "AGENT_MODEL", "REPLICATES_R", "DATASETS",
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
