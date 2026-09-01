"""Render a program.md file from a configuration vector and a dataset.

The fixed part of the file -- the research question, the executor interface and
the JSON response format -- is byte-identical across all 32 configurations of a
given task (thesis, Sec. "The instruction file as a configuration"). It is
parameterised by the dataset because the study crosses the factorial with the
task suite: a run on Wine must be told it is working on Wine.
"""
import sys
from pathlib import Path

from jinja2 import Template

sys.path.insert(0, str(Path(__file__).parent))
from axes import AXES, AXIS_ORDER

TEMPLATE_PATH = Path(__file__).parent / "program_template.md.jinja"

# One entry per dataset in study_config.DATASETS. The wording differs only in
# the task name and the loader, so the treatment paragraphs remain the only
# thing that varies within a task.
TASKS = {
    "breast_cancer": {
        "title": "Tree ensembles on the breast cancer dataset",
        "task_name": "breast cancer diagnostic",
        "loader": "sklearn.datasets.load_breast_cancer",
    },
    "wine": {
        "title": "Tree ensembles on the wine dataset",
        "task_name": "wine cultivar",
        "loader": "sklearn.datasets.load_wine",
    },
    "digits": {
        "title": "Tree ensembles on the digits dataset",
        "task_name": "handwritten digit",
        "loader": "sklearn.datasets.load_digits",
    },
    "iris": {
        "title": "Tree ensembles on the iris dataset",
        "task_name": "iris species",
        "loader": "sklearn.datasets.load_iris",
    },
}

RESEARCH_QUESTION = (
    "How does the choice of hyperparameters affect the generalization accuracy "
    "of tree-ensemble classifiers (Random Forest and Gradient Boosting) on the "
    "{task_name} dataset? Find a well-performing configuration."
)


def task_fields(dataset: str) -> dict:
    if dataset not in TASKS:
        raise KeyError(f"no program.md wording for dataset {dataset!r}; "
                       f"known: {sorted(TASKS)}")
    t = TASKS[dataset]
    return {
        "task_title": t["title"],
        "research_question": RESEARCH_QUESTION.format(task_name=t["task_name"]),
        "dataset_name": f"{t['loader']}, 70/30 stratified train/validation split",
    }


def render_program(config: dict, out_path: Path, dataset: str) -> str:
    """config: axis code -> level int, e.g. {"M": 0, "B": 1, ...}; dataset: task id."""
    template = Template(TEMPLATE_PATH.read_text())
    fields = task_fields(dataset)
    for axis in AXIS_ORDER:
        fields[f"{AXES[axis]['name']}_text"] = AXES[axis]["text"][config[axis]]
    rendered = template.render(**fields)
    out_path.write_text(rendered)
    return rendered


def config_id(config: dict) -> str:
    return "-".join(f"{axis}{config[axis]}" for axis in AXIS_ORDER)


if __name__ == "__main__":
    demo = {"M": 1, "B": 0, "S": 0, "O": 1, "E": 0}
    for ds in ("breast_cancer", "wine"):
        out = Path(__file__).parent / f"program_{config_id(demo)}__{ds}.md"
        render_program(demo, out, ds)
        print(f"wrote {out}")
