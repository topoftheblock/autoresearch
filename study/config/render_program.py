"""Render a program.md file from a configuration vector (Step 2)."""
import sys
from pathlib import Path

from jinja2 import Template

sys.path.insert(0, str(Path(__file__).parent))
from axes import AXES, AXIS_ORDER

TEMPLATE_PATH = Path(__file__).parent / "program_template.md.jinja"

RESEARCH_QUESTION = (
    "How does the choice of hyperparameters affect the generalization accuracy "
    "of tree-ensemble classifiers (Random Forest and Gradient Boosting) on the "
    "breast cancer diagnostic dataset? Find a well-performing configuration."
)
DATASET_NAME = "sklearn.datasets.load_breast_cancer, 70/30 train/validation split"
TASK_TITLE = "Tree ensembles on the breast cancer dataset"


def render_program(config: dict, out_path: Path) -> str:
    """config: dict mapping axis code -> level int, e.g. {"M": 0, "B": 1, ...}"""
    template = Template(TEMPLATE_PATH.read_text())
    fields = {
        "task_title": TASK_TITLE,
        "research_question": RESEARCH_QUESTION,
        "dataset_name": DATASET_NAME,
    }
    for axis in AXIS_ORDER:
        level = config[axis]
        fields[f"{AXES[axis]['name']}_text"] = AXES[axis]["text"][level]
    rendered = template.render(**fields)
    out_path.write_text(rendered)
    return rendered


def config_id(config: dict) -> str:
    return "-".join(f"{axis}{config[axis]}" for axis in AXIS_ORDER)


if __name__ == "__main__":
    demo = {"M": 1, "B": 0, "S": 0, "O": 1, "E": 0}
    out = Path(__file__).parent / f"program_{config_id(demo)}.md"
    render_program(demo, out)
    print(f"wrote {out}")
