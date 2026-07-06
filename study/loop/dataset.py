"""Fixed dataset shared by every run, so runs differ only in program.md + seed."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

_DATA = load_breast_cancer()

X_train, X_val, y_train, y_val = train_test_split(
    _DATA.data, _DATA.target, test_size=0.3, random_state=0, stratify=_DATA.target
)
