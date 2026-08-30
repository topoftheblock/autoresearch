"""
Complete 2^5 factorial over the five components of program.md (thesis 4.4, 5.2).

This replaces the eight-run Plackett-Burman fraction in design/doe.py, which is
retained only because thesis 5.9 extracts a PB subset from these 32 cells in
order to audit the error the fraction incurs.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from study_config import AXIS_ORDER  # noqa: E402


def build_designs():
    """All 2^5 = 32 configurations, in a fixed and reproducible order."""
    return [dict(zip(AXIS_ORDER, levels))
            for levels in itertools.product((0, 1), repeat=len(AXIS_ORDER))]


def config_id(config):
    return "-".join(f"{a}{config[a]}" for a in AXIS_ORDER)


def coded(config):
    """+/-1 coding of eq. (1): level 0 -> -1, level 1 -> +1."""
    return {f"X_{a}": (1 if config[a] == 1 else -1) for a in AXIS_ORDER}


if __name__ == "__main__":
    cfgs = build_designs()
    out = Path(__file__).resolve().parent / "configs_full.json"
    out.write_text(json.dumps(cfgs, indent=2))
    print(f"wrote {len(cfgs)} configurations to {out}")
    # balance check: each axis at level 1 in exactly half the cells (thesis 4.2)
    for a in AXIS_ORDER:
        ones = sum(c[a] for c in cfgs)
        print(f"  {a}: level1 in {ones}/{len(cfgs)} cells")
