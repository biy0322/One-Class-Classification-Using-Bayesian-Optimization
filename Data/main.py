"""Entry point for the paper experiment code.

This clean repository copy contains code for synthetic simulations, NSL-KDD,
forest-fire, and CK+ experiments. Dataset files are intentionally not included.
Shared OC-SVM helper code is in the common folder.
Run from the repository root, for example:

    python main.py synthetic-wallclock -- --help
    python main.py overlap -- --help
    python main.py nsl-kdd -- --help
    python main.py nsl-sensitivity -- --help
    python main.py forest-fire-budget -- --help
    python main.py forest-fire-wallclock -- --help
    python main.py ckplus-wallclock -- --help
    python main.py ckplus-budget -- --help

Arguments after `--` are forwarded to the selected experiment script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "synthetic-wallclock": ROOT / "Simulation" / "run_synthetic_wallclock_ocsvm_hpo.py",
    "overlap": ROOT / "Simulation" / "run_overlap_simulation.py",
    "nsl-kdd": ROOT / "NSL-KDD" / "run_nsl_kdd_ocsvm_hpo.py",
    "nsl-sensitivity": ROOT / "NSL-KDD" / "run_nsl_kdd_sensitivity_sweeps.py",
    "forest-fire-budget": ROOT / "ForestFire" / "run_forest_fire_budget_sweep.py",
    "forest-fire-wallclock": ROOT / "ForestFire" / "run_forest_fire_wallclock.py",
    "ckplus-wallclock": ROOT / "CKPLUS" / "run_ckplus_wallclock.py",
    "ckplus-budget": ROOT / "CKPLUS" / "run_ckplus_budget_sweep.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper experiment script.")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected script. Prefix with -- when needed.",
    )
    return parser.parse_args()


def main() -> int:
    parsed = parse_args()
    script = COMMANDS[parsed.command]
    forwarded = parsed.args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return subprocess.call([sys.executable, str(script), *forwarded], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

