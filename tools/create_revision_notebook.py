"""Create the Phase 3 revision analysis notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "res" / "results_revision.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


def main() -> None:
    notebook = {
        "cells": [
            md(
                """
                # Revision Phase 3 Results

                This notebook aggregates the completed revision batch under
                `/home/abz/Desktop/dif_driven_revision_artifacts`.

                It intentionally starts with a training-completeness audit. Do not use the
                paper-facing summary tables until every intended training run is marked
                `valid_for_revision=True`, or until the manuscript explicitly states why a
                partial checkpoint was used.
                """
            ),
            code(
                """
                from pathlib import Path
                import sys

                import pandas as pd
                from IPython.display import Image, display

                REPO_ROOT = Path.cwd().resolve()
                if REPO_ROOT.name == "res":
                    REPO_ROOT = REPO_ROOT.parent
                if str(REPO_ROOT) not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT))

                from tools.aggregate_revision_results import run_all

                ARTIFACT_ROOT = Path("/home/abz/Desktop/dif_driven_revision_artifacts")
                OUT_DIR = ARTIFACT_ROOT / "res"
                """
            ),
            md(
                """
                ## Regenerate Aggregates

                This cell writes the Phase 3 CSVs and figures into
                `/home/abz/Desktop/dif_driven_revision_artifacts/res`.
                """
            ),
            code(
                """
                outputs = run_all(ARTIFACT_ROOT, OUT_DIR)
                outputs
                """
            ),
            md("## Training Completeness Audit"),
            code(
                """
                audit = pd.read_csv(OUT_DIR / "revision_training_audit.csv")
                display(audit)

                incomplete = audit[~audit["valid_for_revision"]].copy()
                display(incomplete[[
                    "n", "mode", "seed", "episodes_completed", "finished",
                    "launcher_status", "source_note", "run_dir"
                ]])
                """
            ),
            md(
                """
                ## Valid-Training Tables

                These tables exclude generated training runs whose `meta.json` shows fewer
                than 1000 completed episodes. At the current checkpoint, that means several
                `full` seeds are excluded.
                """
            ),
            code(
                """
                for name in [
                    "revision_multiseed_summary_valid.csv",
                    "revision_baseline_comparison_valid.csv",
                    "revision_generalization_valid.csv",
                    "revision_stats_valid.csv",
                ]:
                    print(f"\\n{name}")
                    display(pd.read_csv(OUT_DIR / name))
                """
            ),
            md(
                """
                ## All-Evaluation Tables

                These include every evaluation CSV that exists, including evaluations made
                from partial `full` checkpoints. Use these only for debugging or deciding
                which runs need to be resumed.
                """
            ),
            code(
                """
                for name in [
                    "revision_multiseed_summary_all_evals.csv",
                    "revision_baseline_comparison_all_evals.csv",
                    "revision_generalization_all_evals.csv",
                    "revision_stats_all_evals.csv",
                ]:
                    print(f"\\n{name}")
                    display(pd.read_csv(OUT_DIR / name))
                """
            ),
            md("## Figures"),
            code(
                """
                for name in [
                    "revision_learning_curve_full_valid.png",
                    "revision_rolling_success_full_valid.png",
                    "revision_baseline_success_valid.png",
                    "revision_generalization_success_valid.png",
                ]:
                    path = OUT_DIR / name
                    print(path)
                    if path.exists():
                        display(Image(filename=str(path)))
                """
            ),
            md("## Raw Evaluation Summary"),
            code(
                """
                seed_summary = pd.read_csv(OUT_DIR / "revision_eval_seed_summary.csv")
                display(seed_summary)
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
