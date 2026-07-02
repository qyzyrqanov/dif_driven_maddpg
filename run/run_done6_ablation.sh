#!/usr/bin/env bash
# Repo root is auto-detected from this script's location; override by exporting REPO.
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO/.venv/bin/activate"
python "$REPO/run/train_done6_ablation.py"
