# Launcher: activate the virtual environment and run the training script.
# Artifacts (checkpoints, replay buffers, plots, logs, CSVs) are written to the
# CURRENT working directory, so cd into your desired output folder first.

# Resolve the project root from this script's own location (run/ -> repo root).
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Put the project root on PYTHONPATH (the code uses absolute package imports).
$env:PYTHONPATH = $RepoRoot

# Activate the virtual environment. Override its location with $env:VENV_DIR.
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { Join-Path $RepoRoot ".venv3.10" }
& (Join-Path $VenvDir "Scripts\Activate.ps1")

# Run the training script (writes its artifacts into the current directory).
python (Join-Path $PSScriptRoot "train_adj4.py")
