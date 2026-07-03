# Configuration & paths

All scripts in this repository avoid hardcoded machine-specific paths. The repo
location is auto-detected, and every output/data location is a variable you can
override with an environment variable. This page lists them.

## Repository root

Shell scripts under `run/` and `tools/` auto-detect the repo from their own
location:

```bash
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
```

You normally do not need to set anything. To force a different location, export
`REPO` before calling a script.

For Python, put the repo root on `PYTHONPATH` (absolute imports like
`from rl.maddpg import ...` are used throughout):

```bash
export PYTHONPATH=$(pwd)
```

## Path environment variables

Defaults are portable (`$HOME`-based). Override any of them to point at your own
storage — e.g. a fast local SSD for live runs and an external drive for the
archive.

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `REPO` | `run/*.sh`, `tools/*.sh` | auto-detected | Repository root |
| `ARTIFACT_ROOT` | training / baseline launchers | `$HOME/Desktop/dif_driven_*_artifacts` | Where a run writes checkpoints, buffers, logs |
| `OFFLOAD_ROOT` | offload launchers | `$HOME/dif_driven_archive/...` | Archive/mirror target for finished runs |
| `CKPT_ROOT` | evaluation scripts | see script | Checkpoints to evaluate |
| `OUT_ROOT` | evaluation scripts | `$HOME/Desktop/..._eval` | Evaluation output |
| `REVISION_ROOT` | `res/revision_final_results.ipynb` | repo `revision_logs/` if present | Data root for the results notebook |
| `LEGACY_LOG_ROOT` | `tools/compare_*.py`, `tools/sweep_*.py` | `$HOME/dif_driven_archive` | Legacy reference logs for diagnostics |

Python entry points (`run/run_seeds.py`, `run/run_all.py`, `tools/*.py`) expose
the same locations as `--argument` defaults derived from `Path.home()`; pass the
flag or export the matching variable to change them.

### Examples

```bash
# Train, writing artifacts to a scratch SSD and mirroring to an external drive
export ARTIFACT_ROOT=/mnt/ssd/dif_driven_runs
export OFFLOAD_ROOT=/mnt/external/dif_driven_archive
bash run/run_seeds_offline_replay.sh

# Point the results notebook at a specific data root
REVISION_ROOT=/data/my_runs jupyter nbconvert --to notebook --execute \
    --inplace res/revision_final_results.ipynb
```

## Notebooks

The analysis notebooks under `res/` reference a **data root** (runs / light
logs) that is not shipped with the repository. Set `REVISION_ROOT` (or the
notebook's first-cell variable) to your local copy before executing. Rendered
output cells in the committed notebooks may still show the original author's
local paths — these are frozen results and do not affect re-execution.
