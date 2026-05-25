#!/usr/bin/env bash
# Go/no-go check for the MADDPG-Obs baseline. Reads each run's rewards.csv and
# reports episodes done, last-window success rate + coverage, best single-episode
# coverage, and whether any full-coverage episode has occurred yet.
#
# Interpretation (after ~150-300 episodes):
#   GO   : last-window coverage clearly above the ~5-9% chance floor (rising),
#          ideally everSR>0 == True for at least some runs.
#   NO-GO: still flat near chance (<~12%) and everSR>0 == False everywhere.
#
# Usage: bash run/check_maddpg_obs.sh            # default artifact root
#        ARTIFACT_ROOT=... WINDOW=100 bash run/check_maddpg_obs.sh

set -euo pipefail
REPO="/home/abz/workspace/PycharmProjects/dif_driven_maddpg"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/abz/Desktop/dif_driven_revision_maddpg_obs_artifacts}"
WINDOW="${WINDOW:-100}"
PY="$REPO/.venvLin/bin/python"
[[ -x "$PY" ]] || PY="python"

"$PY" - "$ARTIFACT_ROOT/runs" "$WINDOW" <<'PY'
import sys, glob, os
import pandas as pd
root, W = sys.argv[1], int(sys.argv[2])
runs = sorted(glob.glob(os.path.join(root, "maddpg_obs_*")))
if not runs:
    print(f"No runs found under {root}"); sys.exit(0)
print(f"{'run':28} {'eps':>5} {'lastSR%':>8} {'lastCov%':>9} {'bestCov%':>9} {'everSR>0':>8}")
go_signals = 0
for r in runs:
    n = int(os.path.basename(r).split('_')[2][1:])
    f = os.path.join(r, "rewards.csv")
    if not os.path.exists(f):
        print(f"{os.path.basename(r):28} (no rewards.csv yet)"); continue
    df = pd.read_csv(f, usecols=["episode_id", "done_count"])
    g = df.groupby("episode_id")["done_count"].max()
    eps = len(g); last = g.tail(W)
    last_sr = 100*(last >= n).mean()
    last_cov = 100*(last/n).mean()
    best_cov = 100*(g.max()/n)
    ever = bool((g >= n).any())
    if last_cov > 12 or ever:
        go_signals += 1
    print(f"{os.path.basename(r):28} {eps:5d} {last_sr:8.1f} {last_cov:9.1f} {best_cov:9.0f} {str(ever):>8}")
print()
print(f"GO signals: {go_signals}/{len(runs)} runs show coverage>12% or a full-coverage episode.")
print("GO if signals are appearing and coverage is trending up; NO-GO if all flat near chance.")
PY
