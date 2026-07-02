#!/usr/bin/env bash
# One-button launcher for the full revision sweep:
#   * seeds 1..5, sequentially (seed K finishes fully before K+1 starts)
#   * n ∈ {4, 5, 6}, mode ∈ {full, ablation, nocoll} parallel inside a seed
#   * v_ang_max = pi/2  (confirmed Z7S value)
#   * offline_replay enabled (main_loop, HER-style)
#   * walls unbounded (already commented out in env.py)
#   * offload mirror per-checkpoint, same as before
#
# Usage:
#   bash run_full.sh            # foreground, ~30h
#   nohup bash run_full.sh &    # background, survives terminal close
#
# Override anything by exporting before invocation, e.g.:
#   PARALLEL=2 bash run_full.sh
#   SEEDS="1 2" bash run_full.sh
#   ARTIFACT_ROOT=/tmp/test bash run_full.sh

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# --- Defaults (overridable via env) ---
export PARALLEL="${PARALLEL:-5}"
export SEEDS="${SEEDS:-1 2 3 4 5}"
export NS="${NS:-4 5 6}"
export MODES="${MODES:-full ablation nocoll}"
export EPISODES="${EPISODES:-1000}"
export V_ANG_MAX="${V_ANG_MAX:-pi2}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/Desktop/dif_driven_revision_offline_replay_artifacts_v2}"
export OFFLOAD_ROOT="${OFFLOAD_ROOT:-$HOME/dif_driven_archive/experiments_revision_offline_replay_v2}"
# Per-episode offload mirror. Set OFFLOAD=0 to disable (e.g., when the
# external drive is unmounted, or to skip the post-episode sync delay).
export OFFLOAD="${OFFLOAD:-1}"
# Set FRESH=1 to wipe ARTIFACT_ROOT/runs before launch (preserves logs/master_*.log).
export FRESH="${FRESH:-0}"
export LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/logs}"

MASTER_LOG="$ARTIFACT_ROOT/master_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ARTIFACT_ROOT" "$LOG_DIR"

# Optional fresh-start wipe of run directories (NOT logs).
if [ "${FRESH:-0}" = "1" ]; then
    echo "FRESH=1 -> wiping $ARTIFACT_ROOT/runs/*" | tee -a "$MASTER_LOG"
    rm -rf "$ARTIFACT_ROOT/runs"
    mkdir -p "$ARTIFACT_ROOT/runs"
fi

# --- Sanity banner ---
{
echo "============================================================"
echo "FULL REVISION SWEEP — $(date -Iseconds)"
echo "------------------------------------------------------------"
echo "  seeds        : $SEEDS"
echo "  Ns           : $NS"
echo "  modes        : $MODES"
echo "  episodes     : $EPISODES"
echo "  v_ang_max    : $V_ANG_MAX"
echo "  parallel     : $PARALLEL  (inside each seed)"
echo "  use_offline_replay: yes (main_loop / HER)"
echo "  artifact root: $ARTIFACT_ROOT"
echo "  offload      : $OFFLOAD  (1=mirror per-episode, 0=disabled)"
echo "  offload root : $OFFLOAD_ROOT"
echo "  per-task logs: $LOG_DIR"
echo "  master log   : $MASTER_LOG"
echo "============================================================"
} | tee -a "$MASTER_LOG"

# --- Pre-flight checks ---
if [ ! -f "$REPO_ROOT/.venvLin/bin/activate" ]; then
    echo "ERROR: .venvLin not found at $REPO_ROOT/.venvLin" | tee -a "$MASTER_LOG"
    exit 1
fi
if ! command -v python >/dev/null 2>&1; then
    : # python check happens after venv activation
fi
if [ ! -d "$OFFLOAD_ROOT" ] && ! mkdir -p "$OFFLOAD_ROOT" 2>/dev/null; then
    echo "WARN: OFFLOAD_ROOT $OFFLOAD_ROOT not writable; runs may skip offload." | tee -a "$MASTER_LOG"
fi

# --- Launch the per-seed driver ---
bash "$REPO_ROOT/run/run_seeds_offline_replay.sh" 2>&1 | tee -a "$MASTER_LOG"
exit_code=${PIPESTATUS[0]}

echo "============================================================" | tee -a "$MASTER_LOG"
echo "FULL SWEEP DONE at $(date -Iseconds), exit_code=$exit_code"   | tee -a "$MASTER_LOG"
echo "Local : $ARTIFACT_ROOT/runs/"                                  | tee -a "$MASTER_LOG"
echo "Offload: $OFFLOAD_ROOT/runs/"                                  | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
exit "$exit_code"
