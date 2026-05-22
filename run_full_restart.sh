#!/usr/bin/env bash
# Restart-enabled revision sweep:
#   * Same matrix as run_full.sh (seeds 1..5, n {4,5,6}, modes full/ablation/nocoll)
#   * v_ang_max = pi/2, main_loop / offline_replay (HER) — unchanged
#   * NEW: orbit-basin restart enabled (--use_orbit_restart)
#       * First checks at ep 500 of each attempt, then every 50 eps.
#       * Restart only when last-100 and last-200 strict SR are both <=1%,
#         no recovery signal is present, and the component signature matches:
#         mean comp4>5 AND mean comp8<-800.
#       * Max 3 restart events; 4th attempt runs full episode budget.
#   * NEW artifact + offload paths (..._restart_v3) so prior restart/v2
#     sweeps are preserved.
#
# Usage:
#   bash run_full_restart.sh
#   nohup bash run_full_restart.sh &
#
# Override anything by exporting before invocation:
#   SEEDS="1 2" bash run_full_restart.sh
#   PARALLEL=2 bash run_full_restart.sh

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
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/Desktop/dif_driven_revision_offline_replay_restart_v3_artifacts}"
export OFFLOAD_ROOT="${OFFLOAD_ROOT:-/media/abz/Z7S/experiments_revision_offline_replay_restart_v3}"
export OFFLOAD="${OFFLOAD:-1}"
export OFFLOAD_MODE="${OFFLOAD_MODE:-end}"
export FRESH="${FRESH:-0}"
export LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/logs}"

# Orbit-restart specific
export USE_ORBIT_RESTART="${USE_ORBIT_RESTART:-1}"

MASTER_LOG="$ARTIFACT_ROOT/master_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ARTIFACT_ROOT" "$LOG_DIR"

if [ "${FRESH:-0}" = "1" ]; then
    echo "FRESH=1 -> wiping $ARTIFACT_ROOT/runs/*" | tee -a "$MASTER_LOG"
    rm -rf "$ARTIFACT_ROOT/runs"
    mkdir -p "$ARTIFACT_ROOT/runs"
fi

{
echo "============================================================"
echo "RESTART-ENABLED REVISION SWEEP — $(date -Iseconds)"
echo "------------------------------------------------------------"
echo "  seeds        : $SEEDS"
echo "  Ns           : $NS"
echo "  modes        : $MODES"
echo "  episodes     : $EPISODES (per attempt; up to 2500 with restarts)"
echo "  v_ang_max    : $V_ANG_MAX"
echo "  parallel     : $PARALLEL  (inside each seed)"
echo "  use_offline_replay: yes (main_loop / HER)"
echo "  use_orbit_restart : $USE_ORBIT_RESTART (1=on)"
echo "  restart rule : first attempt-local check at ep 500, then every 50 eps"
echo "  artifact root: $ARTIFACT_ROOT"
echo "  offload      : $OFFLOAD  (1=mirror after each completed run, 0=disabled)"
echo "  offload mode : $OFFLOAD_MODE"
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
if [ ! -d "$OFFLOAD_ROOT" ] && ! mkdir -p "$OFFLOAD_ROOT" 2>/dev/null; then
    echo "WARN: OFFLOAD_ROOT $OFFLOAD_ROOT not writable; runs may skip offload." | tee -a "$MASTER_LOG"
fi

bash "$REPO_ROOT/run/run_seeds_offline_replay.sh" 2>&1 | tee -a "$MASTER_LOG"
exit_code=${PIPESTATUS[0]}

echo "============================================================" | tee -a "$MASTER_LOG"
echo "RESTART SWEEP DONE at $(date -Iseconds), exit_code=$exit_code" | tee -a "$MASTER_LOG"
echo "Local : $ARTIFACT_ROOT/runs/"                                   | tee -a "$MASTER_LOG"
echo "Offload: $OFFLOAD_ROOT/runs/"                                   | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
exit "$exit_code"
