#!/usr/bin/env bash
# MADDPG-Obs within-environment baseline (Reviewer 1 #2 / #17).
#
# CTDE centralized-critic MADDPG using MADDPGSharedActorCriticIndependentObs:
# per-agent reward target + per-agent differentiable actor update (fixes A/B),
# and the centralized critic conditions on the CONCATENATION OF PER-AGENT
# OBSERVATIONS (index order) instead of the distance-sorted global state — so
# critic-head_i <-> agent_i <-> action-slot_i are all consistently bound
# (fixes C, the seed-lottery cause). See rl/maddpg.py for the A/B/C writeup.
#
# Same pipeline as the main method EXCEPT the orbit-restart controller:
#   * --algorithm maddpg_obs       (fixed centralized-critic baseline)
#   * --use_offline_replay         (HER-style relabeling; same as main method)
#   * tagged replay buffer         (on by default in MADDPGBase)
#   * NO --use_orbit_restart       (single attempt per seed)
#   * --disable_episode_offload    (no USB offload; local only)
#   * full reward, v_ang_max=pi2, 1000 episodes
#   * seeds {1,2,3} x n in {4,5,6} = 9 runs (matched seeds vs the main table)
#
# Writes to a FRESH artifact root (does NOT touch the old broken `maddpg` runs).
# Resume-safe: re-running continues unfinished runs from their checkpoints.
#
# GO/NO-GO (no separate pilot needed — the first episodes of these real runs
# ARE the pilot). After ~150-300 episodes run:
#     bash run/check_maddpg_obs.sh
# GO  : last-window coverage clearly above the ~5-9% chance floor, ideally a
#       first SR>0  -> leave it; it trains to 1000 episodes.
# NO-GO: still flat at chance by ~ep 300 -> stop and rethink the approach.
#
# RUN:
#   CONFIRM=1 bash run/run_maddpg_obs_baseline.sh
# Optional overrides:
#   PARALLEL=5  SEEDS="1 2 3"  NS="4 5 6"  ARTIFACT_ROOT=...

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/Desktop/dif_driven_revision_maddpg_obs_artifacts}"
PARALLEL="${PARALLEL:-5}"
SEEDS="${SEEDS:-1 2 3}"
NS="${NS:-4 5 6}"
EPISODES=1000
VANG="pi2"
MODE="full"
# Self-heal: if a run dies before finishing (OOM/crash/kill), re-invoke it; it
# resumes from training_state.pkl. 0 = run once (no auto-resume).
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_BACKOFF="${RETRY_BACKOFF:-10}"

if [[ "${CONFIRM:-0}" != "1" ]]; then
  nruns=$(( $(echo $SEEDS | wc -w) * $(echo $NS | wc -w) ))
  echo "MADDPG-Obs baseline (concat-obs centralized critic, offline-replay, NO restart, NO offload)."
  echo "Will run mode=$MODE, seeds={$SEEDS}, n={$NS} -> $nruns runs,"
  echo "  $PARALLEL parallel, into: $ARTIFACT_ROOT/runs/"
  for s in $SEEDS; do for n in $NS; do echo "  - maddpg_obs_n${n}_${MODE}_seed${s}"; done; done
  echo
  echo "Re-run with CONFIRM=1 to proceed."
  echo "Then after ~150-300 episodes: bash run/check_maddpg_obs.sh"
  exit 1
fi

cd "$REPO"
export PYTHONPATH="$REPO"
for v in .venvLin .venv .venv3.10; do
  if [[ -f "$REPO/$v/bin/activate" ]]; then source "$REPO/$v/bin/activate"; break; fi
done
mkdir -p "$ARTIFACT_ROOT/logs"

# True once meta.json reports the run finished (or already at the episode target).
run_finished() {
  local meta="$1"
  [[ -f "$meta" ]] || return 1
  python - "$meta" "$EPISODES" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
done = bool(m.get("finished")) or int(m.get("episodes_completed") or 0) >= int(sys.argv[2])
sys.exit(0 if done else 1)
PY
}

launch_one() {
  local n="$1" seed="$2"
  local name="maddpg_obs_n${n}_${MODE}_seed${seed}"
  local out_dir="$ARTIFACT_ROOT/runs/$name"
  local log="$ARTIFACT_ROOT/logs/${name}.log"
  local meta="$out_dir/meta.json"
  mkdir -p "$out_dir"
  local attempt=0
  while :; do
    if run_finished "$meta"; then
      echo "[$(date +%T)] DONE $name (finished)"; return 0
    fi
    if (( attempt > MAX_RETRIES )); then
      echo "[$(date +%T)] GAVE UP $name after $MAX_RETRIES resumes (see $log)"; return 1
    fi
    if (( attempt == 0 )); then
      echo "[$(date +%T)] launching $name -> $log"
    else
      echo "[$(date +%T)] RESUME $name (attempt $attempt/$MAX_RETRIES) -> $log"
    fi
    # append to the log so resume history is preserved
    python run/train_seeded.py \
      --algorithm maddpg_obs \
      --n "$n" --mode "$MODE" --seed "$seed" --episodes "$EPISODES" \
      --v_ang_max "$VANG" --use_offline_replay \
      --disable_episode_offload \
      --out_dir "$out_dir" \
      --artifact_root "$ARTIFACT_ROOT" \
      >>"$log" 2>&1 || echo "[$(date +%T)] rc=$? $name (will check meta / maybe resume)" >>"$log"
    attempt=$((attempt+1))
    run_finished "$meta" && { echo "[$(date +%T)] DONE $name"; return 0; }
    sleep "$RETRY_BACKOFF"
  done
}

# Dispatch seed-major, n-minor: seed1{n4,n5,n6}, seed2{n4,n5,n6}, ...
# so every seed gets one n quickly (a full seed-1 sweep lands first).
for seed in $SEEDS; do
  for n in $NS; do
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 2; done
    launch_one "$n" "$seed" &
  done
done
wait
echo "[$(date +%T)] all MADDPG-Obs baseline runs finished. Artifacts under $ARTIFACT_ROOT/runs/"
echo "Next: python tools/export_light_logs.py --artifact_root $ARTIFACT_ROOT --no_media --local_logs revision_logs_maddpg_obs"
