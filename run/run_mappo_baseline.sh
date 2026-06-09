#!/usr/bin/env bash
# MAPPO within-environment baseline (Reviewer round-2, point #3).
#
# Modern on-policy CTDE baseline (Yu et al., 2021) on the SAME task / reward /
# representation / budget as the proposed method, so it is a fair within-env
# reference alongside the maddpg_obs baseline. Implemented in rl/mappo.py:
#   * shared Gaussian actor on per-agent obs o_i
#   * centralized value V(concat per-agent obs, index order) -> per-agent values
#     (same head_i<->agent_i binding fix as maddpg_obs)
#   * per-agent reward + per-agent GAE, value normalization (standard MAPPO)
#   * NO HER offline relabel, NO orbit-restart (those are OUR pipeline, not the
#     baseline) — same env / full reward / v_ang_max=pi2 / no obstacles / 1000 ep
#
# Differs from the maddpg_obs launcher only in: --algorithm mappo, NO
# --use_offline_replay (PPO is on-policy and has no replay buffer).
#
# Scope: seeds {1,2,3,4,5} x n in {4,5,6} = 15 runs (matches the 5-seed main
# table; MAPPO-vs-ours on all 5 shared seeds, MAPPO-vs-MADDPG on the 3 MADDPG ran).
#
# RESUME, NEVER WIPE: re-running this script continues unfinished runs from their
# training_state.pkl checkpoint (train_seeded.py auto-resumes; per-step CSV is
# merged by episode_id, not clobbered). Safe across Ctrl-C / crash / reboot.
#
# RUN:
#   CONFIRM=1 bash run/run_mappo_baseline.sh
# Optional overrides:
#   PARALLEL=3  SEEDS="1 2 3 4 5"  NS="4 5 6"  ARTIFACT_ROOT=...
#   (PARALLEL=3 is the conservative default — each PPO update is heavier than a
#    DDPG learn step; bump to 5 if the GPU is comfortable.)

set -euo pipefail

REPO="/home/abz/workspace/PycharmProjects/dif_driven_maddpg"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/abz/Desktop/dif_driven_revision_mappo_artifacts}"
PARALLEL="${PARALLEL:-3}"
SEEDS="${SEEDS:-1 2 3 4 5}"
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
  echo "MAPPO baseline (on-policy CTDE, concat-obs centralized value, NO restart, NO offload)."
  echo "Will run mode=$MODE, seeds={$SEEDS}, n={$NS} -> $nruns runs,"
  echo "  $PARALLEL parallel, into: $ARTIFACT_ROOT/runs/"
  for s in $SEEDS; do for n in $NS; do echo "  - mappo_n${n}_${MODE}_seed${s}"; done; done
  echo
  echo "Re-run with CONFIRM=1 to proceed."
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
  local name="mappo_n${n}_${MODE}_seed${seed}"
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
      --algorithm mappo \
      --n "$n" --mode "$MODE" --seed "$seed" --episodes "$EPISODES" \
      --v_ang_max "$VANG" \
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
for seed in $SEEDS; do
  for n in $NS; do
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 2; done
    launch_one "$n" "$seed" &
  done
done
wait
echo "[$(date +%T)] all MAPPO baseline runs finished. Artifacts under $ARTIFACT_ROOT/runs/"
echo "Next: python tools/export_light_logs.py --artifact_root $ARTIFACT_ROOT --no_media --local_logs revision_logs_mappo"
