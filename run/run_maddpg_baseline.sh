#!/usr/bin/env bash
# MADDPG within-environment baseline (Reviewer 1 #2 / #17).
#
# Lowe-2017-style joint-action CENTRALIZED-critic MADDPG (MADDPGSharedActorCritic),
# run through the SAME pipeline as the main method EXCEPT the orbit-restart
# controller (which is part of our training pipeline, not the baseline):
#
#   * --algorithm maddpg          (centralized-critic baseline)
#   * --use_offline_replay        (HER-style goal relabeling; same as main method)
#   * tagged replay buffer        (on by default in MADDPGBase)
#   * NO --use_orbit_restart      (single attempt per seed)
#   * --disable_episode_offload   (no USB offload; local only)
#   * full reward, v_ang_max=pi2, 1000 episodes
#   * seeds {1,2,3} x n in {4,5,6} = 9 runs   (same seeds as the main table,
#     so the comparison is on matched seeds)
#
# Up to PARALLEL runs at once. Resume-safe: re-running continues unfinished runs
# from their checkpoints. Each run writes to its own dir under ARTIFACT_ROOT/runs.
#
# PILOT FIRST (recommended): run one seed at n=4 to confirm MADDPG trains here
# and to measure wall-clock, before launching all 9:
#
#   cd /home/abz/workspace/PycharmProjects/dif_driven_maddpg
#   source .venvLin/bin/activate && export PYTHONPATH=$(pwd)
#   python run/train_seeded.py --algorithm maddpg --n 4 --mode full --seed 1 \
#     --episodes 1000 --v_ang_max pi2 --use_offline_replay \
#     --disable_episode_offload \
#     --out_dir /home/abz/Desktop/dif_driven_revision_maddpg_artifacts/runs/maddpg_n4_full_seed1
#
# FULL RUN:
#   CONFIRM=1 bash run/run_maddpg_baseline.sh
# Optional overrides:
#   PARALLEL=5  SEEDS="1 2 3"  NS="4 5 6"  ARTIFACT_ROOT=...

set -euo pipefail

REPO="/home/abz/workspace/PycharmProjects/dif_driven_maddpg"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/abz/Desktop/dif_driven_revision_maddpg_artifacts}"
PARALLEL="${PARALLEL:-5}"
SEEDS="${SEEDS:-1 2 3}"
NS="${NS:-4 5 6}"
EPISODES=1000
VANG="pi2"
MODE="full"

if [[ "${CONFIRM:-0}" != "1" ]]; then
  nruns=$(( $(echo $SEEDS | wc -w) * $(echo $NS | wc -w) ))
  echo "MADDPG baseline (centralized critic, offline-replay, NO restart, NO offload)."
  echo "Will run mode=$MODE, seeds={$SEEDS}, n={$NS} -> $nruns runs,"
  echo "  $PARALLEL parallel, into: $ARTIFACT_ROOT/runs/"
  for n in $NS; do for s in $SEEDS; do echo "  - maddpg_n${n}_${MODE}_seed${s}"; done; done
  echo
  echo "Re-run with CONFIRM=1 to proceed. (Run the n=4 seed1 pilot first — see header.)"
  exit 1
fi

cd "$REPO"
export PYTHONPATH="$REPO"
for v in .venvLin .venv .venv3.10; do
  if [[ -f "$REPO/$v/bin/activate" ]]; then source "$REPO/$v/bin/activate"; break; fi
done
mkdir -p "$ARTIFACT_ROOT/logs"

launch_one() {
  local n="$1" seed="$2"
  local name="maddpg_n${n}_${MODE}_seed${seed}"
  local out_dir="$ARTIFACT_ROOT/runs/$name"
  local log="$ARTIFACT_ROOT/logs/${name}.log"
  mkdir -p "$out_dir"
  echo "[$(date +%T)] launching $name -> $log"
  python run/train_seeded.py \
    --algorithm maddpg \
    --n "$n" --mode "$MODE" --seed "$seed" --episodes "$EPISODES" \
    --v_ang_max "$VANG" --use_offline_replay \
    --disable_episode_offload \
    --out_dir "$out_dir" \
    --artifact_root "$ARTIFACT_ROOT" \
    >"$log" 2>&1 \
    && echo "[$(date +%T)] DONE $name" \
    || echo "[$(date +%T)] EXIT rc=$? $name (see $log)"
}

for n in $NS; do
  for seed in $SEEDS; do
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 2; done
    launch_one "$n" "$seed" &
  done
done
wait
echo "[$(date +%T)] all MADDPG baseline runs finished. Artifacts under $ARTIFACT_ROOT/runs/"
echo "Next: python tools/export_light_logs.py --artifact_root $ARTIFACT_ROOT --no_media  # distil for the notebook"
