#!/usr/bin/env bash
# Eval-only revision passes (NO training). Safe to run alongside the MADDPG
# baseline: models are tiny (~0.07 GB each), so the 6 GB GPU has ample room;
# it only shares compute, won't OOM. Uses a small PARALLEL by default.
#
# Two passes (both eval-only, π/2, full reward, eval seed 42, 200 episodes):
#   A) GENERALIZATION (R2#15): policy eval at env_size=25 on all 45 trained
#      actor checkpoints (read from the Z7S mirror), per n×mode×seed.
#      Also re-evaluates env_size=20 (in-distribution reference).
#   B) HEURISTIC baseline (R1#2 / non-learning reference): greedy Hungarian +
#      proportional controller (Kp=2.0) at π/2, per n. Oracle upper bound.
#
# Results -> $OUT_ROOT/eval/*.csv  (one per-episode CSV per eval).
#
# Usage:
#   CONFIRM=1 bash run/run_eval_revision.sh
# Optional overrides:
#   PARALLEL=3   CKPT_ROOT=...   OUT_ROOT=...   ENV_SIZES="20 25"   EVAL_SEED=42

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
# Trained actor checkpoints: local copy (gitignored) so eval does NOT need the
# Z7S drive mounted. Populate/refresh with:
#   tools/copy_checkpoints_local.sh   (copies shared_actor.pth from the Z7S mirror)
# Falls back to the Z7S mirror if the local copy is absent.
CKPT_ROOT="${CKPT_ROOT:-$REPO/checkpoints_local/runs}"
if [[ ! -d "$CKPT_ROOT" ]]; then
  CKPT_ROOT="$HOME/dif_driven_archive/experiments_revision_offline_replay_restart_v3/runs"
fi
OUT_ROOT="${OUT_ROOT:-$HOME/Desktop/dif_driven_revision_eval}"
PARALLEL="${PARALLEL:-3}"
ENV_SIZES="${ENV_SIZES:-20 25}"
EVAL_SEED="${EVAL_SEED:-42}"
EPISODES=200
VANG="pi2"
MODE="full"          # generalization is reported on the full-method actors
SEEDS="1 2 3 4 5"
NS="4 5 6"

if [[ "${CONFIRM:-0}" != "1" ]]; then
  echo "Eval-only passes (NO training). Reads actors from:"
  echo "  $CKPT_ROOT"
  echo "Writes per-episode CSVs to: $OUT_ROOT/eval/"
  echo "  A) policy eval env_size in {$ENV_SIZES} x n{$NS} x mode=$MODE x seeds{$SEEDS}"
  echo "  B) heuristic baseline (pi2, Kp=2.0) x n{$NS}, eval seed $EVAL_SEED"
  echo "PARALLEL=$PARALLEL. Re-run with CONFIRM=1 to proceed."
  exit 1
fi

cd "$REPO"; export PYTHONPATH="$REPO"
for v in .venvLin .venv .venv3.10; do
  [[ -f "$REPO/$v/bin/activate" ]] && source "$REPO/$v/bin/activate" && break
done
mkdir -p "$OUT_ROOT/eval" "$OUT_ROOT/logs"

throttle() { while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 2; done; }

# --- A) policy generalization eval over the 45 trained actors ---
for envsz in $ENV_SIZES; do
  for n in $NS; do
    for s in $SEEDS; do
      ckpt="$CKPT_ROOT/n${n}_${MODE}_seed${s}/shared_actor.pth"
      [[ -f "$ckpt" ]] || { echo "SKIP missing ckpt: $ckpt"; continue; }
      out="$OUT_ROOT/eval/policy_n${n}_${MODE}_trainseed${s}_env${envsz%.*}_evalseed${EVAL_SEED}.csv"
      [[ -f "$out" ]] && { echo "exists, skip $(basename "$out")"; continue; }
      throttle
      ( python run/eval_policy.py --actor_ckpt "$ckpt" --n "$n" --mode "$MODE" \
          --env_size "$envsz" --v_ang_max "$VANG" --episodes "$EPISODES" \
          --seed "$EVAL_SEED" --out_csv "$out" \
          > "$OUT_ROOT/logs/$(basename "${out%.csv}").log" 2>&1 \
        && echo "done policy n$n seed$s env$envsz" \
        || echo "FAIL policy n$n seed$s env$envsz" ) &
    done
  done
done

# --- B) heuristic non-learning baseline at pi2 ---
for n in $NS; do
  out="$OUT_ROOT/eval/heuristic_n${n}_${MODE}_env20_evalseed${EVAL_SEED}.csv"
  [[ -f "$out" ]] && { echo "exists, skip $(basename "$out")"; continue; }
  throttle
  ( python run/eval_hungarian_p.py --n "$n" --mode "$MODE" --env_size 20 \
      --v_ang_max "$VANG" --Kp 2.0 --episodes "$EPISODES" \
      --seed "$EVAL_SEED" --out_csv "$out" \
      > "$OUT_ROOT/logs/$(basename "${out%.csv}").log" 2>&1 \
    && echo "done heuristic n$n" || echo "FAIL heuristic n$n" ) &
done

wait
echo "[$(date +%T)] all eval passes finished. CSVs in $OUT_ROOT/eval/"
echo "Per-CSV success rate (done_count==n, last column note): inspect with pandas, or"
echo "add a generalization/heuristic loader cell to the notebook."
