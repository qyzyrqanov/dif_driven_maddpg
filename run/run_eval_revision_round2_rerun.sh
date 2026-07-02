#!/usr/bin/env bash
# Round-2: rerun ONLY the 3 interrupted deterministic MAPPO eval jobs.
#
# The full pass (run/run_eval_revision_round2.sh) left 3 jobs partial (no summary
# .json): n4 seed1 env25, n6 seed4 env20, n6 seed5 env20. Their partial CSVs are on
# media, so the skip-safe launcher would skip them. This script targets exactly those
# 3, deletes the stale partial csv/_steps/json first, and writes fresh CSVs into the
# SAME eval dir so they join the 27 complete jobs for re-aggregation.
#
# Checkpoints + eval CSVs were archived to media (Z7S) after the first pass, so the
# defaults below point there. Override CKPT_ROOT / OUT_ROOT if your paths differ.
#
# Eval config matches the first pass: pi/2, full reward, eval seed 42, 200 episodes,
# num_obstacles=0. Each job ~20-25 min (200 episodes) → ~1-1.5 h for all 3.
#
# Usage:
#   CONFIRM=1 bash run/run_eval_revision_round2_rerun.sh
# Optional overrides:
#   PARALLEL=3  CKPT_ROOT=...  OUT_ROOT=...  EVAL_SEED=42  EPISODES=200

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
# MAPPO actors (archived to media after pass 1).
CKPT_ROOT="${CKPT_ROOT:-$HOME/dif_driven_archive/dif_driven_logs_mappo/artifacts_raw/runs}"
# Eval CSV dir holding the 27 complete jobs (so reruns join them for aggregation).
OUT_ROOT="${OUT_ROOT:-$HOME/dif_driven_archive/dif_driven_logs_mappo/eval_raw}"
PARALLEL="${PARALLEL:-3}"
EVAL_SEED="${EVAL_SEED:-42}"
EPISODES="${EPISODES:-200}"
VANG="pi2"
MODE="full"

# The 3 interrupted jobs: "n seed env_size"
JOBS=(
  "4 1 25"
  "6 4 20"
  "6 5 20"
)

if [[ "${CONFIRM:-0}" != "1" ]]; then
  echo "Rerun of the 3 interrupted MAPPO eval jobs (eval-only, NO training):"
  for j in "${JOBS[@]}"; do read -r n s e <<<"$j"; echo "  - n$n seed$s env$e"; done
  echo "Reads actors from : $CKPT_ROOT/mappo_n{N}_${MODE}_seed{S}/mappo_actor.pth"
  echo "Writes CSVs to     : $OUT_ROOT/eval/  (stale partials deleted first)"
  echo "Config: pi2, full, eval seed $EVAL_SEED, $EPISODES episodes, PARALLEL=$PARALLEL."
  echo "Each job ~20-25 min. Re-run with CONFIRM=1 to proceed."
  exit 1
fi

cd "$REPO"; export PYTHONPATH="$REPO"
for v in .venvLin .venv .venv3.10; do
  [[ -f "$REPO/$v/bin/activate" ]] && source "$REPO/$v/bin/activate" && break
done
mkdir -p "$OUT_ROOT/eval" "$OUT_ROOT/logs"

throttle() { while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 2; done; }

for j in "${JOBS[@]}"; do
  read -r n s envsz <<<"$j"
  ckpt="$CKPT_ROOT/mappo_n${n}_${MODE}_seed${s}/mappo_actor.pth"
  if [[ ! -f "$ckpt" ]]; then echo "SKIP missing ckpt: $ckpt"; continue; fi
  base="mappo_n${n}_${MODE}_trainseed${s}_env${envsz}_evalseed${EVAL_SEED}"
  out="$OUT_ROOT/eval/${base}.csv"
  # delete stale partial outputs so this is a clean rerun
  rm -f "$out" "$OUT_ROOT/eval/${base}_steps.csv" "$OUT_ROOT/eval/${base}.json"
  throttle
  ( python run/eval_policy.py --actor_ckpt "$ckpt" --algorithm mappo \
      --n "$n" --mode "$MODE" --env_size "$envsz" --v_ang_max "$VANG" \
      --episodes "$EPISODES" --seed "$EVAL_SEED" --out_csv "$out" \
      > "$OUT_ROOT/logs/${base}.log" 2>&1 \
    && echo "done  n$n seed$s env$envsz" \
    || echo "FAIL  n$n seed$s env$envsz (see $OUT_ROOT/logs/${base}.log)" ) &
done

wait
echo "[$(date +%T)] rerun finished. CSVs in $OUT_ROOT/eval/"
echo
echo "Next — re-aggregate (now 30/30) and rebuild the notebook:"
echo "  cd $REPO && export PYTHONPATH=\$PWD && source .venvLin/bin/activate"
echo "  python tools/aggregate_eval.py --eval_dir $OUT_ROOT/eval --out_dir revision_logs_mappo/eval"
echo "  python tools/create_round2_notebook.py"
echo "  jupyter nbconvert --to notebook --execute --inplace res/round2_revision_results.ipynb"
