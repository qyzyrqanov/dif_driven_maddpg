#!/usr/bin/env bash
# Round-2 long run: DETERMINISTIC eval of the trained MAPPO baseline checkpoints.
# (Training is already finished — this only evaluates saved actors. Eval-only.)
#
# WHY: the round-2 MAPPO head-to-head is reported on the training-window protocol
# (same as MADDPG, §13) and needs NO eval. This pass is the OPTIONAL strengthener:
# a deterministic (noise-free) eval of MAPPO at env_size=20 AND 25, giving MAPPO
# the same treatment as our method's §15 generalization table — so MAPPO can sit
# beside ours in the deterministic SR / coverage / path-length comparison and the
# "limited transfer" (point 5c) discussion. Skip it if you only want the
# training-window head-to-head.
#
# Mirrors run/run_eval_revision.sh (pass A) but loads GaussianActor checkpoints via
# eval_policy.py --algorithm mappo (deterministic action = distribution mean).
#
# Eval config (matched to §15): pi/2, full reward, eval seed 42, 200 episodes,
# num_obstacles=0. 15 checkpoints x {env20, env25} = 30 eval jobs. Models are tiny.
#
# RESUME/skip-safe: existing output CSVs are skipped, so re-running continues.
#
# Usage:
#   CONFIRM=1 bash run/run_eval_revision_round2.sh
# Optional overrides:
#   PARALLEL=3  CKPT_ROOT=...  OUT_ROOT=...  ENV_SIZES="20 25"  EVAL_SEED=42
#   SEEDS="1 2 3 4 5"  NS="4 5 6"

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
# MAPPO training artifacts (contain mappo_actor.pth per run).
CKPT_ROOT="${CKPT_ROOT:-$HOME/Desktop/dif_driven_revision_mappo_artifacts/runs}"
OUT_ROOT="${OUT_ROOT:-$HOME/Desktop/dif_driven_revision_mappo_eval}"
PARALLEL="${PARALLEL:-3}"
ENV_SIZES="${ENV_SIZES:-20 25}"
EVAL_SEED="${EVAL_SEED:-42}"
EPISODES="${EPISODES:-200}"
SEEDS="${SEEDS:-1 2 3 4 5}"
NS="${NS:-4 5 6}"
VANG="pi2"
MODE="full"

if [[ "${CONFIRM:-0}" != "1" ]]; then
  njobs=$(( $(echo $ENV_SIZES | wc -w) * $(echo $NS | wc -w) * $(echo $SEEDS | wc -w) ))
  echo "OPTIONAL deterministic MAPPO eval (eval-only, NO training)."
  echo "Reads MAPPO actors from: $CKPT_ROOT/mappo_n{N}_${MODE}_seed{S}/mappo_actor.pth"
  echo "Writes per-episode CSVs to: $OUT_ROOT/eval/"
  echo "  env_size in {$ENV_SIZES} x n{$NS} x seeds{$SEEDS} = $njobs jobs,"
  echo "  pi2, full reward, eval seed $EVAL_SEED, $EPISODES episodes, PARALLEL=$PARALLEL."
  echo "Re-run with CONFIRM=1 to proceed."
  exit 1
fi

cd "$REPO"; export PYTHONPATH="$REPO"
for v in .venvLin .venv .venv3.10; do
  [[ -f "$REPO/$v/bin/activate" ]] && source "$REPO/$v/bin/activate" && break
done
mkdir -p "$OUT_ROOT/eval" "$OUT_ROOT/logs"

throttle() { while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 2; done; }

for envsz in $ENV_SIZES; do
  for n in $NS; do
    for s in $SEEDS; do
      ckpt="$CKPT_ROOT/mappo_n${n}_${MODE}_seed${s}/mappo_actor.pth"
      [[ -f "$ckpt" ]] || { echo "SKIP missing ckpt: $ckpt"; continue; }
      out="$OUT_ROOT/eval/mappo_n${n}_${MODE}_trainseed${s}_env${envsz%.*}_evalseed${EVAL_SEED}.csv"
      [[ -f "$out" ]] && { echo "exists, skip $(basename "$out")"; continue; }
      throttle
      ( python run/eval_policy.py --actor_ckpt "$ckpt" --algorithm mappo \
          --n "$n" --mode "$MODE" --env_size "$envsz" --v_ang_max "$VANG" \
          --episodes "$EPISODES" --seed "$EVAL_SEED" --out_csv "$out" \
          > "$OUT_ROOT/logs/$(basename "${out%.csv}").log" 2>&1 \
        && echo "done mappo n$n seed$s env$envsz" \
        || echo "FAIL mappo n$n seed$s env$envsz" ) &
    done
  done
done

wait
echo "[$(date +%T)] MAPPO eval finished. CSVs in $OUT_ROOT/eval/"
echo "Next: aggregate (extend tools/aggregate_eval.py with a mappo_ regex) and add to the notebook."
