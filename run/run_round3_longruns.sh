#!/usr/bin/env bash
# Round-3 long jobs — SINGLE entry point for everything that runs long
# (Reviewer 1, round 3). Three phases, selectable via PHASE=:
#
#   TRAIN  (#8, #7)  — new training runs (resume-safe, GPU)
#   EVAL   (#13)     — deterministic transfer-eval sweep of OUR trained actors
#                      across a range of arena sizes (also yields #24 collision
#                      rate and #11 path length at each size)
#
# Everything else in round 3 is text or analysis-of-existing-runs handled by
# tools/create_round3_notebook.py (see .ai/CONTEXT/experiments.md §18).
#
# ── PHASE=train ─────────────────────────────────────────────────────────────
#   #8  Ablate HER (offline-replay relabel) and the orbit-restart controller.
#       --use_orbit_restart REQUIRES --use_offline_replay, so the achievable
#       factorial is a 3-rung ladder at full reward (seeds {1..5} × n {4,5,6}):
#         Full       HER ON , restart ON   = published main sweep (revision_logs/);
#                                            NOT re-run, used as the reference.
#         noRestart  HER ON , restart OFF  = --use_offline_replay           (here)
#         noHER      HER OFF, restart OFF  = (no flags -> train_loop)        (here)
#       restart effect = Full-noRestart ; HER effect = noRestart-noHER. 30 runs.
#   #7  MADDPG baseline seed parity: add seeds {4,5} × n {4,5,6} (6 runs),
#       identical config to run/run_maddpg_obs_baseline.sh.
#   => 36 training runs.  Raw -> $ARTIFACT_ROOT/runs/  (resume via training_state.pkl).
#
# ── PHASE=eval ──────────────────────────────────────────────────────────────
#   #13 Wider transfer: evaluate the 45 trained full-method actors (deterministic,
#       noise off) on a ratio curve of arena sizes ENV_SIZES (default 15 18 20 25
#       30 = 0.75–1.5× of the env20 training arena). env15 is the small floor:
#       landmark placement (pairwise sep 5.0 in an env×env box) is 0% infeasible
#       at env15 for n=6, but fails below env14 — do NOT go under env14.
#       Also re-runs the greedy heuristic (oracle) at each size for #11 normalization.
#       Eval needs the trained actors: CKPT_ROOT/n{N}_full_seed{S}/shared_actor.pth
#       (local copy preferred; populate via tools/copy_checkpoints_local.sh, or it
#       falls back to the Z7S mirror). Missing checkpoints are SKIPPED, not fatal.
#       Raw eval CSVs -> $EVAL_OUT_ROOT/eval/.
#
# RESUME / SKIP, NEVER WIPE: training resumes from training_state.pkl (CSV merged
# by episode_id); eval skips any output CSV that already exists. Self-healing:
# a training run that dies is re-invoked up to MAX_RETRIES.
#
# RUN:
#   CONFIRM=1 bash run/run_round3_longruns.sh                 # all phases
#   CONFIRM=1 PHASE=train  bash run/run_round3_longruns.sh    # #8 + #7 only
#   CONFIRM=1 PHASE=ablation bash run/run_round3_longruns.sh  # #8 only
#   CONFIRM=1 PHASE=maddpg bash run/run_round3_longruns.sh    # #7 only
#   CONFIRM=1 PHASE=eval   bash run/run_round3_longruns.sh    # #13 transfer sweep
# Optional overrides:
#   PARALLEL=3  SEEDS="1 2 3 4 5"  NS="4 5 6"  ENV_SIZES="15 18 20 25 30"
#   ARTIFACT_ROOT=...  CKPT_ROOT=...  EVAL_OUT_ROOT=...  EVAL_SEED=42
#
# AFTER the runs, build the report:
#   python tools/export_light_logs.py --artifact_root "$ARTIFACT_ROOT" \
#          --no_media --local_logs revision_logs_round3                 # train logs
#   python tools/aggregate_eval.py --eval_dir "$EVAL_OUT_ROOT/eval" \
#          --out_dir revision_logs_round3/eval                          # transfer eval
#   python tools/create_round3_notebook.py
#   jupyter nbconvert --to notebook --execute --inplace res/round3_experimental_results.ipynb

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PHASE="${PHASE:-all}"               # all | train | ablation | maddpg | eval
PARALLEL="${PARALLEL:-3}"
SEEDS="${SEEDS:-1 2 3 4 5}"
NS="${NS:-4 5 6}"
VANG="pi2"
MODE="full"

# training
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/Desktop/dif_driven_round3_artifacts}"
EPISODES=1000
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_BACKOFF="${RETRY_BACKOFF:-10}"

# eval (#13 transfer)
CKPT_ROOT="${CKPT_ROOT:-$REPO/checkpoints_local/runs}"
[[ -d "$CKPT_ROOT" ]] || CKPT_ROOT="$HOME/dif_driven_archive/experiments_revision_offline_replay_restart_v3/runs"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-$HOME/Desktop/dif_driven_round3_eval}"
ENV_SIZES="${ENV_SIZES:-15 18 20 25 30}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_EPISODES="${EVAL_EPISODES:-200}"

# Resolve which phases run.
case "$PHASE" in
  all)      DO_ABL=1; DO_MAD=1; DO_EVAL=1 ;;
  train)    DO_ABL=1; DO_MAD=1; DO_EVAL=0 ;;
  ablation) DO_ABL=1; DO_MAD=0; DO_EVAL=0 ;;
  maddpg)   DO_ABL=0; DO_MAD=1; DO_EVAL=0 ;;
  eval)     DO_ABL=0; DO_MAD=0; DO_EVAL=1 ;;
  *) echo "Unknown PHASE=$PHASE (use all|train|ablation|maddpg|eval)"; exit 2 ;;
esac

# ── Build the training job list: "name|algorithm|n|seed|extra_flags" ────────
JOBS=()
if (( DO_ABL )); then
  for seed in $SEEDS; do for n in $NS; do
    JOBS+=("abl_noRestart_n${n}_${MODE}_seed${seed}|iddpg_without_s|$n|$seed|--use_offline_replay")
    JOBS+=("abl_noHER_n${n}_${MODE}_seed${seed}|iddpg_without_s|$n|$seed|")
  done; done
fi
if (( DO_MAD )); then
  for seed in 4 5; do for n in $NS; do
    JOBS+=("maddpg_obs_n${n}_${MODE}_seed${seed}|maddpg_obs|$n|$seed|--use_offline_replay")
  done; done
fi

# ── Dry-run preview ──────────────────────────────────────────────────────────
if [[ "${CONFIRM:-0}" != "1" ]]; then
  echo "Round-3 long jobs — PHASE=$PHASE  PARALLEL=$PARALLEL  mode=$MODE  v_ang_max=$VANG"
  if (( ${#JOBS[@]} )); then
    echo "TRAIN: ${#JOBS[@]} runs (episodes=$EPISODES) -> $ARTIFACT_ROOT/runs/"
    for j in "${JOBS[@]}"; do
      IFS='|' read -r name algo n seed flags <<<"$j"
      echo "    - $name   [$algo ${flags:-train_loop(no-HER,no-restart)}]"
    done
  fi
  if (( DO_EVAL )); then
    ne=$(( $(echo $ENV_SIZES | wc -w) * $(echo $NS | wc -w) * $(echo $SEEDS | wc -w) ))
    nh=$(( $(echo $ENV_SIZES | wc -w) * $(echo $NS | wc -w) ))
    echo "EVAL (#13 transfer): policy $ne jobs + heuristic $nh jobs (episodes=$EVAL_EPISODES)"
    echo "    env_sizes={$ENV_SIZES}  n={$NS}  seeds={$SEEDS}  eval_seed=$EVAL_SEED"
    echo "    actors <- $CKPT_ROOT/n{N}_${MODE}_seed{S}/shared_actor.pth"
    echo "    CSVs   -> $EVAL_OUT_ROOT/eval/"
    [[ -d "$CKPT_ROOT" ]] || echo "    NOTE: CKPT_ROOT not present -> eval will SKIP (populate checkpoints first)."
  fi
  echo; echo "Re-run with CONFIRM=1 to proceed."
  exit 1
fi

cd "$REPO"
export PYTHONPATH="$REPO"
for v in .venvLin .venv .venv3.10; do
  if [[ -f "$REPO/$v/bin/activate" ]]; then source "$REPO/$v/bin/activate"; break; fi
done

# ── Training helpers ─────────────────────────────────────────────────────────
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
  local name="$1" algo="$2" n="$3" seed="$4" flags="$5"
  local out_dir="$ARTIFACT_ROOT/runs/$name"
  local log="$ARTIFACT_ROOT/logs/${name}.log"
  local meta="$out_dir/meta.json"
  mkdir -p "$out_dir"
  local attempt=0
  while :; do
    if run_finished "$meta"; then echo "[$(date +%T)] DONE $name (finished)"; return 0; fi
    if (( attempt > MAX_RETRIES )); then
      echo "[$(date +%T)] GAVE UP $name after $MAX_RETRIES resumes (see $log)"; return 1
    fi
    if (( attempt == 0 )); then echo "[$(date +%T)] launching $name -> $log"
    else echo "[$(date +%T)] RESUME $name (attempt $attempt/$MAX_RETRIES) -> $log"; fi
    # $flags is intentionally unquoted so an empty string adds no argument.
    python run/train_seeded.py \
      --algorithm "$algo" \
      --n "$n" --mode "$MODE" --seed "$seed" --episodes "$EPISODES" \
      --v_ang_max "$VANG" $flags \
      --disable_episode_offload \
      --out_dir "$out_dir" \
      --artifact_root "$ARTIFACT_ROOT" \
      >>"$log" 2>&1 || echo "[$(date +%T)] rc=$? $name (will check meta / maybe resume)" >>"$log"
    attempt=$((attempt+1))
    run_finished "$meta" && { echo "[$(date +%T)] DONE $name"; return 0; }
    sleep "$RETRY_BACKOFF"
  done
}

throttle() { while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 2; done; }

# ── Phase 1: training (#8 + #7) ──────────────────────────────────────────────
if (( ${#JOBS[@]} )); then
  mkdir -p "$ARTIFACT_ROOT/logs"
  echo "[$(date +%T)] === TRAIN phase: ${#JOBS[@]} runs ==="
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name algo n seed flags <<<"$j"
    throttle
    launch_one "$name" "$algo" "$n" "$seed" "$flags" &
  done
  wait
  echo "[$(date +%T)] training done. Artifacts under $ARTIFACT_ROOT/runs/"
fi

# ── Phase 2: #13 transfer eval (deterministic) ───────────────────────────────
if (( DO_EVAL )); then
  mkdir -p "$EVAL_OUT_ROOT/eval" "$EVAL_OUT_ROOT/logs"
  echo "[$(date +%T)] === EVAL phase (#13 transfer): env_sizes={$ENV_SIZES} ==="
  # A) policy transfer sweep over the 45 trained full-method actors
  for envsz in $ENV_SIZES; do
    for n in $NS; do
      for s in $SEEDS; do
        ckpt="$CKPT_ROOT/n${n}_${MODE}_seed${s}/shared_actor.pth"
        [[ -f "$ckpt" ]] || { echo "SKIP missing ckpt: $ckpt"; continue; }
        out="$EVAL_OUT_ROOT/eval/policy_n${n}_${MODE}_trainseed${s}_env${envsz%.*}_evalseed${EVAL_SEED}.csv"
        [[ -f "$out" ]] && { echo "exists, skip $(basename "$out")"; continue; }
        throttle
        ( python run/eval_policy.py --actor_ckpt "$ckpt" --n "$n" --mode "$MODE" \
            --env_size "$envsz" --v_ang_max "$VANG" --episodes "$EVAL_EPISODES" \
            --seed "$EVAL_SEED" --out_csv "$out" \
            > "$EVAL_OUT_ROOT/logs/$(basename "${out%.csv}").log" 2>&1 \
          && echo "done policy n$n seed$s env$envsz" \
          || echo "FAIL policy n$n seed$s env$envsz" ) &
      done
    done
  done
  # B) greedy heuristic (oracle) at each size — for #11 path-length normalization
  for envsz in $ENV_SIZES; do
    for n in $NS; do
      out="$EVAL_OUT_ROOT/eval/heuristic_n${n}_${MODE}_env${envsz%.*}_evalseed${EVAL_SEED}.csv"
      [[ -f "$out" ]] && { echo "exists, skip $(basename "$out")"; continue; }
      throttle
      ( python run/eval_hungarian_p.py --n "$n" --mode "$MODE" --env_size "$envsz" \
          --v_ang_max "$VANG" --Kp 2.0 --episodes "$EVAL_EPISODES" \
          --seed "$EVAL_SEED" --out_csv "$out" \
          > "$EVAL_OUT_ROOT/logs/$(basename "${out%.csv}").log" 2>&1 \
        && echo "done heuristic n$n env$envsz" || echo "FAIL heuristic n$n env$envsz" ) &
    done
  done
  wait
  echo "[$(date +%T)] transfer eval done. CSVs in $EVAL_OUT_ROOT/eval/"
fi

echo
echo "[$(date +%T)] ALL requested round-3 phases finished. Next:"
(( ${#JOBS[@]} )) && echo "  python tools/export_light_logs.py --artifact_root $ARTIFACT_ROOT --no_media --local_logs revision_logs_round3"
(( DO_EVAL ))     && echo "  python tools/aggregate_eval.py --eval_dir $EVAL_OUT_ROOT/eval --out_dir revision_logs_round3/eval"
echo "  python tools/create_round3_notebook.py"
echo "  jupyter nbconvert --to notebook --execute --inplace res/round3_experimental_results.ipynb"
