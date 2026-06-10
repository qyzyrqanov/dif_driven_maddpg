#!/usr/bin/env bash
# Round-3 long jobs with LEAN-LOCAL + OFFLOAD-LANE scheduling.
#
# Same idea as the original run/run_seeds_offline_replay.sh: run PARALLEL=3
# training jobs on the fast LOCAL disk, plus ONE independent "+1" offload
# worker that mirrors each finished run to the external drive (OFFLOAD_ROOT)
# and prunes the heavy local files. Training never blocks on USB I/O; the
# local disk stays lean instead of filling up with replay-buffer snapshots
# (the failure that interrupted the first round-3 launch — disk hit 99%).
#
# Jobs (#8 ablation + #7 MADDPG), identical config to run/run_round3_longruns.sh:
#   abl_noRestart_*  iddpg_without_s --use_offline_replay   (HER on, restart off)
#   abl_noHER_*      iddpg_without_s (train_loop)           (HER off, restart off)
#   maddpg_obs_*     maddpg_obs --use_offline_replay        (seeds 4,5 only)
#
# RESUME / SKIP: a run is "done" if its meta.json says finished EITHER in the
# local ARTIFACT_ROOT OR in OFFLOAD_ROOT (the external drive). So the 20 runs
# already offloaded to Z7S are skipped automatically — only the missing seeds
# (and any partial run copied back local for resume) actually train.
#
# RUN (human launches — provides commands only, per project policy):
#   CONFIRM=1 SEEDS="3 4 5" bash run/run_round3_offload.sh           # #8 + #7
#   CONFIRM=1 SEEDS="3 4 5" PHASE=ablation bash run/run_round3_offload.sh
#   CONFIRM=1 PHASE=maddpg  bash run/run_round3_offload.sh           # seeds 4,5
# Overrides:
#   PARALLEL=3  SEEDS="3 4 5"  NS="4 5 6"  OFFLOAD=1
#   ARTIFACT_ROOT=<local lean>   OFFLOAD_ROOT=<external archive>
#
# To RESUME a partial run that currently lives only on OFFLOAD_ROOT, copy its
# dir back to ARTIFACT_ROOT/runs/ first (see the printed pre-flight hint).
#
# AFTER the runs, build the report off the COMPLETE set on the external drive:
#   python tools/export_light_logs.py --artifact_root "$OFFLOAD_ROOT" \
#          --no_media --local_logs revision_logs_round3
#   python tools/create_round3_notebook.py
#   jupyter nbconvert --to notebook --execute --inplace res/round3_experimental_results.ipynb

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PHASE="${PHASE:-train}"               # train | ablation | maddpg
PARALLEL="${PARALLEL:-3}"
SEEDS="${SEEDS:-3 4 5}"
NS="${NS:-4 5 6}"
VANG="pi2"
MODE="full"
EPISODES="${EPISODES:-1000}"
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_BACKOFF="${RETRY_BACKOFF:-10}"

# Lean local scratch (fast SSD) vs. external archive (the +1 worker mirrors here).
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/abz/Desktop/dif_driven_round3_artifacts}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-/media/abz/Z7S/dif_driven_round3_artifacts}"
OFFLOAD="${OFFLOAD:-1}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/logs}"
OFFLOAD_QUEUE="${OFFLOAD_QUEUE:-$ARTIFACT_ROOT/.offload_queue}"

case "$PHASE" in
  train)    DO_ABL=1; DO_MAD=1 ;;
  ablation) DO_ABL=1; DO_MAD=0 ;;
  maddpg)   DO_ABL=0; DO_MAD=1 ;;
  *) echo "Unknown PHASE=$PHASE (use train|ablation|maddpg)"; exit 2 ;;
esac

# ── Build the job list: "name|algorithm|n|seed|extra_flags" ──────────────────
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
  echo "Round-3 OFFLOAD launcher — PHASE=$PHASE  PARALLEL=$PARALLEL (+1 offload)  seeds={$SEEDS}"
  echo "  local lean   : $ARTIFACT_ROOT"
  echo "  archive (USB): $OFFLOAD_ROOT  (offload=$OFFLOAD)"
  echo "  TRAIN: ${#JOBS[@]} job slots (skipped automatically if finished here OR on the archive):"
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name algo n seed flags <<<"$j"
    echo "    - $name   [$algo ${flags:-train_loop(no-HER,no-restart)}]"
  done
  echo; echo "Re-run with CONFIRM=1 to proceed. (Human launches long runs.)"
  exit 1
fi

cd "$REPO"
export PYTHONPATH="$REPO"
for v in .venvLin .venv .venv3.10; do
  if [[ -f "$REPO/$v/bin/activate" ]]; then source "$REPO/$v/bin/activate"; break; fi
done

mkdir -p "$ARTIFACT_ROOT/runs" "$LOG_DIR" "$OFFLOAD_QUEUE"
rm -f "$OFFLOAD_QUEUE/.done"
[[ -d "$OFFLOAD_ROOT" ]] || mkdir -p "$OFFLOAD_ROOT/runs" 2>/dev/null \
  || echo "WARN: OFFLOAD_ROOT $OFFLOAD_ROOT not writable; runs will not offload." >&2

# A run is finished if EITHER the local OR the archived meta.json says so.
run_finished() {
  local name="$1"
  python - "$ARTIFACT_ROOT/runs/$name/meta.json" "$OFFLOAD_ROOT/runs/$name/meta.json" "$EPISODES" <<'PY'
import json, sys
target = int(sys.argv[3])
for p in (sys.argv[1], sys.argv[2]):
    try:
        m = json.load(open(p))
    except Exception:
        continue
    if bool(m.get("finished")) or int(m.get("episodes_completed") or 0) >= target:
        sys.exit(0)
sys.exit(1)
PY
}

launch_one() {
  local name="$1" algo="$2" n="$3" seed="$4" flags="$5"
  local out_dir="$ARTIFACT_ROOT/runs/$name"
  local log="$LOG_DIR/${name}.log"
  mkdir -p "$out_dir"
  local attempt=0
  while :; do
    if run_finished "$name"; then echo "[$(date +%T)] DONE $name (finished local/archive)"; break; fi
    if (( attempt > MAX_RETRIES )); then
      echo "[$(date +%T)] GAVE UP $name after $MAX_RETRIES resumes (see $log)"; return 1
    fi
    if (( attempt == 0 )); then echo "[$(date +%T)] launching $name -> $log"
    else echo "[$(date +%T)] RESUME $name (attempt $attempt/$MAX_RETRIES)"; fi
    # $flags unquoted so an empty string adds no argument.
    python run/train_seeded.py \
      --algorithm "$algo" \
      --n "$n" --mode "$MODE" --seed "$seed" --episodes "$EPISODES" \
      --v_ang_max "$VANG" $flags \
      --disable_episode_offload \
      --out_dir "$out_dir" \
      --artifact_root "$ARTIFACT_ROOT" \
      >>"$log" 2>&1 || echo "[$(date +%T)] rc=$? $name (will check meta / maybe resume)" >>"$log"
    attempt=$((attempt+1))
    run_finished "$name" && { echo "[$(date +%T)] DONE $name"; break; }
    sleep "$RETRY_BACKOFF"
  done
  # Enqueue the finished local run for the independent offload worker.
  if [[ "$OFFLOAD" == "1" && -f "$out_dir/meta.json" ]]; then
    printf '%s\n' "$out_dir" > "$OFFLOAD_QUEUE/$(date +%s%N)_${name}.job"
    echo "[$(date +%T)] ENQUEUED offload $name"
  fi
}

throttle() { while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 2; done; }

# ── The independent "+1" offload worker ──────────────────────────────────────
# Drains OFFLOAD_QUEUE one job at a time (so at most ONE offload runs at once),
# mirrors each finished run to OFFLOAD_ROOT and prunes the heavy local files.
worker_pid=""
if [[ "$OFFLOAD" == "1" ]]; then
  (
    while true; do
      job="$(ls -1 "$OFFLOAD_QUEUE"/*.job 2>/dev/null | sort | head -n1)"
      if [[ -z "$job" ]]; then
        if [[ -f "$OFFLOAD_QUEUE/.done" ]]; then
          job="$(ls -1 "$OFFLOAD_QUEUE"/*.job 2>/dev/null | sort | head -n1)"
          [[ -z "$job" ]] && break
        else
          sleep 5; continue
        fi
      fi
      run_dir="$(cat "$job" 2>/dev/null)"
      if [[ -n "$run_dir" && -d "$run_dir" ]]; then
        echo "[offload] mirroring $run_dir -> $OFFLOAD_ROOT"
        if python "$REPO/tools/offload_artifacts.py" \
              --run_dir "$run_dir" \
              --source_root "$ARTIFACT_ROOT" \
              --target_root "$OFFLOAD_ROOT" \
              --keep_local_result_csv --include_running --quiet_progress; then
          echo "[offload] done $(basename "$run_dir")"
        else
          echo "[offload] FAILED $(basename "$run_dir") (rerun launcher to retry)"
        fi
      fi
      rm -f "$job"
    done
    echo "[offload] worker exiting (queue drained)"
  ) &
  worker_pid=$!
fi
trap 'kill $worker_pid 2>/dev/null' EXIT INT TERM

# ── Training pool: up to PARALLEL concurrent jobs ────────────────────────────
echo "[$(date +%T)] === ${#JOBS[@]} job slots, PARALLEL=$PARALLEL (+1 offload), seeds={$SEEDS} ==="
for j in "${JOBS[@]}"; do
  IFS='|' read -r name algo n seed flags <<<"$j"
  throttle
  launch_one "$name" "$algo" "$n" "$seed" "$flags" &
done
wait

# Signal the offload worker to finish its final drain, then wait for it.
if [[ "$OFFLOAD" == "1" ]]; then
  : > "$OFFLOAD_QUEUE/.done"
  [[ -n "$worker_pid" ]] && wait "$worker_pid" 2>/dev/null
fi

echo
echo "[$(date +%T)] ALL round-3 offload jobs finished. Build the report off the archive:"
echo "  python tools/export_light_logs.py --artifact_root $OFFLOAD_ROOT --no_media --local_logs revision_logs_round3"
echo "  python tools/create_round3_notebook.py"
echo "  jupyter nbconvert --to notebook --execute --inplace res/round3_experimental_results.ipynb"
