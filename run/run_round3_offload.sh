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
# RESUME / SKIP — every job lands in exactly one of three states, logged as
# "run i/N <verb> <name> at episode E/EPISODES":
#   * DONE    — meta.json says finished (episodes_completed >= EPISODES) in
#               EITHER local ARTIFACT_ROOT OR OFFLOAD_ROOT. Skipped immediately.
#   * RESUME  — a usable checkpoint (replay_buffer.pkl + a *_training_state.pkl)
#               exists locally and/or on the archive. If the archive copy is
#               further along than local — including the common case where the
#               lean-local dir was pruned and the only checkpoint lives on the
#               external drive — its resume files are auto-restored to local
#               first, then train_seeded.py resumes from that episode. (This was
#               previously a manual `cp -r` step; doing it automatically is what
#               keeps partial runs from being orphaned after a reboot/interrupt.)
#   * FRESH   — no checkpoint anywhere → start at episode 0. A stale meta.json
#               (episodes_completed>0 with no checkpoint to back it) is cleared
#               so train_seeded.py's no-restart-from-scratch guard does not trip.
#
# RUN (human launches — provides commands only, per project policy):
#   CONFIRM=1 SEEDS="3 4 5" bash run/run_round3_offload.sh           # #8 + #7
#   CONFIRM=1 SEEDS="3 4 5" PHASE=ablation bash run/run_round3_offload.sh
#   CONFIRM=1 PHASE=maddpg  bash run/run_round3_offload.sh           # seeds 4,5
# Overrides:
#   PARALLEL=3  SEEDS="3 4 5"  NS="4 5 6"  OFFLOAD=1
#   ARTIFACT_ROOT=<local lean>   OFFLOAD_ROOT=<external archive>
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
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/Desktop/dif_driven_round3_artifacts}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-$HOME/dif_driven_archive/dif_driven_round3_artifacts}"
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

# Decide how a NOT-finished run should start. Prints "<action> <episode>" where
# action ∈ {archive,local,clear,fresh} and <episode> is where train_seeded.py
# will resume (0 = fresh). "archive" => the external copy is further along than
# local and should be restored; "clear" => meta claims progress but no usable
# checkpoint exists anywhere (a stale stub) so meta must be wiped.
resume_plan() {
  local name="$1"
  python - "$ARTIFACT_ROOT/runs/$name" "$OFFLOAD_ROOT/runs/$name" <<'PY'
import json, os, re, sys
def meta_completed(d):
    try:
        return int(json.load(open(os.path.join(d, "meta.json"))).get("episodes_completed") or 0)
    except Exception:
        return 0
def resumable_ep(d):
    # train_seeded.py needs the main replay buffer to resume at all.
    if not os.path.exists(os.path.join(d, "replay_buffer.pkl")):
        return -1
    # Mirror train_seeded.py's resume precedence EXACTLY: the bare
    # training_state.pkl wins; only if it is absent does it fall back to the
    # latest episode_*_training_state.pkl. (A dir may also hold stale numbered
    # snapshots from an earlier abandoned attempt — those must NOT be read as
    # progress when a newer bare training_state.pkl is present, or we'd restore
    # a stale checkpoint over a good run.)
    if os.path.exists(os.path.join(d, "training_state.pkl")):
        return meta_completed(d)                  # episode of the live checkpoint
    best = -1
    try:
        for nm in os.listdir(d):
            m = re.match(r"^episode_(\d+)_training_state\.pkl$", nm)
            if m:
                best = max(best, int(m.group(1)))
    except FileNotFoundError:
        pass
    return best
local, arch = sys.argv[1], sys.argv[2]
le, ae = resumable_ep(local), resumable_ep(arch)
mc = max(meta_completed(local), meta_completed(arch))
if ae > le and ae >= 0:
    print(f"archive {ae}")
elif le >= 0:
    print(f"local {le}")
elif mc > 0:
    print("clear 0")
else:
    print("fresh 0")
PY
}

# Per-episode replay snapshots and per-episode model dumps are huge and not
# needed to resume (train_seeded.py reloads bare replay_buffer.pkl + *.pth);
# excluding them keeps the local disk lean during a restore.
RESTORE_EXCLUDES=( --exclude='replay_buffer_*.pkl' --exclude='episode_*__*.pth' --exclude='*.png' )

# Make the local out_dir resumable and echo the start episode (0 = fresh).
# Diagnostics go to stderr so the command-substitution caller captures only the
# episode number on stdout.
prepare_resume() {
  local name="$1" out_dir="$ARTIFACT_ROOT/runs/$name" arch_dir="$OFFLOAD_ROOT/runs/$name"
  local plan action ep
  plan="$(resume_plan "$name")"; action="${plan%% *}"; ep="${plan##* }"
  case "$action" in
    archive)
      echo "[$(date +%T)] RESTORE $name: copying checkpoint (ep $ep) from archive -> local" >&2
      mkdir -p "$out_dir"
      rsync -a "${RESTORE_EXCLUDES[@]}" "$arch_dir/" "$out_dir/" >/dev/null 2>&1 \
        || echo "[$(date +%T)] WARN restore rsync failed for $name (will start fresh)" >&2
      ;;
    clear)
      echo "[$(date +%T)] STALE-META $name: meta claims progress but no checkpoint anywhere; clearing meta to start fresh" >&2
      rm -f "$out_dir/meta.json"
      ;;
  esac
  echo "$ep"
}

launch_one() {
  local name="$1" algo="$2" n="$3" seed="$4" flags="$5" idx="$6" total="$7"
  local out_dir="$ARTIFACT_ROOT/runs/$name"
  local log="$LOG_DIR/${name}.log"
  mkdir -p "$out_dir"

  # DONE → skip without touching anything.
  if run_finished "$name"; then
    echo "[$(date +%T)] job $idx/$total DONE     $name (already finished local/archive — skipping)"
    return 0
  fi

  # RESUME (restoring from archive if needed) or FRESH.
  local start_ep verb
  start_ep="$(prepare_resume "$name")"
  if (( start_ep > 0 )); then verb="resuming"; else verb="starting"; fi
  echo "[$(date +%T)] job $idx/$total $verb $name at episode $start_ep/$EPISODES  [$algo ${flags:-train_loop}] -> $log"

  local attempt=0
  while :; do
    if run_finished "$name"; then echo "[$(date +%T)] job $idx/$total DONE     $name"; break; fi
    if (( attempt > MAX_RETRIES )); then
      echo "[$(date +%T)] job $idx/$total GAVE UP  $name after $MAX_RETRIES resumes (see $log)"; return 1
    fi
    (( attempt > 0 )) && echo "[$(date +%T)] job $idx/$total RESUME   $name (attempt $attempt/$MAX_RETRIES)"
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
    run_finished "$name" && { echo "[$(date +%T)] job $idx/$total DONE     $name"; break; }
    sleep "$RETRY_BACKOFF"
  done
  # Enqueue the finished local run for the independent offload worker.
  if [[ "$OFFLOAD" == "1" && -f "$out_dir/meta.json" ]]; then
    printf '%s\n' "$out_dir" > "$OFFLOAD_QUEUE/$(date +%s%N)_${name}.job"
    echo "[$(date +%T)] job $idx/$total ENQUEUED offload $name"
  fi
}

# PARALLEL = number of concurrent TRAIN jobs. The +1 offload worker is a 4th
# background job, so the total-job ceiling is PARALLEL+1 when OFFLOAD=1 (e.g.
# 3 train + 1 offload = 4) and PARALLEL when offload is off — either way exactly
# PARALLEL trainers run, never PARALLEL-1.
throttle() {
  local cap=$(( OFFLOAD == 1 ? PARALLEL + 1 : PARALLEL ))
  while (( $(jobs -rp | wc -l) >= cap )); do sleep 2; done
}

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
  ) >>"$LOG_DIR/offload.log" 2>&1 &
  worker_pid=$!
fi
trap 'kill $worker_pid 2>/dev/null' EXIT INT TERM

# ── Training pool: up to PARALLEL concurrent jobs ────────────────────────────
TOTAL=${#JOBS[@]}
idx=0
# Pre-pass for an honest counter: how many jobs are already finished (local or
# archive) vs. still to run. The per-job "job X/$TOTAL" index below is a POSITION
# in the list, not a progress count — this summary is the real progress.
finished_count=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r jn _ <<<"$j"
  run_finished "$jn" && finished_count=$((finished_count+1))
done
torun=$(( TOTAL - finished_count ))
echo "[$(date +%T)] === $TOTAL jobs: $finished_count finished, $torun to run | PARALLEL=$PARALLEL (+1 offload), seeds={$SEEDS} ==="
echo "[$(date +%T)]     (offload mirror logs -> $LOG_DIR/offload.log; per-run training logs -> $LOG_DIR/<name>.log)"
for j in "${JOBS[@]}"; do
  IFS='|' read -r name algo n seed flags <<<"$j"
  idx=$((idx+1))
  throttle
  launch_one "$name" "$algo" "$n" "$seed" "$flags" "$idx" "$TOTAL" &
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
