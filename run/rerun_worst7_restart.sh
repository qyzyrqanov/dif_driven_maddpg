#!/usr/bin/env bash
# Rerun the 7 worst runs (last-200 SR < 50%) from the restart-v3 sweep, with
# the Option-B restart rule:
#
#   restart attempt if  last-100 SR <= 50%  AND  coverage < 95%
#
# - check floor: episode 500 (the "safe 500"), re-checked every 50 eps,
#   running through episode 1000 of each attempt;
# - restart budget: up to 5 restarts; the 6th attempt runs the full 1000
#   episodes with no early restart (safety net);
# - on restart everything is rebuilt (actor/critic/targets/optimizers/buffer/
#   counters) but the SEED IS NOT CHANGED and the run is NOT re-seeded
#   (train_seeded.py skips set_seeds when restart_count>0, so each restarted
#   attempt diverges on the same seed).
#
# RESUME, NEVER WIPE: this launcher does NOT delete run directories. It simply
# launches train_seeded.py on the existing dir; the in-Python logic decides:
#   * a finished-but-bad run (1000 eps, failed the SR/cov rule) is detected by
#     the orbit-restart-on-resume check (maddpg.py) and restarted FROM SCRATCH
#     in-process (start_episode->0, fresh nets/buffer, same seed);
#   * a half-finished rerun attempt resumes from its last checkpoint
#     (training_state.pkl / latest episode_*_training_state.pkl) — no progress
#     is ever lost on relaunch.
# So re-running this script after a Ctrl-C / crash / reboot CONTINUES where it
# left off. The original 1000-ep results live in the Z7S backup until a new
# successful run is offloaded over them.
#
# Per-run completion handling (done by THIS launcher, not in-process):
#   * success (rc==0)              -> offload/mirror to OFFLOAD_ROOT
#   * locked by a live trainer (rc==2) -> skip, leave intact (NOT deleted)
#   * interrupt (rc 130/143)       -> leave intact so it can resume later
#   * genuine failure (other rc!=0)-> log to logs/rerun_failures.log and delete
#     the local run dir (set DELETE_FAILED=0 to keep it)
#
# WARNING: n5_full_seed1's prior 49.5% was a real late bootstrap (recovered
# ~ep664). Under a <50%-from-ep500 rule its finished result is eligible to be
# restarted at the ep500 resume check before it can recover. It is included per
# the Option-B request; drop it from WORST7 below if you want to preserve it.
#
# Usage:
#   CONFIRM=1 bash run/rerun_worst7_restart.sh
# Optional env overrides:
#   PARALLEL=3                 # concurrent runs (RTX 4050 6GB; default 3)
#   ARTIFACT_ROOT=...          # local working root
#   OFFLOAD_ROOT=...           # external mirror (set empty to skip offload)

set -euo pipefail

REPO="/home/abz/workspace/PycharmProjects/dif_driven_maddpg"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/abz/Desktop/dif_driven_revision_offline_replay_restart_v3_artifacts}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-/media/abz/Z7S/experiments_revision_offline_replay_restart_v3}"
PARALLEL="${PARALLEL:-3}"
EPISODES=1000
VANG="pi2"

# Option-B restart parameters
CHECK_EP=500
RESUME_CHECK_EP=500
RESTART_MAX=5
SR_THRESHOLD=0.50
COV_THRESHOLD=0.95

# Delete a run dir when its python exits with a genuine (non-interrupt) error.
DELETE_FAILED="${DELETE_FAILED:-1}"

# 7 worst runs:  "n mode seed"
WORST7=(
  "6 ablation 4"   # was SR 0.0  / cov 32.8 (rc3)
  "5 full     5"   # was SR 22.0 / cov 80.6
  "6 full     3"   # was SR 24.5 / cov 84.3
  "6 ablation 3"   # was SR 29.5 / cov 85.4
  "6 full     5"   # was SR 35.5 / cov 87.5
  "4 nocoll   5"   # was SR 43.0 / cov 85.0 (rc2)
  "5 full     1"   # was SR 49.5 / cov 88.6  (late bootstrap — see WARNING)
)

if [[ "${CONFIRM:-0}" != "1" ]]; then
  echo "This will DELETE and rerun ${#WORST7[@]} run directories under:"
  echo "  $ARTIFACT_ROOT/runs/"
  for spec in "${WORST7[@]}"; do
    read -r n mode seed <<<"$spec"
    echo "  - n${n}_${mode}_seed${seed}"
  done
  echo
  echo "Restart rule: SR <= ${SR_THRESHOLD} AND cov < ${COV_THRESHOLD}, floor ep ${CHECK_EP},"
  echo "max ${RESTART_MAX} restarts, no reseed. Re-run with CONFIRM=1 to proceed."
  exit 1
fi

cd "$REPO"
export PYTHONPATH="$REPO"
# activate venv if present
for v in .venvLin .venv .venv3.10; do
  if [[ -f "$REPO/$v/bin/activate" ]]; then source "$REPO/$v/bin/activate"; break; fi
done

mkdir -p "$ARTIFACT_ROOT/logs"

FAILURES_LOG="$ARTIFACT_ROOT/logs/rerun_failures.log"

# Runs one task to completion (blocking); the caller backgrounds this whole
# function so $PARALLEL of them run concurrently. Handles wipe-once-then-resume,
# then offloads on success / logs+deletes on genuine failure.
launch_one() {
  local n="$1" mode="$2" seed="$3"
  local name="n${n}_${mode}_seed${seed}"
  local out_dir="$ARTIFACT_ROOT/runs/$name"
  local log="$ARTIFACT_ROOT/logs/rerun_${name}.log"
  mkdir -p "$out_dir"
  echo "[$(date +%T)] launching $name (resume / orbit-restart decides) -> $log"

  # Offload is handled here, after the run, NOT in-process.
  python run/train_seeded.py \
    --n "$n" --mode "$mode" --seed "$seed" --episodes "$EPISODES" \
    --v_ang_max "$VANG" --use_offline_replay --use_orbit_restart \
    --orbit_restart_check_ep "$CHECK_EP" \
    --orbit_restart_resume_check_ep "$RESUME_CHECK_EP" \
    --orbit_restart_max "$RESTART_MAX" \
    --orbit_restart_sr_threshold "$SR_THRESHOLD" \
    --orbit_restart_cov_threshold "$COV_THRESHOLD" \
    --out_dir "$out_dir" \
    --artifact_root "$ARTIFACT_ROOT" \
    --disable_episode_offload \
    >"$log" 2>&1
  local rc=$?

  if [[ $rc -eq 0 ]]; then
    echo "[$(date +%T)] DONE $name (rc=0)"
    if [[ -n "$OFFLOAD_ROOT" ]]; then
      echo "[$(date +%T)] offloading $name -> $OFFLOAD_ROOT"
      if python tools/offload_artifacts.py \
          --run_dir "$out_dir" \
          --source_root "$ARTIFACT_ROOT" \
          --target_root "$OFFLOAD_ROOT" \
          --keep_local_result_csv --quiet_progress >>"$log" 2>&1; then
        echo "[$(date +%T)] offload OK $name"
      else
        echo "[$(date +%T)] offload FAILED $name (see $log)"
      fi
    fi
  elif [[ $rc -eq 2 ]]; then
    # train_seeded.py exits 2 when another live trainer holds the run lock.
    # Don't touch the dir — another process owns it.
    echo "[$(date +%T)] SKIPPED $name (rc=2: locked by a live trainer) — left intact"
  elif [[ $rc -eq 130 || $rc -eq 143 ]]; then
    # SIGINT / SIGTERM — interrupted, not a real failure. Keep for resume.
    echo "[$(date +%T)] INTERRUPTED $name (rc=$rc) — kept for resume"
  else
    echo "[$(date +%T)] FAILED $name (rc=$rc) — see $log"
    printf '%s\tFAILED\t%s\trc=%s\tlog=%s\n' \
      "$(date -Iseconds)" "$name" "$rc" "$log" >> "$FAILURES_LOG"
    if [[ "$DELETE_FAILED" == "1" ]]; then
      echo "[$(date +%T)] deleting failed run dir $out_dir"
      rm -rf "$out_dir"
    fi
  fi
}

# Background progress monitor: every PROGRESS_INTERVAL seconds, scan
# ARTIFACT_ROOT/runs/*/meta.json + .run_all.lock and redraw one line per
# in-flight run IN PLACE (no scrolling). Same style as
# run/run_seeds_offline_replay.sh. Disable with PROGRESS=0.
PROGRESS="${PROGRESS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
monitor_pid=""
if [[ "$PROGRESS" = "1" ]]; then
    python3 - "$ARTIFACT_ROOT/runs" "$PROGRESS_INTERVAL" "$RESTART_MAX" <<'PYEOF' &
import json, os, sys, time
runs_root, interval, restart_max = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
# Live in-place block goes to the user's terminal so the master log stays clean.
try:
    tty = open("/dev/tty", "w")
except OSError:
    tty = None
def alive(pid):
    if pid <= 0: return False
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True
last_lines = 0
while True:
    time.sleep(interval)
    rows = []
    try:
        entries = sorted(os.listdir(runs_root))
    except FileNotFoundError:
        continue
    for name in entries:
        d = os.path.join(runs_root, name)
        lock = os.path.join(d, ".run_all.lock")
        if not os.path.isfile(lock):
            continue
        try:
            lp = json.load(open(lock))
            if not alive(int(lp.get("pid") or -1)):
                continue
        except Exception:
            continue
        done = total = "?"
        try:
            m = json.load(open(os.path.join(d, "meta.json")))
            done = m.get("episodes_completed", "?")
            total = m.get("n_games_target") or m.get("episodes_requested") or "?"
        except Exception:
            pass
        restarts = 0
        try:
            rs = json.load(open(os.path.join(d, "restart_state.json")))
            restarts = int(rs.get("restart_count") or 0)
        except Exception:
            pass
        rows.append(f"  … {name}: {done}/{total} restarts={restarts}/{restart_max}")
    if tty is not None:
        # Clear previous block, then redraw header + rows in place.
        if last_lines:
            tty.write(f"\x1b[{last_lines}A\x1b[J")
        header = f"[progress {time.strftime('%H:%M:%S')}] in-flight ({len(rows)}):"
        tty.write(header + "\n")
        for r in rows:
            tty.write(r + "\n")
        tty.flush()
        last_lines = 1 + len(rows)
    else:
        # No TTY available (e.g., nohup): fall back to one log block per tick.
        if rows:
            print(f"[progress {time.strftime('%H:%M:%S')}] in-flight:", flush=True)
            for r in rows:
                print(r, flush=True)
PYEOF
    monitor_pid=$!
fi

# Kill the monitor with us, even on Ctrl-C. Unquoted so an empty pid vanishes.
trap 'kill $monitor_pid 2>/dev/null' EXIT INT TERM

# throttle to $PARALLEL concurrent jobs (exclude the monitor from the count)
for spec in "${WORST7[@]}"; do
  read -r n mode seed <<<"$spec"
  while (( $(jobs -rp | grep -vx -e "${monitor_pid:-x}" | wc -l) >= PARALLEL )); do sleep 2; done
  launch_one "$n" "$mode" "$seed" &
done

# Wait only for training workers (exclude the monitor). Don't let a nonzero
# run exit code trip `set -e`.
for pid in $(jobs -rp | grep -vx -e "${monitor_pid:-x}"); do
  wait "$pid" 2>/dev/null || true
done

if [[ -n "$monitor_pid" ]]; then
  kill "$monitor_pid" 2>/dev/null
  wait "$monitor_pid" 2>/dev/null
fi
echo "[$(date +%T)] all ${#WORST7[@]} reruns finished."
