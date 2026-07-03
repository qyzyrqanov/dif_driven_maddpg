#!/usr/bin/env bash
# Sequential per-seed runner:
#   * seeds {1,2,3,4,5} processed one at a time (seed K must finish entirely
#     before seed K+1 starts), so on-disk artifacts grow seed-by-seed in order
#   * inside one seed, the (n, mode) tasks run in parallel up to PARALLEL
#   * uses main_loop (HER-style offline_replay_success)
#   * v_ang_max = pi/2 (confirmed Z7S value)
#   * env walls remain unbounded (already commented out in env.py)
#   * by default, each run is mirrored to OFFLOAD_ROOT once, after completion
#
# Usage:
#   bash run/run_seeds_offline_replay.sh                # default seeds 1..5
#   PARALLEL=3 bash run/run_seeds_offline_replay.sh     # override parallelism
#   SEEDS="7 8" bash run/run_seeds_offline_replay.sh    # override seed list
#
# Environment overrides:
#   PARALLEL       (default 3)      max concurrent (n,mode) jobs inside a seed
#   SEEDS          (default "1 2 3 4 5")
#   MODES          (default "full ablation nocoll")
#   NS             (default "4 5 6")
#   EPISODES       (default 1000)
#   ARTIFACT_ROOT  (default ~/Desktop/dif_driven_revision_offline_replay_artifacts)
#   OFFLOAD_ROOT   (default $HOME/dif_driven_archive/experiments_revision_offline_replay)
#   V_ANG_MAX      (default pi2)
#   LOG_DIR        (default $ARTIFACT_ROOT/logs)
#   OFFLOAD_MODE   (default end)     end | every | every_k
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the canonical Linux venv
source "$REPO_ROOT/.venvLin/bin/activate"
export PYTHONPATH="$REPO_ROOT"

PARALLEL="${PARALLEL:-3}"   # training tasks in parallel; offload runs as an independent +1 worker
SEEDS="${SEEDS:-1 2 3 4 5}"
MODES="${MODES:-full ablation nocoll}"
NS="${NS:-4 5 6}"
EPISODES="${EPISODES:-1000}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/Desktop/dif_driven_revision_offline_replay_artifacts}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-$HOME/dif_driven_archive/experiments_revision_offline_replay}"
V_ANG_MAX="${V_ANG_MAX:-pi2}"
OFFLOAD="${OFFLOAD:-1}"  # 1 = mirror to OFFLOAD_ROOT, 0 = disable
OFFLOAD_MODE="${OFFLOAD_MODE:-end}"   # end | every | every_k — when to mirror to USB
OFFLOAD_EVERY="${OFFLOAD_EVERY:-10}"  # only used when OFFLOAD_MODE=every_k
LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/logs}"
# FIFO queue consumed by a single independent offload worker (the +1 slot).
# Completed runs are enqueued here; training never blocks on USB offload.
OFFLOAD_QUEUE="${OFFLOAD_QUEUE:-$ARTIFACT_ROOT/.offload_queue}"

mkdir -p "$ARTIFACT_ROOT/runs" "$LOG_DIR" "$OFFLOAD_QUEUE"
rm -f "$OFFLOAD_QUEUE/.done"   # clear stale shutdown flag from a prior run

# Verify offload root exists / is writable (best effort warning, not fatal)
if [ ! -d "$OFFLOAD_ROOT" ]; then
    if ! mkdir -p "$OFFLOAD_ROOT" 2>/dev/null; then
        echo "WARN: OFFLOAD_ROOT $OFFLOAD_ROOT not writable; runs will skip offload." >&2
    fi
fi

echo "============================================================"
echo "Sequential per-seed run (offline_replay enabled)"
echo "  seeds       : $SEEDS"
echo "  Ns          : $NS"
echo "  modes       : $MODES"
echo "  episodes    : $EPISODES"
echo "  v_ang_max   : $V_ANG_MAX"
echo "  parallel    : $PARALLEL  (inside each seed)"
echo "  artifact    : $ARTIFACT_ROOT"
echo "  offload     : $OFFLOAD_ROOT"
echo "  offload mode: $OFFLOAD_MODE"
echo "  log dir     : $LOG_DIR"
echo "============================================================"

run_one() {
    local seed="$1" n="$2" mode="$3"
    local out_dir="$ARTIFACT_ROOT/runs/n${n}_${mode}_seed${seed}"
    local log_file="$LOG_DIR/seed${seed}_n${n}_${mode}.log"
    mkdir -p "$out_dir"

    # Skip-if-complete: read episodes_completed from meta.json and bail
    # before spawning Python. With orbit restart enabled, completed runs that
    # have no recovery signal are allowed through so Python can spend restart
    # attempt 2/3 on a failed full run.
    if [ -f "$out_dir/meta.json" ]; then
        local complete_action
        complete_action=$(python3 -c "import json, os, re, sys
out_dir, episodes, restart_enabled = sys.argv[1], int(sys.argv[2]), sys.argv[3] == '1'
try:
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
except Exception:
    print('run 0')
    raise SystemExit
done = int(meta.get('episodes_completed') or 0)
if done < episodes:
    print(f'run {done}')
    raise SystemExit
if not restart_enabled:
    print(f'skip {done}')
    raise SystemExit
restart_count = 0
try:
    state = json.load(open(os.path.join(out_dir, 'restart_state.json')))
    restart_count = int(state.get('restart_count') or 0)
except Exception:
    pass
if restart_count >= 3:
    print(f'skip {done}')
    raise SystemExit
tags = []
episode_re = re.compile(r'^Episode\\s+(\\d+),\\s+Mean Score:\\s+[-+0-9.eE]+,\\s+Tagged count:\\s+(\\d+)')
try:
    with open(os.path.join(out_dir, 'episode_log.txt'), errors='ignore') as f:
        for line in f:
            m = episode_re.search(line)
            if m:
                tags.append(int(m.group(2)))
except Exception:
    print(f'run {done}')
    raise SystemExit
if len(tags) >= 100:
    num_agents = int(meta.get('num_agents') or 0)
    recent = tags[-100:]
    sr100 = sum(t >= num_agents for t in recent) / 100.0 if num_agents else 0.0
    cov100 = sum(min(t, num_agents) / float(num_agents) for t in recent) / 100.0 if num_agents else 0.0
    # Completed run has traction (success/recovery) -> leave complete, skip.
    # Failing (SR<=1% AND coverage<40%) -> allow Python to restart it.
    if not (sr100 <= 0.01 and cov100 < 0.40):
        print(f'skip {done}')
        raise SystemExit
print(f'run {done}')" "$out_dir" "$EPISODES" "${USE_ORBIT_RESTART:-0}" 2>/dev/null)
        if [ "${complete_action%% *}" = "skip" ]; then
            local done_ep="${complete_action#* }"
            echo "[seed=$seed n=$n mode=$mode] SKIP (already complete/recovered: $done_ep/$EPISODES)"
            return 0
        fi
    fi

    local t_start
    t_start=$(date +%s)
    echo "[seed=$seed n=$n mode=$mode] START  $(date -Iseconds)  -> $log_file"
    # Offload is decoupled from training: the trainer never mirrors to USB
    # in-process (it would block the slot). A completed run is enqueued below
    # and mirrored by the independent offload worker.
    local offload_flag=(--disable_episode_offload)
    local restart_flag=()
    if [ "${USE_ORBIT_RESTART:-0}" = "1" ]; then
        restart_flag=(--use_orbit_restart)
    fi
    python run/train_seeded.py \
        --n "$n" --mode "$mode" --seed "$seed" \
        --episodes "$EPISODES" \
        --v_ang_max "$V_ANG_MAX" \
        --use_offline_replay \
        --out_dir "$out_dir" \
        --artifact_root "$ARTIFACT_ROOT" \
        --offload_root "$OFFLOAD_ROOT" \
        --offload_mode "$OFFLOAD_MODE" \
        --offload_every "$OFFLOAD_EVERY" \
        "${offload_flag[@]}" \
        "${restart_flag[@]}" \
        > "$log_file" 2>&1
    local rc=$?
    local t_end elapsed h m s dur
    t_end=$(date +%s)
    elapsed=$(( t_end - t_start ))
    h=$(( elapsed / 3600 ))
    m=$(( (elapsed % 3600) / 60 ))
    s=$(( elapsed % 60 ))
    dur=$(printf "%dh%02dm%02ds" "$h" "$m" "$s")
    local restart_count
    restart_count=$(python3 -c "import json, os, sys
path = os.path.join(sys.argv[1], 'restart_state.json')
try:
    print(int(json.load(open(path)).get('restart_count') or 0))
except Exception:
    print(0)" "$out_dir" 2>/dev/null)
    if [ $rc -eq 0 ]; then
        echo "[seed=$seed n=$n mode=$mode] DONE   rc=0  elapsed=$dur restarts=${restart_count:-0}/3"
        if [ "${OFFLOAD:-1}" = "1" ]; then
            # Enqueue this completed run for the independent offload worker.
            # FIFO ordering via nanosecond timestamp prefix.
            printf '%s\n' "$out_dir" > "$OFFLOAD_QUEUE/$(date +%s%N)_n${n}_${mode}_seed${seed}.job"
            echo "[seed=$seed n=$n mode=$mode] ENQUEUED offload -> $OFFLOAD_QUEUE"
        fi
    else
        echo "[seed=$seed n=$n mode=$mode] FAILED rc=$rc elapsed=$dur restarts=${restart_count:-0}/3 (see $log_file)"
    fi
    return $rc
}

export -f run_one
export ARTIFACT_ROOT OFFLOAD_ROOT LOG_DIR EPISODES V_ANG_MAX OFFLOAD OFFLOAD_MODE OFFLOAD_EVERY USE_ORBIT_RESTART OFFLOAD_QUEUE

# Global job queue: ordered seed-first, then n, then mode.
# A single pool of size $PARALLEL consumes the queue — slots refill
# greedily across seed boundaries, so seed K+1 can begin while a slow
# tail run from seed K is still finishing. Launch ORDER still prioritises
# lower seeds (and within a seed, lower n / earlier mode).
jobs_file="$(mktemp)"
for seed in $SEEDS; do
    for n in $NS; do
        for mode in $MODES; do
            printf "%s\t%s\t%s\n" "$seed" "$n" "$mode" >> "$jobs_file"
        done
    done
done

total_jobs=$(wc -l < "$jobs_file")
echo "Queued $total_jobs jobs (seed-first order), running up to $PARALLEL in parallel."
batch_t_start=$(date +%s)

# Background progress monitor: every PROGRESS_INTERVAL seconds, scan
# ARTIFACT_ROOT/runs/*/meta.json + .run_all.lock and print one line per
# in-flight run with episodes_completed/target. Disable with PROGRESS=0.
PROGRESS="${PROGRESS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
monitor_pid=""
if [ "$PROGRESS" = "1" ]; then
    python3 - "$ARTIFACT_ROOT/runs" "$PROGRESS_INTERVAL" <<'PYEOF' &
import json, os, sys, time
runs_root, interval = sys.argv[1], float(sys.argv[2])
# Live in-place block goes to the user's terminal, NOT through the pipe
# that tee is reading — that way the master log stays clean.
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
        rows.append(f"  … {name}: {done}/{total} restarts={restarts}/3")
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

# ---- Independent offload worker (the +1 slot) ----
# Drains $OFFLOAD_QUEUE one job at a time, so at most ONE offload runs at once
# and training slots are never blocked by USB I/O. Exits once the launcher
# signals .done AND the queue is empty.
worker_pid=""
if [ "${OFFLOAD:-1}" = "1" ]; then
    (
        while true; do
            job="$(ls -1 "$OFFLOAD_QUEUE"/*.job 2>/dev/null | sort | head -n1)"
            if [ -z "$job" ]; then
                if [ -f "$OFFLOAD_QUEUE/.done" ]; then
                    # Final drain: re-scan in case a job slipped in just now.
                    job="$(ls -1 "$OFFLOAD_QUEUE"/*.job 2>/dev/null | sort | head -n1)"
                    [ -z "$job" ] && break
                else
                    sleep 5
                    continue
                fi
            fi
            run_dir="$(cat "$job" 2>/dev/null)"
            if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
                echo "[offload] mirroring $run_dir"
                if python "$REPO_ROOT/tools/offload_artifacts.py" \
                        --run_dir "$run_dir" \
                        --source_root "$ARTIFACT_ROOT" \
                        --target_root "$OFFLOAD_ROOT" \
                        --keep_local_result_csv --include_running --quiet_progress; then
                    echo "[offload] done $run_dir"
                else
                    echo "[offload] FAILED $run_dir (left dequeued; rerun launcher to retry)"
                fi
            fi
            rm -f "$job"
        done
        echo "[offload] worker exiting (queue drained)"
    ) &
    worker_pid=$!
fi

# Kill background helpers with us, even on Ctrl-C. Unquoted so empty pids vanish.
trap 'kill $monitor_pid $worker_pid 2>/dev/null' EXIT INT TERM

while IFS=$'\t' read -r s n m; do
    while [ "$(jobs -rp | grep -vx -e "${monitor_pid:-x}" -e "${worker_pid:-y}" | wc -l)" -ge "$PARALLEL" ]; do
        sleep 2
    done
    run_one "$s" "$n" "$m" &
done < "$jobs_file"

# Wait only for training workers (exclude monitor + offload worker).
for pid in $(jobs -rp | grep -vx -e "${monitor_pid:-x}" -e "${worker_pid:-y}"); do
    wait "$pid" 2>/dev/null
done

# All training finished: tell the offload worker to drain remaining jobs & exit.
if [ -n "$worker_pid" ]; then
    : > "$OFFLOAD_QUEUE/.done"
    echo "All training complete; waiting for offload queue to drain..."
    wait "$worker_pid" 2>/dev/null
    rm -f "$OFFLOAD_QUEUE/.done"
fi
if [ -n "$monitor_pid" ]; then
    kill "$monitor_pid" 2>/dev/null
    wait "$monitor_pid" 2>/dev/null
fi
rm -f "$jobs_file"

batch_elapsed=$(( $(date +%s) - batch_t_start ))
batch_dur=$(printf "%dh%02dm%02ds" $((batch_elapsed/3600)) $(((batch_elapsed%3600)/60)) $((batch_elapsed%60)))
echo ">>>>>>>>>> ALL JOBS DONE : $(date -Iseconds)  elapsed=$batch_dur <<<<<<<<<<"

echo
echo "All seeds complete."
echo "Local artifacts: $ARTIFACT_ROOT/runs/"
echo "Offload mirror: $OFFLOAD_ROOT/runs/"
