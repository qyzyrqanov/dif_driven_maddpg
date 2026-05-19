#!/usr/bin/env bash
# Sequential per-seed runner:
#   * seeds {1,2,3,4,5} processed one at a time (seed K must finish entirely
#     before seed K+1 starts), so on-disk artifacts grow seed-by-seed in order
#   * inside one seed, the (n, mode) tasks run in parallel up to PARALLEL
#   * uses main_loop (HER-style offline_replay_success)
#   * v_ang_max = pi/2 (confirmed Z7S value)
#   * env walls remain unbounded (already commented out in env.py)
#   * each run is mirrored to OFFLOAD_ROOT after every saved episode
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
#   OFFLOAD_ROOT   (default /media/abz/Z7S/experiments_revision_offline_replay)
#   V_ANG_MAX      (default pi2)
#   LOG_DIR        (default $ARTIFACT_ROOT/logs)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the canonical Linux venv
source "$REPO_ROOT/.venvLin/bin/activate"
export PYTHONPATH="$REPO_ROOT"

PARALLEL="${PARALLEL:-3}"
SEEDS="${SEEDS:-1 2 3 4 5}"
MODES="${MODES:-full ablation nocoll}"
NS="${NS:-4 5 6}"
EPISODES="${EPISODES:-1000}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/Desktop/dif_driven_revision_offline_replay_artifacts}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-/media/abz/Z7S/experiments_revision_offline_replay}"
V_ANG_MAX="${V_ANG_MAX:-pi2}"
OFFLOAD="${OFFLOAD:-1}"  # 1 = mirror per-episode to OFFLOAD_ROOT, 0 = disable
LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/logs}"

mkdir -p "$ARTIFACT_ROOT/runs" "$LOG_DIR"

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
echo "  log dir     : $LOG_DIR"
echo "============================================================"

run_one() {
    local seed="$1" n="$2" mode="$3"
    local out_dir="$ARTIFACT_ROOT/runs/n${n}_${mode}_seed${seed}"
    local log_file="$LOG_DIR/seed${seed}_n${n}_${mode}.log"
    mkdir -p "$out_dir"

    # Skip-if-complete: read episodes_completed from meta.json and bail
    # before spawning Python (and before any offload sync), so finished
    # runs cost ~0 seconds.
    if [ -f "$out_dir/meta.json" ]; then
        local done_ep
        done_ep=$(python3 -c "import json,sys
try:
    print(int(json.load(open('$out_dir/meta.json')).get('episodes_completed') or 0))
except Exception:
    print(0)" 2>/dev/null)
        if [ -n "$done_ep" ] && [ "$done_ep" -ge "$EPISODES" ]; then
            echo "[seed=$seed n=$n mode=$mode] SKIP (already complete: $done_ep/$EPISODES)"
            return 0
        fi
    fi

    echo "[seed=$seed n=$n mode=$mode] START  -> $log_file"
    local offload_flag=()
    if [ "${OFFLOAD:-1}" = "0" ]; then
        offload_flag=(--disable_episode_offload)
    fi
    python run/train_seeded.py \
        --n "$n" --mode "$mode" --seed "$seed" \
        --episodes "$EPISODES" \
        --v_ang_max "$V_ANG_MAX" \
        --use_offline_replay \
        --out_dir "$out_dir" \
        --artifact_root "$ARTIFACT_ROOT" \
        --offload_root "$OFFLOAD_ROOT" \
        "${offload_flag[@]}" \
        > "$log_file" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[seed=$seed n=$n mode=$mode] DONE   rc=0"
    else
        echo "[seed=$seed n=$n mode=$mode] FAILED rc=$rc (see $log_file)"
    fi
    return $rc
}

export -f run_one
export ARTIFACT_ROOT OFFLOAD_ROOT LOG_DIR EPISODES V_ANG_MAX OFFLOAD

for seed in $SEEDS; do
    echo
    echo ">>>>>>>>>> SEED $seed START : $(date -Iseconds) <<<<<<<<<<"

    # Build the (seed, n, mode) jobs for this seed
    jobs_file="$(mktemp)"
    for n in $NS; do
        for mode in $MODES; do
            printf "%s\t%s\t%s\n" "$seed" "$n" "$mode" >> "$jobs_file"
        done
    done

    # Run jobs in parallel up to $PARALLEL, but all for THIS seed only.
    # We wait for the whole batch before moving to the next seed.
    while IFS=$'\t' read -r s n m; do
        # cap concurrency
        while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
            sleep 2
        done
        run_one "$s" "$n" "$m" &
    done < "$jobs_file"

    # Block until ALL background jobs for this seed finish
    wait
    rm -f "$jobs_file"

    echo ">>>>>>>>>> SEED $seed DONE  : $(date -Iseconds) <<<<<<<<<<"
done

echo
echo "All seeds complete."
echo "Local artifacts: $ARTIFACT_ROOT/runs/"
echo "Offload mirror: $OFFLOAD_ROOT/runs/"
