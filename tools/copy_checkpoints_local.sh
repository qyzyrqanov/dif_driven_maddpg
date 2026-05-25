#!/usr/bin/env bash
# Copy the 45 trained actor checkpoints (shared_actor.pth) from the Z7S full
# mirror into a local gitignored folder, so eval (run/run_eval_revision.sh) runs
# without needing the Z7S drive mounted. ~40 MB total.
#
# Usage:  bash tools/copy_checkpoints_local.sh
# Override source: SRC=/path/to/runs bash tools/copy_checkpoints_local.sh
set -euo pipefail
REPO="/home/abz/workspace/PycharmProjects/dif_driven_maddpg"
SRC="${SRC:-/media/abz/Z7S/experiments_revision_offline_replay_restart_v3/runs}"
DEST="$REPO/checkpoints_local/runs"
[[ -d "$SRC" ]] || { echo "source not found (Z7S mounted?): $SRC"; exit 1; }
n=0
for d in "$SRC"/n*/; do
  name=$(basename "$d")
  [[ -f "$d/shared_actor.pth" ]] || continue
  mkdir -p "$DEST/$name"
  cp "$d/shared_actor.pth" "$DEST/$name/shared_actor.pth"
  n=$((n+1))
done
echo "copied $n actor checkpoints -> $DEST"
