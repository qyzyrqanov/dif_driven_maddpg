#!/usr/bin/env bash
# Round-3 run dashboard. Run anytime to see how many of the launcher's jobs are
# finished and where each in-progress run currently sits — independent of the
# noisy live launcher terminal. Reads meta.json from BOTH the lean local scratch
# and the Z7S archive (a run counts as finished if EITHER says so).
#
#   bash run/round3_status.sh           # seeds 3 4 5 (default), full mode
#   SEEDS="3 4 5" bash run/round3_status.sh
set -euo pipefail

SEEDS="${SEEDS:-3 4 5}"
NS="${NS:-4 5 6}"
MADDPG_SEEDS="${MADDPG_SEEDS:-4 5}"   # seeds 1-3 already archived separately
TARGET="${EPISODES:-1000}"
R3_LOCAL="$HOME/Desktop/dif_driven_round3_artifacts/runs"
R3_Z7S="$HOME/dif_driven_archive/dif_driven_round3_artifacts/runs"

SEEDS="$SEEDS" NS="$NS" MADDPG_SEEDS="$MADDPG_SEEDS" TARGET="$TARGET" \
R3_LOCAL="$R3_LOCAL" R3_Z7S="$R3_Z7S" python3 - <<'PY'
import json, os
T=int(os.environ["TARGET"])
LOC=os.environ["R3_LOCAL"]; ARC=os.environ["R3_Z7S"]
def best(name):
    ep, fin = 0, False
    for r in (LOC, ARC):
        p=os.path.join(r, name, "meta.json")
        if os.path.exists(p):
            try:
                m=json.load(open(p))
                e=int(m.get("episodes_completed") or 0)
                if m.get("finished") or e>=T: fin=True
                ep=max(ep,e)
            except Exception: pass
    return ep, fin
jobs=[]
for s in os.environ["SEEDS"].split():
    for n in os.environ["NS"].split():
        jobs += [f"abl_noRestart_n{n}_full_seed{s}", f"abl_noHER_n{n}_full_seed{s}"]
for s in os.environ["MADDPG_SEEDS"].split():
    for n in os.environ["NS"].split():
        jobs.append(f"maddpg_obs_n{n}_full_seed{s}")
done=0; rows=[]
for name in jobs:
    ep, fin = best(name)
    if fin: done+=1; continue
    rows.append((ep, f"  {ep:>4}/{T}  {name}"))
print(f"=== {done}/{len(jobs)} finished — {len(rows)} not done ===")
for _, line in sorted(rows, reverse=True):
    print(line)
PY
