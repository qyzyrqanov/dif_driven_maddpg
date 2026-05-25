#!/usr/bin/env python3
"""Export experimental-research detail in two tiers.

Tier split (per the project policy):
  * LOCAL  (repo/revision_logs/, gitignored) = everything the notebook needs to
    run WITHOUT the media drive, kept light: `run_details.csv` (per-run totals) +
    each run's compact `episode_summary.csv` (per-episode done_count + component
    sums, ~50 KB) + `meta.json`/`restart_state.json` + aggregate CSVs + figures +
    the notebook. A few MB total — NO heavy per-step `result*.csv` (150 MB each),
    checkpoints, or replay buffers.
  * MEDIA  (single folder, e.g. /media/abz/Z7S/dif_driven_logs/) = EVERYTHING
    local PLUS the bulkier raw episode-wise text logs (`episode_log.txt`,
    `restart_log.txt`). Disaster-recovery copy.

So the notebook runs fully off the local tier (tables AND episode curves); a
local loss is recoverable from media, and a media loss still leaves a runnable
local copy. The truly huge raw files stay in the artifact root / offload mirror,
not here.

Usage:
    python tools/export_light_logs.py                 # build local + mirror to media
    python tools/export_light_logs.py --no_media       # local totals only
    python tools/export_light_logs.py --window 200
"""
import argparse, glob, json, os, shutil, subprocess
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_META_FILES = ["meta.json", "restart_state.json"]          # per-run totals -> local + media
EPISODEWISE_FILES = ["episode_log.txt", "restart_log.txt"]    # episode-wise -> media only


def num_agents(run_dir, name):
    mp = os.path.join(run_dir, "meta.json")
    if os.path.exists(mp):
        try:
            m = json.load(open(mp))
            return int(m.get("num_agents") or m.get("n"))
        except Exception:
            pass
    return int(name[1:name.index("_")])


def _from_episode_log(run_dir):
    """Fallback when result*.csv is missing: parse episode_log.txt real episodes.
    `Tagged count` = agents done at episode end -> done_count. Components are
    unavailable from the log, so comp1..9 are left NaN for this run."""
    import re
    p = os.path.join(run_dir, "episode_log.txt")
    if not os.path.exists(p):
        return None
    pat = re.compile(r"^Episode (\d+),.*Tagged count:\s*(\d+)", re.M)
    rows = [(int(e), int(t)) for e, t in pat.findall(open(p).read())]
    if not rows:
        return None
    out = pd.DataFrame(rows, columns=["episode_id", "done_count"]).drop_duplicates("episode_id")
    out = out.sort_values("episode_id").reset_index(drop=True)
    for j in range(1, 10):
        out[f"comp{j}"] = np.nan
    return out


def distil_run(run_dir, n):
    """Return per-episode summary DataFrame (episode_id, done_count, comp1..9)."""
    cands = glob.glob(os.path.join(run_dir, "result*.csv")) or \
            glob.glob(os.path.join(run_dir, "rewards.csv"))
    if not cands:
        return _from_episode_log(run_dir)
    f = max(cands, key=os.path.getmtime)
    comp_cols = [f"agent{a}_comp{j}" for a in range(n) for j in range(1, 10)]
    use = set(["episode_id", "done_count"] + comp_cols)
    df = pd.read_csv(f, usecols=lambda c: c in use)
    g = df.groupby("episode_id")
    out = pd.DataFrame(index=sorted(df.episode_id.unique()))
    out.index.name = "episode_id"
    out["done_count"] = g["done_count"].max()
    for j in range(1, 10):
        cols = [f"agent{a}_comp{j}" for a in range(n) if f"agent{a}_comp{j}" in df.columns]
        out[f"comp{j}"] = g[cols].sum().sum(axis=1)
    return out.reset_index()


def run_totals(name, n, summ, meta, restarts, window):
    """One-row dict of per-run totals from the episode summary + meta."""
    parts = name.split("_")
    mode, seed = parts[1], int(parts[2].replace("seed", ""))
    row = dict(run=name, n=n, mode=mode, seed=seed)
    if summ is not None and len(summ):
        last = summ.tail(window)
        row["episodes"] = len(summ)
        row["SR"] = round(100 * (last.done_count >= n).mean(), 1)
        row["coverage"] = round(100 * (last.done_count / n).mean(), 1)
        for j in range(1, 10):
            row[f"comp{j}_mean"] = round(last[f"comp{j}"].mean(), 2)
    row["restarts"] = restarts
    row["finished"] = meta.get("finished")
    row["train_hours"] = meta.get("total_train_hours")
    row["peak_gpu_gb"] = meta.get("peak_gpu_gb")
    row["total_steps"] = meta.get("total_steps")
    row["gpu_name"] = meta.get("gpu_name")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_root",
                    default="/home/abz/Desktop/dif_driven_revision_offline_replay_restart_v3_artifacts")
    ap.add_argument("--local_logs", default=os.path.join(REPO, "revision_logs"))
    ap.add_argument("--media_logs", default="/media/abz/Z7S/dif_driven_logs")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--no_media", action="store_true")
    args = ap.parse_args()

    runs_root = os.path.join(args.artifact_root, "runs")
    local_runs = os.path.join(args.local_logs, "runs")
    local_res = os.path.join(args.local_logs, "res")
    os.makedirs(local_runs, exist_ok=True)
    os.makedirs(local_res, exist_ok=True)
    # media episode-wise staging lives under a sibling tree we build then mirror
    media_stage = os.path.join(args.local_logs, ".episodewise")   # temp, moved to media
    os.makedirs(media_stage, exist_ok=True)

    names = sorted(d for d in os.listdir(runs_root)
                   if os.path.isdir(os.path.join(runs_root, d)) and d.startswith("n"))
    totals = []
    for name in names:
        rd = os.path.join(runs_root, name)
        n = num_agents(rd, name)
        summ = distil_run(rd, n)
        meta = {}
        if os.path.exists(os.path.join(rd, "meta.json")):
            meta = json.load(open(os.path.join(rd, "meta.json")))
        restarts = 0
        rp = os.path.join(rd, "restart_state.json")
        if os.path.exists(rp):
            try: restarts = int(json.load(open(rp)).get("restart_count", 0))
            except Exception: restarts = 0

        # --- LOCAL: per-run meta + compact episode_summary (notebook runs off this) ---
        ld = os.path.join(local_runs, name); os.makedirs(ld, exist_ok=True)
        for fn in RUN_META_FILES:
            src = os.path.join(rd, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(ld, fn))
        if summ is not None:
            summ.to_csv(os.path.join(ld, "episode_summary.csv"), index=False)

        # --- MEDIA staging: bulkier raw episode-wise text logs (media-only) ---
        md = os.path.join(media_stage, "runs", name); os.makedirs(md, exist_ok=True)
        for fn in EPISODEWISE_FILES:
            src = os.path.join(rd, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(md, fn))

        totals.append(run_totals(name, n, summ, meta, restarts, args.window))
        print(f"  {name}: n={n} SR={totals[-1].get('SR')} eps={totals[-1].get('episodes')}")

    # run_details.csv -> LOCAL (the notebook's primary source) and also media
    det = pd.DataFrame(totals).sort_values(["n", "mode", "seed"])
    det.to_csv(os.path.join(args.local_logs, "run_details.csv"), index=False)

    # aggregate csvs + figures + notebook -> LOCAL
    res_src = os.path.join(args.artifact_root, "res")
    if os.path.isdir(res_src):
        for fn in os.listdir(res_src):
            if fn.endswith((".csv", ".png", ".pdf")):
                shutil.copy2(os.path.join(res_src, fn), os.path.join(local_res, fn))
    nb = os.path.join(REPO, "res", "revision_final_results.ipynb")
    if os.path.exists(nb):
        shutil.copy2(nb, os.path.join(args.local_logs, "revision_final_results.ipynb"))

    json.dump({"artifact_root": args.artifact_root, "runs": len(names), "window": args.window,
               "tiers": {"local": "run_details.csv + per-run meta + aggregate csvs + figures + notebook",
                         "media_only": "per-run episode_summary.csv + episode_log.txt (episode-wise)"}},
              open(os.path.join(args.local_logs, "MANIFEST.json"), "w"), indent=2)

    sz = subprocess.run(["du", "-sh", args.local_logs], capture_output=True, text=True).stdout.split()
    print(f"\nLOCAL totals: {args.local_logs}  ({sz[0] if sz else '?'}, {len(names)} runs)")

    if args.no_media:
        print("(--no_media) episode-wise data left in", media_stage, "— not mirrored.")
        return

    media_parent = os.path.dirname(args.media_logs)
    if not os.path.isdir(media_parent):
        print(f"media parent not mounted ({media_parent}) — episode-wise staged at {media_stage}.")
        print(f"  When media is attached: rsync -a {media_stage}/ {args.media_logs}/ "
              f"&& rsync -a {args.local_logs}/ {args.media_logs}/")
        return
    os.makedirs(args.media_logs, exist_ok=True)
    # media = local totals + episode-wise (single consolidated folder)
    subprocess.run(["rsync", "-a", "--exclude", ".episodewise",
                    args.local_logs.rstrip("/") + "/", args.media_logs.rstrip("/") + "/"], check=True)
    subprocess.run(["rsync", "-a",
                    os.path.join(media_stage, "") , args.media_logs.rstrip("/") + "/"], check=True)
    shutil.rmtree(media_stage, ignore_errors=True)
    print(f"MEDIA (single folder, incl. episode-wise): {args.media_logs}")


if __name__ == "__main__":
    main()
