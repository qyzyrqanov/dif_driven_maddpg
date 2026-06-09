#!/usr/bin/env python3
"""Distil the deterministic eval outputs into light, tidy CSVs for the notebook.

Reads the per-episode eval summary CSVs written by ``run/eval_policy.py`` and
``run/eval_hungarian_p.py`` (the small ``*.csv`` files, NOT the large
``*_steps.csv``) and emits two light tables under ``--out_dir``:

  eval_episodes.csv  per-episode rows: kind,env,n,seed,episode_id,success,
                     coverage,completion_time,collision_step_rate,path_length
  eval_summary.csv   aggregated mean(±SD over train-seeds) per (kind,env,n)

Metrics:
  * SR (success rate)  — fraction of episodes where ALL agents reached a goal.
  * coverage           — PARTIAL-success metric: mean fraction of agents that
                         covered a landmark (done_count_final / n), so episodes
                         that fall short of full success still contribute.

Usage:
  python tools/aggregate_eval.py \
    --eval_dir /home/abz/Desktop/dif_driven_revision_eval/eval \
    --out_dir  revision_logs/eval
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

POLICY_RE = re.compile(r"policy_n(\d+)_(\w+?)_trainseed(\d+)_env(\d+)_")
HEUR_RE = re.compile(r"heuristic_n(\d+)_(\w+?)_env(\d+)_")
MAPPO_RE = re.compile(r"mappo_n(\d+)_(\w+?)_trainseed(\d+)_env(\d+)_")


def _episode_rows(path, kind, n, mode, seed, env):
    df = pd.read_csv(path)
    return pd.DataFrame({
        "kind": kind, "env": env, "n": n, "mode": mode, "seed": seed,
        "episode_id": df["episode_id"],
        "success": df["success"].astype(float),
        # partial-success metric: fraction of agents that reached a landmark
        "coverage": df["done_count_final"].astype(float) / n,
        "completion_time": df["completion_time"],
        "collision_step_rate": df.get("collision_step_rate", np.nan),
        "path_length": df.get("path_length_total", np.nan),
    })


def collect(eval_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(eval_dir, "*.csv"))):
        base = os.path.basename(f)
        if base.endswith("_steps.csv"):
            continue
        m = MAPPO_RE.match(base)
        if m:
            # A completed eval writes its summary .json last; skip CSVs without a
            # sibling .json (interrupted / partial runs) so they don't bias means.
            if not os.path.exists(f[:-4] + ".json"):
                print(f"skip (no sibling .json, incomplete): {base}")
                continue
            n, mode, seed, env = int(m[1]), m[2], int(m[3]), int(m[4])
            rows.append(_episode_rows(f, "mappo", n, mode, seed, env))
            continue
        m = POLICY_RE.match(base)
        if m:
            n, mode, seed, env = int(m[1]), m[2], int(m[3]), int(m[4])
            rows.append(_episode_rows(f, "policy", n, mode, seed, env))
            continue
        m = HEUR_RE.match(base)
        if m:
            n, mode, env = int(m[1]), m[2], int(m[3])
            rows.append(_episode_rows(f, "heuristic", n, mode, 0, env))
    if not rows:
        raise SystemExit(f"No eval summary CSVs found under {eval_dir}")
    return pd.concat(rows, ignore_index=True)


def summarize(ep):
    # per-run (per train-seed) means first, then mean±SD across seeds
    per_run = (ep.groupby(["kind", "env", "n", "mode", "seed"])
                 .agg(SR=("success", lambda s: 100 * s.mean()),
                      coverage=("coverage", lambda s: 100 * s.mean()),
                      completion_time=("completion_time",
                                       lambda s: np.nan),  # filled below (success-only)
                      collision_step_rate=("collision_step_rate",
                                           lambda s: 100 * s.mean()),
                      path_length=("path_length", "mean"))
                 .reset_index())
    # success-only completion time per run
    ct = (ep[ep.success == 1].groupby(["kind", "env", "n", "mode", "seed"])
            ["completion_time"].mean().reset_index()
            .rename(columns={"completion_time": "completion_time_succ"}))
    per_run = per_run.merge(ct, on=["kind", "env", "n", "mode", "seed"], how="left")
    per_run["completion_time"] = per_run["completion_time_succ"]
    per_run = per_run.drop(columns=["completion_time_succ"])

    def agg(g):
        return pd.Series({
            "seeds": g.seed.nunique(),
            "SR_mean": round(g.SR.mean(), 1), "SR_sd": round(g.SR.std(ddof=0), 1),
            "cov_mean": round(g.coverage.mean(), 1), "cov_sd": round(g.coverage.std(ddof=0), 1),
            "compT_mean": round(g.completion_time.mean(), 1),
            "coll_mean": round(g.collision_step_rate.mean(), 2),
            "path_mean": round(g.path_length.mean(), 1),
            "SR_per_seed": ", ".join(f"{s}:{v:.0f}" for s, v in zip(g.seed, g.SR)),
        })
    summ = per_run.groupby(["kind", "env", "n", "mode"]).apply(agg).reset_index()
    return per_run, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", default="revision_logs/eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ep = collect(args.eval_dir)
    per_run, summ = summarize(ep)

    ep_path = os.path.join(args.out_dir, "eval_episodes.csv")
    pr_path = os.path.join(args.out_dir, "eval_per_run.csv")
    su_path = os.path.join(args.out_dir, "eval_summary.csv")
    ep.to_csv(ep_path, index=False)
    per_run.to_csv(pr_path, index=False)
    summ.to_csv(su_path, index=False)
    print(f"wrote {ep_path}  ({len(ep)} episode rows)")
    print(f"wrote {pr_path}  ({len(per_run)} run rows)")
    print(f"wrote {su_path}  ({len(summ)} cells)")
    print()
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
