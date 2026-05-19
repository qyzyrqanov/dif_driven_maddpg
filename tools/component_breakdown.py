"""Per-component reward breakdown for a training run.

Reads result*.csv (per-step log written by MADDPGBase.log_env_step_to_csv) and
reports, for each of the 9 reward components, the mean per-episode sum (over
agents) across a final window of episodes.

The hypothesis for the n=5 full shaping-conflict failure is:
  - comp 0 (progressive)  : large negative or oscillating
  - comp 3 (reached goal) : near zero (agents don't land)
  - comps 4/5 (collision) : large negative
  - comp 7 (directional)  : near zero (heading not aligned)
For comparison, a healthy run (e.g. n=5 ablation/nocoll) should show
comp 3 ~= N (all agents reach) and comp 8 (time) small in magnitude.

Usage:
  python tools/component_breakdown.py \
    --run_dir ~/Desktop/dif_driven_revision_offline_replay_artifacts/runs/n5_full_seed1 \
    [--window 200]

  # Compare several runs in one call:
  python tools/component_breakdown.py --window 200 \
    --run_dir .../n5_full_seed1 .../n5_ablation_seed1 .../n5_nocoll_seed1
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

COMP_NAMES = {
    1: "progressive",
    2: "distance",
    3: "base",
    4: "reached",
    5: "agent_coll",
    6: "obs_coll",
    7: "v_lin",
    8: "directional",
    9: "time",
}


def summarize(run_dir: str, window: int):
    csv_paths = sorted(glob.glob(os.path.join(run_dir, "result*.csv")))
    if not csv_paths:
        return None
    csv = csv_paths[0]
    df = pd.read_csv(csv)
    # Identify agents
    agents = sorted({int(m.group(1))
                     for c in df.columns
                     for m in [re.match(r"agent(\d+)_comp1$", c)] if m})
    n = len(agents)

    # Per-episode totals: sum over steps and over agents for each component
    per_ep = {}
    for k in range(1, 10):
        cols = [f"agent{i}_comp{k}" for i in agents]
        per_ep[k] = df.groupby("episode_id")[cols].sum().sum(axis=1)

    # Episode-level success (all N agents reached)
    ep_max_done = df.groupby("episode_id")["done_count"].max()
    n_eps = len(ep_max_done)
    window = min(window, n_eps)
    sl = slice(n_eps - window, n_eps)

    sr = (ep_max_done.iloc[sl] == n).mean() * 100.0
    # Last-window means
    means = {k: float(per_ep[k].iloc[sl].mean()) for k in range(1, 10)}
    stds = {k: float(per_ep[k].iloc[sl].std()) for k in range(1, 10)}

    return {
        "n": n,
        "n_episodes_in_csv": n_eps,
        "window": window,
        "sr_pct": sr,
        "means": means,
        "stds": stds,
    }


def fmt_row(name: str, info: dict) -> str:
    head = (f"{name}  N={info['n']}  eps_in_csv={info['n_episodes_in_csv']}  "
            f"window={info['window']}  SR={info['sr_pct']:.1f}%")
    lines = [head, "  comp                       mean ± std (per-episode, summed over agents)"]
    for k in range(1, 10):
        m = info["means"][k]
        s = info["stds"][k]
        lines.append(f"  {k}.{COMP_NAMES[k]:<14}  {m:>12.2f} ± {s:>10.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", nargs="+", required=True,
                    help="One or more run directories containing result*.csv")
    ap.add_argument("--window", type=int, default=200,
                    help="Last-N episodes window (default 200)")
    args = ap.parse_args()

    blocks = []
    for d in args.run_dir:
        d = os.path.expanduser(d)
        info = summarize(d, args.window)
        if info is None:
            blocks.append(f"{d}: NO result*.csv FOUND")
            continue
        blocks.append(fmt_row(os.path.basename(d.rstrip("/")), info))

    print("\n\n".join(blocks))

    # Side-by-side comparison table if multiple runs given
    infos = []
    for d in args.run_dir:
        d = os.path.expanduser(d)
        info = summarize(d, args.window)
        if info is not None:
            infos.append((os.path.basename(d.rstrip("/")), info))
    if len(infos) > 1:
        print("\n\nSide-by-side per-component means (last-window, per episode):")
        header = ["component"] + [name for name, _ in infos]
        widths = [16] + [max(14, len(h)) for h in header[1:]]
        def row(cells):
            return "  ".join(c.rjust(w) for c, w in zip(cells, widths))
        print(row(header))
        for k in range(1, 10):
            cells = [f"{k}.{COMP_NAMES[k]}"]
            for _, info in infos:
                cells.append(f"{info['means'][k]:.2f}")
            print(row(cells))
        print(row(["SR%"] + [f"{info['sr_pct']:.1f}" for _, info in infos]))


if __name__ == "__main__":
    main()
