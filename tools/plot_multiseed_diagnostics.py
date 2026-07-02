"""Multi-seed diagnostic plots for the revised manuscript (Figures 6-12).

Reads per-step training CSVs (rewards.csv) from the canonical run directories on
the media drive, aggregates across 5 seeds per team size for the FULL reward
configuration only, and writes PDFs (+ PNGs) into
``res/figures_pdf/multiseed_diagnostics/``. Also writes
``multiseed_diagnostic_plot_report.md`` and a reward-decomposition CSV.

Run from the repo root:
    python tools/plot_multiseed_diagnostics.py
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT_CANDIDATES = [
    Path.home() / "dif_driven_archive/experiments_revision_offline_replay_restart_v3/runs",
    REPO / "experiments_revision_offline_replay_restart_v3" / "runs",
]
OUT_DIR = REPO / "res" / "figures_pdf" / "multiseed_diagnostics"
CACHE = OUT_DIR / "_aggregates.pkl"
SEEDS = [1, 2, 3, 4, 5]
NS = [4, 5, 6]
MODE = "full"

# Active reward-component ordering for the revised manuscript (8 channels).
# Old 9-channel order in CSVs (1..9): 1 progressive, 2 distance, 3 base/d_global,
# 4 reached-goal, 5 agent-agent collision, 6 obstacle/inactive-agent collision,
# 7 linear-velocity, 8 directional, 9 time.  Channel 3 has weight 0 and is dropped.
ACTIVE_CHANNELS = [
    ("Progress",                     "comp1", 1.0),
    ("Goal distance",                "comp2", 1.0),
    ("Coverage reward",              "comp4", 10.0),
    ("Agent-agent collision",        "comp5", 10.0),
    ("Inactive/obstacle collision",  "comp6", 10.0),
    ("Linear velocity",              "comp7", 1.0),
    ("Directional alignment",        "comp8", 1.0),
    ("Time penalty",                 "comp9", 1.0),
]
AGENT_COLL_KEY = "comp5"  # named: agent-agent collision

EARLY_LEN = 100
LATE_LEN  = 100
LAST200   = 200
HORIZON   = 500  # episode horizon (steps) used for forward-fill in Figure 8

# ----------------------------- discovery -----------------------------

def find_runs_root() -> Path:
    for c in RUNS_ROOT_CANDIDATES:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"No run root found; tried: {RUNS_ROOT_CANDIDATES}")


def run_dir(root: Path, n: int, seed: int) -> Path:
    return root / f"n{n}_{MODE}_seed{seed}"


# ----------------------------- per-run aggregation -----------------------------

def aggregate_run(csv_path: Path, n: int) -> dict:
    """Read one run's per-step CSV and produce compact aggregates."""
    hung_cols = [f"hung_dist_agent{i}" for i in range(n)]
    vlin_cols = [f"agent{i}_vel_lin"   for i in range(n)]
    vang_cols = [f"agent{i}_vel_ang"   for i in range(n)]
    comp_cols = {k: [f"agent{i}_comp{k}" for i in range(n)] for k in range(1, 10)}

    usecols = (
        ["episode_id", "timestep", "done_count"]
        + hung_cols + vlin_cols + vang_cols
        + [c for v in comp_cols.values() for c in v]
    )
    df = pd.read_csv(csv_path, usecols=usecols)

    df["hung_mean"]      = df[hung_cols].mean(axis=1)
    df["vlin_mean"]      = df[vlin_cols].mean(axis=1)
    df["vang_abs_mean"]  = df[vang_cols].abs().mean(axis=1)

    # Per-row sum across agents of each comp (later sum over timesteps -> per-episode).
    for k in range(1, 10):
        df[f"comp{k}_sum"] = df[comp_cols[k]].sum(axis=1)

    # Per-episode quantities
    eps_sorted = np.sort(df["episode_id"].unique())
    grp = df.groupby("episode_id", sort=True)
    done_max = grp["done_count"].max()
    success = (done_max >= n)
    # First timestep where done_count == n
    mask = df["done_count"] >= n
    first_full = df[mask].groupby("episode_id")["timestep"].min()
    completion_time = first_full.reindex(eps_sorted)  # NaN where never completed

    # Per-episode component sums (raw, unscaled; weights applied later)
    comp_sum_cols = [f"comp{k}_sum" for k in range(1, 10)]
    ep_comp = grp[comp_sum_cols].sum().reindex(eps_sorted)

    # Window selection
    early_eps = eps_sorted[:EARLY_LEN]
    late_eps  = eps_sorted[-LATE_LEN:]
    last200_eps = eps_sorted[-LAST200:]

    def per_t(sub: pd.DataFrame) -> pd.DataFrame:
        return sub.groupby("timestep").agg(
            done_mean=("done_count", "mean"),
            hung_mean=("hung_mean", "mean"),
            vlin_mean=("vlin_mean", "mean"),
            vang_abs_mean=("vang_abs_mean", "mean"),
        )

    early_curve = per_t(df[df["episode_id"].isin(early_eps)])
    late_curve  = per_t(df[df["episode_id"].isin(late_eps)])

    # Forward-filled done_count(t) curves to the horizon (for Figure 8).
    # For each episode in a window, build a length-HORIZON vector of done_count
    # where missing tail timesteps are filled with the last observed value
    # (which equals n for successful episodes and equals the final done_count
    # otherwise). Then average across episodes to get the seed-level curve.
    def ff_done_curve(eps_window) -> np.ndarray:
        sub = df[df["episode_id"].isin(eps_window)][["episode_id", "timestep", "done_count"]]
        if sub.empty:
            return np.full(HORIZON, np.nan)
        # Pivot to (episode_id x timestep) with NaN gaps, ffill across columns
        wide = sub.pivot_table(index="episode_id", columns="timestep",
                                values="done_count", aggfunc="first")
        # Ensure every timestep 1..HORIZON is a column, then forward-fill row-wise.
        wide = wide.reindex(columns=range(1, HORIZON + 1))
        wide = wide.ffill(axis=1)
        # Any leading NaNs (timestep 1 missing) -> 0 (no agents done at start).
        wide = wide.fillna(0.0)
        return wide.values.mean(axis=0)  # length HORIZON

    done_ff_early = ff_done_curve(early_eps)
    done_ff_late  = ff_done_curve(late_eps)

    return dict(
        n=n,
        eps_sorted=np.asarray(eps_sorted),
        success=success.values.astype(bool),
        completion_time=completion_time.values,  # float, NaN if not completed
        ep_comp_sums=ep_comp.values,             # shape (n_ep, 9), unscaled
        early_curve=early_curve,                 # index=timestep
        late_curve=late_curve,
        done_ff_early=done_ff_early,             # shape (HORIZON,), forward-filled
        done_ff_late=done_ff_late,
        last200_eps=last200_eps,
        early_eps=early_eps,
        late_eps=late_eps,
    )


def build_all(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = find_runs_root()
    out = {}
    for n in NS:
        for seed in SEEDS:
            d = run_dir(root, n, seed)
            candidates = [d / "rewards.csv", d / f"result{n}.csv"]
            csv = next((p for p in candidates if p.exists()), None)
            if csv is None:
                warnings.warn(f"MISSING per-step CSV in {d} (tried {candidates})")
                continue
            print(f"  reading n={n} seed={seed} ...", flush=True)
            out[(n, seed)] = aggregate_run(csv, n)
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


# ----------------------------- plotting helpers -----------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS_N = {4: "#1f77b4", 5: "#d62728", 6: "#2ca02c"}
COLORS_EARLY = "#888888"
COLORS_LATE  = "#1f77b4"


def save_fig(fig, name: str):
    pdf = OUT_DIR / f"{name}.pdf"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    return pdf, png


def _curve_mean_std(curves, col):
    """Given a list of DataFrames indexed by timestep, return (t, mean, std)
    aligned on the union of indices; mean over available seeds, std with ddof=0.
    """
    if not curves:
        return np.array([]), np.array([]), np.array([])
    idx = sorted(set().union(*[set(c.index) for c in curves]))
    mat = np.full((len(curves), len(idx)), np.nan, float)
    idx_arr = np.array(idx)
    for i, c in enumerate(curves):
        pos = np.searchsorted(idx_arr, c.index.values)
        mat[i, pos] = c[col].values
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0, ddof=0)
    return idx_arr, mean, std


# ----------------------------- Figure 6 -----------------------------

def fig6_completion_time(data, report):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    positions, dists, labels = [], [], []
    summary_rows = []
    for i, n in enumerate(NS):
        pooled = []
        for s in SEEDS:
            d = data.get((n, s))
            if d is None: continue
            ct = d["completion_time"]
            succ = d["success"]
            eps = d["eps_sorted"]
            sel = np.isin(eps, d["last200_eps"]) & succ & ~np.isnan(ct)
            pooled.extend(ct[sel].tolist())
            summary_rows.append(dict(n=n, seed=s,
                                     n_success=int(sel.sum()),
                                     ct_mean=float(np.mean(ct[sel])) if sel.any() else np.nan,
                                     ct_median=float(np.median(ct[sel])) if sel.any() else np.nan))
        if not pooled:
            continue
        positions.append(i)
        dists.append(pooled)
        labels.append(f"n={n}")
    parts = ax.violinplot(dists, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False, widths=0.7)
    for j, body in enumerate(parts["bodies"]):
        n = NS[positions[j]]
        body.set_facecolor(COLORS_N[n]); body.set_alpha(0.35)
        body.set_edgecolor(COLORS_N[n])
    bp = ax.boxplot(dists, positions=positions, widths=0.18, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("white"); patch.set_alpha(0.9)
    ax.set_xticks(positions); ax.set_xticklabels(labels)
    ax.set_ylabel("Completion time $T_e$ (steps)")
    ax.set_xlabel("Team size")
    ax.set_title("Final-window completion time (Full, last-200 successful episodes)")
    ax.grid(True, axis="y", alpha=0.3)
    pdf, png = save_fig(fig, "figure6_completion_time_multiseed")
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "figure6_completion_time_summary.csv", index=False)
    report["figures"].append(("figure6", str(pdf), "OK"))
    return fig


# ----------------------------- Figure 7 -----------------------------

def fig7_cumulative(data, report):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    # Determine common t range: 1..max horizon observed (500 by config)
    T_MAX = 500
    t_grid = np.arange(1, T_MAX + 1)
    for n in NS:
        seed_curves = []
        for s in SEEDS:
            d = data.get((n, s))
            if d is None: continue
            eps = d["eps_sorted"]
            sel_eps = np.isin(eps, d["last200_eps"])
            ct = d["completion_time"][sel_eps]
            n_ep = sel_eps.sum()
            # F(t) = (# eps with finite ct <= t) / n_ep
            ct_succ = ct[~np.isnan(ct)]
            F = np.searchsorted(np.sort(ct_succ), t_grid, side="right") / max(n_ep, 1)
            seed_curves.append(F)
        if not seed_curves:
            continue
        mat = np.asarray(seed_curves)
        mean = mat.mean(0); std = mat.std(0, ddof=0)
        ax.plot(t_grid, mean, color=COLORS_N[n], lw=2, label=f"n={n}")
        ax.fill_between(t_grid, mean - std, mean + std, color=COLORS_N[n], alpha=0.2)
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel(r"Cumulative completion probability $F_n(t)$")
    ax.set_title("Cumulative team-completion (Full, last-200 episodes; band = SD across seeds)")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, T_MAX)
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    pdf, _ = save_fig(fig, "figure7_cumulative_completion_multiseed")
    report["figures"].append(("figure7", str(pdf), "OK"))
    return fig


# ----------------------------- Figure 8 -----------------------------

def fig8_done_count(data, report):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=False)
    t = np.arange(1, HORIZON + 1)
    for ax, n in zip(axes, NS):
        for label, key, color in [("Early (first 100 ep)", "done_ff_early", COLORS_EARLY),
                                  ("Late (last 100 ep)",   "done_ff_late",  COLORS_LATE)]:
            arrs = [data[(n, s)][key] for s in SEEDS if (n, s) in data]
            if not arrs: continue
            mat = np.stack(arrs)
            m = mat.mean(0); sd = mat.std(0, ddof=0)
            ax.plot(t, m, color=color, lw=2, label=label)
            ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.2)
        ax.axhline(n, color="k", ls=":", lw=1, alpha=0.5, label=f"n={n} (full team)")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean completed agents")
        ax.set_xlim(0, 500); ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Completed-agent progression (Full, mean across 5 seeds, band = SD)",
                 y=1.02)
    fig.tight_layout()
    pdf, _ = save_fig(fig, "figure8_done_count_multiseed")
    report["figures"].append(("figure8", str(pdf), "OK"))
    return fig


# ----------------------------- Figure 9 -----------------------------

def fig9_reward_decomp(data, report):
    """Stacked bars: early vs late scaled component sums, per team size."""
    chan_names  = [c[0] for c in ACTIVE_CHANNELS]
    chan_keys   = [c[1] for c in ACTIVE_CHANNELS]
    chan_alpha  = np.array([c[2] for c in ACTIVE_CHANNELS])
    # CSV channel index 1..9 -> ep_comp_sums column (0..8)
    key_to_idx = {f"comp{k}": k - 1 for k in range(1, 10)}
    sel_idx = np.array([key_to_idx[k] for k in chan_keys])

    rows = []
    # values[n][win] -> (n_seeds, n_channels)
    matrix = {n: {"early": [], "late": []} for n in NS}
    for n in NS:
        for s in SEEDS:
            d = data.get((n, s))
            if d is None: continue
            ep_idx = d["eps_sorted"]
            for win, eps in [("early", d["early_eps"]), ("late", d["late_eps"])]:
                mask = np.isin(ep_idx, eps)
                # Per-episode unscaled sums (n_ep, 9) -> mean over episodes
                mean_unscaled = d["ep_comp_sums"][mask].mean(0)
                scaled = mean_unscaled[sel_idx] * chan_alpha
                matrix[n][win].append(scaled)
                for cname, val in zip(chan_names, scaled):
                    rows.append(dict(n=n, seed=s, window=win, component=cname, value=float(val)))

    fig, ax = plt.subplots(figsize=(11, 5.0))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(chan_names))]
    width = 0.38
    xticks = []; xlabels = []
    for i, n in enumerate(NS):
        for j, win in enumerate(["early", "late"]):
            arr = np.asarray(matrix[n][win])
            if arr.size == 0: continue
            mean = arr.mean(0)
            x = i * 1.2 + (j - 0.5) * width * 1.05
            pos_bottom = 0.0; neg_bottom = 0.0
            for k, val in enumerate(mean):
                bottom = pos_bottom if val >= 0 else neg_bottom
                ax.bar(x, val, width=width, bottom=bottom, color=colors[k],
                       edgecolor="white", linewidth=0.4,
                       label=chan_names[k] if (i == 0 and j == 0) else None)
                if val >= 0:
                    pos_bottom += val
                else:
                    neg_bottom += val
            xticks.append(x); xlabels.append(f"n={n}\n{win}")
            # Net marker
            net = mean.sum()
            ax.plot(x, net, marker="D", color="black", ms=5, zorder=5)
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Scaled component return (episode mean, summed over agents and steps)")
    ax.set_title("Reward decomposition (Full; early vs late; mean across 5 seeds)")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    # Add a black-diamond proxy handle for the Net return marker.
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="D", color="black", linestyle="",
                          markersize=6, label="Net return"))
    labels.append("Net return")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=5, frameon=False)
    fig.tight_layout()
    pdf, _ = save_fig(fig, "figure9_reward_decomposition_multiseed")
    # CSV: seed-level rows + mean
    df_rows = pd.DataFrame(rows)
    df_mean = df_rows.groupby(["n", "window", "component"], as_index=False)["value"].mean()
    df_mean["seed"] = "mean"
    out_csv = pd.concat([df_rows, df_mean[["n", "seed", "window", "component", "value"]]],
                         ignore_index=True)
    out_csv.to_csv(OUT_DIR / "figure9_reward_decomposition_multiseed_summary.csv",
                   index=False)
    report["figures"].append(("figure9", str(pdf), "OK"))
    report["reward_mapping"] = [
        {"name": c[0], "csv_channel": c[1], "alpha": c[2]} for c in ACTIVE_CHANNELS
    ]
    return fig


# ----------------------------- Figure 10 -----------------------------

def fig10_collision(data, report, window: int = 20):
    """Per-episode |agent-agent collision penalty| (named field: comp5), rolling
    mean over episodes per seed, then mean ± SD across seeds, per team size."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    idx5 = 5 - 1  # comp5 column in ep_comp_sums
    alpha5 = 10.0
    max_n_ep = 1000
    for n in NS:
        rolled = []
        for s in SEEDS:
            d = data.get((n, s))
            if d is None: continue
            ep_sums = d["ep_comp_sums"][:, idx5] * alpha5
            mag = np.abs(ep_sums)
            ser = pd.Series(mag).rolling(window, min_periods=1).mean().values
            rolled.append(ser[:max_n_ep])
        if not rolled: continue
        L = min(len(r) for r in rolled)
        mat = np.stack([r[:L] for r in rolled])
        mean = mat.mean(0); sd = mat.std(0, ddof=0)
        x = np.arange(L)
        ax.plot(x, mean, color=COLORS_N[n], lw=2, label=f"n={n}")
        ax.fill_between(x, mean - sd, mean + sd, color=COLORS_N[n], alpha=0.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Agent-agent collision penalty magnitude")
    ax.set_title(f"Agent-agent collision penalty over training "
                 f"(Full; {window}-episode rolling mean; band = SD across 5 seeds)")
    ax.set_ylim(bottom=0)
    ax.legend(); ax.grid(True, alpha=0.3)
    pdf, _ = save_fig(fig, "figure10_collision_penalty_multiseed")
    report["figures"].append(("figure10", str(pdf), "OK"))
    return fig


# ----------------------------- Figure 11 -----------------------------

def fig11_hungarian(data, report):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=False)
    for ax, n in zip(axes, NS):
        for label, key, color in [("Early (first 100 episodes)", "early_curve", COLORS_EARLY),
                                  ("Late (last 100 episodes)",   "late_curve",  COLORS_LATE)]:
            curves = [data[(n, s)][key] for s in SEEDS if (n, s) in data]
            t, m, sd = _curve_mean_std(curves, "hung_mean")
            if t.size == 0: continue
            ax.plot(t, m, color=color, lw=2, label=label)
            ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.2)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean Hungarian distance")
        ax.set_xlim(0, 500); ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Hungarian assignment distance (Full; mean across 5 seeds; band = SD)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf, _ = save_fig(fig, "figure11_hungarian_distance_multiseed")
    report["figures"].append(("figure11", str(pdf), "OK"))
    return fig


# ----------------------------- Figure 12 -----------------------------

def fig12_velocity(data, report):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.0), sharex=True)
    rows = [("vlin_mean", "Mean linear velocity"),
            ("vang_abs_mean", "Mean absolute angular velocity")]
    for r, (col, ylab) in enumerate(rows):
        for c, n in enumerate(NS):
            ax = axes[r, c]
            for label, key, color in [("Early (first 100 episodes)", "early_curve", COLORS_EARLY),
                                      ("Late (last 100 episodes)",   "late_curve",  COLORS_LATE)]:
                curves = [data[(n, s)][key] for s in SEEDS if (n, s) in data]
                t, m, sd = _curve_mean_std(curves, col)
                if t.size == 0: continue
                ax.plot(t, m, color=color, lw=2, label=label)
                ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.2)
            if r == 0:
                ax.set_title(f"n = {n}")
            if r == 1:
                ax.set_xlabel("Timestep")
            if c == 0:
                ax.set_ylabel(ylab)
            ax.set_xlim(0, 500); ax.grid(True, alpha=0.3)
            if r == 0 and c == 2:
                ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Velocity profiles (Full; mean across 5 seeds; band = SD)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf, _ = save_fig(fig, "figure12_velocity_profiles_multiseed")
    report["figures"].append(("figure12", str(pdf), "OK"))
    return fig


# ----------------------------- report -----------------------------

def write_report(data, report):
    detected_columns = [
        "episode_id", "timestep", "done_count",
        "hung_dist_agent{0..n-1}", "agent{i}_vel_lin", "agent{i}_vel_ang",
        "agent{i}_comp{1..9}",
    ]
    seeds_found = sorted({s for (_, s) in data.keys()})
    ns_found    = sorted({n for (n, _) in data.keys()})
    lines = []
    lines.append("# Multi-seed diagnostic plot report\n")
    lines.append(f"- runs root: `{find_runs_root()}`\n")
    lines.append(f"- output dir: `{OUT_DIR}`\n")
    lines.append(f"- detected seeds: {seeds_found}\n")
    lines.append(f"- detected team sizes: {ns_found}\n")
    lines.append(f"- detected reward modes: ['{MODE}'] (full only, per task spec)\n")
    lines.append("- source files used (15 per-step CSVs):\n")
    root = find_runs_root()
    for n in NS:
        for s in SEEDS:
            p = run_dir(root, n, s) / "rewards.csv"
            tag = "OK" if (n, s) in data else "MISSING"
            lines.append(f"    - n={n} seed={s}: `{p}` [{tag}]\n")
    lines.append("- detected log columns: " + ", ".join(f"`{c}`" for c in detected_columns) + "\n")
    lines.append("\n## Reward-component mapping used (revised, 8 active channels)\n")
    lines.append("| order | name | CSV channel | alpha |\n|---|---|---|---|\n")
    for i, (name, key, a) in enumerate(ACTIVE_CHANNELS, start=1):
        lines.append(f"| {i} | {name} | `{key}` | {a} |\n")
    lines.append("\nDropped: `comp3` (base / d_global, alpha=0).\n")
    lines.append("\n## Figures\n")
    for tag, path, status in report["figures"]:
        lines.append(f"- **{tag}**: `{path}` — {status}\n")
    lines.append("\n## Notes / inferred fields\n")
    lines.append("- Completion time T_e = first timestep with `done_count >= n`.\n")
    lines.append("- Success = max(done_count) over episode >= n (failed eps excluded from Fig 6, censored in Fig 7).\n")
    lines.append("- Hungarian distance read from logged `hung_dist_agent{i}` (mean over agents per step).\n")
    lines.append("- Velocities read from logged `agent{i}_vel_lin` / `agent{i}_vel_ang` (latter taken in absolute value).\n")
    lines.append("- Bands across SEEDS (not episodes) with ddof=0 SD.\n")
    (OUT_DIR / "multiseed_diagnostic_plot_report.md").write_text("".join(lines))


# ----------------------------- main -----------------------------

def main(force: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] Aggregating per-run CSVs ...")
    data = build_all(force=force)
    print(f"  loaded {len(data)} runs")
    report = {"figures": []}
    print("[2/3] Plotting figures ...")
    figs = {
        "figure6":  fig6_completion_time(data, report),
        "figure7":  fig7_cumulative(data, report),
        "figure8":  fig8_done_count(data, report),
        "figure9":  fig9_reward_decomp(data, report),
        "figure10": fig10_collision(data, report),
        "figure11": fig11_hungarian(data, report),
        "figure12": fig12_velocity(data, report),
    }
    print("[3/3] Writing report ...")
    write_report(data, report)
    print(f"Done. Outputs in {OUT_DIR}")
    return data, figs, report


if __name__ == "__main__":
    main(force="--force" in sys.argv)
