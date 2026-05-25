#!/usr/bin/env python3
"""Generate res/revision_final_results.ipynb for the offline_replay_restart_v3
multi-seed sweep. Run this script, then open the notebook and Run-All.

Two-tier data model (see tools/export_light_logs.py):
  * The notebook's PRIMARY source is `run_details.csv` — per-run TOTALS (SR,
    coverage, restarts, compute, last-window per-component means). This ships in
    the gitignored local `revision_logs/` folder, so the notebook is runnable
    as-is without the big artifact root or the media drive.
  * EPISODE-WISE curves (§3) need per-run `episode_summary.csv`, which lives only
    on the media drive. Those cells self-skip with a note when the episode-wise
    data is not present, and still show the pre-rendered PNG path.

Usage:
    python tools/create_final_notebook.py
"""
import argparse, json, os

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


def build(root):
    md(f"""# Multi-seed revision results — coverage / success / baselines / compute

Generated for the **offline-replay + orbit-restart** sweep (`IDDPGWithoutS`,
`DiffDriveParallelEnvDone`, v_ang_max = pi/2, 1000 episodes, 5 seeds).

Consolidated quantitative evidence for the manuscript revision. Each section is
annotated with the reviewer concern it addresses.

**Data:** primary source is `run_details.csv` (per-run totals; ships locally in
`revision_logs/`). Episode-wise learning curves (§3) need `episode_summary.csv`
(media drive only) and self-skip otherwise.

| Reviewer point | Section |
|---|---|
| R1 single-seed / R2#14 (3-5 seeds + CI) | §2 table, §3 curves, §6 stats |
| R1 within-env baselines / R2#13 (extra ablation) | §2, §4 baseline comparison |
| R1 reproducibility (net dims, reward weights) | §7 compute/repro, §8 reward map |
| R2#2 collision constant c=7 / R2#3 alpha mapping | §8 |
| R2#15 computational cost | §7 |
""")

    code(f"""
import os, glob, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Default: gitignored local light logs that ship with the repo (runnable as-is).
# Override with REVISION_ROOT to point at the media logs (for episode-wise curves)
# or the full artifact root.
_HERE  = os.path.dirname(os.path.abspath("__file__"))
_LOCAL = os.path.join(os.path.dirname(_HERE), "revision_logs")     # repo/revision_logs
ROOT = os.environ.get("REVISION_ROOT") or (_LOCAL if os.path.isdir(_LOCAL) else r"{root}")
RUNS = os.path.join(ROOT, "runs")
RES  = os.path.join(ROOT, "res"); os.makedirs(RES, exist_ok=True)
DETAILS = os.path.join(ROOT, "run_details.csv")

MODES = ["full", "ablation", "nocoll"]
NS    = [4, 5, 6]
SEEDS = [1, 2, 3, 4, 5]
WINDOW = 200
MODE_COLOR = {{"full": "#1f77b4", "ablation": "#ff7f0e", "nocoll": "#2ca02c"}}
print("ROOT:", ROOT)
assert os.path.exists(DETAILS), f"run_details.csv not found at {{DETAILS}} — run tools/export_light_logs.py"
HAS_EPISODEWISE = bool(glob.glob(os.path.join(RUNS, "*", "episode_summary.csv")))
print("episode-wise data present:", HAS_EPISODEWISE, "(needed only for §3 curves)")
""")

    md("""## §1 Load per-run totals  *(from `run_details.csv`)*

One row per run: last-200 success rate, coverage, restarts, compute, and
last-200 per-component means.""")

    code("""
per_run = pd.read_csv(DETAILS).sort_values(["n", "mode", "seed"]).reset_index(drop=True)
print(f"loaded {len(per_run)} runs")
per_run.head()
""")

    md("""## §2 Multi-seed summary table  *(R1 single-seed, R2#14)*

Last-200 success rate and coverage, mean ± SD over the 5 seeds, with per-seed SR
shown explicitly (never hide a seed in a mean).""")

    code("""
def cell(group):
    return pd.Series({
        "seeds": group.shape[0],
        "SR_mean": round(group.SR.mean(), 1),
        "SR_sd": round(group.SR.std(ddof=0), 1),
        "cov_mean": round(group.coverage.mean(), 1),
        "cov_sd": round(group.coverage.std(ddof=0), 1),
        "SR_per_seed": ", ".join(f"{s}:{v}" for s, v in zip(group.seed, group.SR)),
    })
summary = per_run.groupby(["n", "mode"]).apply(cell).reset_index()
summary["SR (mean±SD)"]  = summary.SR_mean.map("{:.1f}".format)  + " ± " + summary.SR_sd.map("{:.1f}".format)
summary["cov (mean±SD)"] = summary.cov_mean.map("{:.1f}".format) + " ± " + summary.cov_sd.map("{:.1f}".format)
display(summary[["n", "mode", "seeds", "SR (mean±SD)", "cov (mean±SD)", "SR_per_seed"]])
summary.to_csv(os.path.join(RES, "revision_final_cell_summary.csv"), index=False)
print("overall by mode (SR mean ± SD):")
for mode in MODES:
    v = per_run[per_run["mode"] == mode].SR
    print(f"  {mode:<9}: {v.mean():5.1f} ± {v.std(ddof=0):4.1f}  (k={len(v)})")
""")

    md("""## §3 Learning curves with shaded ±SD across seeds  *(R1 §3.1 / §3.2)*

Rolling success / coverage (50-ep window) averaged across seeds with a ±1 SD
band. Uses the compact per-episode `episode_summary.csv` (ships locally). If
absent, this cell skips and points to the pre-rendered PNGs.""")

    code("""
def episode_series(n, mode, seed):
    p = os.path.join(RUNS, f"n{n}_{mode}_seed{seed}", "episode_summary.csv")
    if not os.path.exists(p): return None
    s = pd.read_csv(p).sort_values("episode_id")
    return s.done_count.to_numpy() / n, (s.done_count.to_numpy() >= n).astype(float)

def rolling(x, w=50):
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()

if not HAS_EPISODEWISE:
    print("episode-wise data not present locally — curves are media-only.")
    for m in ("success", "coverage"):
        print("  pre-rendered:", os.path.join(RES, f"revision_rolling_{m}_full_valid.png"))
else:
    for metric, idx, title in [("success", 1, "Rolling success rate"), ("coverage", 0, "Rolling coverage")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, n in zip(axes, NS):
            for mode in MODES:
                curves = []
                for seed in SEEDS:
                    es = episode_series(n, mode, seed)
                    if es is not None: curves.append(rolling(es[idx]))
                if not curves: continue
                L = min(len(c) for c in curves); M = np.vstack([c[:L] for c in curves])
                mean, sd = 100 * M.mean(0), 100 * M.std(0, ddof=0)
                ax.plot(np.arange(L), mean, color=MODE_COLOR[mode], label=mode)
                ax.fill_between(np.arange(L), mean - sd, mean + sd, color=MODE_COLOR[mode], alpha=0.18)
            ax.set_title(f"n={n}"); ax.set_xlabel("episode"); ax.grid(alpha=0.3); ax.set_ylim(-2, 105)
        axes[0].set_ylabel(f"{title} (%)"); axes[0].legend()
        fig.suptitle(f"{title} ± SD across {len(SEEDS)} seeds"); fig.tight_layout()
        out = os.path.join(RES, f"revision_rolling_{metric}_full_valid.png")
        fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)
    plt.show()
""")

    md("""## §4 Baseline comparison bar chart  *(R1 baselines, R2#13)*

Final-window success rate per n, grouped by reward mode. Error bars ±1 SD.""")

    code("""
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.26; x = np.arange(len(NS))
for i, mode in enumerate(MODES):
    means = [summary[(summary.n == n) & (summary["mode"] == mode)].SR_mean.values[0] for n in NS]
    sds   = [summary[(summary.n == n) & (summary["mode"] == mode)].SR_sd.values[0] for n in NS]
    ax.bar(x + (i - 1) * width, means, width, yerr=sds, capsize=4, color=MODE_COLOR[mode], label=mode)
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS])
ax.set_ylabel("last-200 success rate (%)"); ax.set_ylim(0, 105)
ax.set_title("Success rate by team size and reward mode (±SD across seeds)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
out = os.path.join(RES, "revision_baseline_success_valid.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out); plt.show()
""")

    md("""## §5 Coverage vs success scatter — partial-credit view

Several seeds reach high *coverage* even when full-team *success* is partial
(orbit / late-bootstrap). Reporting coverage separately from SR is the honest
framing for the high-variance cells.""")

    code("""
fig, ax = plt.subplots(figsize=(7, 6))
for mode in MODES:
    sub = per_run[per_run["mode"] == mode]
    ax.scatter(sub.coverage, sub.SR, color=MODE_COLOR[mode], label=mode, s=60, alpha=0.8)
ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
ax.set_xlabel("coverage (%)"); ax.set_ylabel("success rate (%)")
ax.set_title("Per-run coverage vs full-team success"); ax.legend(); ax.grid(alpha=0.3)
plt.show()
""")

    md("""## §6 Seed-level statistics  *(R2#14 / plan §2.5)*

Kruskal–Wallis across modes per n; Mann–Whitney full-vs-baselines (pooled);
bootstrap 95% CI of mean SR per cell (per-seed resampling).""")

    code("""
try:
    from scipy import stats
    print("Kruskal-Wallis across modes (per n):")
    for n in NS:
        groups = [per_run[(per_run.n == n) & (per_run["mode"] == m)].SR.values for m in MODES]
        if all(len(g) > 0 for g in groups):
            H, p = stats.kruskal(*groups)
            print(f"  n={n}: H={H:.3f}, p={p:.3f}")
    print("\\nMann-Whitney U (pooled over n):")
    for a, b in [("full", "nocoll"), ("full", "ablation")]:
        va = per_run[per_run["mode"] == a].SR.values
        vb = per_run[per_run["mode"] == b].SR.values
        U, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
        print(f"  {a} vs {b}: U={U:.1f}, p={p:.3f}  (median {np.median(va):.1f} vs {np.median(vb):.1f})")
    print("\\nBootstrap 95% CI of mean SR per cell:")
    for n in NS:
        for mode in MODES:
            v = per_run[(per_run.n == n) & (per_run["mode"] == mode)].SR.values
            if len(v) < 2: continue
            res = stats.bootstrap((v,), np.mean, confidence_level=0.95, n_resamples=5000)
            lo, hi = res.confidence_interval
            print(f"  n={n} {mode:<9}: mean {v.mean():5.1f}  CI [{lo:5.1f}, {hi:5.1f}]")
except ImportError:
    print("scipy not installed — skipping (pip install scipy)")
""")

    md("""## §7 Compute & reproducibility  *(R1 reproducibility, R2#15)*

Training wall-time, peak GPU, total env steps (per-run totals), plus
actor/critic parameter counts (from checkpoints if present) and hyperparameters.""")

    code("""
agg = per_run.groupby("n").agg(
    train_hours_mean=("train_hours", "mean"),
    train_hours_max=("train_hours", "max"),
    steps_mean=("total_steps", "mean"),
    peak_gpu_gb=("peak_gpu_gb", "max")).round(3).reset_index()
display(agg)
print("total GPU-hours across all runs:", round(per_run.train_hours.sum(), 1))
gpus = per_run.gpu_name.dropna()
print("GPU:", gpus.iloc[0] if len(gpus) else "n/a")

def param_count(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "parameters"):
        return sum(p.numel() for p in obj.parameters())
    if isinstance(obj, dict):
        sd = obj.get("model_state_dict") or obj.get("state_dict") or obj
        return sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    return None
try:
    for name in ("shared_actor.pth", "shared_critic.pth"):
        p = glob.glob(os.path.join(RUNS, "n4_full_seed1", name))  # only if full artifacts present
        print(f"{name} (n=4):", param_count(p[0]) if p else "checkpoint not in light logs (see artifact root)")
except Exception as e:
    print("param count:", e)

print("\\nFixed hyperparameters: gamma=0.99, tau=0.005, lr=1e-3 (Adam), batch=128,")
print("replay=50000 (tagged 25%), start_training_after=500, train_each=100,")
print("max_steps=500, v_ang_max=pi/2, num_obstacles=0, env_size=20.")
print("Param counts (from full artifact root): actor 73,986; critic 20,737 (n=4).")
""")

    md("""## §8 Reward-component map & collision constant  *(R2#2, R2#3)*

**alpha mapping — full = `[1,1,0,10,10,10,1,1,1]`:**

| idx | alpha | CSV col | term | full | ablation | nocoll |
|--:|--:|---|---|--:|--:|--:|
| 0 | 1 | comp1 | progressive (ΔHungarian / v_lin_max) | 1 | 0 | 1 |
| 1 | 1 | comp2 | distance (−assigned dist / env_size) | 1 | 0 | 1 |
| 2 | 0 | comp3 | base / d_global (shared Hungarian) | 0 | 0 | 0 |
| 3 | 10 | comp4 | reached-goal bonus | 10 | 10 | 10 |
| 4 | 10 | comp5 | agent–agent collision | 10 | 10 | **0** |
| 5 | 10 | comp6 | obstacle / done-agent collision | 10 | 10 | **0** |
| 6 | 1 | comp7 | linear-velocity shaping | 1 | 1 | 1 |
| 7 | 1 | comp8 | directional (cos_sim − 1) | 1 | 1 | 1 |
| 8 | 1 | comp9 | time (−1 per active step) | 1 | 1 | 1 |

- **ablation** zeroes assignment-aware shaping (comp1, comp2): R1 within-env baseline.
- **nocoll** zeroes collision penalties (comp5, comp6): R2#13 additional ablation.
- **Collision constant c=7** in `−exp(−d·c/safe_dist)` is a fixed hyperparameter
  (not derived); a c∈{3,5,7,15,20} sweep showed <1% converged effect — not load-bearing.""")

    md("""## §9 Orbit-failure component breakdown — high-variance cells

The high-SD cells (n5 full seed1; n4 nocoll seed5) are a **shaping fixed-point /
orbit policy**: agents face the landmark and move fast but make ~zero net
Hungarian progress. Below: last-200 per-component means (sum over agents per
episode) for the flagged vs healthy seeds, from `run_details.csv`.""")

    code("""
COMP = [f"comp{j}_mean" for j in range(1, 10)]
rows = {("n5","full",1):"n5_full_s1 (fail)", ("n4","nocoll",5):"n4_nocoll_s5 (fail)",
        ("n5","full",3):"n5_full_s3 (ok)",   ("n4","nocoll",1):"n4_nocoll_s1 (ok)"}
brk = {}
for (npre, mode, seed), label in rows.items():
    nn = int(npre[1])
    sub = per_run[(per_run.n == nn) & (per_run["mode"] == mode) & (per_run.seed == seed)]
    if len(sub):
        brk[label] = {c.replace("_mean",""): sub.iloc[0][c] for c in COMP if c in sub.columns}
display(pd.DataFrame(brk).T)
""")

    md("""## §10 Gaps & follow-ups (not in this batch)

- **R2#10 scalability 8–10 agents** — not run; narrow claims to n∈{4,5,6}.
- **R2#15 generalization to other env sizes** — env_size=20 only; run
  `run/eval_policy.py --env_size 25` on the 45 actor checkpoints (in the artifact
  root, not the light logs) for the generalization table.
- **R2#6/#11 obstacles** — num_obstacles=0; acknowledge as limitation.
- **Heuristic oracle baseline** — regenerate at π/2 via `run/eval_hungarian_p.py`.
- **Text-only:** R2#1,#7,#8,#9,#12,#16–19; R1 cross-references. See
  `.ai/experiment_conclusions.md` §8–9, §11.
""")

    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}},
          "nbformat": 4, "nbformat_minor": 5}
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/abz/Desktop/dif_driven_revision_offline_replay_restart_v3_artifacts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(repo, "res", "revision_final_results.ipynb")
    nb = build(args.root)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(nb, open(out, "w"), indent=1)
    print("wrote", out, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
