#!/usr/bin/env python3
"""Build the single, comprehensive round-2 revision results notebook.

Generates ``res/round2_revision_results.ipynb`` — one notebook covering every
*experimental* reviewer point of round 2 (robotics-4306086):

  * Point 3  — MAPPO on-policy baseline (the one new experiment)
  * Point 2  — shaping as a sample-efficiency / stability mechanism (re-analysis)
  * Point 4  — n=5 underperformance diagnostic (re-analysis)
  * Point 5a — collision-c and reward-weight sensitivity (surface c-sweep)
  * Points 5d/5e — path length reported numerically

Text-only / reclaim points (1, 5b, 5c, 5f, QMIX/VDN) are author-owned and listed
but not computed here.

Data sources (read-only):
  revision_logs/            main 45-run sweep (ours) + eval/ (deterministic)
  revision_logs_mappo/      MAPPO baseline runs + eval/
  .ai/CONTEXT/data/sweep_coll_c_summary.json   collision-coefficient sweep

Every figure is shown inline *and* saved as a vector PDF to res/figures_round2/.

Run:
    python tools/create_round2_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 res/round2_revision_results.ipynb
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda t: cells.append(new_markdown_cell(t))
code = lambda s: cells.append(new_code_cell(s))

# =========================================================================== #
md(r"""# Round-2 revision — full experimental results

*Manuscript robotics-4306086, round 2. One notebook covering every experimental
reviewer point.* Companion analysis: `res/round2_revision_results.md`.

**Reviewer response map (experimental part).**

| Point | Ask | Response | Section |
|---|---|---|---|
| **3** | Add a modern MARL baseline (MAPPO/IPPO) | New MAPPO baseline, identical protocol | **§3** |
| **2** | Shaping not statistically significant | Reframe: shaping is a *sample-efficiency / stability* mechanism, measured | **§4** |
| **4** | n=5 underperformance (ablation OR diagnostic) | Diagnostic: outlier-driven, not an odd-n structural defect | **§5** |
| **5a** | c=7 and reward weights unjustified | Surface c-sweep + reward-mode sensitivity | **§6** |
| **5d** | Report path length L_e numerically | Path-length tables | **§7** |
| **5e** | Trajectory figs need quantitative path | Report path length (tortuosity optional) | **§7** |

Text-only / reclaim points — **author-owned, not computed here**: 1 (title),
5b (how weights set), 5c (generalization wording), 5f (MAPF framing), QMIX/VDN
(reclaimed as out-of-scope for a continuous-action task).

**Metric conventions.**
- *Training-window SR* = last-200-episode fraction with **all n** agents on a
  landmark (`done_count == n`), exploration noise **on**.
- *Coverage* = mean fraction of agents on a landmark (partial success).
- *Deterministic eval* = noise-off greedy rollout (Gaussian mean for MAPPO), 200 ep/seed.

Every figure is inline **and** saved as PDF to `res/figures_round2/`.""")

# --------------------------------------------------------------------------- #
code(r"""# %% setup --------------------------------------------------------------
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Resolve the repo root (dir containing revision_logs/) by walking up from CWD.
ROOT = Path.cwd()
while not (ROOT / "revision_logs").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / "revision_logs").exists(), f"revision_logs not found from {Path.cwd()}"

MAIN  = ROOT / "revision_logs"            # ours (45-run sweep) + eval
MAPPO = ROOT / "revision_logs_mappo"      # MAPPO baseline + eval
DATA  = ROOT / ".ai" / "CONTEXT" / "data" # c-sweep json
FIG   = ROOT / "res" / "figures_round2"
FIG.mkdir(parents=True, exist_ok=True)

NS, MODES, SEEDS = [4, 5, 6], ["full", "ablation", "nocoll"], [1, 2, 3, 4, 5]
COL  = {"Ours": "#2e86ab", "MADDPG": "#edae49", "MAPPO": "#d1495b"}
MCOL = {"full": "#2e86ab", "ablation": "#a23b72", "nocoll": "#3c887e"}

def fig_show(fig, name):
    "Save a vector PDF and render inline."
    fig.tight_layout()
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.show()
    print("saved", out.relative_to(ROOT))

def msd(vals):
    a = np.asarray(vals, float); return a.mean(), a.std()

# core tables
main_rd  = pd.read_csv(MAIN / "run_details.csv")
mappo_rd = pd.read_csv(MAPPO / "run_details.csv")
ours_eval  = pd.read_csv(MAIN / "eval" / "eval_summary.csv")
mappo_eval = pd.read_csv(MAPPO / "eval" / "eval_summary.csv")
mappo_eval = mappo_eval[mappo_eval.kind == "mappo"]

print("ROOT  :", ROOT)
print("main runs :", len(main_rd), " mappo runs:", len(mappo_rd))
print("figures ->", FIG.relative_to(ROOT))""")

# --------------------------------------------------------------------------- #
md(r"""## 1. Context — our method's performance (what the reviewers challenge)

Our method = **IDDPGWithoutS** (shared actor + shared critic), full reward.
Training-window SR/coverage (last-200) and deterministic eval (env20 train arena,
env25 = +25% larger) provide the reference all comparisons are made against.""")

code(r"""# %% ours headline ------------------------------------------------------
full = main_rd[main_rd["mode"] == "full"]
rows = []
for n in NS:
    g = full[full.n == n]
    sm, ss = msd(g.SR); cm, cs = msd(g.coverage)
    rows.append({"n": n, "SR% (train)": f"{sm:.1f} ± {ss:.1f}",
                 "coverage% (train)": f"{cm:.1f} ± {cs:.1f}",
                 "per-seed SR": ", ".join(f"{v:.0f}" for v in g.sort_values('seed').SR)})
ours_tab = pd.DataFrame(rows)

ev = ours_eval[ours_eval.kind == "policy"]
det = ev.pivot_table(index="n", columns="env", values="SR_mean")
det.columns = [f"det SR% env{c}" for c in det.columns]
display(ours_tab)
display(det.round(1).reset_index())""")

# =========================================================================== #
md(r"""## §3 · Point 3 — MAPPO on-policy baseline (the new experiment)

MAPPO trained under an **identical** protocol to ours (same env, reward, horizon,
parameter-sharing, 1000 episodes, 5 seeds, n∈{4,5,6}). Compared against ours and
the off-policy MADDPG baseline (concat-obs critic, 3 seeds).""")

code(r"""# %% frozen comparison baselines (train-window SR / coverage) -----------
# Ours from main_rd (full mode, read above). MADDPG embedded as frozen constants
# (final values, revision_logs_maddpg_obs/run_details.csv; 3 seeds).
OURS_SR  = {n: full[full.n == n].sort_values("seed").SR.tolist() for n in NS}
OURS_COV = {n: full[full.n == n].sort_values("seed").coverage.tolist() for n in NS}
MADDPG_SR  = {4: [21.5, 23.0, 0.0], 5: [0.0, 0.0, 0.0], 6: [13.5, 18.0, 0.5]}
MADDPG_COV = {4: [73.2, 75.1, 26.2], 5: [28.7, 27.8, 30.4], 6: [75.2, 81.2, 30.3]}
MAPPO_SR   = {n: mappo_rd[mappo_rd.n == n].sort_values("seed").SR.tolist() for n in NS}
MAPPO_COV  = {n: mappo_rd[mappo_rd.n == n].sort_values("seed").coverage.tolist() for n in NS}

sr_tab = pd.DataFrame([{
    "n": n,
    "Ours SR%":   f"{msd(OURS_SR[n])[0]:.1f} ± {msd(OURS_SR[n])[1]:.1f}",
    "MADDPG SR%": f"{msd(MADDPG_SR[n])[0]:.1f} ± {msd(MADDPG_SR[n])[1]:.1f}",
    "MAPPO SR%":  f"{msd(MAPPO_SR[n])[0]:.1f} ± {msd(MAPPO_SR[n])[1]:.1f}",
    "MAPPO seeds": ", ".join(f"{v:.0f}" for v in MAPPO_SR[n]),
} for n in NS])
sr_tab""")

code(r"""# %% head-to-head SR + coverage bars ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
x = np.arange(len(NS)); w = 0.26
for ax, data, ttl in [(axes[0], (OURS_SR, MADDPG_SR, MAPPO_SR), "Success rate"),
                      (axes[1], (OURS_COV, MADDPG_COV, MAPPO_COV), "Coverage")]:
    for i, (lab, d) in enumerate(zip(["Ours", "MADDPG", "MAPPO"], data)):
        means = [msd(d[n])[0] for n in NS]; sds = [msd(d[n])[1] for n in NS]
        ax.bar(x + (i - 1) * w, means, w, yerr=sds, capsize=3, color=COL[lab],
               label=lab, alpha=0.9)
        if ttl == "Success rate":
            for j, n in enumerate(NS):
                ax.scatter([x[j] + (i - 1) * w] * len(d[n]), d[n], s=11,
                           color="k", alpha=0.4, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS])
    ax.set_ylabel(f"{ttl} (%, train-window)"); ax.set_ylim(0, 105)
    ax.set_title(f"{ttl} by team size (mean ± SD)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig_show(fig, "p3_headtohead")""")

code(r"""# %% MAPPO learning curves ----------------------------------------------
def rolling_sr(csv, n, w=50):
    df = pd.read_csv(csv)
    s = (df["done_count"].values >= n).astype(float)
    return df["episode_id"].values, pd.Series(s).rolling(w, min_periods=1).mean().values * 100

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=True)
for ax, n in zip(axes, NS):
    curves = []
    for s in SEEDS:
        c = MAPPO / "runs" / f"mappo_n{n}_full_seed{s}" / "episode_summary.csv"
        if not c.exists():
            continue
        ep, r = rolling_sr(c, n); ax.plot(ep, r, color=COL["MAPPO"], alpha=0.25, lw=1)
        curves.append(r)
    if curves:
        L = min(map(len, curves))
        ax.plot(ep[:L], np.mean([c[:L] for c in curves], 0), color=COL["MAPPO"],
                lw=2.4, label="MAPPO mean")
    ax.axhline(msd(OURS_SR[n])[0], color=COL["Ours"], ls="--", lw=1.6, label="Ours (final)")
    ax.set_title(f"n = {n}"); ax.set_xlabel("episode"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
axes[0].set_ylabel("rolling SR (%, window 50)")
fig.suptitle("MAPPO training success rate (per seed + cross-seed mean)", y=1.02)
fig_show(fig, "p3_mappo_learning_curves")""")

code(r"""# %% MAPPO deterministic eval (env20 -> env25) --------------------------
me = mappo_eval[["env", "n", "seeds", "SR_mean", "SR_sd", "cov_mean",
                 "compT_mean", "path_mean"]].sort_values(["env", "n"])
display(me.reset_index(drop=True))

fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(len(NS)); w = 0.35
for i, env in enumerate([20, 25]):
    vals = [float(mappo_eval[(mappo_eval.env == env) & (mappo_eval.n == n)].SR_mean.iloc[0])
            if len(mappo_eval[(mappo_eval.env == env) & (mappo_eval.n == n)]) else np.nan
            for n in NS]
    ax.bar(x + (i - 0.5) * w, vals, w, label=f"env{env}",
           color=["#2e86ab", "#a23b72"][i], alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS]); ax.set_ylim(0, 105)
ax.set_ylabel("MAPPO deterministic SR (%)")
ax.set_title("MAPPO greedy eval: train arena (env20) vs larger (env25)")
ax.legend(frameon=False)
fig_show(fig, "p3_mappo_eval")""")

md(r"""**Point 3 takeaway.** MAPPO beats the off-policy MADDPG baseline but stays far
below ours at every team size (best MAPPO 37% vs ours 98% at n=4), is highly
seed-sensitive, produces an even weaker *greedy* policy, and collapses on the larger
env25 arena. → supports that the task needs our method, not a stock on-policy MARL
algorithm. *(All 30 deterministic-eval jobs complete — symmetric 5 seeds per cell.)*""")

# =========================================================================== #
md(r"""## §4 · Point 2 — shaping is a sample-efficiency / stability mechanism

The reviewer pre-rejected the bare "shaping affects training dynamics" line. We
**measure** it: asymptotic SR is shaping-invariant (already shown), but full shaping
reaches competence in fewer episodes and bootstraps more reliably. Modes:
`full` (all 9 components), `ablation` (drop progress/distance shaping), `nocoll`
(drop collision terms).""")

code(r"""# %% 4a asymptotic SR by mode x n (should overlap = not significant) -----
def mode_tab(metric):
    rows = []
    for n in NS:
        r = {"n": n}
        for m in MODES:
            g = main_rd[(main_rd.n == n) & (main_rd["mode"] == m)]
            mm, ss = msd(g[metric]); r[m] = f"{mm:.1f} ± {ss:.1f}"
        rows.append(r)
    return pd.DataFrame(rows)

print("Asymptotic training-window SR by mode (mean ± SD over 5 seeds):")
display(mode_tab("SR"))""")

code(r"""# %% 4b sample-efficiency: episodes to rolling-SR >= 80% ----------------
def episodes_to_thr(csv, n, thr=0.8, w=50):
    df = pd.read_csv(csv)
    s = (df["done_count"].values >= n).astype(float)
    roll = pd.Series(s).rolling(w, min_periods=w).mean().values
    idx = np.where(roll >= thr)[0]
    return int(df["episode_id"].values[idx[0]]) if len(idx) else np.nan

def window_sr(csv, n, lo, hi):
    df = pd.read_csv(csv)
    mk = (df["episode_id"] >= lo) & (df["episode_id"] < hi)
    return 100 * (df.loc[mk, "done_count"].values >= n).mean()

eff_rows, ew_rows = [], []
for n in NS:
    for m in MODES:
        e2t, ew, reached = [], [], 0
        for s in SEEDS:
            c = MAIN / "runs" / f"n{n}_{m}_seed{s}" / "episode_summary.csv"
            if not c.exists():
                continue
            t = episodes_to_thr(c, n)
            if not np.isnan(t):
                e2t.append(t); reached += 1
            ew.append(window_sr(c, n, 200, 400))
        eff_rows.append({"n": n, "mode": m,
                         "median ep→80%": int(np.median(e2t)) if e2t else np.nan,
                         "seeds reaching 80%": f"{reached}/{len(SEEDS)}"})
        ew_rows.append({"n": n, "mode": m, "early SR% (ep200-400)": round(np.mean(ew), 1)})
eff = pd.DataFrame(eff_rows); ew = pd.DataFrame(ew_rows)
print("Sample efficiency — episodes to reach rolling-SR >= 80%:")
display(eff)
print("Early-window mean SR (episodes 200-400):")
display(ew.pivot(index="n", columns="mode", values="early SR% (ep200-400)")[MODES].reset_index())""")

code(r"""# %% 4c plots: early-window SR + episodes-to-threshold by mode ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
x = np.arange(len(NS)); w = 0.26
ewp = ew.pivot(index="n", columns="mode", values="early SR% (ep200-400)")
for i, m in enumerate(MODES):
    axes[0].bar(x + (i - 1) * w, ewp[m].values, w, color=MCOL[m], label=m, alpha=0.9)
axes[0].set_xticks(x); axes[0].set_xticklabels([f"n={n}" for n in NS])
axes[0].set_ylabel("early SR % (ep 200–400)"); axes[0].set_ylim(0, 105)
axes[0].set_title("Early-training success rate by mode"); axes[0].legend(frameon=False)

e2tp = eff.pivot(index="n", columns="mode", values="median ep→80%")
for i, m in enumerate(MODES):
    axes[1].bar(x + (i - 1) * w, e2tp[m].values, w, color=MCOL[m], label=m, alpha=0.9)
axes[1].set_xticks(x); axes[1].set_xticklabels([f"n={n}" for n in NS])
axes[1].set_ylabel("median episodes to SR≥80%  (lower=faster)")
axes[1].set_title("Sample efficiency by mode"); axes[1].legend(frameon=False)
fig_show(fig, "p2_sample_efficiency")""")

md(r"""**Point 2 takeaway.** Final SR overlaps across modes (shaping is *not* an
asymptotic-success lever — consistent with §9b), **but** full shaping reaches the
80% competence threshold in fewer episodes and sustains higher early-window SR.
The contribution is therefore a measured **sample-efficiency / training-stability**
mechanism, not an asymptotic-success claim.""")

# =========================================================================== #
md(r"""## §5 · Point 4 — n=5 underperformance is outlier-driven, not structural

Diagnostic on existing data. The elevated n=5 SD is driven by a **single** seed
(s1) caught in the orbit fixed-point basin, not an odd-n structural defect.""")

code(r"""# %% 5a per-seed SR + SD with/without the outlier ------------------------
ps = full.pivot_table(index="seed", columns="n", values="SR")
display(ps.rename_axis(None).reset_index().rename(columns={"index": "seed"}))

n5 = full[full.n == 5].sort_values("seed")
all_sd  = n5.SR.std(ddof=0)
no_s1   = n5[n5.seed != 1].SR
print(f"n=5 full SR: {list(n5.SR.round(1))}")
print(f"  with s1 : mean {n5.SR.mean():.1f}, SD {all_sd:.1f}")
print(f"  drop s1 : mean {no_s1.mean():.1f}, SD {no_s1.std(ddof=0):.1f}  "
      f"(s1={n5[n5.seed==1].SR.iloc[0]:.1f}% is the lone late-bootstrap outlier)")""")

code(r"""# %% 5b per-component last-200 means by n (full) — extend the §9b table --
comp_cols = [f"comp{i}_mean" for i in range(1, 10)]
labels = ["1 progress", "2 distance", "3 d_global", "4 reach-goal",
          "5 agent-coll", "6 obst-coll", "7 direction", "8 v_lin", "9 time"]
cb = full.groupby("n")[comp_cols].mean().round(1); cb.columns = labels
print("Mean per-episode reward components by team size (full mode):")
display(cb)

fig, ax = plt.subplots(figsize=(8.2, 4.2))
x = np.arange(len(NS))
ax.scatter(np.repeat(x, 5) + np.random.default_rng(0).uniform(-0.08, 0.08, len(NS) * 5),
           [v for n in NS for v in full[full.n == n].SR], s=28, color=COL["Ours"],
           alpha=0.7, zorder=3, label="per-seed SR")
ax.plot(x, [full[full.n == n].SR.mean() for n in NS], "k_", ms=26, mew=2, label="mean")
ax.annotate("n=5 seed 1 (low-success)", xy=(1, 49.5), xytext=(1.15, 35),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS]); ax.set_ylim(0, 105)
ax.set_ylabel("training-window SR (%)")
ax.set_title("Per-seed SR by team size (full mode)")
ax.legend(frameon=False)
fig_show(fig, "p4_n5_diagnostic")""")

md(r"""**Point 4 takeaway.** Dropping the single s1 outlier collapses the n=5 SD from
~18 to a level comparable with n=4/n=6, and the component breakdown shows no odd-n
structural anomaly. The mechanism is the orbit fixed-point basin (hit by one seed),
not an odd-team-size defect — a diagnostic conclusion drawn from existing data, no
retraining.""")

# =========================================================================== #
md(r"""## §6 · Point 5a — collision-c and reward-weight sensitivity

The collision coefficient `c=7` and the 10:1 coverage:progress weighting are
justified post-hoc by (i) the existing c-sweep and (ii) the reward-mode ablations
(§4) as a coarse weight-sensitivity study.""")

code(r"""# %% c-sweep summary -----------------------------------------------------
csweep = json.loads((DATA / "sweep_coll_c_summary.json").read_text())
rows = []
for n in NS:
    d = csweep[str(n)]
    ref = d["z_score"]                      # c = 7 (used)
    s5  = d["rows"]["5.0"]["score"]
    s15 = d["rows"]["15.0"]["score"]
    rows.append({"n": n, "score c=5": round(s5, 1), "score c=7 (used)": round(ref, 1),
                 "score c=15": round(s15, 1),
                 "Δ% c5 vs c7": round(100 * (s5 - ref) / abs(ref), 1),
                 "Δ% c15 vs c7": round(100 * (s15 - ref) / abs(ref), 1)})
csw = pd.DataFrame(rows)
display(csw)

fig, ax = plt.subplots(figsize=(7, 4.2))
for n in NS:
    d = csweep[str(n)]
    cs = [5, 7, 15]
    ys = [d["rows"]["5.0"]["score"], d["z_score"], d["rows"]["15.0"]["score"]]
    ax.plot(cs, ys, "o-", label=f"n={n}")
ax.set_xlabel("collision coefficient c"); ax.set_ylabel("converged score")
ax.set_title("Collision-coefficient sensitivity (c=7 used)")
ax.legend(frameon=False)
fig_show(fig, "p5a_csweep")""")

md(r"""**Point 5a takeaway.** Around the chosen `c=7`, the converged score changes
little for n=4 and n=5 (single-digit % for c=5; n=6's c=15 point is noisy, large
error bar). Together with the three reward-mode ablations (§4) — which leave
asymptotic SR statistically unchanged — this is the requested sensitivity evidence
for the hand-set weights (the *how-they-were-set* wording is the author's 5b note).""")

# =========================================================================== #
md(r"""## §7 · Points 5d / 5e — path length reported numerically

Eq.(10) path length `L_e` from the deterministic eval, ours vs the greedy-heuristic
reference, on env20 and env25. Reporting path length satisfies both 5d (report the
number) and 5e (quantitative trajectory metric for Figs 14–16); tortuosity is an
optional extra, not required for the minimal plan.""")

code(r"""# %% path-length table + bars -------------------------------------------
pl = ours_eval[["kind", "env", "n", "path_mean", "compT_mean"]].copy()
tab = pl.pivot_table(index="n", columns=["kind", "env"], values="path_mean")
print("Mean path length L_e (env units):")
display(tab.round(1))

fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(len(NS)); w = 0.35
pol = ours_eval[ours_eval.kind == "policy"]
for i, env in enumerate([20, 25]):
    vals = [float(pol[(pol.env == env) & (pol.n == n)].path_mean.iloc[0]) for n in NS]
    ax.bar(x + (i - 0.5) * w, vals, w, label=f"ours env{env}",
           color=["#2e86ab", "#a23b72"][i], alpha=0.9)
heur = ours_eval[(ours_eval.kind == "heuristic") & (ours_eval.env == 20)]
ax.plot(x, [float(heur[heur.n == n].path_mean.iloc[0]) for n in NS], "k^--",
        label="heuristic (env20)")
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS])
ax.set_ylabel("mean path length L_e"); ax.set_title("Path length by team size")
ax.legend(frameon=False)
fig_show(fig, "p5de_path_length")""")

md(r"""**Points 5d/5e takeaway.** Path length is now reported numerically (table +
figure) for ours (env20 and env25) against the heuristic lower bound; it grows with
team size and arena size, consistent with the completion-time trend. These values go
into the Fig 14–16 captions.""")

# =========================================================================== #
md(r"""## 8. Summary & figure index""")

code(r"""# %% compact summary table ----------------------------------------------
def c(d, n): m, s = msd(d[n]); return f"{m:.1f} ± {s:.1f}"
def ev_sr(df, env, n, kind=None):
    q = df[(df.env == env) & (df.n == n)]
    if kind is not None: q = q[q.kind == kind]
    return f"{q.SR_mean.iloc[0]:.1f}" if len(q) else "—"

summary = pd.DataFrame([{
    "n": n,
    "Ours SR% (train)":   c(OURS_SR, n),
    "MADDPG SR% (train)": c(MADDPG_SR, n),
    "MAPPO SR% (train)":  c(MAPPO_SR, n),
    "Ours det SR env20":  ev_sr(ours_eval, 20, n, "policy"),
    "MAPPO det SR env20": ev_sr(mappo_eval, 20, n),
    "MAPPO det SR env25": ev_sr(mappo_eval, 25, n),
} for n in NS])
summary.to_csv(ROOT / "res" / "round2_summary_table.csv", index=False)
print("wrote res/round2_summary_table.csv")
display(summary)

print("\nFigures (res/figures_round2/):")
for p in sorted(FIG.glob("*.pdf")):
    print("  ", p.name)""")

md(r"""**Provenance.** Ours = `revision_logs/` (45-run sweep + eval). MAPPO =
`revision_logs_mappo/` (raw archived on media `/media/abz/Z7S/dif_driven_logs_mappo/`).
MADDPG numbers are frozen constants from `revision_logs_maddpg_obs/run_details.csv`.
c-sweep from `.ai/CONTEXT/data/sweep_coll_c_summary.json`. Training-window SR
includes exploration noise (disclosed); deterministic eval is reported separately.
Full written analysis: `res/round2_revision_results.md`.""")

# =========================================================================== #
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = "res/round2_revision_results.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
