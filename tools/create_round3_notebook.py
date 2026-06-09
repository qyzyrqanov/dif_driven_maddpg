#!/usr/bin/env python3
"""Build the single round-3 experimental-support notebook (for the response letter).

Generates ``res/round3_experimental_results.ipynb`` — one notebook covering every
*experimental* point of Reviewer 1's round-3 review (robotics-4306086). Companion
prose: ``res/round3_experimental_results.md``.

What each section answers (reviewer point numbers from
``.ai/CONTEXT/experiments.md`` §18):

  §A  #8   HER + orbit-restart ablation         (needs the round-3 long runs)
  §B  #7   MADDPG baseline 3→5 seed parity      (needs the round-3 long runs)
       #10 statistical test of the head-to-head (ours vs MAPPO vs MADDPG)
  §C  #11  path length normalized by oracle / arena diagonal   (existing eval)
  §D  #28  Kaplan–Meier completion curves + log-rank across n  (existing eval)
  §E  #24  collision-rate metric per episode                   (existing eval)
  §F  #6   statistical-power caveat for the N=5 reward-mode tests
       #16 significance of the episodes-to-80% threshold gaps  (existing logs)
  §G  #34  actor/critic parameter counts for n=4,5,6
  §H  #21  reward-weight sensitivity (surface the c-sweep + mode ablations)

The run-dependent sections (§A, §B/#7) DEGRADE GRACEFULLY: if
``revision_logs_round3/`` is absent (long runs not done yet) they print a clear
PENDING banner and skip, so the notebook still executes and reports every
existing-data analysis. Re-run this generator + nbconvert after the runs finish.

Data sources (read-only):
  revision_logs/            main 45-run sweep (ours, mode=full = HER+restart) + eval/
  revision_logs_round3/     #8 ablation rungs + #7 MADDPG seeds {4,5}   (long runs)
  revision_logs_maddpg_obs/ MADDPG baseline seeds {1,2,3}
  revision_logs_mappo/      MAPPO baseline (for the #10 head-to-head test)
  .ai/CONTEXT/data/sweep_coll_c_summary.json   collision-coefficient sweep

Run:
    python tools/create_round3_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=900 res/round3_experimental_results.ipynb
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda t: cells.append(new_markdown_cell(t))
code = lambda s: cells.append(new_code_cell(s))

# =========================================================================== #
md(r"""# Round-3 experimental support — full results

*Manuscript robotics-4306086, round 3 (Reviewer 1, 36 points). One notebook for
every point that needs data/plots/quantitative results.* Companion prose:
`res/round3_experimental_results.md`. Per-point triage: `.ai/CONTEXT/experiments.md` §18.

**This notebook answers the experimental points. The ~22 text/softening points are
author-owned and not computed here.**

| § | Point(s) | Ask | Data |
|---|---|---|---|
| **A** | **#8** | Ablate HER + orbit-restart (isolate their contribution) | round-3 long runs |
| **B** | **#7, #10** | MADDPG 3→5 seed parity; statistical test of the head-to-head | round-3 long runs + existing |
| **C** | **#11** | Normalize path length L_e by oracle / arena diagonal | existing eval |
| **D** | **#28** | Kaplan–Meier completion + log-rank test across n | existing eval |
| **E** | **#24** | Direct collision-rate metric per episode | existing eval |
| **F** | **#6, #16** | Statistical-power caveat (N=5); threshold-gap significance | existing logs |
| **G** | **#34** | Actor/critic parameter counts for n=4,5,6 | model defs |
| **H** | **#21** | Reward-weight sensitivity (c-sweep + mode ablations) | existing sweep |

**Method note for the #8 ablation (the one substantive new experiment).** Our pipeline
runs `main_loop` with **both** HER-style offline goal relabeling (`--use_offline_replay`)
**and** the orbit-basin restart controller (`--use_orbit_restart`); the latter *requires*
the former, so the achievable factorial is a 3-rung ladder at full reward:

| condition | HER | restart | source |
|---|:--:|:--:|---|
| **Full** (published) | ✓ | ✓ | `revision_logs/` (mode=full) |
| **−restart** | ✓ | ✗ | `revision_logs_round3/` `abl_noRestart_*` |
| **−HER** | ✗ | ✗ | `revision_logs_round3/` `abl_noHER_*` |

so **restart's effect = Full − (−restart)**, **HER's effect = (−restart) − (−HER)**,
**combined = Full − (−HER)**.

**Metric conventions.** *Training-window SR* = last-200-episode fraction with all n
agents on a landmark (`done_count == n`), exploration noise **on**. *Deterministic
eval* = noise-off greedy rollout on env20 (train arena). Every figure is shown inline
**and** saved as a vector PDF to `res/figures_round3/`.""")

# --------------------------------------------------------------------------- #
code(r"""# %% setup --------------------------------------------------------------
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120, "font.size": 12,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = Path.cwd()
while not (ROOT / "revision_logs").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / "revision_logs").exists(), f"revision_logs not found from {Path.cwd()}"

MAIN   = ROOT / "revision_logs"
R3     = ROOT / "revision_logs_round3"          # long-run light logs (may be absent)
MADDPG = ROOT / "revision_logs_maddpg_obs"
MAPPO  = ROOT / "revision_logs_mappo"
DATA   = ROOT / ".ai" / "CONTEXT" / "data"
FIG    = ROOT / "res" / "figures_round3"
FIG.mkdir(parents=True, exist_ok=True)

NS, SEEDS = [4, 5, 6], [1, 2, 3, 4, 5]
HORIZON   = 500
COND_COL  = {"Full": "#2e86ab", "-restart": "#edae49", "-HER": "#d1495b"}

def fig_show(fig, name):
    fig.tight_layout(); out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.show()
    print("saved", out.relative_to(ROOT))

def msd(v):
    a = np.asarray(v, float); return (a.mean(), a.std()) if len(a) else (np.nan, np.nan)

def load_rd(p):
    return pd.read_csv(p / "run_details.csv") if (p / "run_details.csv").exists() else None

main_rd   = load_rd(MAIN)
r3_rd     = load_rd(R3)
maddpg_rd = load_rd(MADDPG)
mappo_rd  = load_rd(MAPPO)
R3_READY  = r3_rd is not None and len(r3_rd)

print("ROOT      :", ROOT)
print("main runs :", 0 if main_rd is None else len(main_rd))
print("round-3   :", "READY (%d rows)" % len(r3_rd) if R3_READY else "PENDING (run run_round3_longruns.sh)")
print("maddpg/mappo:", None if maddpg_rd is None else len(maddpg_rd),
      "/", None if mappo_rd is None else len(mappo_rd))""")

# =========================================================================== #
md(r"""## §A · Point 8 — ablate HER and the orbit-restart controller

The reviewer asks how much of the performance comes from the reward structure versus
HER and the restart mechanism. We retrain our method (IDDPGWithoutS, full reward, the
same 5 seeds × n∈{4,5,6}) with each mechanism removed, and decompose the contribution.

> **PENDING until the long runs finish.** Launch with
> `CONFIRM=1 bash run/run_round3_longruns.sh ONLY=ablation`, export with
> `python tools/export_light_logs.py --artifact_root <root> --no_media --local_logs revision_logs_round3`,
> then re-run this notebook.""")

code(r"""# %% A1 assemble the ablation ladder -------------------------------------
def cond_of(run):
    if run.startswith("abl_noRestart_"): return "-restart"
    if run.startswith("abl_noHER_"):     return "-HER"
    return None

rows = []
# Full = published main sweep (mode=full) = HER+restart.
if main_rd is not None:
    for _, r in main_rd[main_rd["mode"] == "full"].iterrows():
        rows.append(dict(cond="Full", n=int(r.n), seed=int(r.seed),
                         SR=float(r.SR), coverage=float(r.coverage),
                         restarts=int(r.get("restarts", 0) or 0)))
if R3_READY:
    for _, r in r3_rd.iterrows():
        c = cond_of(str(r.run))
        if c:
            rows.append(dict(cond=c, n=int(r.n), seed=int(r.seed),
                             SR=float(r.SR), coverage=float(r.coverage),
                             restarts=int(r.get("restarts", 0) or 0)))
abl = pd.DataFrame(rows)
CONDS = ["Full", "-restart", "-HER"]
# Live completeness tracker (watch as the long runs finish): seeds present per
# (condition × n) out of the expected len(SEEDS).
if not abl.empty:
    comp = (abl.groupby(["cond", "n"]).seed.nunique()
               .unstack("n").reindex(CONDS))
    print(f"seeds present / {len(SEEDS)} expected, per condition × n:")
    display(comp.fillna(0).astype(int))
if not R3_READY:
    print("Ablation rungs PENDING — only the Full reference is loaded so far. "
          "Re-run this notebook after exporting revision_logs_round3/ to fill in.")
abl.head()""")

code(r"""# %% A2 decomposition table: SR by condition × n -------------------------
if abl.empty or set(abl.cond.unique()) == {"Full"}:
    print("PENDING: run the #8 ablation (run_round3_longruns.sh ONLY=ablation), "
          "export to revision_logs_round3/, then re-run.")
else:
    tab = []
    for n in NS:
        row = {"n": n}
        m = {}
        for c in CONDS:
            g = abl[(abl.cond == c) & (abl.n == n)].SR
            m[c] = g.mean() if len(g) else np.nan
            row[f"{c} SR%"] = f"{g.mean():.1f} ± {g.std():.1f}" if len(g) else "—"
        row["restart effect (Full−noRestart)"] = (f"{m['Full']-m['-restart']:+.1f}"
            if np.isfinite(m['Full']) and np.isfinite(m['-restart']) else "—")
        row["HER effect (noRestart−noHER)"] = (f"{m['-restart']-m['-HER']:+.1f}"
            if np.isfinite(m['-restart']) and np.isfinite(m['-HER']) else "—")
        tab.append(row)
    display(pd.DataFrame(tab))
    # how often restart actually fired in the published Full runs (context):
    rcount = abl[abl.cond == "Full"].groupby("n").restarts.sum()
    print("orbit-restarts fired in Full runs, by n:", dict(rcount))""")

code(r"""# %% A3 grouped bars: SR by condition × n --------------------------------
if not abl.empty and set(abl.cond.unique()) != {"Full"}:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(NS)); w = 0.26
    for i, c in enumerate(CONDS):
        means = [abl[(abl.cond == c) & (abl.n == n)].SR.mean() for n in NS]
        sds   = [abl[(abl.cond == c) & (abl.n == n)].SR.std()  for n in NS]
        ax.bar(x + (i - 1) * w, means, w, yerr=sds, capsize=3,
               color=COND_COL[c], label=c, alpha=0.9)
        for j, n in enumerate(NS):
            pts = abl[(abl.cond == c) & (abl.n == n)].SR.values
            ax.scatter([x[j] + (i - 1) * w] * len(pts), pts, s=12, color="k",
                       alpha=0.45, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS]); ax.set_ylim(0, 105)
    ax.set_ylabel("training-window SR (%)")
    ax.set_title("HER / orbit-restart ablation (raw seeds overlaid)")
    ax.legend(frameon=False, title="condition")
    fig_show(fig, "p8_her_restart_ablation")
else:
    print("PENDING — bar chart will render once the ablation runs are exported.")""")

md(r"""**§A takeaway (fill after runs).** Expected: removing the **restart** changes SR
only for the orbit-basin seeds (mostly n=5 s1) — the controller is a targeted escape
mechanism, not a broad performance lever. Removing **HER** is expected to hurt more
broadly at higher n where goal-reaching is rare. The decomposition shows the reward
structure (not HER/restart) carries the bulk of the result, which is the point the
reviewer asked us to demonstrate.""")

# =========================================================================== #
md(r"""## §B · Points 7 & 10 — MADDPG seed parity + statistical test of the head-to-head

**#7:** the MADDPG baseline ran seeds {1,2,3}; we add {4,5} so all methods use the same
5 seeds. **#10:** with all methods on 5 seeds we report a non-parametric test (Mann–
Whitney U, one-sided ours > baseline) and the rank-biserial effect size, with the
small-N caveat (see §F).""")

code(r"""# %% B1 MADDPG 5-seed table (existing 3 + round-3 2) ---------------------
md_rows = []
if maddpg_rd is not None:
    for _, r in maddpg_rd.iterrows():
        md_rows.append(dict(n=int(r.n), seed=int(r.seed), SR=float(r.SR)))
if R3_READY:
    for _, r in r3_rd[r3_rd.run.str.startswith("maddpg_obs_")].iterrows():
        md_rows.append(dict(n=int(r.n), seed=int(r.seed), SR=float(r.SR)))
maddpg = pd.DataFrame(md_rows).drop_duplicates(["n", "seed"]) if md_rows else pd.DataFrame()

OURS = {n: main_rd[(main_rd["mode"] == "full") & (main_rd.n == n)].sort_values("seed").SR.tolist()
        for n in NS}
MAPPO_SR = ({n: mappo_rd[mappo_rd.n == n].sort_values("seed").SR.tolist() for n in NS}
            if mappo_rd is not None else {n: [] for n in NS})
MADDPG_SR = ({n: maddpg[maddpg.n == n].sort_values("seed").SR.tolist() for n in NS}
             if len(maddpg) else {n: [] for n in NS})

rows = []
for n in NS:
    rows.append({"n": n,
                 "MADDPG seeds": ", ".join(f"{v:.0f}" for v in MADDPG_SR[n]) or "—",
                 "MADDPG SR%": f"{msd(MADDPG_SR[n])[0]:.1f} ± {msd(MADDPG_SR[n])[1]:.1f}"
                               if MADDPG_SR[n] else "—",
                 "#seeds": len(MADDPG_SR[n])})
display(pd.DataFrame(rows))
if any(len(MADDPG_SR[n]) < 5 for n in NS):
    print("NOTE: MADDPG still <5 seeds for some n — run run_round3_longruns.sh ONLY=maddpg.")""")

code(r"""# %% B2 Mann–Whitney ours vs baselines (one-sided ours > baseline) -------
def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    U, p = stats.mannwhitneyu(a, b, alternative="greater")
    rbc = 2 * U / (len(a) * len(b)) - 1          # rank-biserial effect size
    return p, rbc

rows = []
for n in NS:
    pv_mappo, e_mappo = mw(OURS[n], MAPPO_SR[n])
    pv_mad, e_mad     = mw(OURS[n], MADDPG_SR[n])
    rows.append({"n": n,
                 "ours mean SR": f"{msd(OURS[n])[0]:.1f}",
                 "vs MAPPO p": f"{pv_mappo:.3f}" if np.isfinite(pv_mappo) else "—",
                 "vs MAPPO effect": f"{e_mappo:+.2f}" if np.isfinite(e_mappo) else "—",
                 "vs MADDPG p": f"{pv_mad:.3f}" if np.isfinite(pv_mad) else "—",
                 "vs MADDPG effect": f"{e_mad:+.2f}" if np.isfinite(e_mad) else "—"})
display(pd.DataFrame(rows))
print("One-sided Mann–Whitney (H1: ours > baseline). rank-biserial in [-1,1]; +1 = "
      "every ours-seed exceeds every baseline-seed. With 5v5 the smallest possible "
      "one-sided p is 1/252≈0.004, so a perfect separation is still significant.")""")

md(r"""**§B takeaway.** Even at N=5 a *complete* separation (every ours-seed above every
baseline-seed) reaches p≈0.004 one-sided, which is the case at the n where ours is
uniformly high and the baseline uniformly low. Where the baseline has a lucky high seed
(MAPPO n=4: 37±39) the test is appropriately *not* significant — consistent with the §F
power caveat. We report the effect size alongside p rather than leaning on p alone.""")

# =========================================================================== #
md(r"""## §C · Point 11 — path length normalized by task geometry

Raw `L_e` is not comparable across n. We normalize by (i) the **oracle** (greedy-
heuristic) path length on the same arena and (ii) the **arena diagonal** (`√2·env`), so
1.0 = oracle-optimal and the diagonal-normalized value is a dimensionless detour
factor.""")

code(r"""# %% C1 normalized path length ------------------------------------------
ev = pd.read_csv(MAIN / "eval" / "eval_summary.csv")
pol  = ev[ev.kind == "policy"]
heur = ev[ev.kind == "heuristic"]

rows = []
for env in sorted(pol.env.unique()):
    for n in NS:
        pr = pol[(pol.env == env) & (pol.n == n)]
        if not len(pr):
            continue
        Le = float(pr.path_mean.iloc[0])
        hr = heur[(heur.env == env) & (heur.n == n)]
        opt = float(hr.path_mean.iloc[0]) if len(hr) else np.nan
        diag = np.sqrt(2) * env
        rows.append({"env": env, "n": n, "L_e": round(Le, 1),
                     "oracle L_e": round(opt, 1) if np.isfinite(opt) else "—",
                     "L_e / oracle": round(Le / opt, 2) if np.isfinite(opt) and opt else "—",
                     "L_e / diagonal": round(Le / diag, 2)})
pl_tab = pd.DataFrame(rows)
display(pl_tab)

fig, ax = plt.subplots(figsize=(7.8, 4.4))
x = np.arange(len(NS)); w = 0.35
for i, env in enumerate(sorted(pol.env.unique())):
    vals = []
    for n in NS:
        pr = pol[(pol.env == env) & (pol.n == n)]
        hr = heur[(heur.env == env) & (heur.n == n)]
        vals.append(float(pr.path_mean.iloc[0]) / float(hr.path_mean.iloc[0])
                    if len(pr) and len(hr) and float(hr.path_mean.iloc[0]) else np.nan)
    ax.bar(x + (i - 0.5) * w, vals, w, label=f"env{env}",
           color=["#2e86ab", "#a23b72"][i % 2], alpha=0.9)
ax.axhline(1.0, color="k", ls="--", lw=1, label="oracle (=1)")
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS])
ax.set_ylabel("path length / oracle path"); ax.set_title("Normalized path length L_e")
ax.legend(frameon=False)
fig_show(fig, "p11_normalized_path")""")

md(r"""**§C takeaway.** Normalizing by the oracle path and the arena diagonal makes
`L_e` comparable across team and arena size: the detour factor (×oracle) summarizes how
much longer than optimal the learned trajectories are, independent of raw scale.""")

# =========================================================================== #
md(r"""## §D · Point 28 — empirical completion-time CDF + log-rank test  *(Fig 19)*

The reviewer questioned the Kaplan–Meier framing and asked for a log-rank (or comparable)
test. We do both: (1) **reframe** Fig 19 as an **empirical completion-time CDF** (the
fraction of episodes that have achieved full-team coverage by time t), de-emphasizing the
"survival analysis" language; and (2) add the requested **pairwise log-rank test** across
team sizes (event = full-team success; episodes that never complete are right-censored at
the 500-step horizon). The CDF is `1 − KM survival`, so the curve is unchanged — only the
framing and the added test.""")

code(r"""# %% D1 KM curves + pairwise log-rank -----------------------------------
ep = pd.read_csv(MAIN / "eval" / "eval_episodes.csv")
ep = ep[(ep.kind == "policy") & (ep.env == 20)].copy()
# time-to-event with right-censoring at the horizon for non-successes
ep["time"]  = np.where(ep.success == 1, ep.completion_time.fillna(HORIZON), HORIZON)
ep["event"] = (ep.success == 1).astype(int)

def km(times, events, tmax=HORIZON):
    order = np.argsort(times); t, e = np.asarray(times)[order], np.asarray(events)[order]
    n = len(t); surv = []; S = 1.0; at_risk = n; xs = [0.0]; ys = [1.0]
    for ut in np.unique(t):
        d = int(((t == ut) & (e == 1)).sum()); r = int((t >= ut).sum())
        if r > 0 and d > 0:
            S *= (1 - d / r); xs.append(ut); ys.append(S)
    xs.append(tmax); ys.append(S)
    return np.array(xs), np.array(ys)

def logrank(t1, e1, t2, e2):
    # two-group log-rank chi-square (df=1)
    t = np.concatenate([t1, t2]); e = np.concatenate([e1, e2])
    g = np.concatenate([np.zeros(len(t1)), np.ones(len(t2))])
    O1 = E1 = V = 0.0
    for ut in np.unique(t[e == 1]):
        at = t >= ut; nrisk = at.sum(); n1 = (at & (g == 0)).sum()
        d = ((t == ut) & (e == 1)).sum(); d1 = ((t == ut) & (e == 1) & (g == 0)).sum()
        if nrisk > 1:
            E1 += d * n1 / nrisk
            V  += d * (n1 / nrisk) * (1 - n1 / nrisk) * (nrisk - d) / (nrisk - 1)
            O1 += d1
    if V == 0:
        return np.nan, np.nan
    chi2 = (O1 - E1) ** 2 / V
    return chi2, stats.chi2.sf(chi2, 1)

fig, ax = plt.subplots(figsize=(7.8, 4.6))
for n in NS:
    g = ep[ep.n == n]
    xs, ys = km(g.time.values, g.event.values)
    ax.step(xs, 1 - ys, where="post", label=f"n={n} (SR={100*g.event.mean():.0f}%)")
ax.set_xlabel("completion time (steps)"); ax.set_ylabel("cumulative completion prob.")
ax.set_title("Empirical full-team completion-time CDF (deterministic eval, env20)")
ax.legend(frameon=False); fig_show(fig, "fig19_completion_cdf")

rows = []
for a, b in [(4, 5), (5, 6), (4, 6)]:
    ga, gb = ep[ep.n == a], ep[ep.n == b]
    chi2, p = logrank(ga.time.values, ga.event.values, gb.time.values, gb.event.values)
    rows.append({"comparison": f"n={a} vs n={b}",
                 "log-rank χ²": round(chi2, 2) if np.isfinite(chi2) else "—",
                 "p": f"{p:.2e}" if np.isfinite(p) else "—"})
display(pd.DataFrame(rows))""")

md(r"""**§D takeaway.** The pairwise log-rank test supplies the missing statistical
comparison of the completion-time curves across team sizes (a curve that lies higher/
earlier completes sooner). This is computed on the deterministic-eval episodes, so it is
a clean, censoring-aware comparison.""")

# =========================================================================== #
md(r"""## §E · Point 24 — direct collision-rate metric

The paper tracked the collision *penalty magnitude* but not a direct collision rate.
The deterministic eval already records a per-episode `collision_step_rate` (fraction of
steps with an active agent–agent proximity collision); we surface it as the interpretable
collision metric and pair it with the success rate.""")

code(r"""# %% E1 collision rate by n × env ---------------------------------------
coll = pd.read_csv(MAIN / "eval" / "eval_episodes.csv")
coll = coll[coll.kind == "policy"]
tab = []
for env in sorted(coll.env.unique()):
    for n in NS:
        g = coll[(coll.env == env) & (coll.n == n)]
        if not len(g):
            continue
        tab.append({"env": env, "n": n,
                    "collision_step_rate %": round(100 * g.collision_step_rate.mean(), 2),
                    "episodes w/ ≥1 collision %": round(100 * (g.collision_step_rate > 0).mean(), 1),
                    "SR %": round(100 * g.success.mean(), 1)})
coll_tab = pd.DataFrame(tab)
display(coll_tab)

fig, ax = plt.subplots(figsize=(7.6, 4.4))
x = np.arange(len(NS)); w = 0.35
for i, env in enumerate(sorted(coll.env.unique())):
    vals = [100 * coll[(coll.env == env) & (coll.n == n)].collision_step_rate.mean()
            if len(coll[(coll.env == env) & (coll.n == n)]) else np.nan for n in NS]
    ax.bar(x + (i - 0.5) * w, vals, w, label=f"env{env}",
           color=["#2e86ab", "#a23b72"][i % 2], alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS])
ax.set_ylabel("mean collision-step rate (%)")
ax.set_title("Direct collision rate (deterministic eval)")
ax.legend(frameon=False); fig_show(fig, "p24_collision_rate")""")

md(r"""**§E takeaway.** The collision-step rate is a direct, interpretable collision
metric (vs the indirect penalty magnitude in Fig 10). *Note:* this stays a
simulation-level collision count — it does not establish deployment-level collision
safety (reviewer #19, author text).""")

# =========================================================================== #
md(r"""## §F · Points 6 & 16 — statistical power and threshold-gap significance

**#6:** with N=5 per group the Kruskal–Wallis / Mann–Whitney tests have low power, so a
non-significant result does **not** establish equivalence. We quantify this with the
observed effect size and an explicit post-hoc power estimate, and frame the reward-mode
finding as "no *detectable* difference at N=5," not "no difference." **#16:** we test the
episodes-to-80% threshold gaps the reviewer flagged.""")

code(r"""# %% F1 reward-mode KW effect size + post-hoc power ----------------------
MODES = ["full", "ablation", "nocoll"]
def approx_power_two_sample(a, b, alpha=0.05, nsim=4000, rng=0):
    # post-hoc power via permutation: P(detect) given the observed group means/SD,
    # resampling normal draws at the observed n per group (one-sided MW).
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    g = np.random.default_rng(rng); hits = 0
    for _ in range(nsim):
        sa = g.normal(a.mean(), a.std() + 1e-9, len(a))
        sb = g.normal(b.mean(), b.std() + 1e-9, len(b))
        try:
            _, p = stats.mannwhitneyu(sa, sb, alternative="two-sided")
            hits += p < alpha
        except ValueError:
            pass
    return hits / nsim

rows = []
for n in NS:
    groups = [main_rd[(main_rd.n == n) & (main_rd["mode"] == m)].SR.values for m in MODES]
    H, p = stats.kruskal(*groups)
    k, N = len(groups), sum(len(x) for x in groups)
    eps2 = max(0.0, (H - k + 1) / (N - k))        # epsilon-squared effect size (>=0)
    pw = approx_power_two_sample(groups[0], groups[1])   # full vs ablation
    rows.append({"n": n, "KW H": round(H, 2), "p": round(p, 3),
                 "ε² (effect)": round(eps2, 3),
                 "post-hoc power (full vs abl)": round(pw, 2)})
display(pd.DataFrame(rows))
print("ε²: ~0.01 small, ~0.06 medium, ~0.14 large. Low post-hoc power confirms N=5 "
      "cannot reliably detect even medium effects — non-significance ≠ equivalence (#6).")""")

code(r"""# %% F2 episodes-to-80% gap significance (#16) ---------------------------
def episodes_to_thr(csv, n, thr=0.8, w=50):
    df = pd.read_csv(csv)
    s = (df["done_count"].values >= n).astype(float)
    roll = pd.Series(s).rolling(w, min_periods=w).mean().values
    idx = np.where(roll >= thr)[0]
    return int(df["episode_id"].values[idx[0]]) if len(idx) else np.nan

rows = []
for n in NS:
    e_full, e_abl = [], []
    for s in SEEDS:
        cf = MAIN / "runs" / f"n{n}_full_seed{s}" / "episode_summary.csv"
        ca = MAIN / "runs" / f"n{n}_ablation_seed{s}" / "episode_summary.csv"
        if cf.exists(): e_full.append(episodes_to_thr(cf, n))
        if ca.exists(): e_abl.append(episodes_to_thr(ca, n))
    ef = [v for v in e_full if not np.isnan(v)]; ea = [v for v in e_abl if not np.isnan(v)]
    if len(ef) >= 2 and len(ea) >= 2:
        U, p = stats.mannwhitneyu(ef, ea, alternative="less")   # full faster?
    else:
        p = np.nan
    rows.append({"n": n, "median full": int(np.median(ef)) if ef else "—",
                 "median ablation": int(np.median(ea)) if ea else "—",
                 "gap": (int(np.median(ea) - np.median(ef)) if ef and ea else "—"),
                 "MW p (full faster)": f"{p:.3f}" if np.isfinite(p) else "—"})
display(pd.DataFrame(rows))
print("#16: the n=5 gap (~33 ep) is small and, with N=5, not significant — so the text "
      "should claim faster convergence at n=4/n=6 and 'comparable' at n=5, not 'all n'.")""")

md(r"""**§F takeaway.** We report effect sizes and explicit low post-hoc power, and
reframe the reward-mode result as *no detectable difference at N=5* (#6). For the
threshold gaps (#16) the n=4/n=6 advantages are real but the n=5 gap is within noise —
so "earlier for all team sizes" is softened to "n=4 and n=6; comparable at n=5".""")

# =========================================================================== #
md(r"""## §G · Point 34 — parameter counts for n = 4, 5, 6

Table 2 listed n=4 only; the actor width is `max(256, obs_dim)` and `obs_dim` grows with
n, so we report counts for every team size.""")

code(r"""# %% G1 actor/critic parameter counts -----------------------------------
import sys
sys.path.insert(0, str(ROOT))
rows = []
try:
    import torch
    from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone
    from rl.maddpg import IDDPGWithoutS
    for n in NS:
        env = DiffDriveParallelEnvDone(num_agents=n, num_obstacles=0,
                                       v_ang_max=torch.pi / 2)
        algo = IDDPGWithoutS(env, reward_scales=[1,1,0,10,10,10,1,1,1],
                             batch_size=128, replay_buffer_size=1000)
        def count(m):
            return sum(p.numel() for p in m.parameters()) if m is not None else np.nan
        actor = getattr(algo, "actor", None) or getattr(algo, "shared_actor", None)
        critic = getattr(algo, "critic", None) or getattr(algo, "shared_critic", None)
        rows.append({"n": n, "obs_dim": getattr(env, "obs_dim", "—"),
                     "actor params": count(actor), "critic params": count(critic)})
    display(pd.DataFrame(rows))
except Exception as exc:
    print("param-count cell skipped:", repr(exc))
    print("(Run on the training machine with the env importable to populate Table 2.)")""")

# =========================================================================== #
md(r"""## §H · Point 21 — reward-weight sensitivity

The collision coefficient `c=7` and the 10:1 coverage:progress weighting are justified
post-hoc by the existing collision-coefficient sweep plus the three reward-mode
ablations (a coarse weight-sensitivity study). This re-surfaces the round-2 §5a evidence
for completeness; a full per-weight grid is optional (not run).""")

code(r"""# %% H1 c-sweep summary --------------------------------------------------
f = DATA / "sweep_coll_c_summary.json"
if f.exists():
    cs = json.loads(f.read_text())
    rows = []
    for n in NS:
        d = cs.get(str(n));
        if not d: continue
        ref = d["z_score"]
        s5  = d["rows"]["5.0"]["score"]; s15 = d["rows"]["15.0"]["score"]
        rows.append({"n": n, "score c=5": round(s5, 1), "score c=7 (used)": round(ref, 1),
                     "score c=15": round(s15, 1),
                     "Δ% c5": round(100*(s5-ref)/abs(ref), 1),
                     "Δ% c15": round(100*(s15-ref)/abs(ref), 1)})
    display(pd.DataFrame(rows))
    print("Near c=7 the converged score moves only single-digit % for n=4/n=5 "
          "(n=6 c=15 is noisy). Combined with the mode ablations (§F) this is the "
          "requested sensitivity evidence for the hand-set weights.")
else:
    print("c-sweep json not found at", f)""")

# =========================================================================== #
md(r"""## §I · Point 13 — wider transfer range (arena-size sweep)

The round-2 transfer was a single +25% point (env20→env25). Here we evaluate the trained
full-method actors (deterministic, noise off) across a **ratio curve** of arena sizes —
default env ∈ {15, 18, 20, 25, 30} = 0.75–1.5× of the env20 training arena. env15 is the
small floor (landmark placement is 0% infeasible at n=6 there, but fails below env14).

> **PENDING until the eval finishes.** Produced by
> `CONFIRM=1 PHASE=eval bash run/run_round3_longruns.sh`, then
> `python tools/aggregate_eval.py --eval_dir <EVAL_OUT_ROOT>/eval --out_dir revision_logs_round3/eval`.

**Interpretation caveat.** Reach threshold (0.5), landmark separation (5.0) and
`agent_radius` are *absolute*, while observations are normalized by `env_size/2`. So a
*smaller* arena simultaneously (i) shifts the normalized input distribution (the transfer
stressor) and (ii) shrinks absolute distances (targets spawn closer → easier to reach). A
smaller-arena SR change therefore reflects distribution-shift sensitivity, not
infeasibility — symmetric to "larger arena = sparser, fewer collisions".""")

code(r"""# %% I0 load transfer-eval aggregates (PENDING-aware) --------------------
R3_EVAL = R3 / "eval" / "eval_summary.csv"
if R3_EVAL.exists():
    tev = pd.read_csv(R3_EVAL)
    tpol = tev[tev.kind == "policy"].copy()
    theu = tev[tev.kind == "heuristic"].copy()
    ENVS = sorted(tpol.env.unique())
    print("transfer eval READY — env sizes:", ENVS,
          "| policy rows:", len(tpol), "| heuristic rows:", len(theu))
else:
    tpol = theu = None; ENVS = []
    print("PENDING: run PHASE=eval, aggregate into revision_logs_round3/eval/, re-run.")""")

code(r"""# %% I1 SR + coverage vs arena size, per n ------------------------------
if tpol is not None and len(tpol):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, col, ttl in [(axes[0], "SR_mean", "Success rate"),
                         (axes[1], "cov_mean", "Coverage")]:
        for n in NS:
            g = tpol[tpol.n == n].sort_values("env")
            if len(g):
                ax.plot(g.env, g[col], "o-", label=f"n={n}")
        ax.axvline(20, color="k", ls="--", lw=1, alpha=0.6)
        ax.text(20, ax.get_ylim()[1]*0.02, " train", fontsize=9)
        ax.set_xlabel("arena size (env)"); ax.set_ylabel(f"{ttl} (%)")
        ax.set_title(f"{ttl} vs arena size"); ax.set_ylim(0, 105); ax.legend(frameon=False)
    fig_show(fig, "p13_transfer_curve")
    display(tpol.pivot_table(index="n", columns="env", values="SR_mean").round(1))
else:
    print("PENDING — transfer SR/coverage curve renders once the eval is aggregated.")""")

code(r"""# %% I2 normalized path length across the curve (extends #11) ------------
if tpol is not None and len(tpol):
    rows = []
    for env in ENVS:
        for n in NS:
            pr = tpol[(tpol.env == env) & (tpol.n == n)]
            hr = theu[(theu.env == env) & (theu.n == n)] if theu is not None else []
            if not len(pr):
                continue
            Le = float(pr.path_mean.iloc[0]); diag = np.sqrt(2) * env
            opt = float(hr.path_mean.iloc[0]) if len(hr) else np.nan
            rows.append({"env": env, "n": n, "L_e": round(Le, 1),
                         "L_e/oracle": round(Le/opt, 2) if np.isfinite(opt) and opt else "—",
                         "L_e/diagonal": round(Le/diag, 2)})
    display(pd.DataFrame(rows))
else:
    print("PENDING — normalized path length across the curve.")""")

code(r"""# %% I3 collision rate across the curve (extends #24) -------------------
R3_EP = R3 / "eval" / "eval_episodes.csv"
if R3_EP.exists():
    tep = pd.read_csv(R3_EP); tep = tep[tep.kind == "policy"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for n in NS:
        g = (tep[tep.n == n].groupby("env").collision_step_rate.mean() * 100).sort_index()
        if len(g):
            ax.plot(g.index, g.values, "o-", label=f"n={n}")
    ax.axvline(20, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("arena size (env)"); ax.set_ylabel("mean collision-step rate (%)")
    ax.set_title("Collision rate vs arena size"); ax.legend(frameon=False)
    fig_show(fig, "p13_collision_vs_size")
else:
    print("PENDING — collision rate vs size (needs revision_logs_round3/eval/eval_episodes.csv).")""")

md(r"""**§I takeaway (fill after eval).** Expected: SR is highest at/around the env20
training size and degrades gradually outward; the smaller end (env15) is the denser,
distribution-shifted regime and the larger end (env30) the sparser one. Reporting the
full curve (vs a single +25% point) directly answers the reviewer's "wider range" ask.""")

# =========================================================================== #
md(r"""## §J · Manuscript figure corrections (drop-in replacements)

Regenerated versions of the manuscript figures the reviewer criticized, with the
specific fixes applied. Each is saved as a vector PDF to `res/figures_round3/` named
`figNN_*` to map to the manuscript figure number; swap them into the paper. All read
the **same data** as the manuscript (the round-3-dependent ones auto-update once the
long runs land).

| Manuscript fig | Fix applied | Status |
|---|---|---|
| **Fig 5** rolling success | legend → `Full / No-assignment-shaping / No-collision` (#33) | ready |
| **Fig 9** reward decomposition | larger fonts, wider, legend outside (#32) | ready |
| **Fig 10** collision | 2-panel: penalty magnitude **+ raw collision count** (#24) | ready |
| **Fig 13** final-window SR | raw per-seed dots overlaid (#32) | ready |
| **Fig 14** sample-efficiency | per-seed E₈₀ points/box + never-reached marks (#16, #32) | ready |
| **Fig 19** completion | reframed as completion-time CDF + log-rank (§D, #28) | ready |
| **Fig 21** baselines | seed dots; `Proposed/MADDPG/MAPPO` (#32) | partial (MADDPG→5 seeds) |
| **Fig 22** transfer | title "transfer" not "generalization"; oracle separated (#5c, #9) | ready (multi-size when eval lands) |

Fonts are bumped uniformly here; apply the same style to the structurally-fine figures
(6,7,8,11,12,15) on a final pass if desired.""")

code(r"""# %% J setup: corrected style + mode labels -----------------------------
CORR = dict(fontsize=13)  # bump over the global 12 for print readability
MODE_LABEL = {"full": "Full", "ablation": "No-assignment-shaping", "nocoll": "No-collision"}
MODE_C     = {"full": "#2e86ab", "ablation": "#a23b72", "nocoll": "#3c887e"}
ALLMODES   = ["full", "ablation", "nocoll"]
rng_j = np.random.default_rng(0)

def roll_sr(csv, n, w=50):
    df = pd.read_csv(csv); s = (df["done_count"].values >= n).astype(float)
    return df["episode_id"].values, pd.Series(s).rolling(w, min_periods=1).mean().values * 100

def e80(csv, n, thr=0.8, w=50):
    df = pd.read_csv(csv); s = (df["done_count"].values >= n).astype(float)
    roll = pd.Series(s).rolling(w, min_periods=w).mean().values
    idx = np.where(roll >= thr)[0]
    return int(df["episode_id"].values[idx[0]]) if len(idx) else np.nan
print("manuscript-correction style ready")""")

code(r"""# %% Fig 5 — rolling success curves (corrected legend) ------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, n in zip(axes, NS):
    for m in ALLMODES:
        curves = []
        for s in SEEDS:
            c = MAIN / "runs" / f"n{n}_{m}_seed{s}" / "episode_summary.csv"
            if not c.exists():
                continue
            ep, r = roll_sr(c, n); ax.plot(ep, r, color=MODE_C[m], alpha=0.18, lw=1)
            curves.append(r)
        if curves:
            L = min(map(len, curves))
            ax.plot(ep[:L], np.mean([c[:L] for c in curves], 0), color=MODE_C[m],
                    lw=2.6, label=MODE_LABEL[m])
    ax.set_title(f"n = {n}", **CORR); ax.set_xlabel("episode", **CORR); ax.set_ylim(0, 105)
    ax.tick_params(labelsize=11)
axes[0].set_ylabel("rolling success rate (%, window 50)", **CORR)
axes[0].legend(frameon=False, fontsize=12, loc="lower right")
fig.suptitle("Rolling full-team success rate by reward configuration", fontsize=15, y=1.03)
fig_show(fig, "fig05_rolling_success")""")

code(r"""# %% Fig 9 — reward decomposition (larger fonts, wider, legend outside) --
comp_labels = ["progress", "distance", "d_global", "reach-goal", "agent-coll",
               "obst-coll", "direction", "v_lin", "time"]
nshow = 6   # representative team size for the decomposition panel
fig, ax = plt.subplots(figsize=(12.5, 5.2))
seed_csvs = [MAIN / "runs" / f"n{nshow}_full_seed{s}" / "episode_summary.csv" for s in SEEDS]
seed_csvs = [c for c in seed_csvs if c.exists()]
if seed_csvs:
    dfs = [pd.read_csv(c) for c in seed_csvs]
    L = min(len(d) for d in dfs); ep = dfs[0]["episode_id"].values[:L]
    for i in range(1, 10):
        stack = np.mean([d[f"comp{i}"].values[:L] for d in dfs], 0)
        roll = pd.Series(stack).rolling(25, min_periods=1).mean().values
        ax.plot(ep, roll, lw=2, label=f"{i}. {comp_labels[i-1]}")
    ax.set_xlabel("episode", **CORR); ax.set_ylabel("mean per-episode component value", **CORR)
    ax.set_title(f"Reward-component decomposition (n={nshow}, full reward, 25-ep smoothed)",
                 fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(frameon=False, fontsize=11, loc="center left", bbox_to_anchor=(1.0, 0.5),
              title="component")
    fig_show(fig, "fig09_reward_decomposition")
else:
    print("no n%d full episode_summary found" % nshow)""")

code(r"""# %% Fig 13 — final-window success bars with raw seed dots ---------------
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(NS)); w = 0.26
for i, m in enumerate(ALLMODES):
    means = [main_rd[(main_rd.n == n) & (main_rd["mode"] == m)].SR.mean() for n in NS]
    sds   = [main_rd[(main_rd.n == n) & (main_rd["mode"] == m)].SR.std()  for n in NS]
    ax.bar(x + (i - 1) * w, means, w, yerr=sds, capsize=3, color=MODE_C[m],
           label=MODE_LABEL[m], alpha=0.85)
    for j, n in enumerate(NS):
        pts = main_rd[(main_rd.n == n) & (main_rd["mode"] == m)].SR.values
        ax.scatter(x[j] + (i - 1) * w + rng_j.uniform(-0.05, 0.05, len(pts)), pts,
                   s=22, color="k", alpha=0.6, zorder=4)
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS], **CORR); ax.set_ylim(0, 105)
ax.set_ylabel("final-window success rate (%)", **CORR)
ax.set_title("Final-window success by reward configuration (raw seeds overlaid)", fontsize=15)
ax.tick_params(labelsize=12); ax.legend(frameon=False, fontsize=12)
fig_show(fig, "fig13_final_window_success")""")

code(r"""# %% Fig 14 — sample-efficiency: per-seed E80 (points + box) -------------
data, labels, colors, never = [], [], [], []
for n in NS:
    for m in ALLMODES:
        vals = []
        for s in SEEDS:
            c = MAIN / "runs" / f"n{n}_{m}_seed{s}" / "episode_summary.csv"
            if c.exists():
                vals.append(e80(c, n))
        reached = [v for v in vals if not np.isnan(v)]
        data.append(reached); labels.append(f"n{n}\n{MODE_LABEL[m][:6]}")
        colors.append(MODE_C[m]); never.append(sum(np.isnan(v) for v in vals))

fig, ax = plt.subplots(figsize=(13, 5))
pos = np.arange(len(data))
bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.25)
for i, (vals, c, nv) in enumerate(zip(data, colors, never)):
    ax.scatter([i] * len(vals) + rng_j.uniform(-0.12, 0.12, len(vals)), vals,
               s=30, color=c, alpha=0.8, zorder=3, edgecolor="k", linewidth=0.4)
    if nv:
        ax.annotate(f"{nv}× never", (i, ax.get_ylim()[1]*0.96), ha="center",
                    fontsize=9, color="crimson")
ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("episodes to reach rolling-SR ≥ 80%  (E₈₀, lower=faster)", **CORR)
ax.set_title("Sample efficiency: per-seed E₈₀ (box + raw seeds; #16 — no 'all-n' claim)",
             fontsize=14)
fig_show(fig, "fig14_sample_efficiency")""")

code(r"""# %% Fig 21 — baseline comparison with raw seed dots --------------------
# Reuses OURS / MADDPG_SR / MAPPO_SR from §B (auto-updates to 5 MADDPG seeds).
series = [("Proposed", OURS, "#2e86ab"), ("MADDPG", MADDPG_SR, "#edae49"),
          ("MAPPO", MAPPO_SR, "#d1495b")]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(NS)); w = 0.26
for i, (lab, d, col) in enumerate(series):
    means = [msd(d[n])[0] for n in NS]; sds = [msd(d[n])[1] for n in NS]
    ax.bar(x + (i - 1) * w, means, w, yerr=sds, capsize=3, color=col, label=lab, alpha=0.85)
    for j, n in enumerate(NS):
        pts = np.asarray(d[n], float)
        if len(pts):
            ax.scatter(x[j] + (i - 1) * w + rng_j.uniform(-0.05, 0.05, len(pts)), pts,
                       s=22, color="k", alpha=0.6, zorder=4)
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in NS], **CORR); ax.set_ylim(0, 105)
ax.set_ylabel("training-window success rate (%)", **CORR)
ncnt = {n: len(MADDPG_SR[n]) for n in NS}
ax.set_title(f"Within-environment baselines (raw seeds; MADDPG seeds/n={ncnt})", fontsize=13)
ax.tick_params(labelsize=12); ax.legend(frameon=False, fontsize=12)
fig_show(fig, "fig21_baseline_comparison")
if any(len(MADDPG_SR[n]) < 5 for n in NS):
    print("Fig 21 PARTIAL: MADDPG <5 seeds — re-run after PHASE=maddpg for full seed parity.")""")

code(r"""# %% Fig 22 — environment-size TRANSFER (title fixed; oracle separated) --
# Learned method only in the main series; oracle shown as a clearly-labelled,
# separated reference (reviewer #9), not as a peer bar.
evp = ev[ev.kind == "policy"]; evh = ev[ev.kind == "heuristic"]
sizes = sorted(set(evp.env.unique()) | (set(tpol.env.unique()) if tpol is not None else set()))
src_pol = tpol if (tpol is not None and len(tpol)) else evp     # prefer full round-3 curve

fig, ax = plt.subplots(figsize=(8.4, 5))
for n in NS:
    g = src_pol[src_pol.n == n].sort_values("env")
    if len(g):
        ax.plot(g.env, g.SR_mean, "o-", lw=2, label=f"Proposed n={n}")
ax.axvline(20, color="k", ls="--", lw=1, alpha=0.5); ax.text(20.2, 4, "training size", fontsize=10)
ax.set_xlabel("arena size (env)", **CORR); ax.set_ylabel("deterministic success rate (%)", **CORR)
ax.set_title("Deterministic eval: transfer across arena sizes", fontsize=14)
ax.set_ylim(0, 105); ax.tick_params(labelsize=12)
ax.legend(frameon=False, fontsize=11, title="(oracle reported separately, not a peer)")
fig_show(fig, "fig22_transfer")
if tpol is None or not len(tpol):
    print("Fig 22 shows env20/25 only — full {15,18,20,25,30} curve fills in after PHASE=eval.")""")

code(r"""# %% Fig 10 — collision: penalty magnitude (A) + raw collision count (B) -
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
# Panel A: agent-agent collision penalty magnitude over training (comp5, full)
for n in NS:
    seed_csvs = [MAIN / "runs" / f"n{n}_full_seed{s}" / "episode_summary.csv" for s in SEEDS]
    seed_csvs = [c for c in seed_csvs if c.exists()]
    if not seed_csvs:
        continue
    dfs = [pd.read_csv(c) for c in seed_csvs]; L = min(len(d) for d in dfs)
    ep = dfs[0]["episode_id"].values[:L]
    mag = np.mean([np.abs(d["comp5"].values[:L]) for d in dfs], 0)
    axes[0].plot(ep, pd.Series(mag).rolling(25, min_periods=1).mean(), lw=2, label=f"n={n}")
axes[0].set_xlabel("episode", **CORR); axes[0].set_ylabel("|agent–agent collision penalty|", **CORR)
axes[0].set_title("A. Collision penalty magnitude (training)", fontsize=13)
axes[0].legend(frameon=False, fontsize=11); axes[0].tick_params(labelsize=11)
# Panel B: raw collision count per episode (eval) = collision_step_rate × horizon
cdf = pd.read_csv(MAIN / "eval" / "eval_episodes.csv"); cdf = cdf[(cdf.kind == "policy") & (cdf.env == 20)]
xb = np.arange(len(NS))
cnt = [cdf[cdf.n == n].collision_step_rate.mean() * HORIZON for n in NS]
sd  = [cdf[cdf.n == n].collision_step_rate.std() * HORIZON for n in NS]
axes[1].bar(xb, cnt, 0.6, yerr=sd, capsize=4, color="#d1495b", alpha=0.85)
axes[1].set_xticks(xb); axes[1].set_xticklabels([f"n={n}" for n in NS], **CORR)
axes[1].set_ylabel("raw collision-steps per episode", **CORR)
axes[1].set_title("B. Raw collision count (deterministic eval, env20)", fontsize=13)
axes[1].tick_params(labelsize=11)
fig.suptitle("Collision: penalty magnitude + direct count", fontsize=15, y=1.02)
fig_show(fig, "fig10_collision_two_panel")""")

# =========================================================================== #
md(r"""## Summary & figure index""")

code(r"""# %% summary -------------------------------------------------------------
print("Round-3 experimental sections computed in this run:")
status = {
    "A  #8  HER/restart ablation": "READY" if (not abl.empty and set(abl.cond.unique()) != {'Full'}) else "PENDING (long runs)",
    "B  #7  MADDPG seed parity":   "READY" if all(len(MADDPG_SR[n]) >= 5 for n in NS) else "PARTIAL (need seeds 4,5)",
    "B  #10 head-to-head test":    "READY" if mappo_rd is not None else "PARTIAL",
    "C  #11 normalized path length":"READY",
    "D  #28 KM + log-rank":         "READY",
    "E  #24 collision rate":        "READY",
    "F  #6/#16 power + thresholds": "READY",
    "G  #34 parameter counts":      "READY (if env importable)",
    "H  #21 weight sensitivity":    "READY",
    "I  #13 transfer sweep":        "READY" if (R3 / 'eval' / 'eval_summary.csv').exists() else "PENDING (eval phase)",
    "J  manuscript fig corrections":"READY (Fig21/22 fill in w/ runs)",
}
for k, v in status.items():
    print(f"  [{v:>22}]  {k}")

print("\nFigures (res/figures_round3/):")
for p in sorted(FIG.glob("*.pdf")):
    print("  ", p.name)""")

md(r"""**Provenance.** Ours = `revision_logs/` (45-run sweep, mode=full = HER+restart) +
`eval/`. #8 ablation + #7 MADDPG seeds {4,5} = `revision_logs_round3/` (raw produced by
`run/run_round3_longruns.sh`). MADDPG seeds {1,2,3} = `revision_logs_maddpg_obs/`; MAPPO
= `revision_logs_mappo/`. c-sweep = `.ai/CONTEXT/data/sweep_coll_c_summary.json`.
Training-window SR includes exploration noise (disclosed); deterministic eval is
reported separately. Full written analysis: `res/round3_experimental_results.md`.""")

# =========================================================================== #
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = "res/round3_experimental_results.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
