"""Compact supplementary-materials zip — light logs + figures + scripts only.

Drops the bulky per-step training CSVs (~5.4 GB raw) that dominated the original
``multiseed_supplementary.zip``. Keeps everything needed to verify the
manuscript numbers and regenerate the figures from already-aggregated data.

Contents:
- ``training_logs/n{N}_{mode}_seed{S}/``  per-run light log
    - ``episode_summary.csv`` (per-episode totals, 9 reward components + done_count)
    - ``meta.json``           (algorithm, env, config, seed, timings)
- ``maddpg_baseline_logs/n{N}_full_seed{S}/`` same shape, fixed CTDE MADDPG baseline
- ``eval/``                   distilled deterministic-eval CSVs (env20 + env25 + heuristic oracle)
- ``aggregates/``             per-run + per-cell summary CSVs, MANIFEST.json
- ``figures/``                Figures 6-12 (PDF + PNG) + reports + notebook + CSV summaries
- ``main_notebook/revision_final_results.ipynb`` rendered manuscript notebook
- ``scripts/``                plotting + aggregation scripts
- ``README.md``               this archive's index

Run from repo root:
    python tools/build_supplementary_compact_zip.py
"""
from __future__ import annotations
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_ZIP = REPO / "res" / "supplementary" / "multiseed_supplementary_compact.zip"
STAGE   = REPO / "res" / "supplementary" / "_stage_compact"

LIGHT     = REPO / "revision_logs"
MADDPG    = REPO / "revision_logs_maddpg_obs"
FIGURES   = REPO / "res" / "figures_pdf" / "multiseed_diagnostics"
MAIN_NB   = REPO / "res" / "revision_final_results.ipynb"
TOOLS_DIR = REPO / "tools"

KEEP_SCRIPTS = [
    "plot_multiseed_diagnostics.py",
    "build_supplementary_compact_zip.py",
    "build_supplementary_zip.py",
    "final_aggregate.py",
    "create_final_notebook.py",
    "export_light_logs.py",
    "aggregate_eval.py",
]


def copy_tree(src: Path, dst: Path, name_filter=None):
    if not src.exists():
        return 0
    n = 0
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        if name_filter and not name_filter(p):
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    counts = {}

    # 1. main-sweep light logs
    counts["training_logs"] = copy_tree(
        LIGHT / "runs", STAGE / "training_logs",
        name_filter=lambda p: p.name in ("episode_summary.csv", "meta.json"),
    )

    # 2. eval distilled CSVs
    counts["eval"] = copy_tree(LIGHT / "eval", STAGE / "eval")

    # 3. aggregates at the top of revision_logs
    agg_dst = STAGE / "aggregates"
    agg_dst.mkdir(parents=True, exist_ok=True)
    for name in ("run_details.csv", "MANIFEST.json"):
        src = LIGHT / name
        if src.exists():
            shutil.copy2(src, agg_dst / name)
    if (LIGHT / "res").exists():
        counts["aggregates_extra"] = copy_tree(LIGHT / "res", agg_dst / "res")

    # 4. MADDPG baseline light logs
    counts["maddpg_baseline_logs"] = copy_tree(
        MADDPG, STAGE / "maddpg_baseline_logs",
        name_filter=lambda p: p.suffix in (".csv", ".json"),
    )

    # 5. figures + reports (multiseed diagnostics) — skip the cache pickle
    counts["figures"] = copy_tree(
        FIGURES, STAGE / "figures",
        name_filter=lambda p: p.suffix in (".pdf", ".png", ".csv", ".md", ".ipynb"),
    )

    # 6. main manuscript notebook
    if MAIN_NB.exists():
        dst = STAGE / "main_notebook" / "revision_final_results.ipynb"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MAIN_NB, dst)

    # 7. scripts
    scripts_dst = STAGE / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    for name in KEEP_SCRIPTS:
        src = TOOLS_DIR / name
        if src.exists():
            shutil.copy2(src, scripts_dst / name)

    readme = STAGE / "README.md"
    readme.write_text(
        "# Supplementary materials (compact) — multi-seed coverage-MARL revision\n\n"
        "This compact archive contains only light logs, aggregates, distilled eval data, "
        "and the figure bundle. The bulky per-step training CSVs (~5.4 GB raw) are "
        "available in the separate `multiseed_supplementary.zip` archive.\n\n"
        "## Layout\n"
        "- `training_logs/n{N}_{mode}_seed{S}/`\n"
        "  - `episode_summary.csv` — per-episode totals (9 reward components + `done_count`).\n"
        "  - `meta.json` — algorithm, env, config, seed, timings.\n"
        "- `maddpg_baseline_logs/...` — same shape, fixed CTDE MADDPG baseline (9 runs).\n"
        "- `eval/` — distilled deterministic-eval CSVs (env20, env25, heuristic oracle).\n"
        "- `aggregates/` — `run_details.csv`, `MANIFEST.json`, and per-cell summaries.\n"
        "- `figures/` — Figures 6-12 (PDF + PNG), executed notebook, reports, summary CSVs.\n"
        "- `main_notebook/revision_final_results.ipynb` — manuscript-results notebook.\n"
        "- `scripts/` — plotting and aggregation scripts.\n\n"
        "## Sweep coverage\n"
        "- Main: 45 runs = n ∈ {4,5,6} × mode ∈ {full, ablation, nocoll} × 5 seeds.\n"
        "- MADDPG baseline: 9 runs = n ∈ {4,5,6} × `full` × 3 seeds.\n"
        "- 1000 episodes/run, horizon 500 steps, `env_size=20`, `v_ang_max=π/2`, `num_obstacles=0`.\n\n"
        "## 9 reward components in CSVs\n"
        "1. progressive — 2. distance — 3. base/d_global (alpha=0, dropped) — "
        "4. reached-goal — 5. agent-agent collision — 6. obstacle/inactive-agent collision — "
        "7. linear velocity — 8. directional alignment — 9. time.\n"
        "Full-reward alpha = (1, 1, 0, 10, 10, 10, 1, 1, 1).\n"
    )

    # quick manifest
    manifest = {"counts": counts}
    (STAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    print(f"Writing zip -> {OUT_ZIP}")
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(STAGE)))
    print(f"Done. {OUT_ZIP}  ({OUT_ZIP.stat().st_size/1e6:.2f} MB)")
    print(f"Stage retained at: {STAGE}")
    print("Counts:", counts)


if __name__ == "__main__":
    main()
