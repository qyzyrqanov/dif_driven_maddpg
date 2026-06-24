"""Build the compact supplementary-materials zip (data + figures only).

Bundles the multi-seed training logs, the deterministic transfer evaluation, the
per-run aggregates, and the figure bundle into a single self-contained archive.
Reader-facing content only: light logs, eval CSVs, figures, and a neutral
README. No notebooks and no scripts are shipped.

Each per-run light log = ``episode_summary.csv`` (per-episode totals: 9 reward
components + done_count) + ``meta.json`` (algorithm, env, config, seed, timings).

Run from repo root:
    python tools/build_round3_supplementary_zip.py
"""
from __future__ import annotations
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_ZIP = REPO / "res" / "supplementary" / "coverage_marl_supplementary_compact.zip"
STAGE   = REPO / "res" / "supplementary" / "_stage_round3"

R3      = REPO / "revision_logs_round3"           # ablation + maddpg seeds 4,5 + eval
MADDPG3 = REPO / "revision_logs_maddpg_obs"        # maddpg seeds 1-3
MAIN    = REPO / "revision_logs"                   # main sweep (full-method reference)
FIGURES = REPO / "res" / "figures_round3"

LIGHT_FILTER = lambda p: p.name in ("episode_summary.csv", "meta.json")


def write_sanitized_json(src: Path, dst: Path):
    """Copy a JSON file, dropping any string value that is an absolute filesystem
    path. Those carry only local build-machine provenance (and the internal
    artifact-tree names); every scientific/config field is kept."""
    d = json.loads(src.read_text())
    clean = {k: v for k, v in d.items()
             if not (isinstance(v, str) and v.startswith("/"))}
    dst.write_text(json.dumps(clean, indent=2))


def copy_light(src: Path, dst: Path):
    """Copy a per-run light-log file, sanitizing meta.json paths en route."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.name == "meta.json":
        write_sanitized_json(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_runs(src_runs: Path, dst: Path, accept) -> int:
    """Copy light-log files for every run dir under src_runs whose name accept()s."""
    if not src_runs.exists():
        return 0
    n = 0
    for run in sorted(src_runs.iterdir()):
        if not run.is_dir() or not accept(run.name):
            continue
        for f in run.iterdir():
            if f.is_file() and LIGHT_FILTER(f):
                copy_light(f, dst / run.name / f.name)
                n += 1
    return n


def copy_tree(src: Path, dst: Path, name_filter=None) -> int:
    if not src.exists():
        return 0
    n = 0
    for p in src.rglob("*"):
        if p.is_dir() or (name_filter and not name_filter(p)):
            continue
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


README = """\
# Supplementary materials — cooperative multi-landmark coverage

Multi-seed training logs, deterministic transfer evaluation, per-run aggregates,
and figures for the differential-drive coverage experiments.

## Layout
- `ablation_logs/` — HER and orbit-restart ablation. 30 runs: two rungs
  (`abl_noRestart_*` = HER only; `abl_noHER_*` = neither HER nor restart) ×
  team size n ∈ {4,5,6} × 5 seeds.
- `proposed_full_logs/` — proposed method (HER + orbit-restart), n ∈ {4,5,6} ×
  5 seeds. The full-method reference for the ablation rungs.
- `maddpg_baseline_logs/` — fixed-CTDE MADDPG baseline, n ∈ {4,5,6} × 5 seeds.
- `transfer_eval/` — deterministic evaluation across arena sizes
  env ∈ {15,18,20,25,30} (training size env=20). Per-episode and per-run CSVs.
- `aggregates/` — per-run summary table and dataset manifest.
- `figures/` — figure bundle (PDF).

## Per-run light log
- `episode_summary.csv` — per-episode totals (9 reward components + `done_count`).
- `meta.json` — algorithm, environment, configuration, seed, timings.

## Protocol
- 1000 episodes per run, horizon 500 steps, `env_size=20` (training),
  `v_ang_max=π/2`, `num_obstacles=0`.
- Headline metric = training-window full-team success rate (with exploration
  noise). The transfer evaluation is deterministic (no exploration noise).

## 9 reward components in `episode_summary.csv`
1. progressive · 2. distance · 3. base/d_global (alpha=0, dropped) ·
4. reached-goal · 5. agent-agent collision · 6. obstacle/inactive-agent
collision · 7. linear velocity · 8. directional alignment · 9. time.
Full-reward alpha = (1, 1, 0, 10, 10, 10, 1, 1, 1).
"""


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    counts = {}

    # 1. HER/orbit-restart ablation rungs — 30 runs
    counts["ablation_logs"] = copy_runs(
        R3 / "runs", STAGE / "ablation_logs",
        accept=lambda nm: nm.startswith("abl_"))

    # 2. full-method reference (HER + restart) = published main sweep, full mode
    counts["proposed_full_logs"] = copy_runs(
        MAIN / "runs", STAGE / "proposed_full_logs",
        accept=lambda nm: nm.startswith("n") and "_full_" in nm
                          and nm.endswith(("seed1", "seed2", "seed3", "seed4", "seed5")))

    # 3. MADDPG baseline — full 5 seeds (seeds 1-3 + seeds 4,5)
    m = copy_runs(MADDPG3 / "runs", STAGE / "maddpg_baseline_logs",
                  accept=lambda nm: nm.startswith("maddpg_obs_"))
    m += copy_runs(R3 / "runs", STAGE / "maddpg_baseline_logs",
                   accept=lambda nm: nm.startswith("maddpg_obs_"))
    counts["maddpg_baseline_logs"] = m

    # 4. transfer eval (distilled CSVs)
    counts["transfer_eval"] = copy_tree(R3 / "eval", STAGE / "transfer_eval")

    # 5. aggregates (MANIFEST.json sanitized of absolute paths)
    agg = STAGE / "aggregates"; agg.mkdir(parents=True, exist_ok=True)
    if (R3 / "run_details.csv").exists():
        shutil.copy2(R3 / "run_details.csv", agg / "run_details.csv")
    if (R3 / "MANIFEST.json").exists():
        write_sanitized_json(R3 / "MANIFEST.json", agg / "MANIFEST.json")

    # 6. figures
    counts["figures"] = copy_tree(
        FIGURES, STAGE / "figures",
        name_filter=lambda p: p.suffix in (".pdf", ".png"))

    # 7. README
    (STAGE / "README.md").write_text(README)
    (STAGE / "MANIFEST.json").write_text(json.dumps({"counts": counts}, indent=2))

    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    print(f"Writing zip -> {OUT_ZIP}")
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(STAGE)))
    print(f"Done. {OUT_ZIP}  ({OUT_ZIP.stat().st_size/1e6:.2f} MB)")
    print("Counts:", counts)


if __name__ == "__main__":
    main()
