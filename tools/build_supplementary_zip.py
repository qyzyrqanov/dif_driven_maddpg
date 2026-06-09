"""Build a supplementary-materials zip bundling all per-step training reward CSVs
(45 runs × 5 seeds × {full, ablation, nocoll} × n∈{4,5,6}) plus light-log
episode summaries, run metadata, the multi-seed diagnostic figures, and a
README. Per-step CSVs are gzipped individually (CSVs compress ~10-20x);
the outer container is zip (stored, since contents are already compressed).

Run from repo root:
    python tools/build_supplementary_zip.py
"""
from __future__ import annotations
import json
import shutil
import gzip
import zipfile
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = Path("/media/abz/Z7S/experiments_revision_offline_replay_restart_v3/runs")
LIGHT_ROOT = REPO / "revision_logs" / "runs"
FIGURES_DIR = REPO / "res" / "figures_pdf" / "multiseed_diagnostics"
STAGE = REPO / "res" / "supplementary" / "_stage"
OUT_ZIP = REPO / "res" / "supplementary" / "multiseed_supplementary.zip"

NS = [4, 5, 6]
MODES = ["full", "ablation", "nocoll"]
SEEDS = [1, 2, 3, 4, 5]


def find_per_step_csv(d: Path, n: int) -> Path | None:
    for name in ("rewards.csv", f"result{n}.csv"):
        p = d / name
        if p.exists():
            return p
    return None


def sha256(p: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def gzip_to(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    manifest = {"runs": [], "figures": [], "notes": [
        "Per-step training-reward CSVs (one per run) are gzipped; columns: "
        "total_step, episode_id, timestep, done_count, hung_dist_agent{i}, "
        "agent{i}_vel_lin, agent{i}_vel_ang, agent{i}_comp{1..9}.",
        "9 reward components in CSV order: 1 progressive, 2 distance, "
        "3 base/d_global (alpha=0), 4 reached-goal, 5 agent-agent collision, "
        "6 obstacle/inactive-agent collision, 7 linear-velocity, 8 directional, 9 time.",
        "alpha (full)={1,1,0,10,10,10,1,1,1}; ablation zeros comp1,2,3; nocoll zeros comp3,5,6.",
        "Light-log episode_summary.csv = per-episode totals of these 9 components plus done_count.",
    ]}

    print("[1/3] Staging per-run data ...")
    for n in NS:
        for mode in MODES:
            for s in SEEDS:
                run = f"n{n}_{mode}_seed{s}"
                src_d = RUNS_ROOT / run
                dst_d = STAGE / "training_logs" / run
                dst_d.mkdir(parents=True, exist_ok=True)
                entry = {"run": run, "n": n, "mode": mode, "seed": s, "files": {}}

                csv = find_per_step_csv(src_d, n)
                if csv is not None:
                    gz_dst = dst_d / "rewards.csv.gz"
                    print(f"  gzip {run}/{csv.name} ({csv.stat().st_size/1e6:.0f} MB) ...", flush=True)
                    gzip_to(csv, gz_dst)
                    entry["files"]["per_step_csv_gz"] = {
                        "path": str(gz_dst.relative_to(STAGE)),
                        "source_name": csv.name,
                        "uncompressed_bytes": csv.stat().st_size,
                        "gz_bytes": gz_dst.stat().st_size,
                    }
                else:
                    entry["files"]["per_step_csv_gz"] = None

                meta_src = src_d / "meta.json"
                if meta_src.exists():
                    shutil.copy2(meta_src, dst_d / "meta.json")
                    entry["files"]["meta"] = "meta.json"
                light_src = LIGHT_ROOT / run / "episode_summary.csv"
                if light_src.exists():
                    shutil.copy2(light_src, dst_d / "episode_summary.csv")
                    entry["files"]["episode_summary"] = "episode_summary.csv"
                manifest["runs"].append(entry)

    print("[2/3] Copying figures + report ...")
    fig_dst = STAGE / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(FIGURES_DIR.iterdir()):
        if p.suffix in (".pdf", ".png", ".csv", ".md", ".ipynb"):
            shutil.copy2(p, fig_dst / p.name)
            manifest["figures"].append(p.name)

    shutil.copy2(REPO / "tools" / "plot_multiseed_diagnostics.py",
                 STAGE / "plot_multiseed_diagnostics.py")
    shutil.copy2(REPO / "tools" / "build_supplementary_zip.py",
                 STAGE / "build_supplementary_zip.py")

    readme = STAGE / "README.md"
    readme.write_text(
        "# Supplementary materials — multi-seed coverage-MARL revision\n\n"
        "Contents:\n\n"
        "- `training_logs/n{N}_{mode}_seed{S}/`\n"
        "  - `rewards.csv.gz` — per-step training log (gzip)\n"
        "  - `episode_summary.csv` — per-episode totals (light log)\n"
        "  - `meta.json` — run metadata (algorithm, env, config, seed, timings)\n"
        "- `figures/` — Figures 6-12 PDFs/PNGs, report, notebook, CSV summaries\n"
        "- `plot_multiseed_diagnostics.py` — reusable plotting script\n"
        "- `build_supplementary_zip.py` — this archive's builder\n"
        "- `MANIFEST.json` — per-file inventory with sizes and hashes (top level only)\n\n"
        f"Runs: 45 = N×mode×seed; n∈{NS}, modes={MODES}, seeds={SEEDS}.\n"
        "Algorithm: IDDPGWithoutS (shared actor + shared decentralized critic, "
        "no global state). Env: DiffDriveParallelEnvDone, env_size=20, "
        "v_ang_max=π/2, num_obstacles=0, horizon 500. See `meta.json` per run.\n"
    )

    (STAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    print(f"[3/3] Writing zip -> {OUT_ZIP}")
    # CSVs are already gz'd; use ZIP_STORED for them. zip everything with deflate
    # by default; for .gz files use stored.
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_dir():
                continue
            arc = p.relative_to(STAGE)
            comp = zipfile.ZIP_STORED if p.suffix == ".gz" else zipfile.ZIP_DEFLATED
            zf.write(p, arcname=str(arc), compress_type=comp)

    sz = OUT_ZIP.stat().st_size
    print(f"Done. {OUT_ZIP}  ({sz/1e6:.1f} MB)")
    print(f"Stage retained at: {STAGE}")


if __name__ == "__main__":
    main()
